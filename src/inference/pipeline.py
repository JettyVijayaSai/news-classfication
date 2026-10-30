"""Production Inference Pipeline for Fake News Classification.

Encapsulates preprocessing, model inference, calibrated thresholding,
and explainability into a unified high-throughput predictor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib

from src.config import AppConfig, ModelConfig, PreprocessingConfig
from src.data.preprocessor import TextPreprocessor
from src.explainability.explainer import NewsExplainer
from src.models.base import BaseNewsClassifier
from src.models.classical import ClassicalNewsClassifier
from src.models.transformer import TransformerNewsClassifier

logger = logging.getLogger(__name__)


class NewsInferenceEngine:
    """Production inference engine packaging preprocessor, model, and explainer."""

    def __init__(
        self,
        model: BaseNewsClassifier,
        preprocessor: Optional[TextPreprocessor] = None,
        explainer: Optional[NewsExplainer] = None,
        decision_threshold: float = 0.42
    ):
        self.model = model
        self.preprocessor = preprocessor or TextPreprocessor()
        self.explainer = explainer or NewsExplainer(model=self.model)
        self.decision_threshold = decision_threshold

    @classmethod
    def load_from_artifacts(
        cls,
        model_path: Union[str, Path],
        preprocessor_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.42
    ) -> NewsInferenceEngine:
        """Loads serialized model and preprocessor artifacts into active inference engine."""
        m_path = Path(model_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {m_path}")

        # Try loading as classical model first, then transformer
        try:
            model = ClassicalNewsClassifier.load(m_path)
        except Exception:
            model = TransformerNewsClassifier.load(m_path)

        preprocessor = None
        if preprocessor_path and Path(preprocessor_path).exists():
            preprocessor = joblib.load(preprocessor_path)
        else:
            preprocessor = TextPreprocessor()

        explainer = NewsExplainer(model=model)
        return cls(model=model, preprocessor=preprocessor, explainer=explainer, decision_threshold=threshold)

    def predict_news(
        self,
        title: str,
        text: str,
        author: str = "Unknown",
        explain: bool = True
    ) -> Dict[str, Any]:
        """Classifies a news article and returns label, confidence, and explainability.

        Args:
            title: Headline of article.
            text: Body text of article.
            author: Byline / source author.
            explain: Whether to run feature attribution.

        Returns:
            Dictionary matching requirement:
            {
                "label": "fake" | "real",
                "confidence": float,
                "fake_probability": float,
                "real_probability": float,
                "decision_threshold": float,
                "top_contributing_keywords": [...],
                "rationale": str
            }
        """
        combined_text = self.preprocessor.combine_metadata(title=title, text=text, author=author)
        
        # Softmax probability distribution
        probs = self.model.predict_proba([combined_text])[0]
        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        # Apply threshold tuned to minimize false negatives on fake news
        is_fake = fake_prob >= self.decision_threshold
        predicted_label = "fake" if is_fake else "real"
        confidence = fake_prob if is_fake else real_prob

        top_keywords = []
        rationale = ""

        if explain:
            explanation = self.explainer.explain_instance(combined_text)
            top_keywords = (
                explanation["top_fake_contributors"]
                if is_fake
                else explanation["top_real_contributors"]
            )
            rationale = explanation["rationale"]

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "decision_threshold": self.decision_threshold,
            "top_contributing_keywords": top_keywords,
            "rationale": rationale
        }

    def batch_predict(
        self,
        items: List[Dict[str, str]],
        explain: bool = False
    ) -> List[Dict[str, Any]]:
        """Processes a batch of articles with high throughput."""
        results = []
        for item in items:
            title = item.get("title", "")
            text = item.get("text", "")
            author = item.get("author", "Unknown")
            results.append(self.predict_news(title=title, text=text, author=author, explain=explain))
        return results


# Module-level convenience singleton
_GLOBAL_ENGINE: Optional[NewsInferenceEngine] = None


def get_inference_engine(model_path: Optional[Union[str, Path]] = None) -> NewsInferenceEngine:
    """Retrieves or creates active global inference engine instance."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        target_path = Path(model_path) if model_path else AppConfig.model_artifact_path
        if not target_path.exists():
            raise FileNotFoundError(
                f"No trained model artifact found at {target_path}. "
                "Run 'python scripts/train_and_evaluate.py' first."
            )
        _GLOBAL_ENGINE = NewsInferenceEngine.load_from_artifacts(target_path)
    return _GLOBAL_ENGINE


def predict_news(title: str, text: str, author: str = "Unknown") -> Dict[str, Any]:
    """Production top-level inference function complying directly with deliverable specification.

    Args:
        title: Headline of the news article.
        text: Main body text of the news article.
        author: Byline / publisher source.

    Returns:
        dict: predicted label, confidence score, and top contributing keywords/rationale.
    """
    engine = get_inference_engine()
    return engine.predict_news(title=title, text=text, author=author, explain=True)
