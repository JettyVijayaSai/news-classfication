"""Model Explainability and Local Interpretability Module.

Implements model-agnostic local feature attribution (LIME-style token perturbation),
linear coefficient decomposition, and automated factual rationale generation.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.linear_model import Ridge

from src.models.base import BaseNewsClassifier

logger = logging.getLogger(__name__)


class NewsExplainer:
    """Interprets predictions by quantifying individual word and phrase contributions."""

    def __init__(
        self,
        model: BaseNewsClassifier,
        num_perturbations: int = 150,
        mask_prob: float = 0.25,
        random_state: int = 42
    ):
        self.model = model
        self.num_perturbations = num_perturbations
        self.mask_prob = mask_prob
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def explain_instance(
        self,
        text: str,
        num_features: int = 8
    ) -> Dict[str, Any]:
        """Explains prediction on a text document using local linear perturbation.

        Args:
            text: Cleaned or synthesized input text.
            num_features: Maximum number of influential words to extract.

        Returns:
            Dictionary with predicted class, probabilities, top driving words, and rationale.
        """
        words = text.split()
        if not words:
            return {
                "base_prediction": "Real",
                "confidence": 0.5,
                "top_fake_contributors": [],
                "top_real_contributors": [],
                "rationale": "Empty document provided; default neutral score applied."
            }

        # Predict probability for original document
        orig_proba = self.model.predict_proba([text])[0]
        fake_prob = float(orig_proba[1])
        predicted_label = "Fake" if fake_prob >= 0.5 else "Real"
        confidence = fake_prob if predicted_label == "Fake" else (1.0 - fake_prob)

        # Handle very short documents
        if len(words) <= 3:
            return {
                "base_prediction": predicted_label,
                "fake_probability": round(fake_prob, 4),
                "confidence": round(confidence, 4),
                "top_fake_contributors": [(w, 0.1) for w in words],
                "top_real_contributors": [],
                "rationale": f"Short text classified as {predicted_label} based on keyword presence."
            }

        # Fast direct explanation if underlying model has linear coefficients
        direct_explanation = self._try_direct_linear_attribution(text, num_features)
        if direct_explanation is not None:
            fake_contribs, real_contribs = direct_explanation
        else:
            # Model-agnostic perturbation explanation (LIME-style)
            fake_contribs, real_contribs = self._perturbation_attribution(words, num_features)

        # Synthesize explainability rationale
        rationale = self._generate_rationale(
            predicted_label=predicted_label,
            confidence=confidence,
            fake_contributors=fake_contribs,
            real_contributors=real_contribs
        )

        return {
            "base_prediction": predicted_label,
            "fake_probability": round(fake_prob, 4),
            "confidence": round(confidence, 4),
            "top_fake_contributors": fake_contribs,
            "top_real_contributors": real_contribs,
            "rationale": rationale
        }

    def _try_direct_linear_attribution(
        self,
        text: str,
        num_features: int
    ) -> Optional[Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]]:
        """Attempts exact linear feature attribution if model exposes TF-IDF coefficients."""
        try:
            from src.models.classical import ClassicalNewsClassifier
            if not isinstance(self.model, ClassicalNewsClassifier):
                return None

            pipeline = self.model.pipeline
            if "tfidf" not in pipeline.named_steps or "clf" not in pipeline.named_steps:
                return None

            tfidf = pipeline.named_steps["tfidf"]
            clf = pipeline.named_steps["clf"]

            # Check if classifier has direct coef_
            if hasattr(clf, "coef_"):
                coefs = clf.coef_[0]
            elif hasattr(clf, "estimators_"):  # CalibratedClassifierCV
                coefs = np.mean([e.coef_[0] for e in clf.estimators_ if hasattr(e, "coef_")], axis=0)
            else:
                return None

            # Compute TF-IDF vector for this document
            vec = tfidf.transform([text])
            feature_names = np.array(tfidf.get_feature_names_out())
            nonzero_idx = vec.indices
            nonzero_vals = vec.data

            if len(nonzero_idx) == 0:
                return [], []

            # Contribution = TF-IDF value * Weight
            contributions = nonzero_vals * coefs[nonzero_idx]
            words = feature_names[nonzero_idx]

            fake_idx = np.argsort(contributions)[::-1]
            real_idx = np.argsort(contributions)

            fake_contributors = [
                (words[i], round(float(contributions[i]), 4))
                for i in fake_idx if contributions[i] > 0
            ][:num_features]

            real_contributors = [
                (words[i], round(float(abs(contributions[i])), 4))
                for i in real_idx if contributions[i] < 0
            ][:num_features]

            return fake_contributors, real_contributors
        except Exception as e:
            logger.debug("Direct linear attribution skipped: %s", e)
            return None

    def _perturbation_attribution(
        self,
        words: List[str],
        num_features: int
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Estimates token importance using local perturbation sampling."""
        unique_words = list(dict.fromkeys(words))
        num_words = len(unique_words)
        word_to_idx = {w: i for i, w in enumerate(unique_words)}

        # Sample binary mask perturbations (1 = word included, 0 = word masked)
        binary_masks = self.rng.binomial(1, 1 - self.mask_prob, size=(self.num_perturbations, num_words))
        # Ensure row 0 is the original unperturbed document
        binary_masks[0, :] = 1

        perturbed_texts = []
        for row in binary_masks:
            active_words = {unique_words[i] for i in range(num_words) if row[i] == 1}
            perturbed = " ".join([w for w in words if w in active_words])
            perturbed_texts.append(perturbed if perturbed else "neutral")

        # Query model on perturbed texts
        preds = self.model.predict_proba(perturbed_texts)[:, 1]

        # Calculate cosine similarity distances as instance sample weights
        distances = np.sum(binary_masks != 1, axis=1) / num_words
        kernel_width = 0.75
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        # Fit local interpretable surrogate Ridge regression
        ridge = Ridge(alpha=1.0, random_state=self.random_state)
        ridge.fit(binary_masks, preds, sample_weight=weights)

        coefs = ridge.coef_
        sorted_indices = np.argsort(coefs)

        # Positive coef = drives prediction towards Fake (1)
        top_fake_idx = sorted_indices[::-1]
        fake_contributors = [
            (unique_words[i], round(float(coefs[i]), 4))
            for i in top_fake_idx if coefs[i] > 0
        ][:num_features]

        # Negative coef = drives prediction towards Real (0)
        real_contributors = [
            (unique_words[i], round(float(abs(coefs[i])), 4))
            for i in sorted_indices if coefs[i] < 0
        ][:num_features]

        return fake_contributors, real_contributors

    def _generate_rationale(
        self,
        predicted_label: str,
        confidence: float,
        fake_contributors: List[Tuple[str, float]],
        real_contributors: List[Tuple[str, float]]
    ) -> str:
        """Composes human-readable explanation synthesizing the model evidence."""
        conf_pct = round(confidence * 100, 1)

        if predicted_label == "Fake":
            if fake_contributors:
                tokens_str = ", ".join([f"'{token}' (+{score})" for token, score in fake_contributors[:4]])
                return (
                    f"Classified as Fake News with {conf_pct}% confidence. "
                    f"Key misinformation drivers identified: {tokens_str}. "
                    "These patterns correlate with sensationalist framing or unreliable sourcing."
                )
            return f"Classified as Fake News with {conf_pct}% confidence based on aggregate lexical structure."
        else:
            if real_contributors:
                tokens_str = ", ".join([f"'{token}' (+{score})" for token, score in real_contributors[:4]])
                return (
                    f"Classified as Real News with {conf_pct}% confidence. "
                    f"Strong credibility indicators detected: {tokens_str}. "
                    "Consistent with journalistic citation standards and neutral phrasing."
                )
            return f"Classified as Real News with {conf_pct}% confidence based on standard factual syntax."
