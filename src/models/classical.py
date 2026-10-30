"""Classical Machine Learning Models for Fake News Classification.

Implements TF-IDF vectorization paired with:
1. Logistic Regression (L2 regularized, balanced class weighting)
2. Calibrated Linear Support Vector Classifier (Platt scaling for probability calibration)
3. Multinomial Naive Bayes
4. Soft-Voting / Stacking Ensemble
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import ModelConfig
from src.models.base import BaseNewsClassifier

logger = logging.getLogger(__name__)


class ClassicalNewsClassifier(BaseNewsClassifier):
    """Unified wrapper for TF-IDF based classical machine learning classifiers."""

    SUPPORTED_ALGORITHMS = ["logistic_regression", "linear_svc", "naive_bayes", "ensemble"]

    def __init__(
        self,
        algorithm: str = "logistic_regression",
        config: Optional[ModelConfig] = None
    ):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm '{algorithm}' not in {self.SUPPORTED_ALGORITHMS}")

        self.algorithm = algorithm
        self.config = config or ModelConfig()
        self.pipeline: Optional[Pipeline] = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Constructs scikit-learn Pipeline with TF-IDF vectorizer and estimator."""
        vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            max_features=self.config.max_features,
            sublinear_tf=self.config.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b"
        )

        if self.algorithm == "logistic_regression":
            estimator = LogisticRegression(
                C=1.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state,
                solver="lbfgs"
            )
        elif self.algorithm == "linear_svc":
            base_svc = LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=self.config.random_state,
                dual=False
            )
            # CalibratedClassifierCV wraps LinearSVC to output well-calibrated posterior probabilities
            estimator = CalibratedClassifierCV(estimator=base_svc, cv=3)
        elif self.algorithm == "naive_bayes":
            estimator = MultinomialNB(alpha=0.5)
        elif self.algorithm == "ensemble":
            # Soft-voting requires calibrated probabilities from each constituent
            from sklearn.ensemble import VotingClassifier
            lr = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", random_state=self.config.random_state)
            svc = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=self.config.random_state, dual=False), cv=3)
            nb = MultinomialNB(alpha=0.5)
            estimator = VotingClassifier(
                estimators=[("lr", lr), ("svc", svc), ("nb", nb)],
                voting="soft"
            )

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", estimator)
        ])

    def fit(self, texts: List[str], y: Union[np.ndarray, List[int]]) -> ClassicalNewsClassifier:
        """Fits TF-IDF vectorizer and classification model.

        Args:
            texts: Preprocessed document strings.
            y: Binary target labels (1=Fake, 0=Real).

        Returns:
            self
        """
        logger.info("Training %s classifier on %d samples...", self.algorithm, len(texts))
        self.pipeline.fit(texts, y)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predicts class probabilities [P(Real), P(Fake)].

        Args:
            texts: List of preprocessed texts.

        Returns:
            Probability matrix of shape (N, 2).
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary labels using configurable decision threshold.

        Args:
            texts: List of preprocessed texts.
            threshold: Probability threshold for Fake label (defaults to config threshold).

        Returns:
            Binary predictions array of shape (N,).
        """
        th = threshold if threshold is not None else self.config.decision_threshold
        probs = self.predict_proba(texts)
        fake_probs = probs[:, 1]
        return (fake_probs >= th).astype(int)

    def get_feature_importances(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Extracts top indicative words for Real and Fake news (for linear models)."""
        if self.algorithm not in ["logistic_regression", "naive_bayes"]:
            return {"fake_indicators": [], "real_indicators": []}

        tfidf: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        feature_names = np.array(tfidf.get_feature_names_out())

        if self.algorithm == "logistic_regression":
            coefs = clf.coef_[0]
            top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
            top_real_idx = np.argsort(coefs)[:top_n]
            return {
                "fake_indicators": [(feature_names[i], round(float(coefs[i]), 4)) for i in top_fake_idx],
                "real_indicators": [(feature_names[i], round(float(abs(coefs[i])), 4)) for i in top_real_idx]
            }
        elif self.algorithm == "naive_bayes":
            # Log-odds ratio: log P(w|fake) - log P(w|real)
            log_odds = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            top_fake_idx = np.argsort(log_odds)[-top_n:][::-1]
            top_real_idx = np.argsort(log_odds)[:top_n]
            return {
                "fake_indicators": [(feature_names[i], round(float(log_odds[i]), 4)) for i in top_fake_idx],
                "real_indicators": [(feature_names[i], round(float(abs(log_odds[i])), 4)) for i in top_real_idx]
            }

    def save(self, output_path: Union[str, Path]) -> None:
        """Saves fitted pipeline to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"algorithm": self.algorithm, "config": self.config, "pipeline": self.pipeline}, path)
        logger.info("Saved classical model (%s) to %s", self.algorithm, path)

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> ClassicalNewsClassifier:
        """Loads fitted pipeline from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        instance = cls(algorithm=data["algorithm"], config=data["config"])
        instance.pipeline = data["pipeline"]
        return instance
