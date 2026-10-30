"""Model Evaluation and False Negative Mitigation Module.

Computes precision, recall, F1 (macro/weighted), ROC-AUC, confusion matrix,
and optimizes decision threshold to minimize false negatives on subtle fake news.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    fbeta_score
)


@dataclass
class EvaluationMetrics:
    """Dataclass holding complete model performance metrics."""
    accuracy: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    f1_macro: float
    f1_weighted: float
    roc_auc: float
    f2_score: float  # Weights recall twice as high as precision (False Negative minimization)
    confusion_matrix: List[List[int]]
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    decision_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """Evaluates classification models with misinformation-focused error analysis."""

    @staticmethod
    def evaluate(
        y_true: Union[np.ndarray, List[int]],
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> EvaluationMetrics:
        """Evaluates model performance at specified decision threshold.

        Args:
            y_true: Ground truth binary labels (1=Fake, 0=Real).
            y_pred_proba: Predicted probabilities for class 1 (Fake), shape (N,) or (N, 2).
            threshold: Probability cutoff for classifying as Fake.

        Returns:
            EvaluationMetrics object.
        """
        y_true = np.asarray(y_true, dtype=int)
        
        # Handle 1D or 2D probability outputs
        if y_pred_proba.ndim == 2:
            fake_probs = y_pred_proba[:, 1]
        else:
            fake_probs = y_pred_proba

        y_pred = (fake_probs >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        try:
            auc = round(float(roc_auc_score(y_true, fake_probs)), 4)
        except Exception:
            auc = 0.5

        return EvaluationMetrics(
            accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
            precision_fake=round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            recall_fake=round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            f1_fake=round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            f1_macro=round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            f1_weighted=round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            roc_auc=auc,
            f2_score=round(float(fbeta_score(y_true, y_pred, beta=2.0, pos_label=1, zero_division=0)), 4),
            confusion_matrix=cm.tolist(),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_positives=int(tp),
            decision_threshold=threshold
        )

    @classmethod
    def optimize_decision_threshold(
        cls,
        y_true: Union[np.ndarray, List[int]],
        y_pred_proba: np.ndarray,
        beta: float = 2.0,
        min_threshold: float = 0.2,
        max_threshold: float = 0.7,
        steps: int = 51
    ) -> Tuple[float, EvaluationMetrics]:
        """Finds the decision threshold that maximizes F-beta score.
        
        Using beta=2.0 heavily penalizes False Negatives (critical in misinformation
        containment where undetected fake news poses higher viral risk than false alarms).

        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted fake news probabilities.
            beta: Weight of recall relative to precision (2.0 = recall weighted 2x).
            min_threshold: Minimum search boundary.
            max_threshold: Maximum search boundary.
            steps: Number of grid increments.

        Returns:
            Tuple of (optimal_threshold, EvaluationMetrics).
        """
        thresholds = np.linspace(min_threshold, max_threshold, steps)
        best_threshold = 0.5
        best_fbeta = -1.0
        best_metrics: Optional[EvaluationMetrics] = None

        if y_pred_proba.ndim == 2:
            fake_probs = y_pred_proba[:, 1]
        else:
            fake_probs = y_pred_proba

        y_true = np.asarray(y_true, dtype=int)

        for th in thresholds:
            y_pred = (fake_probs >= th).astype(int)
            fb = fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)
            if fb > best_fbeta:
                best_fbeta = fb
                best_threshold = round(float(th), 3)

        best_metrics = cls.evaluate(y_true, fake_probs, threshold=best_threshold)
        return best_threshold, best_metrics
