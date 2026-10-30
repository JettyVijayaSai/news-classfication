"""Abstract base classifier interface for Fake News detection."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np


class BaseNewsClassifier(ABC):
    """Abstract Base Class for all Fake News Classifiers."""

    @abstractmethod
    def fit(self, texts: List[str], y: Union[np.ndarray, List[int]]) -> "BaseNewsClassifier":
        """Trains the model on text documents and binary labels."""
        pass

    @abstractmethod
    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """Predicts binary labels (1=Fake, 0=Real) applying the decision threshold."""
        pass

    @abstractmethod
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predicts class probabilities [P(Real), P(Fake)] of shape (N, 2)."""
        pass

    @abstractmethod
    def save(self, output_path: Union[str, Path]) -> None:
        """Serializes model artifacts to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_path: Union[str, Path]) -> "BaseNewsClassifier":
        """Deserializes and reconstructs model instance from disk."""
        pass
