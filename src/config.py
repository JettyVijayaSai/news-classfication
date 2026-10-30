"""Configuration module for the Fake News Detection pipeline.

Centralizes paths, random seeds, hyperparameters, and feature engineering rules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Ensure artifact subdirectories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessingConfig:
    """Configuration for data cleaning, normalization, and tokenization."""
    lowercase: bool = True
    strip_html: bool = True
    remove_urls: bool = True
    remove_special_characters: bool = True
    lemmatize: bool = True
    remove_stopwords: bool = True
    custom_stopwords: List[str] = field(default_factory=lambda: [
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "but", "if"
    ])
    author_prefix: str = "[AUTHOR]"
    title_prefix: str = "[TITLE]"
    text_prefix: str = "[CONTENT]"
    min_token_length: int = 2
    max_token_length: int = 35


@dataclass
class ModelConfig:
    """Hyperparameters and configuration for modeling and evaluation."""
    random_state: int = 42
    test_size: float = 0.20
    val_size: float = 0.10
    
    # TF-IDF Settings
    max_features: int = 15000
    ngram_range: tuple = (1, 2)
    sublinear_tf: bool = True
    
    # False Negative Mitigation: Optimal threshold for fake news detection
    # fake=1, real=0. Lower threshold increases recall for fake news (minimizes FN)
    decision_threshold: float = 0.42
    
    # Transformer Settings
    transformer_model_name: str = "distilbert-base-uncased"
    max_seq_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01


@dataclass
class AppConfig:
    """Service host and environment settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    model_artifact_path: Path = MODELS_DIR / "best_model.joblib"
    preprocessor_artifact_path: Path = MODELS_DIR / "preprocessor.joblib"
    explainer_artifact_path: Path = MODELS_DIR / "explainer.joblib"
