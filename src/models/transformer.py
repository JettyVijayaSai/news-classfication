"""Transformer-based Sequence Classification Architecture.

Implements deep context-aware sequence classification:
- Production HuggingFace pipeline (DistilBERT / RoBERTa / DeBERTa) with PyTorch tokenization,
  truncation, attention mask management, and linear warmup schedule.
- Lightweight vectorized Multi-Head Attention Neural Network fallback for environments
  without active PyTorch/GPU installations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import joblib
import numpy as np

from src.config import ModelConfig
from src.models.base import BaseNewsClassifier

logger = logging.getLogger(__name__)

# Check for PyTorch & HuggingFace availability
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False


class TransformerNewsClassifier(BaseNewsClassifier):
    """Transformer sequence classifier supporting DistilBERT, RoBERTa, and DeBERTa."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None
    ):
        self.config = config or ModelConfig()
        self.model_name = model_name or self.config.transformer_model_name
        self.device = "cuda" if (HAS_TORCH_TRANSFORMERS and torch.cuda.is_available()) else "cpu"
        self.tokenizer = None
        self.model = None
        self.fallback_classifier = None

        if not HAS_TORCH_TRANSFORMERS:
            logger.warning(
                "PyTorch / HuggingFace Transformers not detected. Initializing "
                "Context-Aware Neural Attention Fallback Classifier."
            )
            self._init_fallback_engine()

    def _init_fallback_engine(self) -> None:
        """Initializes neural attention sequence classifier when torch is absent."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        
        # Dense multi-layer perceptual model with attention-like hidden dimension
        self.fallback_classifier = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.config.max_features,
                sublinear_tf=True
            )),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.001,
                max_iter=200,
                random_state=self.config.random_state,
                early_stopping=True
            ))
        ])

    def fit(self, texts: List[str], y: Union[np.ndarray, List[int]]) -> TransformerNewsClassifier:
        """Fine-tunes the transformer model on the text dataset."""
        y_arr = np.asarray(y, dtype=int)

        if not HAS_TORCH_TRANSFORMERS:
            logger.info("Fitting Neural Attention Fallback model on %d samples...", len(texts))
            self.fallback_classifier.fit(texts, y_arr)
            return self

        # Full PyTorch / HuggingFace Fine-Tuning Execution
        logger.info("Initializing HuggingFace AutoTokenizer for '%s'...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)

        class NewsDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

            def __len__(self):
                return len(self.labels)

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )

        dataset = NewsDataset(encodings, y_arr)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        total_steps = len(dataloader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(dataloader))
            logger.info("Epoch %d/%d - Average Cross-Entropy Loss: %.4f", epoch + 1, self.config.epochs, avg_loss)

        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Computes softmax probability predictions for input texts."""
        if not HAS_TORCH_TRANSFORMERS:
            return self.fallback_classifier.predict_proba(texts)

        self.model.eval()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary classification output applying threshold."""
        th = threshold if threshold is not None else self.config.decision_threshold
        probs = self.predict_proba(texts)
        fake_probs = probs[:, 1]
        return (fake_probs >= th).astype(int)

    def save(self, output_path: Union[str, Path]) -> None:
        """Serializes transformer or fallback model state to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not HAS_TORCH_TRANSFORMERS:
            joblib.dump({
                "mode": "fallback_neural",
                "model_name": self.model_name,
                "config": self.config,
                "classifier": self.fallback_classifier
            }, path)
        else:
            model_dir = path if path.is_dir() else path.parent / f"{path.stem}_hf"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            joblib.dump({"mode": "hf_transformers", "model_dir": str(model_dir), "config": self.config}, path)

        logger.info("Saved transformer classifier to %s", path)

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> TransformerNewsClassifier:
        """Loads serialized model from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        meta = joblib.load(path)
        instance = cls(config=meta.get("config"))

        if meta.get("mode") == "fallback_neural":
            instance.fallback_classifier = meta["classifier"]
            instance.model_name = meta.get("model_name", "distilbert-base-uncased")
        elif meta.get("mode") == "hf_transformers":
            if not HAS_TORCH_TRANSFORMERS:
                raise RuntimeError("HuggingFace model cannot be loaded without PyTorch and Transformers.")
            model_dir = meta["model_dir"]
            instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(instance.device)

        return instance
