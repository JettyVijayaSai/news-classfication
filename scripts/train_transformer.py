"""GPU Transformer Fine-Tuning Pipeline for Fake News Classification.

Optimized for NVIDIA RTX GPUs (such as RTX 3050 6GB):
- Mixed-Precision FP16 training with PyTorch torch.cuda.amp for fast tensor core execution
- Gradient Accumulation to support effective batch sizes within 6GB VRAM limits
- RoBERTa / DeBERTa / DistilBERT fine-tuning with attention mask management
- Validation checkpointing targeting False Negative minimization (F2 Score)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ModelConfig
from src.data.loader import NewsDataLoader
from src.data.preprocessor import TextPreprocessor
from src.models.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TRAIN_TRANSFORMER")


def check_gpu_environment():
    """Validates PyTorch CUDA environment and logs GPU hardware specs."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA is NOT detected by PyTorch. Execution will default to CPU.")
            return False, "cpu"
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        cuda_version = torch.version.cuda
        logger.info("Found GPU: %s | Total VRAM: %.0f MB | CUDA: %s", gpu_name, vram_mb, cuda_version)
        return True, "cuda"
    except ImportError:
        logger.error("PyTorch is not installed. Run: pip install torch transformers")
        return False, "none"


def train_gpu_transformer(
    dataset_path: Path,
    model_name: str = "roberta-base",
    batch_size: int = 8,
    accum_steps: int = 4,
    epochs: int = 3,
    lr: float = 2e-5,
    max_length: int = 256
):
    """Executes production fine-tuning on GPU with FP16 and gradient accumulation."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    has_gpu, device_str = check_gpu_environment()
    device = torch.device(device_str)

    logger.info("Loading dataset from %s...", dataset_path)
    loader = NewsDataLoader()
    df = loader.load_data(dataset_path, handle_imbalance="resample")
    X_train, X_test, y_train, y_test = loader.split_data(df)

    preprocessor = TextPreprocessor()
    train_texts = preprocessor.transform(X_train)
    test_texts = preprocessor.transform(X_test)
    y_train_arr = y_train.values
    y_test_arr = y_test.values

    logger.info("Loading HuggingFace Tokenizer: '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # PyTorch Dataset
    class NewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_ds = NewsDataset(train_texts, y_train_arr, tokenizer, max_length)
    test_ds = NewsDataset(test_texts, y_test_arr, tokenizer, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device_str == "cuda"),
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=(device_str == "cuda")
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // accum_steps) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = (device_str == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info("Starting GPU Training: Epochs=%d | Effective Batch Size=%d (%d x %d accum)",
                epochs, batch_size * accum_steps, batch_size, accum_steps)

    best_f2 = -1.0
    best_threshold = 0.5
    save_dir = MODELS_DIR / "transformer_roberta"
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_loss = epoch_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)

        val_probs_arr = np.array(val_probs)
        opt_th, metrics = ModelEvaluator.optimize_decision_threshold(y_test_arr, val_probs_arr, beta=2.0)

        logger.info(
            "Epoch %d/%d | Loss: %.4f | Val F1: %.4f | Val Recall: %.4f | Val F2: %.4f | ROC-AUC: %.4f | Best Thresh: %.2f",
            epoch + 1, epochs, avg_loss, metrics.f1_fake, metrics.recall_fake, metrics.f2_score, metrics.roc_auc, opt_th
        )

        if metrics.f2_score > best_f2:
            best_f2 = metrics.f2_score
            best_threshold = opt_th
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info("Saved new champion checkpoint to %s (F2: %.4f)", save_dir, best_f2)

    total_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds. Best F2: %.4f (Threshold: %.2f)", total_time, best_f2, best_threshold)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer on GPU for Fake News Classification.")
    parser.add_argument("--dataset", type=str, default=str(DATA_DIR / "sample_news.csv"))
    parser.add_argument("--model", type=str, default="roberta-base", help="HuggingFace model hub identifier")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size (fits 6GB VRAM)")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    train_gpu_transformer(
        dataset_path=Path(args.dataset),
        model_name=args.model,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
