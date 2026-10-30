"""Data Ingestion, Schema Validation, and Imbalance Balancing Pipeline.

Handles loading CSV/JSON datasets, schema conformance checks, deduplication,
missing value imputation, and class balancing strategies (resampling & class weighting).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.config import ModelConfig

logger = logging.getLogger(__name__)


class NewsDataLoader:
    """Production data loading, cleaning, and dataset splitting pipeline."""

    REQUIRED_COLUMNS = ["title", "text", "label"]
    LABEL_MAPPING = {
        "fake": 1, "1": 1, 1: 1, True: 1, "unreliable": 1, "false": 1,
        "real": 0, "0": 0, 0: 0, False: 0, "reliable": 0, "true": 0
    }

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

    def load_data(
        self,
        file_path: Union[str, Path],
        handle_imbalance: Optional[str] = "resample"
    ) -> pd.DataFrame:
        """Loads dataset from CSV or JSON, performs data cleansing and normalization.

        Args:
            file_path: Path to dataset file.
            handle_imbalance: Method for addressing class skew ('resample', 'none').

        Returns:
            Cleaned and normalized pandas DataFrame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file does not exist at {path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in [".json", ".jsonl"]:
            df = pd.read_json(path, lines=(path.suffix.lower() == ".jsonl"))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info("Raw dataset loaded: %d rows, %d columns", len(df), len(df.columns))

        # Check required schema
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise KeyError(f"Dataset missing mandatory column: '{col}'. Present: {list(df.columns)}")

        # Ensure 'author' column exists
        if "author" not in df.columns:
            df["author"] = "Unknown"

        cleaned_df = self._clean_and_validate(df)

        if handle_imbalance == "resample":
            cleaned_df = self._balance_classes(cleaned_df)

        return cleaned_df

    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies schema validation, missing text handling, and deduplication.

        Args:
            df: Raw DataFrame.

        Returns:
            Validated DataFrame.
        """
        df = df.copy()

        # Fill missing values
        df["title"] = df["title"].fillna("").astype(str).str.strip()
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df["author"] = df["author"].fillna("Unknown").astype(str).str.strip()

        # Remove records where both title and text are completely empty
        initial_len = len(df)
        df = df[~((df["title"] == "") & (df["text"] == ""))]
        dropped_empty = initial_len - len(df)
        if dropped_empty > 0:
            logger.warning("Dropped %d records with missing both title and text.", dropped_empty)

        # Map labels to binary integer format (1 = Fake, 0 = Real)
        df["label"] = df["label"].map(self.LABEL_MAPPING)
        if df["label"].isna().any():
            invalid_count = df["label"].isna().sum()
            logger.warning("Dropping %d records with unmapped or invalid labels.", invalid_count)
            df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        # Deduplication based on combined title and body hash
        dedup_subset = ["title", "text"]
        before_dedup = len(df)
        df = df.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
        dropped_dups = before_dedup - len(df)
        if dropped_dups > 0:
            logger.info("Removed %d duplicate news records.", dropped_dups)

        return df

    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balances classes via stratified resampling if imbalance ratio > 1.5.

        Args:
            df: Cleaned DataFrame.

        Returns:
            Class-balanced DataFrame.
        """
        counts = df["label"].value_counts()
        if len(counts) < 2:
            return df

        min_class = counts.idxmin()
        max_class = counts.idxmax()
        ratio = counts[max_class] / counts[min_class]

        # Only balance if noticeable imbalance exists (> 1.5:1)
        if ratio > 1.5:
            logger.info("Class imbalance detected (ratio %.2f). Balancing dataset...", ratio)
            df_major = df[df["label"] == max_class]
            df_minor = df[df["label"] == min_class]

            # Upsample minority class to 85% of majority size to preserve diversity
            target_size = int(len(df_major) * 0.85)
            df_minor_upsampled = resample(
                df_minor,
                replace=True,
                n_samples=target_size,
                random_state=self.config.random_state
            )

            df_balanced = pd.concat([df_major, df_minor_upsampled]).sample(
                frac=1.0, random_state=self.config.random_state
            ).reset_index(drop=True)
            return df_balanced

        return df

    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Performs stratified train/test split.

        Args:
            df: Processed DataFrame.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X = df[["title", "text", "author"]]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        logger.info(
            "Stratified split completed: Train=%d (Fake=%d, Real=%d), Test=%d (Fake=%d, Real=%d)",
            len(X_train), (y_train == 1).sum(), (y_train == 0).sum(),
            len(X_test), (y_test == 1).sum(), (y_test == 0).sum()
        )
        return X_train, X_test, y_train, y_test
