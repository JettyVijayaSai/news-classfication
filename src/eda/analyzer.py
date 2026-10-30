"""Exploratory Data Analysis (EDA) module for Fake News detection.

Computes summary statistics, token distributions, n-gram divergences,
lexical richness, sensationalism indicators, and readability metrics.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.data.preprocessor import TextPreprocessor


@dataclass
class EDAReport:
    """Dataclass holding structured exploratory data analysis results."""
    total_samples: int
    class_distribution: Dict[str, int]
    class_percentages: Dict[str, float]
    token_length_stats: Dict[str, Dict[str, float]]
    lexical_diversity: Dict[str, float]
    sensationalism_signals: Dict[str, Dict[str, float]]
    readability_scores: Dict[str, Dict[str, float]]
    top_fake_unigrams: List[Tuple[str, int]]
    top_real_unigrams: List[Tuple[str, int]]
    top_fake_bigrams: List[Tuple[str, int]]
    top_real_bigrams: List[Tuple[str, int]]


class NewsEDAAnalyzer:
    """Comprehensive linguistic and statistical analyzer for news articles."""

    # Curated sensationalist/clickbait trigger lexicon
    SENSATIONAL_MARKERS = {
        "shocking", "bombshell", "unbelievable", "secret", "exposed", "conspiracy",
        "hoax", "hidden", "urgent", "must-see", "mainstream media", "truth about",
        "they don't want you to know", "miracle", "banned", "proven", "leaked",
        "shocker", "cover-up", "rigged", "treason", "crisis"
    }

    # Basic sentiment polarity lexicon for fast heuristic scoring
    POSITIVE_WORDS = {
        "good", "great", "excellent", "positive", "growth", "success", "confirmed",
        "benefit", "improve", "support", "progress", "evidence", "stable", "peace",
        "agreement", "victory", "solution", "reliable", "trustworthy", "promising"
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "disaster", "crisis", "illegal", "criminal", "fraud",
        "corrupt", "death", "kill", "threat", "destroy", "scandal", "lie",
        "false", "hoax", "panic", "collapse", "fake", "treason", "failure"
    }

    def __init__(self, preprocessor: TextPreprocessor = None):
        self.preprocessor = preprocessor or TextPreprocessor()

    def _count_syllables(self, word: str) -> int:
        """Estimates syllable count using heuristic vowel group counting."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
        word = re.sub(r"^y", "", word)
        matches = re.findall(r"[aeiouy]{1,2}", word)
        return max(1, len(matches))

    def _flesch_reading_ease(self, raw_text: str) -> float:
        """Calculates Flesch Reading Ease score.
        
        Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        Higher score = easier to read (plain English: 60-70).
        """
        sentences = re.split(r"[.!?]+", raw_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_sentences = max(1, len(sentences))

        words = re.findall(r"\b[A-Za-z]+\b", raw_text)
        total_words = len(words)
        if total_words == 0:
            return 0.0

        total_syllables = sum(self._count_syllables(w) for w in words)
        score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def _calculate_sentiment_signal(self, words: List[str]) -> float:
        """Computes net sentiment polarity ratio in range [-1.0, 1.0]."""
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        denom = max(1, pos + neg)
        return round((pos - neg) / denom, 3)

    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generates n-grams from token sequence."""
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def analyze(self, df: pd.DataFrame) -> EDAReport:
        """Executes full linguistic and statistical analysis over the dataset.

        Args:
            df: DataFrame containing 'title', 'text', 'author', 'label'.

        Returns:
            Structured EDAReport with statistical metrics and top discriminative patterns.
        """
        total_samples = len(df)
        label_counts = df["label"].value_counts().to_dict()
        real_count = label_counts.get(0, 0)
        fake_count = label_counts.get(1, 0)

        distribution = {"Real (0)": real_count, "Fake (1)": fake_count}
        percentages = {
            "Real (%)": round((real_count / total_samples) * 100, 2) if total_samples > 0 else 0,
            "Fake (%)": round((fake_count / total_samples) * 100, 2) if total_samples > 0 else 0
        }

        # Token length & lexical statistics per class
        token_lengths = {"real": [], "fake": []}
        all_tokens = {"real": [], "fake": []}
        all_bigrams = {"real": [], "fake": []}
        readability = {"real": [], "fake": []}
        sentiment_scores = {"real": [], "fake": []}
        caps_ratios = {"real": [], "fake": []}
        exclamation_counts = {"real": [], "fake": []}
        sensational_counts = {"real": [], "fake": []}

        for _, row in df.iterrows():
            label_key = "fake" if row["label"] == 1 else "real"
            raw_title = str(row.get("title", ""))
            raw_text = str(row.get("text", ""))
            full_raw = f"{raw_title} {raw_text}"

            # Stylometric & sensationalism metrics
            all_words_raw = re.findall(r"\b[A-Za-z]+\b", full_raw)
            if all_words_raw:
                upper_words = sum(1 for w in all_words_raw if w.isupper() and len(w) > 1)
                caps_ratios[label_key].append(upper_words / len(all_words_raw))
            else:
                caps_ratios[label_key].append(0.0)

            exclamations = full_raw.count("!") + full_raw.count("?")
            exclamation_counts[label_key].append(exclamations)

            lower_raw = full_raw.lower()
            marker_hits = sum(1 for marker in self.SENSATIONAL_MARKERS if marker in lower_raw)
            sensational_counts[label_key].append(marker_hits)

            # Readability
            readability[label_key].append(self._flesch_reading_ease(full_raw))

            # Clean tokens
            cleaned = self.preprocessor.clean_text(full_raw)
            tokens = cleaned.split()
            token_lengths[label_key].append(len(tokens))
            all_tokens[label_key].extend(tokens)
            
            # Bigrams
            bigrams = self._extract_ngrams(tokens, 2)
            all_bigrams[label_key].extend(bigrams)

            # Sentiment signal
            sentiment_scores[label_key].append(self._calculate_sentiment_signal(tokens))

        # Compute summary statistics
        def summarize(series: List[float]) -> Dict[str, float]:
            if not series:
                return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            arr = np.array(series)
            return {
                "mean": round(float(np.mean(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2)
            }

        length_stats = {
            "Real News": summarize(token_lengths["real"]),
            "Fake News": summarize(token_lengths["fake"])
        }

        # Type-Token Ratio (Lexical Diversity)
        lexical_diversity = {
            "Real News (TTR)": round(len(set(all_tokens["real"])) / max(1, len(all_tokens["real"])), 4),
            "Fake News (TTR)": round(len(set(all_tokens["fake"])) / max(1, len(all_tokens["fake"])), 4)
        }

        sensational_signals = {
            "All-Caps Word Ratio": {
                "Real News": round(float(np.mean(caps_ratios["real"] or [0])), 4),
                "Fake News": round(float(np.mean(caps_ratios["fake"] or [0])), 4)
            },
            "Punctuation Intensity (!/?)": {
                "Real News": round(float(np.mean(exclamation_counts["real"] or [0])), 2),
                "Fake News": round(float(np.mean(exclamation_counts["fake"] or [0])), 2)
            },
            "Clickbait Lexicon Frequency": {
                "Real News": round(float(np.mean(sensational_counts["real"] or [0])), 2),
                "Fake News": round(float(np.mean(sensational_counts["fake"] or [0])), 2)
            },
            "Sentiment Polarity Score": {
                "Real News": round(float(np.mean(sentiment_scores["real"] or [0])), 3),
                "Fake News": round(float(np.mean(sentiment_scores["fake"] or [0])), 3)
            }
        }

        readability_scores = {
            "Flesch Reading Ease (Higher=Simpler)": {
                "Real News": round(float(np.mean(readability["real"] or [0])), 2),
                "Fake News": round(float(np.mean(readability["fake"] or [0])), 2)
            }
        }

        # Top N-grams
        top_fake_unigrams = Counter(all_tokens["fake"]).most_common(10)
        top_real_unigrams = Counter(all_tokens["real"]).most_common(10)
        top_fake_bigrams = Counter(all_bigrams["fake"]).most_common(10)
        top_real_bigrams = Counter(all_bigrams["real"]).most_common(10)

        return EDAReport(
            total_samples=total_samples,
            class_distribution=distribution,
            class_percentages=percentages,
            token_length_stats=length_stats,
            lexical_diversity=lexical_diversity,
            sensationalism_signals=sensational_signals,
            readability_scores=readability_scores,
            top_fake_unigrams=top_fake_unigrams,
            top_real_unigrams=top_real_unigrams,
            top_fake_bigrams=top_fake_bigrams,
            top_real_bigrams=top_real_bigrams
        )
