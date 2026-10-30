"""CLI script to run Exploratory Data Analysis (EDA) on news dataset."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, REPORTS_DIR
from src.data.loader import NewsDataLoader
from src.eda.analyzer import NewsEDAAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EDA")


def main():
    data_file = DATA_DIR / "sample_news.csv"
    if not data_file.exists():
        logger.error("Dataset not found at %s", data_file)
        sys.exit(1)

    logger.info("Loading news dataset from %s...", data_file)
    loader = NewsDataLoader()
    df = loader.load_data(data_file, handle_imbalance="none")

    logger.info("Executing comprehensive linguistic and statistical EDA...")
    analyzer = NewsEDAAnalyzer()
    report = analyzer.analyze(df)

    print("\n" + "=" * 70)
    print("           EXPLORATORY DATA ANALYSIS (EDA) REPORT           ")
    print("=" * 70)
    print(f"Total Analyzed Samples: {report.total_samples}")
    print(f"Class Balance: {report.class_distribution} ({report.class_percentages})")
    print("\n--- TOKEN LENGTH STATISTICS ---")
    for group, stats in report.token_length_stats.items():
        print(f"  {group}: Mean={stats['mean']} | Median={stats['median']} | Std={stats['std']} | Min={stats['min']} | Max={stats['max']}")

    print("\n--- LEXICAL DIVERSITY (Type-Token Ratio) ---")
    for group, ttr in report.lexical_diversity.items():
        print(f"  {group}: {ttr}")

    print("\n--- SENSATIONALISM & AFFECT SIGNALS ---")
    for metric, vals in report.sensationalism_signals.items():
        print(f"  {metric}: Real={vals['Real News']} vs Fake={vals['Fake News']}")

    print("\n--- READABILITY (Flesch Reading Ease) ---")
    for metric, vals in report.readability_scores.items():
        print(f"  {metric}: Real={vals['Real News']} vs Fake={vals['Fake News']}")

    print("\n--- TOP DISCRIMINATIVE UNIGRAMS ---")
    print(f"  Fake News: {report.top_fake_unigrams[:7]}")
    print(f"  Real News: {report.top_real_unigrams[:7]}")

    print("\n--- TOP DISCRIMINATIVE BIGRAMS ---")
    print(f"  Fake News: {report.top_fake_bigrams[:5]}")
    print(f"  Real News: {report.top_real_bigrams[:5]}")
    print("=" * 70 + "\n")

    # Export JSON report
    report_dict = {
        "total_samples": report.total_samples,
        "class_distribution": report.class_distribution,
        "class_percentages": report.class_percentages,
        "token_length_stats": report.token_length_stats,
        "lexical_diversity": report.lexical_diversity,
        "sensationalism_signals": report.sensationalism_signals,
        "readability_scores": report.readability_scores,
        "top_fake_unigrams": report.top_fake_unigrams,
        "top_real_unigrams": report.top_real_unigrams,
        "top_fake_bigrams": report.top_fake_bigrams,
        "top_real_bigrams": report.top_real_bigrams
    }

    json_path = REPORTS_DIR / "eda_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved structured EDA report to %s", json_path)


if __name__ == "__main__":
    main()
