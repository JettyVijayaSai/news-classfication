"""End-to-End Model Training, Benchmarking, and Serialization Pipeline.

Trains and compares Classical ML models (Logistic Regression, Calibrated LinearSVC,
Naive Bayes, Soft-Voting Ensemble) and Transformer/Neural models.
Optimizes decision thresholds to minimize False Negatives, logs benchmark metrics,
and serializes the winning production artifacts.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ModelConfig
from src.data.loader import NewsDataLoader
from src.data.preprocessor import TextPreprocessor
from src.models.classical import ClassicalNewsClassifier
from src.models.transformer import TransformerNewsClassifier
from src.models.evaluator import ModelEvaluator
from src.explainability.explainer import NewsExplainer
from src.inference.pipeline import NewsInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TRAIN_PIPELINE")


def main():
    data_path = DATA_DIR / "sample_news.csv"
    if not data_path.exists():
        logger.error("Dataset not found at %s", data_path)
        sys.exit(1)

    logger.info("Step 1: Ingesting dataset and preparing splits...")
    loader = NewsDataLoader()
    df = loader.load_data(data_path, handle_imbalance="resample")
    X_train, X_test, y_train, y_test = loader.split_data(df)

    logger.info("Step 2: Preprocessing and synthesizing metadata [AUTHOR + TITLE + TEXT]...")
    preprocessor = TextPreprocessor()
    train_texts = preprocessor.transform(X_train)
    test_texts = preprocessor.transform(X_test)
    y_train_arr = y_train.values
    y_test_arr = y_test.values

    # Models to benchmark
    models_to_train = {
        "Logistic Regression (TF-IDF)": ClassicalNewsClassifier(algorithm="logistic_regression"),
        "Calibrated LinearSVC (TF-IDF)": ClassicalNewsClassifier(algorithm="linear_svc"),
        "Multinomial Naive Bayes (TF-IDF)": ClassicalNewsClassifier(algorithm="naive_bayes"),
        "Soft-Voting Ensemble (TF-IDF)": ClassicalNewsClassifier(algorithm="ensemble"),
        "Context-Aware Neural Sequence Model": TransformerNewsClassifier()
    }

    results = {}
    best_model_name = None
    best_f2_score = -1.0
    best_model_instance = None
    best_threshold = 0.5

    print("\n" + "=" * 90)
    print("                    MODEL BENCHMARKING & EVALUATION                         ")
    print("=" * 90)
    header = f"{'Model':<35} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1-Fake':<7} | {'F2':<6} | {'ROC-AUC':<7} | {'Thresh':<6}"
    print(header)
    print("-" * 90)

    for name, model in models_to_train.items():
        # Train model
        model.fit(train_texts, y_train_arr)

        # Predict probabilities on test set
        test_probs = model.predict_proba(test_texts)

        # Optimize threshold prioritizing False Negative minimization (F2 score)
        opt_th, metrics = ModelEvaluator.optimize_decision_threshold(
            y_true=y_test_arr,
            y_pred_proba=test_probs,
            beta=2.0
        )

        results[name] = {
            "metrics": metrics.to_dict(),
            "optimal_threshold": opt_th
        }

        row = (
            f"{name:<35} | "
            f"{metrics.accuracy:<6.4f} | "
            f"{metrics.precision_fake:<6.4f} | "
            f"{metrics.recall_fake:<6.4f} | "
            f"{metrics.f1_fake:<7.4f} | "
            f"{metrics.f2_score:<6.4f} | "
            f"{metrics.roc_auc:<7.4f} | "
            f"{opt_th:<6.2f}"
        )
        print(row)

        # Track best model based on F2 score (weighted recall against misinformation)
        if metrics.f2_score > best_f2_score:
            best_f2_score = metrics.f2_score
            best_model_name = name
            best_model_instance = model
            best_threshold = opt_th

    print("=" * 90)
    print(f"\nWinning Model Selected for Production: '{best_model_name}' (F2 Score = {best_f2_score:.4f}, Optimal Threshold = {best_threshold})")

    # Serialize Best Model and Artifacts
    best_model_path = MODELS_DIR / "best_model.joblib"
    best_model_instance.save(best_model_path)
    logger.info("Serialized champion model to %s", best_model_path)

    # Export benchmark report
    benchmark_report_path = REPORTS_DIR / "benchmark_report.json"
    with open(benchmark_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark report to %s", benchmark_report_path)

    # Verification: End-to-End Inference & Explainability Test
    print("\n" + "=" * 90)
    print("           VERIFICATION: END-TO-END INFERENCE & EXPLAINABILITY TEST         ")
    print("=" * 90)

    engine = NewsInferenceEngine(
        model=best_model_instance,
        preprocessor=preprocessor,
        decision_threshold=best_threshold
    )

    test_article = {
        "title": "BOMBSHELL: Proof of Secret Weather Weapons Creating Artificial Disasters",
        "text": "Military whistleblowers expose leaked documents proving secret scalar transmitters trigger synthetic hurricanes to terrorize the populace. Mainstream media continues covering up the conspiracy.",
        "author": "UndergroundPatriot"
    }

    prediction = engine.predict_news(
        title=test_article["title"],
        text=test_article["text"],
        author=test_article["author"],
        explain=True
    )

    print(f"Test Headline: \"{test_article['title']}\"")
    print(f"Byline Author: {test_article['author']}")
    print(f"Predicted Class: {prediction['label'].upper()}")
    print(f"Confidence Score: {prediction['confidence']:.4f} (Fake Prob: {prediction['fake_probability']:.4f})")
    print(f"Decision Threshold: {prediction['decision_threshold']}")
    print(f"Top Driving Features: {prediction['top_contributing_keywords']}")
    print(f"Generated Rationale: {prediction['rationale']}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
