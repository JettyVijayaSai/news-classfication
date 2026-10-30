"""Unit tests for models and evaluation metrics."""

import unittest
import numpy as np
from src.models.classical import ClassicalNewsClassifier
from src.models.transformer import TransformerNewsClassifier
from src.models.evaluator import ModelEvaluator


class TestModelsAndEvaluation(unittest.TestCase):

    def setUp(self):
        self.train_texts = [
            "[AUTHOR] reuters [TITLE] federal reserve rate [CONTENT] economic growth stable",
            "[AUTHOR] ap [TITLE] european parliament regulation [CONTENT] approved artificial intelligence legislation",
            "[AUTHOR] who [TITLE] clinical trial malaria [CONTENT] published medical science journal",
            "[AUTHOR] truthwarrior [TITLE] secret underground vault [CONTENT] miracle cure banned mainstream media",
            "[AUTHOR] hoaxes [TITLE] 5g towers earthquake [CONTENT] scalar transmitters secret military weather",
            "[AUTHOR] leakcentral [TITLE] illuminati cash confiscation [CONTENT] bank collapse digital prison buy gold"
        ]
        self.y_train = [0, 0, 0, 1, 1, 1]

    def test_classical_models(self):
        for algo in ["logistic_regression", "naive_bayes"]:
            clf = ClassicalNewsClassifier(algorithm=algo)
            clf.fit(self.train_texts, self.y_train)
            probs = clf.predict_proba(self.train_texts)
            self.assertEqual(probs.shape, (6, 2))
            preds = clf.predict(self.train_texts)
            self.assertEqual(len(preds), 6)

    def test_evaluator_metrics_and_threshold_optimization(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([0.1, 0.3, 0.6, 0.9])
        
        metrics = ModelEvaluator.evaluate(y_true, y_probs, threshold=0.5)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.precision_fake, 1.0)
        self.assertEqual(metrics.recall_fake, 1.0)

        # Threshold optimization prioritizing F2
        opt_th, opt_metrics = ModelEvaluator.optimize_decision_threshold(y_true, y_probs, beta=2.0)
        self.assertGreater(opt_th, 0.0)
        self.assertLess(opt_th, 1.0)
        self.assertGreaterEqual(opt_metrics.recall_fake, 1.0)


if __name__ == "__main__":
    unittest.main()
