"""Unit tests for interpretability and explainability layer."""

import unittest
from src.models.classical import ClassicalNewsClassifier
from src.explainability.explainer import NewsExplainer


class TestExplainability(unittest.TestCase):

    def setUp(self):
        texts = [
            "[AUTHOR] reuters [TITLE] federal reserve monetary policy [CONTENT] central bank report economic stability",
            "[AUTHOR] conspiracy [TITLE] secret underground vault cure [CONTENT] miracle truth exposed banned mainstream"
        ]
        y = [0, 1]
        self.clf = ClassicalNewsClassifier(algorithm="logistic_regression")
        self.clf.fit(texts, y)
        self.explainer = NewsExplainer(model=self.clf, num_perturbations=50)

    def test_explain_instance_output_structure(self):
        test_text = "secret underground bunker with miracle cure exposed"
        result = self.explainer.explain_instance(test_text, num_features=4)

        self.assertIn("base_prediction", result)
        self.assertIn("confidence", result)
        self.assertIn("top_fake_contributors", result)
        self.assertIn("top_real_contributors", result)
        self.assertIn("rationale", result)
        self.assertIsInstance(result["rationale"], str)
        self.assertGreater(len(result["rationale"]), 10)


if __name__ == "__main__":
    unittest.main()
