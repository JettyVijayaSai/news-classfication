"""Unit tests for inference pipeline and API endpoints."""

import unittest
from src.inference.pipeline import predict_news, get_inference_engine
from src.api.app import app


class TestInferenceAPI(unittest.TestCase):

    def test_predict_news_function(self):
        result = predict_news(
            title="NASA Discovers New Exoplanet in Habitable Zone",
            text="Astronomers confirmed spectral data matching atmospheric water signatures.",
            author="ScienceBureau"
        )
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("top_contributing_keywords", result)
        self.assertIn("rationale", result)
        self.assertIn(result["label"], ["fake", "real"])
        self.assertIsInstance(result["confidence"], float)

    def test_api_asgi_app_routes(self):
        # Verify routes are configured properly
        route_paths = [route.path for route in app.routes]
        self.assertIn("/health", route_paths)
        self.assertIn("/predict", route_paths)


if __name__ == "__main__":
    unittest.main()
