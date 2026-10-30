"""Unit tests for data loader, validation, and preprocessor."""

import unittest
import pandas as pd
from src.data.loader import NewsDataLoader
from src.data.preprocessor import TextPreprocessor, RuleBasedLemmatizer


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.loader = NewsDataLoader()

    def test_lemmatizer_rules(self):
        lemmatizer = RuleBasedLemmatizer()
        self.assertEqual(lemmatizer.lemmatize("conspiracies"), "conspiracy")
        self.assertEqual(lemmatizer.lemmatize("whistleblowers"), "whistleblower")
        self.assertEqual(lemmatizer.lemmatize("running"), "run")
        self.assertEqual(lemmatizer.lemmatize("leaked"), "leak")
        self.assertEqual(lemmatizer.lemmatize("cat"), "cat")  # Short word unchanged

    def test_clean_text_contractions_and_entities(self):
        raw = "DON'T drink this! &amp; visit http://fake.com for <b>cures</b>."
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("cure", cleaned)

    def test_combine_metadata(self):
        combined = self.preprocessor.combine_metadata(
            title="Secret Vault Found",
            text="Underground base discovered.",
            author="PatriotInsider"
        )
        self.assertIn("[AUTHOR] patriotinsider", combined)
        self.assertIn("[TITLE] secret vault found", combined)
        self.assertIn("[CONTENT] underground base discov", combined)

    def test_loader_validation_and_cleaning(self):
        raw_data = pd.DataFrame([
            {"title": "Real News 1", "text": "Genuine economic report", "author": "Reuters", "label": "real"},
            {"title": "Real News 1", "text": "Genuine economic report", "author": "Reuters", "label": "real"},  # Duplicate
            {"title": "", "text": "", "author": "Spam", "label": "fake"},  # Empty text
            {"title": "Fake News 2", "text": "Secret cure hoax", "author": "Hoaxer", "label": "fake"},
        ])
        cleaned = self.loader._clean_and_validate(raw_data)
        # Should drop duplicate and empty record
        self.assertEqual(len(cleaned), 2)
        self.assertIn("label", cleaned.columns)
        self.assertEqual(set(cleaned["label"].unique()), {0, 1})


if __name__ == "__main__":
    unittest.main()
