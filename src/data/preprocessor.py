"""Text Preprocessing and Feature Normalization Pipeline.

Includes text cleaning, contraction expansion, lemmatization/stemming,
stopword removal, and metadata synthesis for fake news classification.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import PreprocessingConfig

# Common English contractions dictionary
CONTRACTION_MAP = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
    "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"
}

# Standard English stopwords
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class RuleBasedLemmatizer:
    """Lightweight, deterministic rule-based lemmatizer/stemmer.
    
    Reduces morphological variants (e.g. running -> run, conspiracies -> conspiracy)
    without requiring heavyweight binary models, ensuring cross-platform stability.
    """

    def __init__(self):
        # Suffix replacement rules: (suffix_pattern, replacement, min_stem_length)
        self.suffix_rules = [
            (r"sses$", "ss", 2),
            (r"ies$", "y", 2),
            (r"ss$", "ss", 2),
            (r"s$", "", 3),
            (r"eed$", "ee", 3),
            (r"ational$", "ate", 3),
            (r"tional$", "tion", 3),
            (r"enci$", "ence", 3),
            (r"anci$", "ance", 3),
            (r"izer$", "ize", 3),
            (r"bli$", "ble", 3),
            (r"alli$", "al", 3),
            (r"entli$", "ent", 3),
            (r"eli$", "e", 3),
            (r"ousli$", "ous", 3),
            (r"ization$", "ize", 3),
            (r"ation$", "ate", 3),
            (r"ator$", "ate", 3),
            (r"alism$", "al", 3),
            (r"iveness$", "ive", 3),
            (r"fulness$", "ful", 3),
            (r"ousness$", "ous", 3),
            (r"aliti$", "al", 3),
            (r"iviti$", "ive", 3),
            (r"biliti$", "ble", 3),
        ]
        self.compiled_rules = [(re.compile(pat), rep, m_len) for pat, rep, m_len in self.suffix_rules]
        self.vowel_pattern = re.compile(r"[aeiouy]")

    def lemmatize(self, word: str) -> str:
        """Lemmatizes a single token using morphological suffix stripping."""
        if len(word) <= 3:
            return word

        # Handle 'ed' and 'ing' suffixes safely if preceded by vowel
        if word.endswith("ed") and len(word) > 4:
            stem = word[:-2]
            if self.vowel_pattern.search(stem):
                if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "ls":
                    stem = stem[:-1]
                return stem
        elif word.endswith("ing") and len(word) > 5:
            stem = word[:-3]
            if self.vowel_pattern.search(stem):
                if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "ls":
                    stem = stem[:-1]
                return stem

        for pattern, replacement, min_stem_length in self.compiled_rules:
            if pattern.search(word):
                stem = pattern.sub(replacement, word)
                if len(stem) >= min_stem_length:
                    return stem
        return word


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Production-grade text cleaning and normalization transformer.
    
    Implements scikit-learn's TransformerMixin interface for seamless
    integration into training and inference pipelines.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.lemmatizer = RuleBasedLemmatizer() if self.config.lemmatize else None
        self.stopwords = set(DEFAULT_STOPWORDS).union(set(self.config.custom_stopwords))
        
        # Precompile regex patterns for high-throughput batch execution
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.html_pattern = re.compile(r"<.*?>")
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
        self.whitespace_pattern = re.compile(r"\s+")
        self.contraction_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
            flags=re.IGNORECASE
        )

    def _expand_contractions(self, text: str) -> str:
        """Expands common English contractions."""
        def replace(match):
            key = match.group(0).lower()
            return CONTRACTION_MAP.get(key, match.group(0))
        return self.contraction_pattern.sub(replace, text)

    def clean_text(self, text: Optional[str]) -> str:
        """Cleans and normalizes a single text string.

        Args:
            text: Raw input text.

        Returns:
            Normalized, cleaned string.
        """
        if not text or not isinstance(text, str) or text.strip() == "":
            return ""

        # Decode HTML entities (&amp; -> &, &quot; -> ", etc.)
        cleaned = html.unescape(text)

        # Unicode normalization (NFKD decomposes ligatures and accented characters)
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("utf-8", "ignore")

        # Remove URLs and email addresses
        if self.config.remove_urls:
            cleaned = self.url_pattern.sub(" ", cleaned)
            cleaned = self.email_pattern.sub(" ", cleaned)

        # Remove HTML tags
        if self.config.strip_html:
            cleaned = self.html_pattern.sub(" ", cleaned)

        # Lowercase conversion
        if self.config.lowercase:
            cleaned = cleaned.lower()

        # Expand contractions
        cleaned = self._expand_contractions(cleaned)

        # Strip special characters and punctuation
        if self.config.remove_special_characters:
            cleaned = self.special_char_pattern.sub(" ", cleaned)

        # Tokenization & stopword removal / lemmatization
        tokens = cleaned.split()
        processed_tokens: List[str] = []

        for token in tokens:
            if len(token) < self.config.min_token_length or len(token) > self.config.max_token_length:
                continue
            if self.config.remove_stopwords and token in self.stopwords:
                continue
            if self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            processed_tokens.append(token)

        return " ".join(processed_tokens)

    def combine_metadata(
        self,
        title: Optional[str],
        text: Optional[str],
        author: Optional[str]
    ) -> str:
        """Combines headline, author, and article body into structured input.
        
        Structuring with semantic tags allows both classical n-gram extractors
        and Transformer attention heads to leverage author and headline authority.

        Args:
            title: Headline string.
            text: Article body.
            author: Author or source name.

        Returns:
            Synthesized document text.
        """
        author_clean = self.clean_text(author) or "unknown"
        title_clean = self.clean_text(title) or ""
        body_clean = self.clean_text(text) or ""

        # Structured representation
        combined = (
            f"{self.config.author_prefix} {author_clean} "
            f"{self.config.title_prefix} {title_clean} "
            f"{self.config.text_prefix} {body_clean}"
        ).strip()
        
        return self.whitespace_pattern.sub(" ", combined)

    def fit(self, X: Union[pd.DataFrame, List[str]], y=None) -> TextPreprocessor:
        """Fit transformer (stateless)."""
        return self

    def transform(self, X: Union[pd.DataFrame, List[dict], List[str]]) -> List[str]:
        """Transforms batch of news items into cleaned, combined text documents.

        Args:
            X: DataFrame with 'title', 'text', 'author' columns OR list of dicts/strings.

        Returns:
            List of processed strings ready for vectorization/tokenization.
        """
        if isinstance(X, pd.DataFrame):
            results = []
            for _, row in X.iterrows():
                title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
                text = str(row.get("text", "")) if pd.notna(row.get("text")) else ""
                author = str(row.get("author", "")) if pd.notna(row.get("author")) else ""
                results.append(self.combine_metadata(title=title, text=text, author=author))
            return results

        elif isinstance(X, list):
            results = []
            for item in X:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    text = item.get("text", "")
                    author = item.get("author", "")
                    results.append(self.combine_metadata(title=title, text=text, author=author))
                elif isinstance(item, str):
                    results.append(self.clean_text(item))
                else:
                    results.append("")
            return results

        raise TypeError(f"Unsupported input type: {type(X)}")
