"""Data package initialization."""
from src.data.loader import NewsDataLoader
from src.data.preprocessor import TextPreprocessor

__all__ = ["NewsDataLoader", "TextPreprocessor"]
