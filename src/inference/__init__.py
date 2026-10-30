"""Inference package initialization."""
from src.inference.pipeline import NewsInferenceEngine, predict_news, get_inference_engine

__all__ = ["NewsInferenceEngine", "predict_news", "get_inference_engine"]
