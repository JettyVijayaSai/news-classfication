"""Production REST API Service for Fake News Classification.

Provides high-throughput ASGI endpoints for real-time inference,
batch processing, model diagnostics, and explainability payloads.
Works with FastAPI if installed, and falls back seamlessly to Starlette ASGI.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional

from src.config import AppConfig
from src.inference.pipeline import get_inference_engine, NewsInferenceEngine

logger = logging.getLogger(__name__)

# Try importing FastAPI & Pydantic
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route


if HAS_FASTAPI:
    class NewsItemRequest(BaseModel):
        title: str = Field(..., description="Headline of the news article")
        text: str = Field(..., description="Full body content of the article")
        author: Optional[str] = Field("Unknown", description="Author or publishing organization")
        explain: Optional[bool] = Field(True, description="Whether to include feature attribution")

        model_config = {
            "json_schema_extra": {
                "example": {
                    "title": "Shocking Revelation: Secret Underground Base Exposed!",
                    "text": "Whistleblowers reveal unbelievable hidden operations suppressed by mainstream media.",
                    "author": "ConspiracyInsider",
                    "explain": True
                }
            }
        }

    class BatchNewsRequest(BaseModel):
        items: List[NewsItemRequest]

    class PredictionResponse(BaseModel):
        label: str
        confidence: float
        fake_probability: float
        real_probability: float
        decision_threshold: float
        top_contributing_keywords: List[Any]
        rationale: str

    app = FastAPI(
        title="Fake News Detection & Explainability API",
        description="Production microservice classifying news articles with local interpretability rationale.",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
    def health_check() -> Dict[str, Any]:
        """Performs liveness check on model service."""
        try:
            engine = get_inference_engine()
            return {
                "status": "healthy",
                "model_loaded": True,
                "decision_threshold": engine.decision_threshold
            }
        except Exception as e:
            return {
                "status": "degraded",
                "model_loaded": False,
                "error": str(e)
            }

    @app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
    def predict_endpoint(payload: NewsItemRequest) -> Dict[str, Any]:
        """Classifies an individual article and generates explanation."""
        try:
            engine = get_inference_engine()
            return engine.predict_news(
                title=payload.title,
                text=payload.text,
                author=payload.author or "Unknown",
                explain=payload.explain if payload.explain is not None else True
            )
        except Exception as e:
            logger.error("Inference failure: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch-predict", response_model=List[PredictionResponse], tags=["Inference"])
    def batch_predict_endpoint(payload: BatchNewsRequest) -> List[Dict[str, Any]]:
        """Processes a batch of articles."""
        try:
            engine = get_inference_engine()
            batch_data = [item.model_dump() for item in payload.items]
            return engine.batch_predict(batch_data, explain=False)
        except Exception as e:
            logger.error("Batch inference failure: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

else:
    # Starlette ASGI Implementation Fallback
    async def health_check_route(request):
        try:
            engine = get_inference_engine()
            return JSONResponse({
                "status": "healthy",
                "model_loaded": True,
                "decision_threshold": engine.decision_threshold
            })
        except Exception as e:
            return JSONResponse({
                "status": "degraded",
                "model_loaded": False,
                "error": str(e)
            }, status_code=503)

    async def predict_route(request):
        try:
            body = await request.json()
            title = body.get("title", "")
            text = body.get("text", "")
            author = body.get("author", "Unknown")
            explain = body.get("explain", True)

            engine = get_inference_engine()
            result = engine.predict_news(title=title, text=text, author=author, explain=explain)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)

    async def batch_predict_route(request):
        try:
            body = await request.json()
            items = body.get("items", [])
            engine = get_inference_engine()
            results = engine.batch_predict(items, explain=False)
            return JSONResponse(results)
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)

    routes = [
        Route("/health", health_check_route, methods=["GET"]),
        Route("/predict", predict_route, methods=["POST"]),
        Route("/batch-predict", batch_predict_route, methods=["POST"])
    ]
    app = Starlette(debug=False, routes=routes)
