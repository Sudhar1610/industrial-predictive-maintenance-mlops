"""FastAPI inference service: `/predict`, `/predict-batch`, `/health`.

Fails loud at startup (no model -> the process refuses to come up
healthy) but fails safe per-request (a malformed request or a transient
model error returns a structured HTTP error, never a crash) -- see the
module docstring convention described in the project README.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST

from pdm.config import settings
from pdm.data.factory import get_datasource
from pdm.models.base import Model
from pdm.monitoring.prediction_logger import PredictionLogger
from pdm.monitoring.prometheus_metrics import record_prediction, render_latest
from pdm.serving.inference import build_model_input
from pdm.serving.model_loader import load_production_model
from pdm.serving.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    PredictionResponse,
    PredictRequest,
)


class _AppState:
    """Holds the process-lifetime singletons the API depends on. A plain
    class (not a dict) so attribute access is type-checked."""

    model: Model | None = None
    model_backend: str | None = None
    prediction_logger: PredictionLogger = PredictionLogger()


state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_config = settings.get_model_config()
    state.model = load_production_model(model_config)
    state.model_backend = model_config.active_model
    logger.info("Serving Production model (backend={})", state.model_backend)
    yield
    state.model = None


app = FastAPI(
    title="Predictive Maintenance Inference API",
    description="Failure-classification and RUL-regression inference for turbofan sensor telemetry.",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_model() -> Model:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return state.model


def _predict_one(request: PredictRequest) -> PredictionResponse:
    model = _require_model()
    model_config = settings.get_model_config()
    try:
        X, latest_cycle = build_model_input(request, model_config)
    except Exception as exc:  # noqa: BLE001 - malformed input, not a server error
        raise HTTPException(status_code=422, detail=f"Could not build features: {exc}") from exc

    start = time.perf_counter()
    result = model.predict(X)
    latency = time.perf_counter() - start

    response = PredictionResponse(
        unit_id=request.unit_id,
        cycle=latest_cycle,
        failure_probability=float(result.failure_probability[0]),
        will_fail=bool(result.will_fail[0]),
        remaining_useful_life=float(result.remaining_useful_life[0]),
        model_backend=state.model_backend or "unknown",
    )

    record_prediction(
        model_backend=response.model_backend,
        will_fail=response.will_fail,
        failure_probability=response.failure_probability,
        latency_seconds=latency,
    )
    state.prediction_logger.log(request, response)

    return response


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    return _predict_one(request)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=[_predict_one(unit) for unit in request.units])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    datasource_reachable = False
    try:
        datasource_config = settings.get_datasource_config()
        datasource_reachable = get_datasource(datasource_config).health_check()
    except Exception as exc:  # noqa: BLE001 - health check must never raise
        logger.error("Health check datasource probe failed: {}", exc)

    model_loaded = state.model is not None
    return HealthResponse(
        status="ok" if model_loaded and datasource_reachable else "degraded",
        model_loaded=model_loaded,
        model_backend=state.model_backend,
        datasource_reachable=datasource_reachable,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=render_latest(), media_type=CONTENT_TYPE_LATEST)
