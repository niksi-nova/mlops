"""FastAPI inference service for the Delhi bus segment travel-time model.

The model predicts travel_time_s (seconds) for one stop-to-stop segment,
given the 25 pre-computed features in data/processed/feature_spec.json.
It does NOT compute real-time features (e.g. prev_bus_tt_s, headway_s)
from a live bus feed - see docs/api.md.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.model_service import model_service
from src.api.schemas import BatchRequest, BatchResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("delhi_bus_delay_api")

# The underlying model wrapper (src.models.evaluate.predict) already clips
# raw predictions at >= 1.0 second before returning them. This floor at 0.0
# is a defensive second layer at the API boundary: if model_service is ever
# pointed at a model that does NOT do its own clipping, the API still never
# reports a negative travel time to a caller.
MIN_RETURNED_PREDICTION_S = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load()
        logger.info('{"event": "startup", "model_loaded": true}')
    except Exception:
        logger.error(
            '{"event": "startup", "model_loaded": false, "load_error": "%s"}' % model_service.load_error
        )
    yield


app = FastAPI(
    title="Delhi Bus Delay Prediction API",
    description=(
        "Predicts stop-to-stop segment travel time (seconds) for Delhi bus "
        "routes from pre-computed features. Model: lightgbm_l1, served from "
        "the W&B model registry (delhi-bus-delay-model)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error(
            '{"request_id": "%s", "path": "%s", "status": 500, "latency_ms": %.2f}'
            % (request_id, request.url.path, latency_ms)
        )
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        '{"request_id": "%s", "path": "%s", "status": %d, "latency_ms": %.2f}'
        % (request_id, request.url.path, response.status_code, latency_ms)
    )
    response.headers["X-Request-ID"] = request_id
    return response


def _require_model_loaded() -> None:
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")


def _predict_one(request: PredictionRequest, request_id: str) -> PredictionResponse:
    df = pd.DataFrame([request.model_dump()])
    raw_pred = model_service.predict(df)[0]
    predicted = max(float(raw_pred), MIN_RETURNED_PREDICTION_S)
    info = model_service.info()
    return PredictionResponse(
        predicted_travel_time_s=predicted,
        model_alias=info["alias"],
        model_version=info["version"],
        request_id=request_id,
    )


@app.get("/health")
async def health():
    if not model_service.is_loaded:
        content = {"status": "degraded", "model_loaded": False}
        if model_service.load_error:
            content["load_error"] = model_service.load_error
        return JSONResponse(status_code=503, content=content)
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
async def model_info():
    _require_model_loaded()
    return model_service.info()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, http_request: Request):
    _require_model_loaded()
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    return _predict_one(request, request_id)


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest, http_request: Request):
    _require_model_loaded()
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    predictions = [_predict_one(req, f"{request_id}-{i}") for i, req in enumerate(batch.requests)]
    return BatchResponse(predictions=predictions)
