"""Tests for the FastAPI inference service (src/api).

All tests except the one marked "integration" inject a fake model via
model_service.load(model=...) and block the real src.models.registry
load_model() from ever being called, so nothing here touches W&B or the
network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model_service import model_service
from src.api.schemas import EXAMPLE_ROW, MAX_BATCH_SIZE

CATEGORICAL_FEATURES = ["route", "direction", "segment_id", "relative_weekday"]
NUMERIC_FEATURES = [k for k in EXAMPLE_ROW if k not in CATEGORICAL_FEATURES]


class FakeModel:
    """Duck-types src.models.registry.ProductionModel without touching
    LightGBM, W&B, or the filesystem."""

    def __init__(self, predict_fn=None, alias: str = "fake-staging", version: str = "v0-fake"):
        self.alias = alias
        self.version = version
        self.model_name = "fake_model"
        self.categorical_features = CATEGORICAL_FEATURES
        self.numeric_features = NUMERIC_FEATURES
        self.expected_feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        self._predict_fn = predict_fn or (lambda df: df["typical_segment_hour_s"].to_numpy())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._predict_fn(df), dtype=float)


@pytest.fixture(autouse=True)
def _block_real_model_load(request, monkeypatch):
    """Prevent the app lifespan's model_service.load() from ever reaching
    the real registry/W&B during unit tests. The one test marked
    "integration" opts out and exercises the real load_model('production')
    path instead."""
    if request.node.get_closest_marker("integration"):
        yield
        return

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("blocked in unit tests: no network/W&B calls allowed")

    monkeypatch.setattr("src.api.model_service.load_model", _blocked)
    yield
    model_service.unload()


@pytest.fixture
def client(_block_real_model_load):
    with TestClient(app) as c:
        yield c


def _inject_fake_model(**kwargs) -> FakeModel:
    fake = FakeModel(**kwargs)
    model_service.load(model=fake)
    return fake


def test_health_returns_503_when_model_not_loaded(client):
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    # The blocked-in-unit-tests load failure surfaces as a short load_error.
    assert "load_error" in body


def test_health_ok_when_model_loaded(client):
    _inject_fake_model()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    fake = _inject_fake_model()
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["alias"] == fake.alias
    assert body["version"] == fake.version
    assert len(body["feature_columns"]) == 25
    assert body["feature_columns"] == CATEGORICAL_FEATURES + NUMERIC_FEATURES


def test_model_info_503_when_not_loaded(client):
    response = client.get("/model-info")
    assert response.status_code == 503


def test_predict_success(client):
    fake = _inject_fake_model()
    response = client.post("/predict", json=EXAMPLE_ROW)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_travel_time_s"] == pytest.approx(EXAMPLE_ROW["typical_segment_hour_s"])
    assert body["model_alias"] == fake.alias
    assert body["model_version"] == fake.version
    assert body["request_id"]


def test_predict_503_when_not_loaded(client):
    response = client.post("/predict", json=EXAMPLE_ROW)
    assert response.status_code == 503


def test_negative_prediction_is_clipped_to_zero(client):
    """If the underlying model could ever emit a negative value, the API
    floors it at 0.0 before returning (see MIN_RETURNED_PREDICTION_S in
    src/api/main.py) - a travel time can never be negative."""
    _inject_fake_model(predict_fn=lambda df: np.full(len(df), -50.0))
    response = client.post("/predict", json=EXAMPLE_ROW)
    assert response.status_code == 200
    assert response.json()["predicted_travel_time_s"] == 0.0


def test_predict_invalid_hour_returns_422(client):
    _inject_fake_model()
    bad_row = {**EXAMPLE_ROW, "hour": 24}
    response = client.post("/predict", json=bad_row)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    _inject_fake_model()
    bad_row = {k: v for k, v in EXAMPLE_ROW.items() if k != "route"}
    response = client.post("/predict", json=bad_row)
    assert response.status_code == 422


def test_predict_extra_unknown_field_is_ignored(client):
    """Pydantic v2's default extra='ignore' means unrecognized fields are
    silently dropped rather than rejected; documented here as the current,
    intentional behaviour."""
    _inject_fake_model()
    row_with_extra = {**EXAMPLE_ROW, "some_unexpected_field": "surprise"}
    response = client.post("/predict", json=row_with_extra)
    assert response.status_code == 200


def test_predict_batch_of_three(client):
    _inject_fake_model()
    response = client.post("/predict/batch", json={"requests": [EXAMPLE_ROW, EXAMPLE_ROW, EXAMPLE_ROW]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 3


def test_predict_batch_over_limit_returns_422(client):
    _inject_fake_model()
    too_many = [EXAMPLE_ROW] * (MAX_BATCH_SIZE + 1)
    response = client.post("/predict/batch", json={"requests": too_many})
    assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.needs_data  # load_model() always needs a real data/processed/feature_spec.json on disk
@pytest.mark.skipif(
    not (os.environ.get("WANDB_API_KEY") or os.environ.get("MODEL_PATH")),
    reason="requires WANDB_API_KEY or MODEL_PATH to load the real production model",
)
def test_integration_load_real_production_model():
    os.environ["MODEL_ALIAS"] = "production"
    model_service.load()
    assert model_service.is_loaded

    df = pd.DataFrame([EXAMPLE_ROW])
    pred = model_service.predict(df)[0]
    assert 0.0 <= pred <= 3600.0
