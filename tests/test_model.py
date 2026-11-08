"""Inference and contract tests for the trained LightGBM production model (lightgbm_l1).

Verifies model artifact loading, 25-feature schema specification, exact booster feature
ordering (categoricals + numerics), prediction output shape, non-null/finite bounds,
plausible positive range, determinism, error handling on missing/invalid columns,
cold-start handling (has_prev_bus = 0), hermetic offline execution without W&B,
and warm single-row inference latency configured via params.yaml.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import wandb
import yaml

from src.models.evaluate import predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@pytest.fixture(autouse=True)
def disable_wandb_and_network(monkeypatch: pytest.MonkeyPatch):
    """Ensure all model tests execute in disabled tracking mode."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    assert PARAMS_PATH.exists(), f"params.yaml not found at {PARAMS_PATH}"
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def feature_spec() -> dict[str, Any]:
    spec_path = PROCESSED_DIR / "feature_spec.json"
    assert spec_path.exists(), f"feature_spec.json not found at {spec_path}"
    with open(spec_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def expected_feature_columns(feature_spec: dict[str, Any]) -> list[str]:
    """LightGBM training orders features as categorical_features + numeric_features."""
    return feature_spec["categorical_features"] + feature_spec["numeric_features"]


@pytest.fixture(scope="module")
def lightgbm_model() -> lgb.LGBMRegressor:
    model_path = MODELS_DIR / "lightgbm_l1.joblib"
    assert model_path.exists(), f"Model artifact lightgbm_l1.joblib not found at {model_path}"
    model = joblib.load(model_path)
    return model


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    test_path = PROCESSED_DIR / "test.parquet"
    assert test_path.exists(), f"test.parquet not found at {test_path}"
    return pd.read_parquet(test_path)


# -----------------------------------------------------------------------------
# 1. Successful loading of the actual trained model and exact booster feature order
# -----------------------------------------------------------------------------
def test_load_trained_model(
    lightgbm_model: lgb.LGBMRegressor, feature_spec: dict[str, Any], expected_feature_columns: list[str]
):
    """Verify that lightgbm_l1.joblib loads successfully as an LGBMRegressor

    and its booster feature names match expected categorical + numeric ordering.
    """
    assert lightgbm_model is not None
    assert isinstance(lightgbm_model, lgb.LGBMRegressor)
    assert hasattr(lightgbm_model, "predict")
    assert hasattr(lightgbm_model, "n_features_")
    assert lightgbm_model.n_features_ == 25

    expected_columns = (
        feature_spec["categorical_features"]
        + feature_spec["numeric_features"]
    )
    assert lightgbm_model.booster_.feature_name() == expected_columns
    assert expected_feature_columns == expected_columns


# -----------------------------------------------------------------------------
# 2. Verify complete 25-feature schema: 21 numeric and 4 categorical
# -----------------------------------------------------------------------------
def test_feature_schema_specification(feature_spec: dict[str, Any], test_data: pd.DataFrame):
    """Verify the 25-feature contract: 21 numeric, 4 categorical, no overlap."""
    features = feature_spec.get("features", [])
    numeric = feature_spec.get("numeric_features", [])
    categorical = feature_spec.get("categorical_features", [])

    assert len(features) == 25, f"Expected 25 total features, got {len(features)}"
    assert len(numeric) == 21, f"Expected 21 numeric features, got {len(numeric)}"
    assert len(categorical) == 4, f"Expected 4 categorical features, got {len(categorical)}"

    # Ensure sets partition the feature list completely without duplicates
    assert set(numeric).isdisjoint(set(categorical)), "Numeric and categorical sets must be disjoint"
    assert set(numeric) | set(categorical) == set(features), "Union of numeric and categorical must equal features"

    # Verify all 25 features exist in the dataset
    for col in features:
        assert col in test_data.columns, f"Feature column '{col}' missing from test dataset"


# -----------------------------------------------------------------------------
# 3. Verify prediction output shape
# -----------------------------------------------------------------------------
def test_prediction_output_shape(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
):
    """Verify predict returns a 1D ndarray matching input row count for single and batch rows."""
    categoricals = feature_spec["categorical_features"]

    # Single-row prediction
    single_row = test_data.head(1)
    preds_single = predict("lightgbm_l1", lightgbm_model, single_row, expected_feature_columns, categoricals)
    assert isinstance(preds_single, np.ndarray)
    assert preds_single.shape == (1,)

    # Multi-row batch prediction (e.g., 25 rows)
    batch_rows = test_data.head(25)
    preds_batch = predict("lightgbm_l1", lightgbm_model, batch_rows, expected_feature_columns, categoricals)
    assert isinstance(preds_batch, np.ndarray)
    assert preds_batch.shape == (25,)


# -----------------------------------------------------------------------------
# 4. Verify predictions contain no NaN or infinity
# -----------------------------------------------------------------------------
def test_prediction_no_nan_or_infinity(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
):
    """Verify predictions on a sample of test data contain zero NaN or inf values."""
    sample = test_data.head(100)
    preds = predict(
        "lightgbm_l1", lightgbm_model, sample, expected_feature_columns, feature_spec["categorical_features"]
    )

    assert not np.isnan(preds).any(), "Predictions must not contain NaN"
    assert not np.isinf(preds).any(), "Predictions must not contain infinity"


# -----------------------------------------------------------------------------
# 5. Verify predictions are positive and within plausible range
# -----------------------------------------------------------------------------
def test_prediction_positive_and_plausible_range(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
    params: dict[str, Any],
):
    """Verify predictions are strictly positive and within segments bounds from params.yaml."""
    min_bound = params.get("segments", {}).get("min_segment_seconds", 1.0)
    max_bound = params.get("segments", {}).get("max_segment_seconds", 1800.0)

    sample = test_data.head(200)
    preds = predict(
        "lightgbm_l1", lightgbm_model, sample, expected_feature_columns, feature_spec["categorical_features"]
    )

    assert (preds > 0).all(), "All predicted travel times must be strictly positive"
    assert (preds >= min_bound).all(), f"Predictions must be >= min_segment_seconds ({min_bound})"
    assert (preds <= max_bound).all(), f"Predictions must be <= max_segment_seconds ({max_bound})"


# -----------------------------------------------------------------------------
# 6. Test deterministic predictions
# -----------------------------------------------------------------------------
def test_prediction_determinism(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
):
    """Verify identical inputs yield numerically identical outputs across repeated runs."""
    sample = test_data.head(50)
    categoricals = feature_spec["categorical_features"]

    preds_run1 = predict("lightgbm_l1", lightgbm_model, sample, expected_feature_columns, categoricals)
    preds_run2 = predict("lightgbm_l1", lightgbm_model, sample, expected_feature_columns, categoricals)

    np.testing.assert_array_equal(preds_run1, preds_run2, err_msg="Model predictions must be deterministic")


# -----------------------------------------------------------------------------
# 7. Test missing and invalid feature columns with clear errors
# -----------------------------------------------------------------------------
def test_missing_and_invalid_feature_columns(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
):
    """Verify missing required columns raise KeyError and incompatible numeric types raise an error."""
    sample = test_data.head(5).copy()
    categoricals = feature_spec["categorical_features"]

    # 1. Missing required column
    sample_missing = sample.drop(columns=["hour"])
    with pytest.raises(KeyError, match=r"hour"):
        predict("lightgbm_l1", lightgbm_model, sample_missing, expected_feature_columns, categoricals)

    # 2. Incompatible numeric value (string injected into a numeric feature)
    sample_invalid_type = sample.copy()
    sample_invalid_type["elapsed_trip_s"] = "unparseable_text_string"
    with pytest.raises((ValueError, TypeError)):
        predict("lightgbm_l1", lightgbm_model, sample_invalid_type, expected_feature_columns, categoricals)


# -----------------------------------------------------------------------------
# 8. Test cold-start rows where has_prev_bus = 0
# -----------------------------------------------------------------------------
def test_cold_start_rows_prediction(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
    params: dict[str, Any],
):
    """Verify inference succeeds on cold-start segments where no preceding bus is present."""
    cold_start_df = test_data[test_data["has_prev_bus"] == 0]
    assert len(cold_start_df) > 0, "Test set must contain cold-start rows where has_prev_bus == 0"

    sample = cold_start_df.head(50)
    categoricals = feature_spec["categorical_features"]

    preds = predict("lightgbm_l1", lightgbm_model, sample, expected_feature_columns, categoricals)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(sample)
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert (preds > 0).all()

    max_bound = params.get("segments", {}).get("max_segment_seconds", 1800.0)
    assert (preds <= max_bound).all()


# -----------------------------------------------------------------------------
# 9. Measure warm single-row inference latency against params.yaml threshold
# -----------------------------------------------------------------------------
def test_single_row_inference_latency(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
    params: dict[str, Any],
):
    """Measure single-row prediction latency and verify median warm latency satisfies params.yaml."""
    max_latency_ms = params.get("quality_gate", {}).get("max_inference_latency_ms", 50.0)

    single_row = test_data.head(1).copy()
    categoricals = feature_spec["categorical_features"]

    # Warm-up phase (10 iterations)
    for _ in range(10):
        _ = predict("lightgbm_l1", lightgbm_model, single_row, expected_feature_columns, categoricals)

    # Measurement phase (30 iterations)
    durations_ms: list[float] = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = predict("lightgbm_l1", lightgbm_model, single_row, expected_feature_columns, categoricals)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(durations_ms))
    p95_ms = float(np.percentile(durations_ms, 95))
    min_ms = float(np.min(durations_ms))

    # Assert median warm inference latency satisfies configured threshold (50 ms),
    # with P95 and min reported diagnostically.
    assert median_ms <= max_latency_ms, (
        f"Single-row median inference latency ({median_ms:.2f} ms) exceeded maximum "
        f"allowed threshold ({max_latency_ms:.2f} ms). Diagnostic: min={min_ms:.2f} ms, P95={p95_ms:.2f} ms."
    )


# -----------------------------------------------------------------------------
# 10. Ensure model inference does not require W&B authentication or init a run
# -----------------------------------------------------------------------------
def test_hermetic_offline_execution(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that model inference does not require W&B authentication or initialize a W&B run.

    Explicitly proves that the model inference path does not invoke wandb.init().
    """
    assert os.environ.get("WANDB_MODE") == "disabled"

    def fail_if_wandb_init_called(*args, **kwargs):
        raise AssertionError("wandb.init was unexpectedly invoked during model inference")

    monkeypatch.setattr(wandb, "init", fail_if_wandb_init_called)

    sample = test_data.head(5)
    preds = predict(
        "lightgbm_l1", lightgbm_model, sample, expected_feature_columns, feature_spec["categorical_features"]
    )
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 5
    assert not np.isnan(preds).any()
    assert wandb.run is None, "Model inference must not initialize a W&B run"


# -----------------------------------------------------------------------------
# 11. Sensitivity check: prev_bus_tt_s increase must not cause sharp drop
# -----------------------------------------------------------------------------
def test_prev_bus_sensitivity_no_sharp_drop(
    lightgbm_model: lgb.LGBMRegressor,
    expected_feature_columns: list[str],
    feature_spec: dict[str, Any],
    test_data: pd.DataFrame,
    params: dict[str, Any],
):
    """Verify that when prev_bus_tt_s increases with all other features held constant,

    predicted travel time does not drop sharply. LightGBM was not trained with monotonic
    constraints, so small leaf-boundary fluctuations are expected; however, across a
    representative slice of rows with preceding buses (has_prev_bus == 1), the mean
    effect of +60s previous bus delay must be positive, and individual drops must not
    exceed max_prev_bus_drop_s (15s) from params.yaml.
    """
    max_allowed_drop_s = params.get("quality_gate", {}).get("max_prev_bus_drop_s", 15.0)

    # Select representative rows that have an active previous bus signal
    rows = test_data[test_data["has_prev_bus"] == 1].head(100).copy()
    assert len(rows) > 0, "Expected test rows with has_prev_bus == 1"

    categoricals = feature_spec["categorical_features"]
    base_preds = predict("lightgbm_l1", lightgbm_model, rows, expected_feature_columns, categoricals)

    # Increase prev_bus_tt_s by 60 seconds while holding every other feature constant
    perturbed_rows = rows.copy()
    perturbed_rows["prev_bus_tt_s"] = perturbed_rows["prev_bus_tt_s"] + 60.0
    perturbed_preds = predict("lightgbm_l1", lightgbm_model, perturbed_rows, expected_feature_columns, categoricals)

    diff = perturbed_preds - base_preds

    # 1. On average, increasing previous bus travel time must push predicted travel time up
    assert diff.mean() > 0, f"Mean sensitivity should be positive, got {diff.mean():.2f}s"

    # 2. No individual row may exhibit a sharp drop exceeding the configured tolerance
    worst_drop = float(-diff.min()) if diff.min() < 0 else 0.0
    assert worst_drop <= max_allowed_drop_s, (
        f"Individual row exhibited an unexpectedly sharp drop of {worst_drop:.2f}s when "
        f"prev_bus_tt_s increased, exceeding max_prev_bus_drop_s ({max_allowed_drop_s:.2f}s)."
    )
