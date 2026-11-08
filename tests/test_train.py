"""Unit tests for src/models/train.py and src/models/baselines.py.

All tests use small hand-made DataFrames; no real data files are read and
no W&B run is started (WANDB_MODE=disabled).
"""

import os

import numpy as np
import pandas as pd

from src.models.baselines import ColumnLookupRegressor, GlobalMedianRegressor
from src.models.train import (
    MIN_PREDICTION_S,
    clip_and_backtransform,
    prep_lightgbm_features,
    regression_metrics,
    select_best_baseline,
    train_lightgbm,
    train_random_forest,
    train_ridge,
)

os.environ["WANDB_MODE"] = "disabled"


# --------------------------------------------------------------------------
# GlobalMedianRegressor
# --------------------------------------------------------------------------
def test_global_median_predicts_constant_train_median():
    train = pd.DataFrame({"travel_time_s": [5.0, 10.0, 10.0, 15.0, 100.0]})
    model = GlobalMedianRegressor().fit(train, train["travel_time_s"])

    other_rows = pd.DataFrame({"travel_time_s": [0.0, 0.0, 0.0]})
    pred = model.predict(other_rows)

    assert (pred == 10.0).all()  # median of [5, 10, 10, 15, 100]


# --------------------------------------------------------------------------
# ColumnLookupRegressor - typical_baseline / recent_buses_baseline
# --------------------------------------------------------------------------
def test_column_lookup_returns_typical_segment_hour_column():
    df = pd.DataFrame({"typical_segment_hour_s": [42.0, 99.0], "other_col": [1, 2]})
    model = ColumnLookupRegressor(column="typical_segment_hour_s").fit(df)
    pred = model.predict(df)
    np.testing.assert_array_equal(pred, [42.0, 99.0])


def test_column_lookup_returns_prev3_median_column():
    df = pd.DataFrame({"prev3_median_tt_s": [15.0, 200.0]})
    model = ColumnLookupRegressor(column="prev3_median_tt_s").fit(df)
    pred = model.predict(df)
    np.testing.assert_array_equal(pred, [15.0, 200.0])


def test_column_lookup_reflects_featurize_fallback_values():
    # featurize.py already fills "no previous bus" rows with the historical
    # typical value - the baseline just needs to read that column as-is,
    # fallback or not.
    df = pd.DataFrame({"prev3_median_tt_s": [80.0], "has_prev_bus": [0]})
    model = ColumnLookupRegressor(column="prev3_median_tt_s").fit(df)
    pred = model.predict(df)
    assert pred[0] == 80.0


# --------------------------------------------------------------------------
# predictions are never below MIN_PREDICTION_S
# --------------------------------------------------------------------------
def test_clip_and_backtransform_floors_at_min_prediction():
    raw = np.array([-5.0, -0.1, 0.0, 0.5, 50.0])
    out = clip_and_backtransform(raw, log_target=False)
    assert (out >= MIN_PREDICTION_S).all()
    assert out[-1] == 50.0


# --------------------------------------------------------------------------
# log1p / expm1 round trip
# --------------------------------------------------------------------------
def test_log1p_expm1_round_trip_recovers_original_values():
    original = np.array([1.0, 10.0, 137.0])  # all >= MIN_PREDICTION_S, so clip is a no-op
    transformed = np.log1p(original)
    recovered = clip_and_backtransform(transformed, log_target=True)
    np.testing.assert_allclose(recovered, original, rtol=1e-10)


# --------------------------------------------------------------------------
# best-baseline selection
# --------------------------------------------------------------------------
def test_select_best_baseline_picks_lower_val_mae():
    results = {
        "typical_baseline": {"val_mae": 50.0},
        "recent_buses_baseline": {"val_mae": 30.0},
    }
    assert select_best_baseline(results) == "recent_buses_baseline"


def test_select_best_baseline_other_direction():
    results = {
        "typical_baseline": {"val_mae": 20.0},
        "recent_buses_baseline": {"val_mae": 45.0},
    }
    assert select_best_baseline(results) == "typical_baseline"


# --------------------------------------------------------------------------
# Regression Metrics
# --------------------------------------------------------------------------
def test_regression_metrics_computes_expected_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])  # abs errors: 2, 2, 5 -> mean 3.0, median 2.0
    metrics = regression_metrics(y_true, y_pred, prefix="val")

    assert metrics["val_mae"] == 3.0
    assert metrics["val_median_ae"] == 2.0
    assert metrics["val_within_30s_pct"] == 100.0
    assert metrics["val_within_60s_pct"] == 100.0
    assert "val_rmse" in metrics
    assert "val_r2" in metrics


# --------------------------------------------------------------------------
# Model Training Smoke Tests
# --------------------------------------------------------------------------
def _make_dummy_dataset(n_samples: int = 40):
    np.random.seed(42)
    numeric_features = ["nodes_spanned", "elapsed_trip_s", "typical_segment_hour_s"]
    categorical_features = ["route", "direction", "segment_id"]
    df = pd.DataFrame(
        {
            "nodes_spanned": np.random.randint(1, 4, size=n_samples),
            "elapsed_trip_s": np.random.uniform(50.0, 500.0, size=n_samples),
            "typical_segment_hour_s": np.random.uniform(60.0, 180.0, size=n_samples),
            "route": np.random.choice(["425", "501"], size=n_samples),
            "direction": np.random.choice(["up", "down"], size=n_samples),
            "segment_id": np.random.choice(["seg_1", "seg_2", "seg_3"], size=n_samples),
            "travel_time_s": np.random.uniform(30.0, 300.0, size=n_samples),
        }
    )
    return df, numeric_features, categorical_features


def test_prep_lightgbm_features_categoricals_and_categories_preservation():
    df, num_cols, cat_cols = _make_dummy_dataset(20)
    feat_cols = cat_cols + num_cols

    X_train = prep_lightgbm_features(df, feat_cols, cat_cols)
    for col in cat_cols:
        assert str(X_train[col].dtype) == "category"

    train_categories = {col: X_train[col].cat.categories for col in cat_cols}

    val_df = pd.DataFrame(
        {
            "nodes_spanned": [1],
            "elapsed_trip_s": [100.0],
            "typical_segment_hour_s": [80.0],
            "route": ["unseen_route"],
            "direction": ["up"],
            "segment_id": ["seg_1"],
            "travel_time_s": [90.0],
        }
    )
    X_val = prep_lightgbm_features(val_df, feat_cols, cat_cols, train_categories=train_categories)
    assert pd.isna(X_val["route"].iloc[0])  # unseen mapped to NaN/missing category code
    assert X_val["direction"].iloc[0] == "up"


def test_train_ridge_smoke():
    df, num_cols, cat_cols = _make_dummy_dataset(30)
    train_df, val_df = df.iloc[:20], df.iloc[20:]

    model, train_pred, val_pred, elapsed = train_ridge(
        train_df, val_df, "travel_time_s", num_cols, cat_cols, {"alpha": 1.0}
    )
    assert hasattr(model, "predict")
    assert len(train_pred) == len(train_df)
    assert len(val_pred) == len(val_df)
    assert (train_pred >= MIN_PREDICTION_S).all()
    assert (val_pred >= MIN_PREDICTION_S).all()
    assert elapsed > 0


def test_train_random_forest_smoke():
    df, num_cols, cat_cols = _make_dummy_dataset(30)
    train_df, val_df = df.iloc[:20], df.iloc[20:]

    rf_params = {"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1}
    model, train_pred, val_pred, elapsed, feat_names, importances = train_random_forest(
        train_df, val_df, "travel_time_s", num_cols, cat_cols, rf_params, random_state=42
    )
    assert len(feat_names) == len(cat_cols) + len(num_cols)
    assert len(importances) == len(feat_names)
    assert len(train_pred) == len(train_df)
    assert len(val_pred) == len(val_df)
    assert (train_pred >= MIN_PREDICTION_S).all()
    assert elapsed > 0


def test_train_lightgbm_smoke():
    df, num_cols, cat_cols = _make_dummy_dataset(40)
    train_df, val_df = df.iloc[:25], df.iloc[25:]

    lgbm_params = {
        "n_estimators": 10,
        "num_leaves": 7,
        "learning_rate": 0.1,
        "early_stopping_rounds": 5,
    }
    model, train_pred, val_pred, elapsed, feat_cols, importances = train_lightgbm(
        train_df, val_df, "travel_time_s", num_cols, cat_cols, lgbm_params, log_target=False, random_state=42
    )
    assert len(feat_cols) == len(cat_cols) + len(num_cols)
    assert len(importances) == len(feat_cols)
    assert len(train_pred) == len(train_df)
    assert len(val_pred) == len(val_df)
    assert (train_pred >= MIN_PREDICTION_S).all()
    assert (val_pred >= MIN_PREDICTION_S).all()
    assert elapsed > 0
