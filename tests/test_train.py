"""Unit tests for src/models/train.py and src/models/baselines.py.

All tests use small hand-made DataFrames; no real data files are read and
no W&B run is started (WANDB_MODE=disabled).
"""

import os

import numpy as np
import pandas as pd

from src.models.baselines import ColumnLookupRegressor, GlobalMedianRegressor
from src.models.train import MIN_PREDICTION_S, clip_and_backtransform, select_best_baseline

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
