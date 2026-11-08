"""Unit tests for src/models/evaluate.py.

Small hand-made DataFrames/arrays only; no real data files are read and no
W&B run is started.
"""

import numpy as np
import pandas as pd

from src.models.baselines import ColumnLookupRegressor, GlobalMedianRegressor
from src.models.evaluate import (
    bucket_travel_time,
    compute_daily_mae,
    delay_view,
    mae_by,
    mae_by_bucket,
    predict,
    within_n_seconds_pct,
    worst_segments,
)


def test_bucket_boundary_60_goes_to_60_120():
    assert bucket_travel_time(60) == "60-120"


def test_bucket_boundary_600_goes_to_600_plus():
    assert bucket_travel_time(600) == "600+"


def test_bucket_assignment_interior_values():
    assert bucket_travel_time(0) == "0-60"
    assert bucket_travel_time(59.9) == "0-60"
    assert bucket_travel_time(119.9) == "60-120"
    assert bucket_travel_time(120) == "120-300"
    assert bucket_travel_time(299.9) == "120-300"
    assert bucket_travel_time(300) == "300-600"
    assert bucket_travel_time(599.9) == "300-600"
    assert bucket_travel_time(1800) == "600+"


def test_within_n_seconds_pct_tiny_example():
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([100.0, 110.0, 135.0, 200.0])  # errors: 0, 10, 35, 100
    assert within_n_seconds_pct(y_true, y_pred, 30) == 50.0  # 2 of 4 within 30s
    assert within_n_seconds_pct(y_true, y_pred, 60) == 75.0  # 3 of 4 within 60s


def test_daily_mae_labels_val_and_test_days_correctly():
    val_df = pd.DataFrame({"day_index": [62, 62, 70], "travel_time_s": [100.0, 100.0, 100.0]})
    val_pred = np.array([110.0, 90.0, 100.0])  # abs errors: 10, 10, 0

    test_df = pd.DataFrame({"day_index": [77, 92], "travel_time_s": [100.0, 100.0]})
    test_pred = np.array([120.0, 100.0])  # abs errors: 20, 0

    daily = compute_daily_mae(val_df, val_pred, test_df, test_pred, "travel_time_s")

    val_rows = daily[daily["day_index"].isin([62, 70])]
    test_rows = daily[daily["day_index"].isin([77, 92])]

    assert (val_rows["split"] == "val").all()
    assert (test_rows["split"] == "test").all()
    assert set(daily["day_index"]) == {62, 70, 77, 92}

    day62_mae = daily[(daily["day_index"] == 62) & (daily["split"] == "val")]["mae"].iloc[0]
    assert day62_mae == 10.0


# --------------------------------------------------------------------------
# predict function tests
# --------------------------------------------------------------------------
def test_predict_baseline_model():
    df = pd.DataFrame({"typical_segment_hour_s": [45.0, 90.0]})
    model = ColumnLookupRegressor(column="typical_segment_hour_s").fit(df)
    preds = predict("typical_baseline", model, df, ["typical_segment_hour_s"], [])
    assert len(preds) == 2
    assert (preds >= 1.0).all()
    np.testing.assert_array_equal(preds, [45.0, 90.0])


def test_predict_global_median_model():
    df = pd.DataFrame({"travel_time_s": [100.0, 200.0]})
    model = GlobalMedianRegressor().fit(df, df["travel_time_s"])
    preds = predict("global_median", model, df, [], [])
    assert len(preds) == 2
    assert (preds == 150.0).all()


# --------------------------------------------------------------------------
# error-analysis aggregations
# --------------------------------------------------------------------------
def test_mae_by_and_worst_segments():
    df = pd.DataFrame(
        {
            "segment_id": ["seg_A", "seg_A", "seg_B", "seg_B"],
            "abs_error": [10.0, 20.0, 50.0, 70.0],
        }
    )
    res = mae_by(df, "segment_id")
    assert res.iloc[0]["segment_id"] == "seg_B"
    assert res.iloc[0]["mae"] == 60.0
    assert res.iloc[1]["segment_id"] == "seg_A"
    assert res.iloc[1]["mae"] == 15.0

    worst = worst_segments(df, 1)
    assert len(worst) == 1
    assert worst.iloc[0]["segment_id"] == "seg_B"


def test_mae_by_bucket_preserves_order():
    df = pd.DataFrame(
        {
            "travel_time_s": [30.0, 90.0, 200.0, 450.0, 800.0],
            "abs_error": [5.0, 10.0, 20.0, 30.0, 50.0],
        }
    )
    res = mae_by_bucket(df)
    assert list(res["bucket"]) == ["0-60", "60-120", "120-300", "300-600", "600+"]
    assert res.loc[res["bucket"] == "0-60", "mae"].iloc[0] == 5.0
    assert res.loc[res["bucket"] == "600+", "mae"].iloc[0] == 50.0


def test_delay_view_computes_breakdown():
    df = pd.DataFrame(
        {
            "travel_time_s": [120.0, 180.0, 50.0],
            "delay_s": [70.0, 100.0, 10.0],  # 2 of 3 > 60s
            "abs_error": [20.0, 40.0, 5.0],
        }
    )
    typical_pred = np.array([100.0, 120.0, 50.0])
    recent_pred = np.array([110.0, 150.0, 48.0])

    view = delay_view(df, typical_pred, recent_pred, threshold_s=60.0)
    assert abs(view["share_of_rows_delayed_gt_60s"] - (2 / 3)) < 1e-6
    assert view["best_model_mae_on_delayed_rows"] == 30.0  # mean of [20, 40]
    assert view["typical_baseline_mae_on_delayed_rows"] == 40.0  # mean of [|120-100|, |180-120|] = [20, 60] / 2
