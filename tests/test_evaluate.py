"""Unit tests for src/models/evaluate.py.

Small hand-made DataFrames/arrays only; no real data files are read and no
W&B run is started.
"""

import numpy as np
import pandas as pd

from src.models.evaluate import bucket_travel_time, compute_daily_mae, within_n_seconds_pct


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
