import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root directory is in sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.time_features import add_time_features


def test_rush_hour_monday_0800():
    df = pd.DataFrame({"timestamp": ["2026-03-09 08:00:00"]})
    res = add_time_features(df)
    assert res.loc[0, "is_rush_hour"] == 1


def test_rush_hour_saturday_0800():
    df = pd.DataFrame({"timestamp": ["2026-03-14 08:00:00"]})
    res = add_time_features(df)
    assert res.loc[0, "is_rush_hour"] == 0


def test_rush_hour_monday_1300():
    df = pd.DataFrame({"timestamp": ["2026-03-09 13:00:00"]})
    res = add_time_features(df)
    assert res.loc[0, "is_rush_hour"] == 0


def test_is_weekend():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-03-14 10:00:00",  # Saturday
                "2026-03-15 10:00:00",  # Sunday
                "2026-03-09 10:00:00",  # Monday
            ]
        }
    )
    res = add_time_features(df)
    assert list(res["is_weekend"]) == [1, 1, 0]


def test_cyclic_encoding_range():
    hours = [f"2026-03-09 {h:02d}:00:00" for h in range(24)]
    df = pd.DataFrame({"timestamp": hours})
    res = add_time_features(df)
    assert (res["hour_sin"] >= -1.0).all() and (res["hour_sin"] <= 1.0).all()
    assert (res["hour_cos"] >= -1.0).all() and (res["hour_cos"] <= 1.0).all()


def test_equivalent_positions_24h_cycle():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-03-09 08:00:00",  # Monday 08:00
                "2026-03-10 08:00:00",  # Tuesday 08:00
                "2026-03-14 08:00:00",  # Saturday 08:00
            ]
        }
    )
    res = add_time_features(df)
    assert res.loc[0, "hour_sin"] == pytest.approx(res.loc[1, "hour_sin"])
    assert res.loc[0, "hour_sin"] == pytest.approx(res.loc[2, "hour_sin"])
    assert res.loc[0, "hour_cos"] == pytest.approx(res.loc[1, "hour_cos"])
    assert res.loc[0, "hour_cos"] == pytest.approx(res.loc[2, "hour_cos"])


def test_input_dataframe_not_modified():
    df = pd.DataFrame({"timestamp": ["2026-03-09 08:00:00"], "extra_col": [42]})
    df_copy = df.copy()
    _ = add_time_features(df)
    pd.testing.assert_frame_equal(df, df_copy)


def test_timezone_aware_toronto():
    ts_toronto = pd.date_range("2026-03-09 07:00:00", periods=3, freq="h", tz="America/Toronto")
    df = pd.DataFrame({"timestamp": ts_toronto})
    res = add_time_features(df)
    assert list(res["hour"]) == [7, 8, 9]
    assert list(res["day_of_week"]) == [0, 0, 0]
    assert list(res["is_rush_hour"]) == [1, 1, 1]
    assert list(res["is_weekend"]) == [0, 0, 0]
