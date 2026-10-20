import numpy as np
import pandas as pd
import pytest

from src.features.time_features import add_time_features, add_time_of_day_features


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


def test_time_of_day_hour_and_minute():
    df = pd.DataFrame({"depart_time_s": [0, 3661, 86399]})
    res = add_time_of_day_features(df)
    assert list(res["hour"]) == [0, 1, 23]
    assert list(res["minute_of_day"]) == [0, 61, 1439]


def test_time_of_day_wraps_past_midnight():
    # a midnight-crossing trip's depart_time_s can exceed 86400
    df = pd.DataFrame({"depart_time_s": [86400 + 3600]})  # "25:00" -> wraps to hour 1
    res = add_time_of_day_features(df)
    assert res.loc[0, "hour"] == 1


def test_is_peak_am_boundaries():
    df = pd.DataFrame({"depart_time_s": [7 * 3600, 8 * 3600, 10 * 3600, 11 * 3600]})
    res = add_time_of_day_features(df, peak_am=(8, 10))
    assert list(res["is_peak_am"]) == [0, 1, 1, 0]


def test_is_peak_pm_boundaries():
    df = pd.DataFrame({"depart_time_s": [16 * 3600, 17 * 3600, 20 * 3600, 21 * 3600]})
    res = add_time_of_day_features(df, peak_pm=(17, 20))
    assert list(res["is_peak_pm"]) == [0, 1, 1, 0]


def test_time_of_day_cyclic_encoding_range():
    df = pd.DataFrame({"depart_time_s": [h * 3600 for h in range(24)]})
    res = add_time_of_day_features(df)
    assert (res["hour_sin"] >= -1.0).all() and (res["hour_sin"] <= 1.0).all()
    assert (res["hour_cos"] >= -1.0).all() and (res["hour_cos"] <= 1.0).all()


def test_relative_weekday_becomes_category():
    df = pd.DataFrame({"depart_time_s": [0, 3600], "relative_weekday": [0, 3]})
    res = add_time_of_day_features(df)
    assert isinstance(res["relative_weekday"].dtype, pd.CategoricalDtype)
