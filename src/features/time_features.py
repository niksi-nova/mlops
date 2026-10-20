import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Extract temporal features from a timestamp column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the timestamp column.
    ts_col : str, default="timestamp"
        Name of the column containing timestamp data.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the following new temporal feature columns:
        - hour: Hour of the day (0-23)
        - day_of_week: Day of the week (Monday=0, Sunday=6)
        - month: Month of the year (1-12)
        - year: Calendar year
        - is_weekend: 1 if Saturday or Sunday, otherwise 0
        - is_rush_hour: 1 on weekdays when hour is 7-9 or 16-18 inclusive, otherwise 0
        - hour_sin: Sine cyclic encoding of hour (period 24)
        - hour_cos: Cosine cyclic encoding of hour (period 24)
        - month_sin: Sine cyclic encoding of month (period 12)
        - month_cos: Cosine cyclic encoding of month (period 12)
    """
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])

    out["hour"] = ts.dt.hour.astype(int)
    out["day_of_week"] = ts.dt.dayofweek.astype(int)
    out["month"] = ts.dt.month.astype(int)
    out["year"] = ts.dt.year.astype(int)

    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    is_weekday = ts.dt.dayofweek < 5
    is_morning_rush = (ts.dt.hour >= 7) & (ts.dt.hour <= 9)
    is_evening_rush = (ts.dt.hour >= 16) & (ts.dt.hour <= 18)
    out["is_rush_hour"] = (is_weekday & (is_morning_rush | is_evening_rush)).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12.0)

    return out


def add_time_of_day_features(
    df: pd.DataFrame,
    seconds_col: str = "depart_time_s",
    peak_am: tuple[int, int] = (8, 10),
    peak_pm: tuple[int, int] = (17, 20),
) -> pd.DataFrame:
    """Extract time-of-day features from a seconds-since-midnight column.

    Used for the Delhi bus data, where there is no real timestamp - only
    `seconds_col` (seconds since midnight, possibly > 86400 for a
    midnight-crossing trip) and a separate `relative_weekday` column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing `seconds_col` and (optionally)
        `relative_weekday`.
    seconds_col : str, default="depart_time_s"
        Name of the column holding seconds since midnight.
    peak_am, peak_pm : tuple[int, int]
        Inclusive (start_hour, end_hour) ranges that define the morning and
        evening peaks (e.g. from params.yaml's `featurize.peak_am/peak_pm`).

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the following new columns:
        - hour: hour of day, 0-23 (seconds_col wrapped to a single day)
        - minute_of_day: minutes since midnight, 0-1439
        - hour_sin, hour_cos: cyclic encoding of hour (period 24)
        - is_peak_am: 1 if hour is within `peak_am` (inclusive), else 0
        - is_peak_pm: 1 if hour is within `peak_pm` (inclusive), else 0
        `relative_weekday`, if present, is cast to a category dtype.
    """
    out = df.copy()
    seconds_of_day = out[seconds_col] % 86400

    out["hour"] = (seconds_of_day // 3600).astype(int)
    out["minute_of_day"] = (seconds_of_day // 60).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    out["is_peak_am"] = out["hour"].between(peak_am[0], peak_am[1]).astype(int)
    out["is_peak_pm"] = out["hour"].between(peak_pm[0], peak_pm[1]).astype(int)

    if "relative_weekday" in out.columns:
        out["relative_weekday"] = out["relative_weekday"].astype("category")

    return out
