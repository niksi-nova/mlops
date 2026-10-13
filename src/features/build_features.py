"""Build the train/val/test feature tables for the TTC delay model.

Reads:
  - data/interim/merged.parquet
  - params.yaml (featurize section)

Writes:
  - data/processed/train.parquet
  - data/processed/val.parquet
  - data/processed/test.parquet
  - data/processed/feature_spec.json
  - reports/featurize_report.json

The split is by time, not random: train = timestamp <= train_end,
val = year == val_year, test = year == test_year. Rare-value grouping for
route/location is fitted on the train split only, then applied unchanged to
val/test (unseen or rare values become "OTHER"), so no information from
val/test leaks into feature construction.

Usage:
    uv run python src/features/build_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import holidays
import pandas as pd
import yaml

from src.features.time_features import add_time_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
INPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "merged.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SPEC_PATH = OUTPUT_DIR / "feature_spec.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "featurize_report.json"

TIMEZONE = "America/Toronto"
TARGET = "min_delay"
OTHER_LABEL = "OTHER"
UNKNOWN_LABEL = "UNKNOWN"

WEATHER_NUMERIC_FEATURES = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]
TIME_NUMERIC_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
FLAG_FEATURES = ["is_holiday", "covid_period", "is_snowing", "heavy_precip"]
NUMERIC_FEATURES = WEATHER_NUMERIC_FEATURES + TIME_NUMERIC_FEATURES + FLAG_FEATURES
CATEGORICAL_FEATURES = ["route", "incident", "direction", "location"]
RARE_GROUPED_COLUMNS = ["route", "location"]


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def split_by_time(df: pd.DataFrame, featurize_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into train/val/test by timestamp (not randomly).

    train = timestamp <= train_end (inclusive of the whole last day)
    val   = year == val_year
    test  = year == test_year
    """
    train_end = pd.Timestamp(featurize_params["train_end"]).tz_localize(TIMEZONE)
    train_end_inclusive = train_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    train = df[df["timestamp"] <= train_end_inclusive].copy()
    val = df[df["timestamp"].dt.year == featurize_params["val_year"]].copy()
    test = df[df["timestamp"].dt.year == featurize_params["test_year"]].copy()
    return train, val, test


def fill_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing `incident`/`location` values with the UNKNOWN label."""
    df = df.copy()
    df["incident"] = df["incident"].fillna(UNKNOWN_LABEL)
    df["location"] = df["location"].fillna(UNKNOWN_LABEL)
    return df


def fit_rare_value_sets(train_df: pd.DataFrame, featurize_params: dict) -> dict[str, set[str]]:
    """Fit the set of values to keep (count >= threshold, in TRAIN only)
    for each rare-grouped categorical column."""
    thresholds = {
        "route": featurize_params["rare_route_min_count"],
        "location": featurize_params["rare_location_min_count"],
    }
    kept: dict[str, set[str]] = {}
    for column in RARE_GROUPED_COLUMNS:
        counts = train_df[column].value_counts()
        kept[column] = set(counts[counts >= thresholds[column]].index)
    return kept


def apply_rare_value_sets(df: pd.DataFrame, kept_values: dict[str, set[str]]) -> pd.DataFrame:
    """Replace any value not in the fitted kept set with OTHER_LABEL.
    Applied identically to train/val/test, so unseen (val/test-only)
    values are also mapped to OTHER."""
    df = df.copy()
    for column, kept in kept_values.items():
        df[column] = df[column].where(df[column].isin(kept), OTHER_LABEL)
    return df


def add_derived_features(df: pd.DataFrame, featurize_params: dict, ca_on_holidays: holidays.HolidayBase) -> pd.DataFrame:
    """Add time features (via `add_time_features`), the holiday flag, the
    COVID-period flag, and the weather flags. Stateless: safe to call on
    train/val/test independently."""
    df = add_time_features(df, ts_col="timestamp")

    dates = df["timestamp"].dt.date
    df["is_holiday"] = dates.map(lambda d: d in ca_on_holidays).astype(int)

    covid_start = pd.Timestamp(featurize_params["covid_start"]).tz_localize(TIMEZONE)
    covid_end = pd.Timestamp(featurize_params["covid_end"]).tz_localize(TIMEZONE) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    df["covid_period"] = ((df["timestamp"] >= covid_start) & (df["timestamp"] <= covid_end)).astype(int)

    df["is_snowing"] = (df["snowfall"] > 0).astype(int)
    df["heavy_precip"] = (df["precipitation"] > featurize_params["heavy_precip_mm"]).astype(int)

    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only feature columns + target + timestamp (for reference)."""
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "timestamp"]].reset_index(drop=True)


def other_share(df: pd.DataFrame, column: str) -> float:
    """Share of rows in `df` where `column` equals OTHER_LABEL."""
    if len(df) == 0:
        return 0.0
    return float((df[column] == OTHER_LABEL).mean())


def date_range(df: pd.DataFrame) -> dict[str, str]:
    """Min/max timestamp in `df`, as ISO strings."""
    return {"min": str(df["timestamp"].min()), "max": str(df["timestamp"].max())}


def main() -> int:
    params = load_params()
    featurize_params = params["featurize"]

    df = pd.read_parquet(INPUT_PARQUET)
    train_raw, val_raw, test_raw = split_by_time(df, featurize_params)

    ca_on_holidays = holidays.Canada(subdiv="ON")

    train = add_derived_features(fill_missing_categoricals(train_raw), featurize_params, ca_on_holidays)
    val = add_derived_features(fill_missing_categoricals(val_raw), featurize_params, ca_on_holidays)
    test = add_derived_features(fill_missing_categoricals(test_raw), featurize_params, ca_on_holidays)

    rare_value_sets = fit_rare_value_sets(train, featurize_params)
    train = apply_rare_value_sets(train, rare_value_sets)
    val = apply_rare_value_sets(val, rare_value_sets)
    test = apply_rare_value_sets(test, rare_value_sets)

    train_out = select_output_columns(train)
    val_out = select_output_columns(val)
    test_out = select_output_columns(test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_out.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test_out.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    print(f"Wrote {len(train_out):,} / {len(val_out):,} / {len(test_out):,} rows "
          f"(train/val/test) to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    feature_spec = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "kept_routes": sorted(rare_value_sets["route"]),
        "kept_locations": sorted(rare_value_sets["location"]),
        "params": featurize_params,
    }
    FEATURE_SPEC_PATH.write_text(json.dumps(feature_spec, indent=2))
    print(f"Wrote feature spec to {FEATURE_SPEC_PATH.relative_to(PROJECT_ROOT)}")

    report = {
        "rows": {"train": len(train_out), "val": len(val_out), "test": len(test_out)},
        "date_range": {
            "train": date_range(train_out),
            "val": date_range(val_out),
            "test": date_range(test_out),
        },
        "n_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "n_kept_routes": len(rare_value_sets["route"]),
        "n_kept_locations": len(rare_value_sets["location"]),
        "other_share": {
            "train": {"route": other_share(train_out, "route"), "location": other_share(train_out, "location")},
            "val": {"route": other_share(val_out, "route"), "location": other_share(val_out, "location")},
            "test": {"route": other_share(test_out, "route"), "location": other_share(test_out, "location")},
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote featurize report to {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
