"""Build the train/val/test feature tables for the Delhi bus travel-time model.

Reads:
  - data/interim/segments.parquet
  - params.yaml (featurize section)

Writes:
  - data/processed/train.parquet
  - data/processed/val.parquet
  - data/processed/test.parquet
  - data/processed/feature_spec.json
  - reports/featurize_report.json

The split is by day_index (not random): train/val/test are disjoint,
non-overlapping day_index ranges from params.yaml. Since every trip's
segments share a single day_index (see docs/data_inventory.md /
src/data/clean.py), a trip is always wholly inside one split - segment
pairs are never cut across train/val/test.

Historical "typical travel time" and vehicle target-encoding tables are
fitted on TRAIN only; train rows get out-of-fold values (GroupKFold by
day_index) so a row never sees its own day's statistics, while val/test get
values from the table fitted on the *full* train split. Real-time
"recent conditions" features (previous bus, headway, ...) use only buses
that have *already completed* the segment by the time the current bus
departs, within the same split (days never cross a split boundary).

Usage:
    uv run python -m src.features.build_features
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from src.features.time_features import add_time_of_day_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
INPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "segments.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SPEC_PATH = OUTPUT_DIR / "feature_spec.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "featurize_report.json"

TARGET = "travel_time_s"
OTHER_LABEL = "OTHER"
TE_VEHICLE_SMOOTHING = 20.0
TE_RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["route", "direction", "segment_id", "relative_weekday"]
NUMERIC_FEATURES = [
    "from_node",
    "nodes_spanned",
    "route_position",
    "stops_into_trip",
    "elapsed_trip_s",
    "hour",
    "minute_of_day",
    "hour_sin",
    "hour_cos",
    "is_peak_am",
    "is_peak_pm",
    "prev_bus_tt_s",
    "prev3_median_tt_s",
    "headway_s",
    "n_buses_recent",
    "has_prev_bus",
    "own_prev_segment_tt_s",
    "own_prev_segment_ratio",
    "typical_segment_hour_s",
    "typical_segment_s",
    "te_vehicle",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PREV_FEATURES_FOR_REPORT = [
    "prev_bus_tt_s",
    "prev3_median_tt_s",
    "headway_s",
    "n_buses_recent",
    "own_prev_segment_tt_s",
    "own_prev_segment_ratio",
]
# non-feature columns kept in the output parquet for traceability/evaluation
EXTRA_OUTPUT_COLUMNS = ["trip_id", "day_index", "vehicle", TARGET, "delay_s"]


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------- static features


def add_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["segment_id"] = df["route"] + "_" + df["direction"] + "_" + df["from_node"].astype(str)
    return df


def add_route_position(df: pd.DataFrame) -> pd.DataFrame:
    """route_position = from_node / max node reached on that route+direction.

    The max-node table is a static topology fact (not derived from the
    target), so it is safe to compute from the full dataset.
    """
    df = df.copy()
    max_node = df.groupby(["route", "direction"])["to_node"].transform("max")
    df["route_position"] = df["from_node"] / max_node
    return df


def add_trip_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """stops_into_trip (1-based segment index within the trip) and
    elapsed_trip_s (seconds since the trip's first stop arrival)."""
    df = df.copy()
    df = df.sort_values(["trip_id", "depart_time_s"])
    df["stops_into_trip"] = df.groupby("trip_id").cumcount() + 1
    trip_start_s = df.groupby("trip_id")["depart_time_s"].transform("min")
    df["elapsed_trip_s"] = df["depart_time_s"] - trip_start_s
    return df.sort_index()


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_segment_id(df)
    df = add_route_position(df)
    df = add_trip_position_features(df)
    return df


# --------------------------------------------------------------------- splitting


def split_by_day(df: pd.DataFrame, featurize_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by day_index into disjoint train/val/test ranges (inclusive)."""
    train_lo, train_hi = featurize_params["train_days"]
    val_lo, val_hi = featurize_params["val_days"]
    test_lo, test_hi = featurize_params["test_days"]

    train = df[df["day_index"].between(train_lo, train_hi)].reset_index(drop=True)
    val = df[df["day_index"].between(val_lo, val_hi)].reset_index(drop=True)
    test = df[df["day_index"].between(test_lo, test_hi)].reset_index(drop=True)
    return train, val, test


# --------------------------------------------------------------------- historical (typical) features


def make_day_folds(train_df: pd.DataFrame, n_folds: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold splits of train_df's positional index, grouped by day_index,
    so a fold's holdout rows never share a day with that fold's fit rows."""
    kfold = GroupKFold(n_splits=n_folds)
    return list(kfold.split(train_df, groups=train_df["day_index"]))


def fit_historical_tables(df: pd.DataFrame) -> dict[str, dict]:
    """Median travel_time_s per (segment_id, hour), per segment_id, and per
    (route, direction), as plain dicts keyed by tuples (or segment_id)."""
    hour_table = df.groupby(["segment_id", "hour"])[TARGET].median().to_dict()
    segment_table = df.groupby("segment_id")[TARGET].median().to_dict()
    route_dir_table = df.groupby(["route", "direction"])[TARGET].median().to_dict()
    return {"hour": hour_table, "segment": segment_table, "route_direction": route_dir_table}


def apply_historical_tables(df: pd.DataFrame, tables: dict[str, dict]) -> tuple[pd.Series, pd.Series]:
    """Look up typical_segment_hour_s / typical_segment_s for every row in df,
    falling back hour -> segment -> route+direction when a key is unseen."""
    hour_keys = pd.Series(list(zip(df["segment_id"], df["hour"])), index=df.index)
    route_dir_keys = pd.Series(list(zip(df["route"], df["direction"])), index=df.index)

    val_hour = hour_keys.map(tables["hour"])
    val_segment = df["segment_id"].map(tables["segment"])
    val_route_dir = route_dir_keys.map(tables["route_direction"])

    typical_segment_hour_s = val_hour.fillna(val_segment).fillna(val_route_dir)
    typical_segment_s = val_segment.fillna(val_route_dir)
    return typical_segment_hour_s, typical_segment_s


def compute_oof_historical_features(
    train_df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold typical_segment_hour_s / typical_segment_s for every TRAIN
    row: each fold's holdout rows are scored using tables fitted only on
    that fold's fit rows (which exclude the holdout rows' own days)."""
    hour_out = np.empty(len(train_df), dtype=float)
    segment_out = np.empty(len(train_df), dtype=float)
    for fit_idx, holdout_idx in folds:
        tables = fit_historical_tables(train_df.iloc[fit_idx])
        hour_vals, segment_vals = apply_historical_tables(train_df.iloc[holdout_idx], tables)
        hour_out[holdout_idx] = hour_vals.to_numpy()
        segment_out[holdout_idx] = segment_vals.to_numpy()
    return hour_out, segment_out


# --------------------------------------------------------------------- vehicle target encoding


def fit_rare_vehicle_set(train_df: pd.DataFrame, min_count: int) -> set[str]:
    counts = train_df["vehicle"].value_counts()
    return set(counts[counts >= min_count].index)


def apply_vehicle_grouping(df: pd.DataFrame, kept_vehicles: set[str]) -> pd.Series:
    """Map rare/unseen vehicles to OTHER_LABEL; kept vehicles pass through."""
    return df["vehicle"].where(df["vehicle"].isin(kept_vehicles), OTHER_LABEL)


def fit_smoothed_vehicle_means(df: pd.DataFrame, smoothing: float, global_mean: float) -> pd.Series:
    grouped = df.groupby("vehicle_grouped")[TARGET].agg(["sum", "count"])
    return (grouped["sum"] + global_mean * smoothing) / (grouped["count"] + smoothing)


def apply_smoothed_vehicle_means(df: pd.DataFrame, table: pd.Series, global_mean: float) -> pd.Series:
    return df["vehicle_grouped"].map(table).fillna(global_mean).astype(float)


def compute_oof_te_vehicle(
    train_df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], smoothing: float, global_mean: float
) -> np.ndarray:
    out = np.empty(len(train_df), dtype=float)
    for fit_idx, holdout_idx in folds:
        table = fit_smoothed_vehicle_means(train_df.iloc[fit_idx], smoothing, global_mean)
        out[holdout_idx] = apply_smoothed_vehicle_means(train_df.iloc[holdout_idx], table, global_mean).to_numpy()
    return out


# --------------------------------------------------------------------- recent-conditions (previous bus) features


def compute_recent_bus_features(df: pd.DataFrame, prev_n_buses: int, recent_window_s: float) -> pd.DataFrame:
    """For each segment (route, direction, from_node, to_node) on a given
    day, compute, for every bus, the travel time of the most recent EARLIER
    bus whose arrival at to_node is <= this bus's depart_time_s (not simply
    the previous row in departure order - a slower earlier bus may still be
    "in flight"), the median of up to the last prev_n_buses such buses, the
    headway since that previous bus's own departure, and how many buses
    completed the segment within the last recent_window_s.

    Rows with no qualifying previous bus get has_prev_bus=0; prev_bus_tt_s
    and prev3_median_tt_s are left NaN here (the caller fills them with the
    historical typical value) and headway_s is filled with 0 (no measurable
    gap - this is the first bus of the day on this segment).
    """
    df = df.copy()
    df["arrival_time_s"] = df["depart_time_s"] + df[TARGET]

    n = len(df)
    prev_tt = np.full(n, np.nan)
    prev3_median = np.full(n, np.nan)
    headway = np.full(n, np.nan)
    n_recent = np.zeros(n, dtype=int)
    has_prev = np.zeros(n, dtype=int)

    group_cols = ["route", "direction", "from_node", "to_node", "day_index"]
    for _, group in df.groupby(group_cols, sort=False):
        ordered = group.sort_values("arrival_time_s")
        arrivals = ordered["arrival_time_s"].to_numpy()
        travel_times = ordered[TARGET].to_numpy()
        departs_in_arrival_order = ordered["depart_time_s"].to_numpy()

        depart_query = group["depart_time_s"].to_numpy()
        n_qualifying = np.searchsorted(arrivals, depart_query, side="right")
        has_prev_group = n_qualifying > 0
        last_idx = np.clip(n_qualifying - 1, 0, None)

        prev_tt_group = np.where(has_prev_group, travel_times[last_idx], np.nan)
        prev_depart_group = np.where(has_prev_group, departs_in_arrival_order[last_idx], np.nan)
        headway_group = depart_query - prev_depart_group

        lower = np.searchsorted(arrivals, depart_query - recent_window_s, side="right")
        n_recent_group = n_qualifying - lower

        prev3_group = np.full(len(group), np.nan)
        for i, k in enumerate(n_qualifying):
            if k > 0:
                window = travel_times[max(0, k - prev_n_buses) : k]
                prev3_group[i] = float(np.median(window))

        idx = group.index.to_numpy()
        prev_tt[idx] = prev_tt_group
        prev3_median[idx] = prev3_group
        headway[idx] = headway_group
        n_recent[idx] = n_recent_group
        has_prev[idx] = has_prev_group.astype(int)

    df["prev_bus_tt_s"] = prev_tt
    df["prev3_median_tt_s"] = prev3_median
    df["headway_s"] = headway
    df["n_buses_recent"] = n_recent
    df["has_prev_bus"] = has_prev
    return df.drop(columns=["arrival_time_s"])


def fill_missing_recent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Where has_prev_bus == 0, fill prev_bus_tt_s / prev3_median_tt_s with
    the row's own historical typical value, and headway_s with 0."""
    df = df.copy()
    no_prev = df["has_prev_bus"] == 0
    df.loc[no_prev, "prev_bus_tt_s"] = df.loc[no_prev, "typical_segment_hour_s"]
    df.loc[no_prev, "prev3_median_tt_s"] = df.loc[no_prev, "typical_segment_hour_s"]
    df.loc[no_prev, "headway_s"] = 0.0
    return df


# --------------------------------------------------------------------- own-previous-segment features


def add_own_prev_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """This bus's own travel time on its previous segment in the same trip,
    and that time divided by the (already-computed) typical time for that
    previous segment. The first segment of a trip has no previous segment:
    own_prev_segment_tt_s falls back to the row's own typical_segment_s and
    own_prev_segment_ratio is set to the neutral value 1.0.
    """
    df = df.copy()
    df = df.sort_values(["trip_id", "depart_time_s"])
    grouped = df.groupby("trip_id", sort=False)

    own_prev_tt = grouped[TARGET].shift(1)
    prev_typical = grouped["typical_segment_s"].shift(1)

    has_prev_segment = own_prev_tt.notna()
    df["own_prev_segment_tt_s"] = own_prev_tt.where(has_prev_segment, df["typical_segment_s"])
    df["own_prev_segment_ratio"] = (own_prev_tt / prev_typical).where(has_prev_segment, 1.0)
    return df.sort_index()


# --------------------------------------------------------------------- orchestration per split


def add_historical_features_to_train(
    train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[pd.DataFrame, dict, pd.Series]:
    hour_oof, segment_oof = compute_oof_historical_features(train, folds)
    train = train.copy()
    train["typical_segment_hour_s"] = hour_oof
    train["typical_segment_s"] = segment_oof

    full_train_tables = fit_historical_tables(train)

    global_mean = float(train[TARGET].mean())
    vehicle_table = fit_smoothed_vehicle_means(train, TE_VEHICLE_SMOOTHING, global_mean)
    train["te_vehicle"] = compute_oof_te_vehicle(train, folds, TE_VEHICLE_SMOOTHING, global_mean)

    return train, full_train_tables, vehicle_table


def add_historical_features_to_eval_split(
    df: pd.DataFrame, tables: dict, vehicle_table: pd.Series, global_mean: float
) -> pd.DataFrame:
    df = df.copy()
    hour_vals, segment_vals = apply_historical_tables(df, tables)
    df["typical_segment_hour_s"] = hour_vals
    df["typical_segment_s"] = segment_vals
    df["te_vehicle"] = apply_smoothed_vehicle_means(df, vehicle_table, global_mean)
    return df


def finish_split(df: pd.DataFrame, featurize_params: dict) -> pd.DataFrame:
    """Shared steps once typical_segment_hour_s/typical_segment_s/te_vehicle
    are present: delay_s, recent-bus features, own-prev-segment features."""
    df = df.copy()
    df["delay_s"] = df[TARGET] - df["typical_segment_hour_s"]

    df = compute_recent_bus_features(
        df, featurize_params["prev_n_buses"], featurize_params["recent_window_minutes"] * 60
    )
    df = fill_missing_recent_features(df)
    df = add_own_prev_segment_features(df)
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES + EXTRA_OUTPUT_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------- reporting


def day_range(df: pd.DataFrame) -> dict[str, int]:
    return {"min_day_index": int(df["day_index"].min()), "max_day_index": int(df["day_index"].max())}


def nested_table_to_json(table: dict) -> dict[str, float]:
    out = {}
    for key, value in table.items():
        key_str = "||".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        out[key_str] = float(value)
    return out


def main() -> int:
    params = load_params()
    featurize_params = params["featurize"]

    df = pd.read_parquet(INPUT_PARQUET)
    df = add_static_features(df)
    df = add_time_of_day_features(
        df,
        seconds_col="depart_time_s",
        peak_am=tuple(featurize_params["peak_am"]),
        peak_pm=tuple(featurize_params["peak_pm"]),
    )

    train, val, test = split_by_day(df, featurize_params)

    kept_vehicles = fit_rare_vehicle_set(train, featurize_params["rare_vehicle_min_count"])
    train["vehicle_grouped"] = apply_vehicle_grouping(train, kept_vehicles)
    val["vehicle_grouped"] = apply_vehicle_grouping(val, kept_vehicles)
    test["vehicle_grouped"] = apply_vehicle_grouping(test, kept_vehicles)

    folds = make_day_folds(train, featurize_params["te_folds"], TE_RANDOM_STATE)
    train, historical_tables, vehicle_table = add_historical_features_to_train(train, folds)
    global_mean = float(train[TARGET].mean())

    val = add_historical_features_to_eval_split(val, historical_tables, vehicle_table, global_mean)
    test = add_historical_features_to_eval_split(test, historical_tables, vehicle_table, global_mean)

    train = finish_split(train, featurize_params)
    val = finish_split(val, featurize_params)
    test = finish_split(test, featurize_params)

    train_out = select_output_columns(train)
    val_out = select_output_columns(val)
    test_out = select_output_columns(test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_out.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test_out.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    print(
        f"Wrote {len(train_out):,} / {len(val_out):,} / {len(test_out):,} rows "
        f"(train/val/test) to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}"
    )

    feature_spec = {
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "extra_output_columns": EXTRA_OUTPUT_COLUMNS,
        "kept_vehicles": sorted(kept_vehicles),
        "other_label": OTHER_LABEL,
        "te_vehicle_smoothing": TE_VEHICLE_SMOOTHING,
        "global_mean_travel_time_s": global_mean,
        "typical_segment_hour_s_table": nested_table_to_json(historical_tables["hour"]),
        "typical_segment_s_table": nested_table_to_json(historical_tables["segment"]),
        "route_direction_median_table": nested_table_to_json(historical_tables["route_direction"]),
        "te_vehicle_table": nested_table_to_json(vehicle_table.to_dict()),
        "params": featurize_params,
    }
    FEATURE_SPEC_PATH.write_text(json.dumps(feature_spec, indent=2))
    print(f"Wrote feature spec to {FEATURE_SPEC_PATH.relative_to(PROJECT_ROOT)}")

    report = {
        "rows": {"train": len(train_out), "val": len(val_out), "test": len(test_out)},
        "day_range": {
            "train": day_range(train_out),
            "val": day_range(val_out),
            "test": day_range(test_out),
        },
        "n_features": len(FEATURES),
        "has_prev_bus_share": {
            "train": float(train_out["has_prev_bus"].mean()),
            "val": float(val_out["has_prev_bus"].mean()),
            "test": float(test_out["has_prev_bus"].mean()),
        },
        "prev_feature_means": {
            split_name: {col: float(split_df[col].mean()) for col in PREV_FEATURES_FOR_REPORT}
            for split_name, split_df in [("train", train_out), ("val", val_out), ("test", test_out)]
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote featurize report to {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
