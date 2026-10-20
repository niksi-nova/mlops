"""Build stop-to-stop travel-time segments from cleaned stop-arrival events.

Reads data/interim/stop_events.parquet (one row per stop arrival, written by
src/data/clean.py) and, for each trip, turns consecutive arrivals into one
row per stop-to-stop segment. Writes:
  - data/interim/segments.parquet
  - reports/segments_report.json

Usage:
    uv run python -m src.data.build_segments
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
STOP_EVENTS_PATH = PROJECT_ROOT / "data" / "interim" / "stop_events.parquet"
OUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "segments.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "segments_report.json"

PERCENTILES = [0.50, 0.75, 0.90, 0.95, 0.99]


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def build_segments(stop_events: pd.DataFrame) -> pd.DataFrame:
    """Turn consecutive stop arrivals within each trip into stop-to-stop segments."""
    stop_events = stop_events.sort_values(["trip_id", "time_s"])
    group_cols = ["route", "direction", "vehicle", "trip_id", "month", "day_index", "relative_weekday"]

    segments = stop_events.copy()
    grouped = segments.groupby("trip_id", sort=False)
    segments["to_node"] = grouped["node"].shift(-1)
    segments["arrive_time_s"] = grouped["time_s"].shift(-1)
    segments = segments.rename(columns={"node": "from_node", "time_s": "depart_time_s"})
    segments = segments.dropna(subset=["to_node", "arrive_time_s"])

    segments["to_node"] = segments["to_node"].astype(int)
    segments["nodes_spanned"] = segments["to_node"] - segments["from_node"]
    segments["travel_time_s"] = segments["arrive_time_s"] - segments["depart_time_s"]
    segments["seconds_per_stop"] = segments["travel_time_s"] / segments["nodes_spanned"]

    out_cols = group_cols + [
        "from_node",
        "to_node",
        "nodes_spanned",
        "depart_time_s",
        "travel_time_s",
        "seconds_per_stop",
    ]
    return segments[out_cols].reset_index(drop=True)


def apply_bounds(segments: pd.DataFrame, min_segment_seconds: float, max_segment_seconds: float) -> pd.DataFrame:
    """Drop segments whose travel_time_s falls outside [min_segment_seconds, max_segment_seconds]."""
    in_bounds = segments["travel_time_s"].between(min_segment_seconds, max_segment_seconds)
    return segments[in_bounds].reset_index(drop=True)


def drop_zero_span_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Drop segments where consecutive stop arrivals record the same node twice in a
    row (nodes_spanned <= 0) - not a real stop-to-stop transition."""
    return segments[segments["nodes_spanned"] >= 1].reset_index(drop=True)


def main() -> int:
    params = load_params()
    min_segment_seconds = params["segments"]["min_segment_seconds"]
    max_segment_seconds = params["segments"]["max_segment_seconds"]

    stop_events = pd.read_parquet(STOP_EVENTS_PATH)
    segments = build_segments(stop_events)
    segments_built = len(segments)

    segments = drop_zero_span_segments(segments)
    segments_removed_zero_span = segments_built - len(segments)
    segments_before_bounds = len(segments)

    segments = apply_bounds(segments, min_segment_seconds, max_segment_seconds)
    segments_after_bounds = len(segments)

    rows_per_route_direction_month = {
        f"{route}_{direction}_month{month}": int(n)
        for (route, direction, month), n in segments.groupby(["route", "direction", "month"]).size().items()
    }

    share_multi_node = float((segments["nodes_spanned"] > 1).mean()) if len(segments) else 0.0

    travel_time_percentiles = {
        f"p{int(q * 100)}": float(v)
        for q, v in segments["travel_time_s"].quantile(PERCENTILES).items()
    }

    report = {
        "segments_built": segments_built,
        "segments_removed_zero_span": segments_removed_zero_span,
        "segments_before_bounds_filter": segments_before_bounds,
        "segments_removed_out_of_bounds": segments_before_bounds - segments_after_bounds,
        "segments_after_bounds_filter": segments_after_bounds,
        "rows_per_route_direction_month": rows_per_route_direction_month,
        "share_segments_nodes_spanned_gt_1": share_multi_node,
        "travel_time_s_percentiles": travel_time_percentiles,
    }

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(OUT_PARQUET, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {OUT_PARQUET.relative_to(PROJECT_ROOT)} ({segments_after_bounds:,} rows)")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
