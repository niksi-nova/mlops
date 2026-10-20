"""Clean raw Delhi bus node-crossing data into one-row-per-stop-arrival events.

Reads every raw file under data/raw/delhi_gps/ (Month_<m>/<route>/Day<d>.xlsx,
see docs/data_inventory.md for the exact format), derives route/direction/
day_index/relative_weekday, splits each vehicle's readings into trips, drops
bad trips, and writes:
  - data/interim/stop_events.parquet (one row per stop arrival)
  - reports/cleaning_report.json

There are no calendar dates or coordinates in the source data: `day_index`
is a synthetic 1..92 index over Month_1..3 x Day1..31, and `Node` is a
direction-specific stop-sequence index that increases over time for both
directions.

Usage:
    uv run python -m src.data.clean
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delhi_gps"
OUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "stop_events.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "cleaning_report.json"

DAY_RE = re.compile(r"Day(\d+)\.xlsx$")
# day_index = MONTH_DAY_OFFSET[month] + day, giving 1..92 as Month_1 Day1..31,
# Month_2 Day1..30 -> 32..61, Month_3 Day1..31 -> 62..92.
MONTH_DAY_OFFSET = {1: 0, 2: 31, 3: 61}
MIDNIGHT_THRESHOLD_S = 12 * 3600


@dataclass(frozen=True)
class RawFile:
    path: Path
    route: str
    month: int
    day: int

    @property
    def day_index(self) -> int:
        return MONTH_DAY_OFFSET[self.month] + self.day


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def discover_files(raw_dir: Path) -> list[RawFile]:
    """Find every Day*.xlsx file under raw_dir and parse its route/month/day."""
    files: list[RawFile] = []
    for month_dir in sorted(raw_dir.glob("Month_*")):
        if not month_dir.is_dir():
            continue
        month = int(month_dir.name.split("_")[1])
        for route_dir in sorted(month_dir.iterdir()):
            if not route_dir.is_dir():
                continue
            route = route_dir.name
            for f in route_dir.glob("Day*.xlsx"):
                match = DAY_RE.search(f.name)
                if not match:
                    continue
                files.append(RawFile(path=f, route=route, month=month, day=int(match.group(1))))
    return sorted(files, key=lambda r: (r.month, r.route, r.day))


def read_raw_file(rec: RawFile) -> pd.DataFrame:
    """Read one raw file into a tidy per-row frame, preserving capture order.

    The row order within a file is the actual capture/arrival order, which
    is used downstream for trip splitting instead of sorting by time_s
    (time_s wraps at midnight and cannot be used directly as a sort key).
    """
    raw = pd.read_excel(rec.path)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    time_of_day = pd.to_datetime(raw["Time"], format="%H:%M:%S:%f").dt.time
    return pd.DataFrame(
        {
            "route": rec.route,
            "direction": raw["Route"].astype(str).str.strip(),
            "vehicle": raw["Number"].astype(str).str.strip(),
            "node": raw["Node"].astype(int),
            "time_s": time_of_day.apply(
                lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            ),
            "month": rec.month,
            "day": rec.day,
            "day_index": rec.day_index,
            "row_order": range(len(raw)),
        }
    )


def add_calendar_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative_weekday = (day_index - 1) % 7."""
    df = df.copy()
    df["relative_weekday"] = (df["day_index"] - 1) % 7
    return df


def assign_trips(df: pd.DataFrame, trip_gap_minutes: float) -> tuple[pd.DataFrame, int]:
    """Split rows into trips per (route, direction, day_index, vehicle).

    df must already be sorted by (route, direction, day_index, vehicle,
    row_order). A new trip starts when the node decreases relative to the
    previous record, or the adjusted time gap exceeds trip_gap_minutes. A
    raw backward time jump of more than 12 hours inside a trip is treated as
    a midnight crossing: 86400s is added to a running offset instead of
    starting a new trip, so `time_s` in the output is midnight-corrected and
    may exceed 86400.

    Returns the frame with trip_id and corrected time_s, plus the count of
    distinct trips that received at least one midnight correction.
    """
    trip_gap_s = trip_gap_minutes * 60
    trip_ids: list[int] = []
    adjusted_times: list[float] = []
    midnight_trip_ids: set[int] = set()

    trip_id = 0
    prev_key = None
    prev_node = None
    prev_raw_time = None
    prev_adjusted_time = None
    running_offset = 0.0
    adjusted_time = 0.0

    for row in df.itertuples(index=False):
        key = (row.route, row.direction, row.day_index, row.vehicle)
        if key != prev_key:
            trip_id += 1
            running_offset = 0.0
            adjusted_time = row.time_s
        else:
            delta_raw = row.time_s - prev_raw_time
            midnight = delta_raw < -MIDNIGHT_THRESHOLD_S
            if midnight:
                running_offset += 86400
            adjusted_time = row.time_s + running_offset
            gap_s = adjusted_time - prev_adjusted_time
            node_decreased = row.node < prev_node
            if node_decreased or (gap_s > trip_gap_s and not midnight):
                trip_id += 1
                running_offset = 0.0
                adjusted_time = row.time_s
            elif midnight:
                midnight_trip_ids.add(trip_id)

        trip_ids.append(trip_id)
        adjusted_times.append(adjusted_time)

        prev_key = key
        prev_node = row.node
        prev_raw_time = row.time_s
        prev_adjusted_time = adjusted_time

    df = df.copy()
    df["trip_id"] = trip_ids
    df["time_s"] = adjusted_times
    return df, len(midnight_trip_ids)


def find_bad_trips(df: pd.DataFrame, min_nodes_per_trip: int) -> dict[str, set[int]]:
    """Return trip_ids to drop, keyed by the rule that flagged them (sets may overlap)."""
    bad: dict[str, set[int]] = {"node_backward": set(), "time_backward": set(), "too_few_nodes": set()}
    for trip_id, g in df.groupby("trip_id"):
        g = g.sort_values("row_order")
        if (g["node"].diff().dropna() < 0).any():
            bad["node_backward"].add(trip_id)
        if (g["time_s"].diff().dropna() < 0).any():
            bad["time_backward"].add(trip_id)
        if len(g) < min_nodes_per_trip:
            bad["too_few_nodes"].add(trip_id)
    return bad


def main() -> int:
    params = load_params()
    trip_gap_minutes = params["clean"]["trip_gap_minutes"]
    min_nodes_per_trip = params["clean"]["min_nodes_per_trip"]

    files = discover_files(RAW_DIR)
    if not files:
        raise SystemExit(f"no raw files found under {RAW_DIR}")

    raw_df = pd.concat((read_raw_file(rec) for rec in files), ignore_index=True)
    raw_rows = len(raw_df)

    dedup_cols = ["route", "direction", "vehicle", "node", "time_s", "day_index"]
    duplicate_rows_removed = int(raw_df.duplicated(subset=dedup_cols).sum())
    df = raw_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    rows_after_dedup = len(df)

    df = add_calendar_fields(df)
    df = df.sort_values(["route", "direction", "day_index", "vehicle", "row_order"]).reset_index(drop=True)

    df, midnight_crossing_trips_fixed = assign_trips(df, trip_gap_minutes)
    trips_before_filtering = int(df["trip_id"].nunique())

    bad = find_bad_trips(df, min_nodes_per_trip)
    all_bad_trip_ids = set().union(*bad.values())

    clean_df = df[~df["trip_id"].isin(all_bad_trip_ids)].copy()
    final_rows = len(clean_df)
    final_trips = int(clean_df["trip_id"].nunique())

    trips_per_route_direction = {
        f"{route}_{direction}": int(n)
        for (route, direction), n in clean_df.groupby(["route", "direction"])["trip_id"].nunique().items()
    }

    report = {
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_dedup": rows_after_dedup,
        "trips_before_filtering": trips_before_filtering,
        "trips_removed": {
            "node_backward": len(bad["node_backward"]),
            "time_backward": len(bad["time_backward"]),
            "too_few_nodes": len(bad["too_few_nodes"]),
            "total_distinct_trips_removed": len(all_bad_trip_ids),
        },
        "rows_removed_with_bad_trips": rows_after_dedup - final_rows,
        "final_rows": final_rows,
        "final_trips": final_trips,
        "trips_per_route_direction": trips_per_route_direction,
        "midnight_crossing_trips_fixed": midnight_crossing_trips_fixed,
    }

    out_cols = [
        "route",
        "direction",
        "vehicle",
        "trip_id",
        "month",
        "day",
        "day_index",
        "relative_weekday",
        "node",
        "time_s",
    ]
    clean_df = clean_df[out_cols].sort_values(["route", "direction", "trip_id", "time_s"]).reset_index(drop=True)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(OUT_PARQUET, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {OUT_PARQUET.relative_to(PROJECT_ROOT)} ({final_rows:,} rows, {final_trips:,} trips)")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
