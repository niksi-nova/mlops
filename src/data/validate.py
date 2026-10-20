"""Validate the Delhi bus stop-to-stop segments against a pandera schema
before they are used downstream.

Reads:
  - data/interim/segments.parquet
  - params.yaml (segments.min_segment_seconds / segments.max_segment_seconds bounds)

Writes:
  - reports/validation_report.json

On a failing check, prints the failing checks and exits non-zero so a
DVC pipeline stops instead of continuing on bad data.

Usage:
    uv run python -m src.data.validate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.errors import SchemaErrors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
INPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "segments.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "validation_report.json"

ALLOWED_ROUTES = {"425", "534"}
ALLOWED_DIRECTIONS = {"up", "down"}
MIN_DAY_INDEX, MAX_DAY_INDEX = 1, 92
MIN_RELATIVE_WEEKDAY, MAX_RELATIVE_WEEKDAY = 0, 6
# depart_time_s can exceed 86400 for midnight-crossing trips; allow generous headroom.
MAX_DEPART_TIME_S = 2 * 86400


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def build_schema(segments_params: dict) -> pa.DataFrameSchema:
    """Build the pandera schema for segments.parquet, using the
    min/max_segment_seconds bounds from params.yaml."""
    return pa.DataFrameSchema(
        {
            "route": pa.Column(pa.String, nullable=False, checks=pa.Check.isin(ALLOWED_ROUTES)),
            "direction": pa.Column(pa.String, nullable=False, checks=pa.Check.isin(ALLOWED_DIRECTIONS)),
            "vehicle": pa.Column(pa.String, nullable=False),
            "trip_id": pa.Column(int, nullable=False),
            "day_index": pa.Column(
                int, nullable=False, checks=pa.Check.in_range(MIN_DAY_INDEX, MAX_DAY_INDEX)
            ),
            "relative_weekday": pa.Column(
                int,
                nullable=False,
                checks=pa.Check.in_range(MIN_RELATIVE_WEEKDAY, MAX_RELATIVE_WEEKDAY),
            ),
            "from_node": pa.Column(int, nullable=False),
            "to_node": pa.Column(int, nullable=False),
            "nodes_spanned": pa.Column(int, nullable=False, checks=pa.Check.ge(1)),
            "depart_time_s": pa.Column(
                float, nullable=False, checks=pa.Check.in_range(0, MAX_DEPART_TIME_S)
            ),
            "travel_time_s": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.in_range(
                    segments_params["min_segment_seconds"], segments_params["max_segment_seconds"]
                ),
            ),
        },
        checks=pa.Check(
            lambda df: df["from_node"] < df["to_node"],
            element_wise=False,
            error="from_node must be < to_node",
        ),
        strict=False,
    )


def main() -> int:
    params = load_params()
    segments_params = params["segments"]
    schema = build_schema(segments_params)

    df = pd.read_parquet(INPUT_PARQUET)

    checks_run = [
        "required columns present, no nulls",
        f"route in {sorted(ALLOWED_ROUTES)}",
        f"direction in {sorted(ALLOWED_DIRECTIONS)}",
        "from_node < to_node",
        "nodes_spanned >= 1",
        f"day_index in [{MIN_DAY_INDEX}, {MAX_DAY_INDEX}]",
        f"relative_weekday in [{MIN_RELATIVE_WEEKDAY}, {MAX_RELATIVE_WEEKDAY}]",
        f"depart_time_s in [0, {MAX_DEPART_TIME_S}]",
        f"travel_time_s in [{segments_params['min_segment_seconds']}, {segments_params['max_segment_seconds']}]",
    ]

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        failure_cases = exc.failure_cases.to_dict(orient="records")
        print("Validation FAILED:")
        for case in failure_cases:
            print(f"  {case}")
        report = {
            "status": "failed",
            "row_count": len(df),
            "checks_run": checks_run,
            "failure_cases": failure_cases,
        }
        OUTPUT_REPORT.write_text(json.dumps(report, indent=2, default=str))
        return 1

    report = {
        "status": "passed",
        "row_count": len(df),
        "checks_run": checks_run,
    }
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"Validation passed for {len(df):,} rows.")
    print(f"Wrote validation report to {OUTPUT_REPORT.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
