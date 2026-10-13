"""Validate the cleaned TTC bus delay data against a pandera schema
before it is used downstream.

Reads:
  - data/interim/historical_clean.parquet
  - params.yaml (clean.min_delay / clean.max_delay bounds)

Writes:
  - reports/validation_report.json

On a failing check, prints the failing checks and exits non-zero so a
DVC pipeline stops instead of continuing on bad data.

Usage:
    uv run python src/data/validate.py
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
INPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "historical_clean.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "validation_report.json"

ALLOWED_DIRECTIONS = {"N", "S", "E", "W", "B", "UNKNOWN"}
MIN_DATE = pd.Timestamp("2018-01-01", tz="America/Toronto")
MAX_DATE = pd.Timestamp("2024-12-31", tz="America/Toronto") + pd.Timedelta(days=1)


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def build_schema(clean_params: dict) -> pa.DataFrameSchema:
    """Build the pandera schema for historical_clean.parquet, using the
    min_delay/max_delay bounds from params.yaml."""
    return pa.DataFrameSchema(
        {
            "route": pa.Column(pa.String, nullable=False),
            "direction": pa.Column(
                pa.String,
                nullable=False,
                checks=pa.Check.isin(ALLOWED_DIRECTIONS),
            ),
            "min_delay": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.in_range(
                    clean_params["min_delay"], clean_params["max_delay"]
                ),
            ),
            "timestamp": pa.Column(
                "datetime64[us, America/Toronto]",
                nullable=False,
                checks=pa.Check.in_range(MIN_DATE, MAX_DATE),
            ),
        },
        strict=False,
    )


def main() -> int:
    params = load_params()
    clean_params = params["clean"]
    schema = build_schema(clean_params)

    df = pd.read_parquet(INPUT_PARQUET)

    checks_run = [
        "required columns present",
        f"min_delay in [{clean_params['min_delay']}, {clean_params['max_delay']}]",
        "timestamp not null and within 2018-01-01 to 2024-12-31",
        f"direction in {sorted(ALLOWED_DIRECTIONS)}",
        "route not null",
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
