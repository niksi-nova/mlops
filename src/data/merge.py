"""Merge cleaned TTC bus delay incidents with hourly weather, by
rounding each incident's timestamp down to the hour and left-joining
the hourly weather table.

Reads:
  - data/interim/historical_clean.parquet
  - data/external/weather_hourly.parquet

Writes:
  - data/interim/merged.parquet
  - reports/merge_report.json

Usage:
    uv run python src/data/merge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELAYS_PARQUET = PROJECT_ROOT / "data" / "interim" / "historical_clean.parquet"
WEATHER_PARQUET = PROJECT_ROOT / "data" / "external" / "weather_hourly.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "merged.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "merge_report.json"

WEATHER_COLUMNS = ["temperature_2m", "precipitation", "rain", "snowfall", "wind_speed_10m"]


def main() -> int:
    delays = pd.read_parquet(DELAYS_PARQUET)
    weather = pd.read_parquet(WEATHER_PARQUET)

    delays["weather_hour"] = delays["timestamp"].dt.floor("h")
    merged = delays.merge(
        weather,
        left_on="weather_hour",
        right_on="timestamp",
        how="left",
        suffixes=("", "_weather"),
    )
    merged = merged.drop(columns=["weather_hour", "timestamp_weather"])

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Wrote {len(merged):,} rows to {OUTPUT_PARQUET.relative_to(PROJECT_ROOT)}")

    matched = merged[WEATHER_COLUMNS].notna().all(axis=1).sum()
    match_pct = round(100 * matched / len(merged), 2) if len(merged) else 0.0

    report = {
        "row_count": len(merged),
        "weather_match_count": int(matched),
        "weather_match_pct": match_pct,
    }
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"Weather match: {match_pct}% ({matched:,}/{len(merged):,} rows)")
    print(f"Wrote merge report to {OUTPUT_REPORT.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
