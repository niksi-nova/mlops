"""Fetch hourly historical weather for Toronto from the Open-Meteo
archive API, covering the date range of the cleaned TTC delay data.

Reads:
  - data/interim/historical_clean.parquet  (for the min/max incident date)

Writes:
  - data/external/weather_hourly.parquet

Usage:
    uv run python src/data/fetch_weather.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
INPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "historical_clean.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "external" / "weather_hourly.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def date_range_from_clean_data(path: Path) -> tuple[str, str]:
    """Return (min_date, max_date) as YYYY-MM-DD strings from the
    cleaned delay data's timestamp column."""
    df = pd.read_parquet(path, columns=["timestamp"])
    min_date = df["timestamp"].min().strftime("%Y-%m-%d")
    max_date = df["timestamp"].max().strftime("%Y-%m-%d")
    return min_date, max_date


def year_chunks(min_date: str, max_date: str) -> list[tuple[str, str]]:
    """Split [min_date, max_date] into one (start, end) pair per calendar
    year, clipped to the overall range (the archive API is friendlier to
    yearly requests than one multi-year request)."""
    start_year = int(min_date[:4])
    end_year = int(max_date[:4])
    chunks = []
    for year in range(start_year, end_year + 1):
        chunk_start = max(min_date, f"{year}-01-01")
        chunk_end = min(max_date, f"{year}-12-31")
        chunks.append((chunk_start, chunk_end))
    return chunks


def fetch_chunk(weather_params: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one date-range chunk of hourly weather from the archive API."""
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": weather_params["latitude"],
            "longitude": weather_params["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(weather_params["hourly_variables"]),
            "timezone": weather_params["timezone"],
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.DataFrame(hourly)


def main() -> int:
    params = load_params()
    weather_params = params["weather"]

    min_date, max_date = date_range_from_clean_data(INPUT_PARQUET)
    print(f"Fetching weather from {min_date} to {max_date} ...")

    frames = []
    for chunk_start, chunk_end in year_chunks(min_date, max_date):
        print(f"  requesting {chunk_start} to {chunk_end}")
        frames.append(fetch_chunk(weather_params, chunk_start, chunk_end))

    weather = pd.concat(frames, ignore_index=True)
    weather["time"] = pd.to_datetime(weather["time"]).dt.tz_localize(
        weather_params["timezone"], ambiguous="NaT", nonexistent="shift_forward"
    )
    weather = weather.rename(columns={"time": "timestamp"})
    weather = weather.drop_duplicates(subset="timestamp").sort_values("timestamp")

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Wrote {len(weather):,} hourly rows to {OUTPUT_PARQUET.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
