"""Clean the raw TTC bus delay historical workbooks into one tidy table.

Reads every monthly sheet of every yearly workbook in
data/raw/historical/, standardises the schema (column names and
values differ across years - see docs/data_inventory.md), applies the
cleaning rules from params.yaml, and writes:

  - data/interim/historical_clean.parquet  (the cleaned table)
  - reports/cleaning_report.json           (row counts per rule, per year)

Usage:
    uv run python src/data/clean.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "interim" / "historical_clean.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "reports" / "cleaning_report.json"

TIMEZONE = "America/Toronto"

# Column name variants seen across years and odd one-off sheets (see
# docs/data_inventory.md): e.g. 2018-2019 use "Report Date"/"Min Delay", 2020
# shortens to "Delay"/"Gap", and a couple of 2021 monthly sheets use "Line"/
# "Bound" instead of "Route"/"Direction". Rather than renaming (which would
# collide into duplicate columns once different years are concatenated),
# each final column is built by coalescing its first non-null candidate.
CANDIDATE_COLUMNS = {
    "date": ["Date", "Report Date"],
    "route": ["Route", "Line"],
    "time": ["Time"],
    "day": ["Day"],
    "location": ["Location"],
    "incident": ["Incident"],
    "min_delay": ["Min Delay", "Delay"],
    "min_gap": ["Min Gap", "Gap"],
    "direction": ["Direction", "Bound"],
    "vehicle": ["Vehicle"],
}
FINAL_COLUMNS = list(CANDIDATE_COLUMNS.keys())

# A direction cell that, once stripped of everything but letters, equals
# one of these keys is mapped to the compass bound it represents. Only
# exact matches are trusted; anything else (free text, ids, typos) becomes
# UNKNOWN rather than guessed at.
DIRECTION_MAP = {
    "N": "N", "NB": "N",
    "S": "S", "SB": "S",
    "E": "E", "EB": "E",
    "W": "W", "WB": "W",
    "B": "B",  # both ways
}


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def read_historical_workbooks(historical_dir: Path) -> pd.DataFrame:
    """Read every sheet of every yearly workbook and concatenate them."""
    files = sorted(historical_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"no .xlsx files found under {historical_dir}")

    frames = []
    for path in files:
        sheets = pd.read_excel(path, sheet_name=None)
        for sheet_name, df in sheets.items():
            df = df.copy()
            df["__source_file__"] = path.name
            df["__source_sheet__"] = sheet_name
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def coalesce_series(series_list: list[pd.Series], index: pd.Index) -> pd.Series:
    """Return the first non-null value, per row, across a list of Series."""
    result = pd.Series(pd.NA, index=index, dtype="object")
    for series in series_list:
        result = result.where(result.notna(), series)
    return result


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse columns that share a name (e.g. "Min Delay" and the
    whitespace variant " Min Delay", after stripping) into one, by
    taking the first non-null value per row among the duplicates."""
    if not df.columns.duplicated().any():
        return df
    deduped = {}
    for name in df.columns.unique():
        cols = df.loc[:, df.columns == name]
        if cols.shape[1] == 1:
            deduped[name] = cols.iloc[:, 0]
        else:
            deduped[name] = coalesce_series([cols.iloc[:, i] for i in range(cols.shape[1])], df.index)
    return pd.DataFrame(deduped, index=df.index)


def coalesce_columns(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first non-null value, per row, across `candidates`
    (columns that don't exist in this frame are treated as all-null)."""
    series_list = [df[name] for name in candidates if name in df.columns]
    return coalesce_series(series_list, df.index)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names, then build the final
    snake_case schema by coalescing known column-name variants."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = dedupe_columns(df)

    standardised = pd.DataFrame(index=df.index)
    for final_name, candidates in CANDIDATE_COLUMNS.items():
        standardised[final_name] = coalesce_columns(df, candidates)

    standardised["__source_file__"] = df["__source_file__"]
    standardised["__source_sheet__"] = df["__source_sheet__"]
    return standardised


def normalise_time_string(value: str) -> str:
    """Pad a time string to HH:MM:SS so every row uses the same format
    before parsing (the raw files mix "HH:MM" and "HH:MM:SS")."""
    parts = value.split(":")
    parts = (parts + ["00", "00"])[:3] if len(parts) < 3 else parts[:3]
    try:
        return ":".join(p.zfill(2) for p in parts)
    except (ValueError, AttributeError):
        return "00:00:00"


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Combine the date and time columns into one tz-aware timestamp
    column (local Toronto time)."""
    df = df.copy()
    date_part = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_part = df["time"].astype(str).str.strip().map(normalise_time_string)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time"] = time_part
    combined = date_part.str.cat(time_part, sep=" ", na_rep="")
    timestamp = pd.to_datetime(combined, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["timestamp"] = timestamp.dt.tz_localize(
        TIMEZONE, ambiguous="NaT", nonexistent="shift_forward"
    )
    return df


def clean_route(series: pd.Series) -> pd.Series:
    """Format route as a stripped string (e.g. 9.0 -> "9"); missing stays NaN."""

    def fmt(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        text = str(value).strip()
        return text if text else pd.NA

    return series.map(fmt)


def clean_text_field(series: pd.Series) -> pd.Series:
    """Strip, uppercase, and collapse repeated whitespace in a text field."""
    text = series.astype(str).str.strip().str.upper()
    return text.str.replace(r"\s+", " ", regex=True)


def normalise_direction(series: pd.Series) -> pd.Series:
    """Map direction values to N/S/E/W/B, or UNKNOWN for anything else."""

    def normalise_one(value: object) -> str:
        if pd.isna(value):
            return "UNKNOWN"
        letters_only = re.sub(r"[^A-Za-z]", "", str(value)).upper()
        return DIRECTION_MAP.get(letters_only, "UNKNOWN")

    return series.map(normalise_one)


def apply_cleaning_rules(df: pd.DataFrame, clean_params: dict, counts: dict) -> pd.DataFrame:
    """Apply all value-level cleaning rules, recording the row count
    remaining after each one into `counts` (mutated in place)."""
    df = df.copy()

    df["route"] = clean_route(df["route"])
    df = df[df["route"].notna()]
    counts["drop_missing_route"] = len(df)

    df["location"] = clean_text_field(df["location"])
    df["incident"] = clean_text_field(df["incident"])

    df["direction"] = normalise_direction(df["direction"])

    df["min_delay"] = pd.to_numeric(df["min_delay"], errors="coerce")
    df["min_gap"] = pd.to_numeric(df["min_gap"], errors="coerce")
    df = df[df["min_delay"].notna() & df["timestamp"].notna()]
    counts["drop_missing_min_delay_or_timestamp"] = len(df)

    min_delay = clean_params["min_delay"]
    df = df[df["min_delay"] >= min_delay]
    counts["apply_min_delay_bound"] = len(df)

    max_delay = clean_params["max_delay"]
    df = df[df["min_delay"] <= max_delay]
    counts["apply_max_delay_bound"] = len(df)

    if clean_params["drop_zero_delay"]:
        df = df[df["min_delay"] != 0]
    counts["drop_zero_delay"] = len(df)

    df = df.drop_duplicates(subset=FINAL_COLUMNS + ["timestamp"])
    counts["drop_duplicates"] = len(df)

    return df


def finalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every column to one consistent dtype so the parquet file
    (and pyarrow) doesn't choke on columns that are a mix of types
    across years (e.g. route as int in some years, str in others)."""
    df = df.copy()
    string_columns = ["route", "time", "day", "location", "incident", "direction"]
    for col in string_columns:
        df[col] = df[col].astype(str).astype("string")
    df["vehicle"] = pd.to_numeric(df["vehicle"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_report(df_raw_len: int, counts: dict, df_final: pd.DataFrame) -> dict:
    """Assemble the cleaning_report.json payload."""
    rows_per_year = (
        df_final["timestamp"].dt.year.value_counts().sort_index().astype(int).to_dict()
    )
    return {
        "rows_raw": df_raw_len,
        "rows_after_each_rule": counts,
        "rows_final": len(df_final),
        "rows_per_year": {str(year): count for year, count in rows_per_year.items()},
        "min_timestamp": df_final["timestamp"].min().isoformat(),
        "max_timestamp": df_final["timestamp"].max().isoformat(),
    }


def main() -> int:
    params = load_params()
    clean_params = params["clean"]

    print(f"Reading historical workbooks from {HISTORICAL_DIR} ...")
    raw = read_historical_workbooks(HISTORICAL_DIR)
    rows_raw = len(raw)
    print(f"  {rows_raw:,} rows read across all sheets")

    standardised = standardise_columns(raw)
    with_timestamp = build_timestamp(standardised)

    counts: dict[str, int] = {}
    cleaned = apply_cleaning_rules(with_timestamp, clean_params, counts)
    cleaned = cleaned[FINAL_COLUMNS + ["timestamp"]].reset_index(drop=True)
    cleaned = finalise_dtypes(cleaned)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Wrote {len(cleaned):,} rows to {OUTPUT_PARQUET.relative_to(PROJECT_ROOT)}")

    report = build_report(rows_raw, counts, cleaned)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"Wrote cleaning report to {OUTPUT_REPORT.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
