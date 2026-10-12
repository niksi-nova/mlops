"""Unit tests for the helper functions in src/data/clean.py.

All tests use small hand-made DataFrames; no real data files are read.
"""

import os
import sys

import pandas as pd
import pytest

# Ensure project root directory is in sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.clean import (  # noqa: E402
    FINAL_COLUMNS,
    apply_cleaning_rules,
    build_timestamp,
    clean_route,
    clean_text_field,
    coalesce_columns,
    coalesce_series,
    dedupe_columns,
    normalise_direction,
    normalise_time_string,
    standardise_columns,
)

CLEAN_PARAMS = {"min_delay": 0, "max_delay": 300, "drop_zero_delay": False}


# --------------------------------------------------------------------------
# normalise_direction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("N", "N"),
        ("n", "N"),
        ("NB", "N"),
        ("nb", "N"),
        ("N/B", "N"),       # punctuation is stripped before matching
        (" s b ", "S"),     # whitespace is stripped too
        ("EB", "E"),
        ("W", "W"),
        ("wb", "W"),
        ("B", "B"),         # "both ways"
        ("b", "B"),
    ],
)
def test_normalise_direction_known_values(raw, expected):
    assert normalise_direction(pd.Series([raw])).iloc[0] == expected


@pytest.mark.parametrize("raw", ["xyz", "12345", "", "???", "NORTH/SOUTH", "Nrth"])
def test_normalise_direction_junk_is_unknown(raw):
    assert normalise_direction(pd.Series([raw])).iloc[0] == "UNKNOWN"


def test_normalise_direction_missing_is_unknown():
    result = normalise_direction(pd.Series([None, float("nan"), pd.NA]))
    assert (result == "UNKNOWN").all()


def test_normalise_direction_full_words_are_unknown_by_design():
    # clean.py only trusts exact short codes ("only exact matches are
    # trusted"), so full words like "Northbound" and "b/w" currently map to
    # UNKNOWN. This pins down the current behaviour.
    result = normalise_direction(pd.Series(["Northbound", "b/w", "Both Ways"]))
    assert (result == "UNKNOWN").all()


def test_normalise_direction_keeps_length_and_index():
    s = pd.Series(["N", "junk", "B"], index=[10, 11, 12])
    result = normalise_direction(s)
    assert list(result.index) == [10, 11, 12]
    assert list(result) == ["N", "UNKNOWN", "B"]


# --------------------------------------------------------------------------
# normalise_time_string
# --------------------------------------------------------------------------
def test_time_hh_mm_and_hh_mm_ss_give_same_result():
    assert normalise_time_string("02:08") == normalise_time_string("02:08:00")
    assert normalise_time_string("02:08") == "02:08:00"


def test_time_already_full_is_unchanged():
    assert normalise_time_string("00:17:00") == "00:17:00"


def test_time_single_digit_hour_is_zero_padded():
    assert normalise_time_string("2:08") == "02:08:00"
    assert normalise_time_string("9:5:3") == "09:05:03"


def test_time_extra_parts_are_truncated():
    assert normalise_time_string("01:02:03:04") == "01:02:03"


def test_time_unparseable_text_becomes_nat_downstream():
    # Text that is not a time must end up NaT once the timestamp is built,
    # so the row is dropped later by apply_cleaning_rules.
    df = pd.DataFrame({"date": ["2024-01-05"], "time": ["garbage"]})
    out = build_timestamp(df)
    assert out["timestamp"].isna().all()


@pytest.mark.xfail(
    reason="SUSPECTED BUG: a missing Time value stays NaN after astype(str) "
    "(pandas 3), so normalise_time_string calls .split on a float and "
    "build_timestamp raises AttributeError instead of producing NaT",
    raises=AttributeError,
    strict=True,
)
def test_time_missing_value_becomes_nat_downstream():
    df = pd.DataFrame({"date": ["2024-01-05"], "time": [None]})
    out = build_timestamp(df)
    assert out["timestamp"].isna().all()


# --------------------------------------------------------------------------
# coalesce_series / coalesce_columns
# --------------------------------------------------------------------------
def test_coalesce_series_takes_first_non_null_per_row():
    idx = pd.RangeIndex(3)
    a = pd.Series(["a1", None, None], index=idx)
    b = pd.Series([None, "b2", None], index=idx)
    c = pd.Series(["c1", "c2", "c3"], index=idx)
    result = coalesce_series([a, b, c], idx)
    assert list(result) == ["a1", "b2", "c3"]


def test_coalesce_series_empty_list_is_all_null():
    idx = pd.RangeIndex(2)
    assert coalesce_series([], idx).isna().all()


def test_coalesce_columns_report_date_and_date_lose_no_values():
    # Two different years: one uses "Report Date", the other "Date".
    df = pd.DataFrame(
        {
            "Report Date": ["2018-01-01", "2018-01-02", None, None],
            "Date": [None, None, "2021-03-01", "2021-03-02"],
        }
    )
    result = coalesce_columns(df, ["Date", "Report Date"])
    assert list(result) == ["2018-01-01", "2018-01-02", "2021-03-01", "2021-03-02"]
    assert result.notna().all()


def test_coalesce_columns_prefers_first_candidate_when_both_present():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Report Date": ["1999-09-09"]})
    assert coalesce_columns(df, ["Date", "Report Date"]).iloc[0] == "2024-01-01"


def test_coalesce_columns_missing_candidates_are_all_null():
    df = pd.DataFrame({"Other": [1, 2]})
    assert coalesce_columns(df, ["Date", "Report Date"]).isna().all()


def test_coalesce_columns_line_instead_of_route():
    df = pd.DataFrame({"Line": [7, 8], "Route": [None, None]})
    result = coalesce_columns(df, ["Route", "Line"])
    assert list(result) == [7, 8]


# --------------------------------------------------------------------------
# dedupe_columns (" Min Delay" vs "Min Delay")
# --------------------------------------------------------------------------
def test_dedupe_columns_no_duplicates_returns_same_frame():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = dedupe_columns(df)
    assert list(out.columns) == ["a", "b"]
    assert out.equals(df)


def test_dedupe_columns_merges_duplicate_names():
    df = pd.DataFrame([[5, None, 1], [None, 7, 2]], columns=["Min Delay", "Min Delay", "x"])
    out = dedupe_columns(df)
    assert list(out.columns) == ["Min Delay", "x"]
    assert list(out["Min Delay"]) == [5, 7]


def test_dedupe_columns_first_non_null_wins():
    df = pd.DataFrame([[5, 9]], columns=["Min Delay", "Min Delay"])
    assert dedupe_columns(df)["Min Delay"].iloc[0] == 5


def test_standardise_columns_merges_leading_space_min_delay():
    df = pd.DataFrame(
        {
            "Date": ["2018-01-01", "2018-01-02"],
            "Route": [1, 2],
            "Time": ["01:00", "02:00"],
            "Min Delay": [10, None],
            " Min Delay": [None, 20],
            "__source_file__": ["f.xlsx"] * 2,
            "__source_sheet__": ["Jan"] * 2,
        }
    )
    out = standardise_columns(df)
    assert list(out["min_delay"]) == [10, 20]


def test_standardise_columns_final_schema_and_variant_names():
    df = pd.DataFrame(
        {
            "Report Date": ["2018-01-01"],
            "Line": [5],
            "Time": ["01:00"],
            "Day": ["Monday"],
            "Location": ["Main St"],
            "Incident": ["Mechanical"],
            "Delay": [12],
            "Gap": [20],
            "Bound": ["N"],
            "Vehicle": [1234],
            "__source_file__": ["f.xlsx"],
            "__source_sheet__": ["Jan"],
        }
    )
    out = standardise_columns(df)
    assert list(out.columns) == FINAL_COLUMNS + ["__source_file__", "__source_sheet__"]
    row = out.iloc[0]
    assert row["date"] == "2018-01-01"
    assert row["route"] == 5
    assert row["min_delay"] == 12
    assert row["min_gap"] == 20
    assert row["direction"] == "N"


def test_standardise_columns_does_not_modify_input():
    df = pd.DataFrame(
        {"Date": ["2024-01-01"], "__source_file__": ["f"], "__source_sheet__": ["s"]}
    )
    before = df.copy()
    standardise_columns(df)
    assert df.equals(before)


# --------------------------------------------------------------------------
# build_timestamp
# --------------------------------------------------------------------------
def test_build_timestamp_is_toronto_tz_aware():
    df = pd.DataFrame({"date": ["2024-01-05"], "time": ["02:08"]})
    ts = build_timestamp(df)["timestamp"].iloc[0]
    assert ts.tz is not None
    assert str(ts.tz) == "America/Toronto"
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute) == (2024, 1, 5, 2, 8)


def test_build_timestamp_mixed_time_formats_agree():
    df = pd.DataFrame(
        {"date": ["2024-01-05", "2024-01-05"], "time": ["02:08", "02:08:00"]}
    )
    ts = build_timestamp(df)["timestamp"]
    assert ts.iloc[0] == ts.iloc[1]


def test_build_timestamp_bad_date_becomes_nat():
    df = pd.DataFrame({"date": ["not a date"], "time": ["02:08"]})
    assert build_timestamp(df)["timestamp"].isna().all()


def test_build_timestamp_nonexistent_dst_time_is_shifted_not_dropped():
    # 2024-03-10 02:30 does not exist in Toronto (spring-forward).
    df = pd.DataFrame({"date": ["2024-03-10"], "time": ["02:30"]})
    ts = build_timestamp(df)["timestamp"].iloc[0]
    assert pd.notna(ts)


def test_build_timestamp_ambiguous_dst_time_is_nat():
    # 2024-11-03 01:30 happens twice in Toronto (fall-back) -> NaT.
    df = pd.DataFrame({"date": ["2024-11-03"], "time": ["01:30"]})
    assert build_timestamp(df)["timestamp"].isna().all()


def test_build_timestamp_does_not_modify_input():
    df = pd.DataFrame({"date": ["2024-01-05"], "time": ["02:08"]})
    before = df.copy()
    build_timestamp(df)
    assert df.equals(before)


# --------------------------------------------------------------------------
# clean_route
# --------------------------------------------------------------------------
def test_clean_route_float_integers_become_plain_strings():
    result = clean_route(pd.Series([9.0, 102.0, 7]))
    assert list(result) == ["9", "102", "7"]


def test_clean_route_strips_whitespace_and_keeps_letters():
    assert list(clean_route(pd.Series([" 36A ", "941"]))) == ["36A", "941"]


def test_clean_route_missing_and_blank_become_na():
    result = clean_route(pd.Series([None, float("nan"), "", "   "]))
    assert result.isna().all()


# --------------------------------------------------------------------------
# clean_text_field
# --------------------------------------------------------------------------
def test_clean_text_field_strips_uppercases_collapses_whitespace():
    result = clean_text_field(pd.Series(["  main   st  ", "Mechanical", "a\tb"]))
    assert list(result) == ["MAIN ST", "MECHANICAL", "A B"]


def test_clean_text_field_missing_stays_missing():
    result = clean_text_field(pd.Series(["Main St", None]))
    assert result.iloc[0] == "MAIN ST"
    assert pd.isna(result.iloc[1])


# --------------------------------------------------------------------------
# apply_cleaning_rules (delay bounds, duplicates, missing values)
# --------------------------------------------------------------------------
def _row(**overrides):
    base = {
        "date": pd.Timestamp("2024-01-05"),
        "route": 9,
        "time": "02:08:00",
        "day": "Friday",
        "location": "Main St",
        "incident": "Mechanical",
        "min_delay": 10,
        "min_gap": 20,
        "direction": "N",
        "vehicle": 1234,
        "timestamp": pd.Timestamp("2024-01-05 02:08:00", tz="America/Toronto"),
    }
    base.update(overrides)
    return base


def _frame(rows):
    return pd.DataFrame(rows)


def test_negative_delays_are_removed():
    df = _frame([_row(min_delay=-5), _row(min_delay=10, location="Other")])
    out = apply_cleaning_rules(df, CLEAN_PARAMS, {})
    assert list(out["min_delay"]) == [10]


def test_delays_above_max_are_removed_and_max_itself_kept():
    df = _frame(
        [
            _row(min_delay=301, location="A"),
            _row(min_delay=300, location="B"),
            _row(min_delay=5000, location="C"),
        ]
    )
    out = apply_cleaning_rules(df, CLEAN_PARAMS, {})
    assert list(out["min_delay"]) == [300]


def test_zero_delay_kept_by_default_and_dropped_when_configured():
    df = _frame([_row(min_delay=0, location="A"), _row(min_delay=4, location="B")])
    kept = apply_cleaning_rules(df, CLEAN_PARAMS, {})
    assert sorted(kept["min_delay"]) == [0, 4]

    dropped = apply_cleaning_rules(df, {**CLEAN_PARAMS, "drop_zero_delay": True}, {})
    assert list(dropped["min_delay"]) == [4]


def test_duplicate_rows_are_removed():
    df = _frame([_row(), _row(), _row(location="Elsewhere")])
    out = apply_cleaning_rules(df, CLEAN_PARAMS, {})
    assert len(out) == 2


def test_rows_with_missing_route_delay_or_timestamp_are_removed():
    df = _frame(
        [
            _row(route=None, location="A"),
            _row(min_delay=None, location="B"),
            _row(timestamp=pd.NaT, location="C"),
            _row(location="D"),
        ]
    )
    out = apply_cleaning_rules(df, CLEAN_PARAMS, {})
    assert list(out["location"]) == ["D"]


def test_apply_cleaning_rules_normalises_fields():
    df = _frame([_row(route=9.0, location=" main  st ", direction="nb")])
    out = apply_cleaning_rules(df, CLEAN_PARAMS, {})
    row = out.iloc[0]
    assert row["route"] == "9"
    assert row["location"] == "MAIN ST"
    assert row["direction"] == "N"


def test_apply_cleaning_rules_records_row_counts():
    df = _frame([_row(min_delay=-1, location="A"), _row(min_delay=999, location="B"),
                 _row(location="C"), _row(location="C")])
    counts = {}
    apply_cleaning_rules(df, CLEAN_PARAMS, counts)
    assert counts["drop_missing_route"] == 4
    assert counts["apply_min_delay_bound"] == 3
    assert counts["apply_max_delay_bound"] == 2
    assert counts["drop_zero_delay"] == 2
    assert counts["drop_duplicates"] == 1


def test_apply_cleaning_rules_does_not_modify_input():
    df = _frame([_row(min_delay=-5), _row()])
    before = df.copy()
    apply_cleaning_rules(df, CLEAN_PARAMS, {})
    assert df.equals(before)
