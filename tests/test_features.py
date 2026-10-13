"""Unit tests for src/features/build_features.py.

All tests use small hand-made DataFrames; no real data files are read.
"""

import holidays
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    add_derived_features,
    add_target_encoding,
    apply_rare_value_sets,
    apply_smoothed_group_stats,
    compute_oof_target_encoding,
    fill_missing_categoricals,
    fit_rare_value_sets,
    fit_smoothed_group_stats,
    select_output_columns,
    split_by_time,
)

TIMEZONE = "America/Toronto"

FEATURIZE_PARAMS = {
    "train_end": "2022-12-31",
    "val_year": 2023,
    "test_year": 2024,
    "rare_location_min_count": 2,
    "rare_route_min_count": 2,
    "heavy_precip_mm": 2.5,
    "covid_start": "2020-03-15",
    "covid_end": "2021-12-31",
    "te_smoothing": 20,
    "te_folds": 5,
}


def make_merged_row(timestamp: str, route: str = "1", location: str = "Main St") -> dict:
    """A minimal hand-made row shaped like data/interim/merged.parquet."""
    return {
        "timestamp": pd.Timestamp(timestamp, tz=TIMEZONE),
        "route": route,
        "incident": "Mechanical",
        "direction": "N",
        "location": location,
        "min_delay": 10.0,
        "min_gap": 20.0,
        "vehicle": 1234.0,
        "temperature_2m": 5.0,
        "precipitation": 0.0,
        "snowfall": 0.0,
        "wind_speed_10m": 10.0,
    }


# --------------------------------------------------------------------------
# excluded columns
# --------------------------------------------------------------------------
def test_leakage_and_excluded_columns_never_appear_in_features():
    df = pd.DataFrame(
        [make_merged_row("2022-06-01 08:00"), make_merged_row("2022-06-02 09:00")]
    )
    df = fill_missing_categoricals(df)
    df = add_derived_features(df, FEATURIZE_PARAMS, holidays.Canada(subdiv="ON"))
    kept = fit_rare_value_sets(df, FEATURIZE_PARAMS)
    df = apply_rare_value_sets(df, kept)
    small_te_params = dict(FEATURIZE_PARAMS, te_folds=2)
    add_target_encoding(df, df.copy(), df.copy(), small_te_params)
    out = select_output_columns(df)

    assert "min_gap" not in out.columns
    assert "vehicle" not in out.columns
    assert "year" not in out.columns
    assert "date" not in out.columns
    assert "time" not in out.columns
    assert TARGET in out.columns
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "timestamp"]) == set(out.columns)


# --------------------------------------------------------------------------
# time split
# --------------------------------------------------------------------------
def test_split_is_strictly_time_ordered():
    df = pd.DataFrame(
        [
            make_merged_row("2022-12-31 23:00"),
            make_merged_row("2023-01-01 01:00"),
            make_merged_row("2023-12-31 23:00"),
            make_merged_row("2024-01-01 02:00"),
        ]
    )
    train, val, test = split_by_time(df, FEATURIZE_PARAMS)

    assert len(train) == 1
    assert len(val) == 2
    assert len(test) == 1
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].min() < test["timestamp"].min()


# --------------------------------------------------------------------------
# rare-value grouping, fitted on train only
# --------------------------------------------------------------------------
def test_rare_grouping_is_fitted_on_train_only():
    train = pd.DataFrame(
        [make_merged_row("2022-01-01", route="RARE") for _ in range(1)]
        + [make_merged_row("2022-01-02", route="COMMON") for _ in range(5)]
    )
    # "RARE" is common in test but was rare (count=1) in train, so it must
    # still be grouped into OTHER in test, because the map was fit on train.
    test = pd.DataFrame([make_merged_row("2024-01-01", route="RARE") for _ in range(10)])

    kept = fit_rare_value_sets(train, FEATURIZE_PARAMS)
    test_mapped = apply_rare_value_sets(test, kept)

    assert "RARE" not in kept["route"]
    assert (test_mapped["route"] == "OTHER").all()


def test_unseen_category_in_val_becomes_other():
    train = pd.DataFrame([make_merged_row("2022-01-01", route="1") for _ in range(5)])
    val = pd.DataFrame([make_merged_row("2023-01-01", route="NEVER_SEEN")])

    kept = fit_rare_value_sets(train, FEATURIZE_PARAMS)
    val_mapped = apply_rare_value_sets(val, kept)

    assert val_mapped["route"].iloc[0] == "OTHER"


# --------------------------------------------------------------------------
# is_holiday
# --------------------------------------------------------------------------
def test_is_holiday_flags_canada_day_and_not_a_normal_weekday():
    df = pd.DataFrame(
        [
            make_merged_row("2024-07-01 10:00"),  # Canada Day
            make_merged_row("2024-07-10 10:00"),  # an ordinary Wednesday
        ]
    )
    out = add_derived_features(df, FEATURIZE_PARAMS, holidays.Canada(subdiv="ON"))

    assert out["is_holiday"].iloc[0] == 1
    assert out["is_holiday"].iloc[1] == 0


# --------------------------------------------------------------------------
# target encoding
# --------------------------------------------------------------------------
def test_oof_target_encoding_never_uses_a_rows_own_target():
    # 2 rows, same route, 2 folds -> each row's holdout fold has exactly
    # the OTHER row in its training fold, so with smoothing=0 its encoded
    # value must equal the other row's target exactly, not its own.
    train = pd.DataFrame({"route": ["1", "1"], "min_delay": [10.0, 100.0]})
    global_mean_value = float(train["min_delay"].mean())

    oof = compute_oof_target_encoding(
        train, ["route"], global_mean_value, smoothing=0, n_folds=2, random_state=42
    )
    assert oof[0] == 100.0  # row 0's own target (10.0) is never used
    assert oof[1] == 10.0  # row 1's own target (100.0) is never used

    # the in-sample (non-OOF) group mean, which DOES use each row's own
    # target, differs from the OOF values above.
    in_sample_stats = fit_smoothed_group_stats(train, ["route"], global_mean_value, smoothing=0)
    in_sample = apply_smoothed_group_stats(train, ["route"], in_sample_stats, global_mean_value).to_numpy()
    assert not (oof == in_sample).all()


def test_unseen_group_in_val_gets_global_train_mean():
    train = pd.DataFrame({"location": ["A", "A", "B"], "min_delay": [10.0, 20.0, 30.0]})
    global_mean_value = float(train["min_delay"].mean())
    stats = fit_smoothed_group_stats(train, ["location"], global_mean_value, smoothing=20)

    val = pd.DataFrame({"location": ["NEVER_SEEN"]})
    encoded = apply_smoothed_group_stats(val, ["location"], stats, global_mean_value)

    assert encoded.iloc[0] == global_mean_value


def test_target_encoding_columns_have_no_nan_in_any_split():
    train = pd.DataFrame(
        [
            make_merged_row("2022-01-01", route="1", location="A"),
            make_merged_row("2022-01-02", route="1", location="A"),
            make_merged_row("2022-01-03", route="2", location="B"),
            make_merged_row("2022-01-04", route="2", location="B"),
        ]
    )
    train["min_delay"] = [10.0, 100.0, 5.0, 50.0]
    val = pd.DataFrame([make_merged_row("2023-01-01", route="1", location="NEVER_SEEN")])
    val["min_delay"] = [20.0]
    test = pd.DataFrame([make_merged_row("2024-01-01", route="2", location="B")])
    test["min_delay"] = [30.0]

    small_te_params = dict(FEATURIZE_PARAMS, te_folds=2)
    spec = add_target_encoding(train, val, test, small_te_params)
    te_cols = list(spec.keys())

    assert not train[te_cols].isna().any().any()
    assert not val[te_cols].isna().any().any()
    assert not test[te_cols].isna().any().any()
