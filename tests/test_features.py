import pandas as pd

from src.features.build_features import (
    FEATURES,
    OTHER_LABEL,
    apply_historical_tables,
    apply_vehicle_grouping,
    compute_recent_bus_features,
    fit_historical_tables,
    fit_rare_vehicle_set,
    load_params,
)

BASE_SEGMENT = dict(route="425", direction="up", from_node=1, to_node=2, day_index=1)


def test_prev_bus_ignores_earlier_bus_still_in_flight():
    # Bus A departs first but takes 1000s (arrives at 1100) - still "in flight"
    # when bus C departs at 300. Bus B departs after A but finishes quickly
    # (arrives at 250), so it IS a qualifying previous bus for C.
    df = pd.DataFrame(
        [
            {**BASE_SEGMENT, "vehicle": "A", "depart_time_s": 100.0, "travel_time_s": 1000.0},
            {**BASE_SEGMENT, "vehicle": "B", "depart_time_s": 200.0, "travel_time_s": 50.0},
            {**BASE_SEGMENT, "vehicle": "C", "depart_time_s": 300.0, "travel_time_s": 40.0},
        ]
    )
    out = compute_recent_bus_features(df, prev_n_buses=3, recent_window_s=10_000)
    bus_c = out[out["vehicle"] == "C"].iloc[0]
    assert bus_c["has_prev_bus"] == 1
    assert bus_c["prev_bus_tt_s"] == 50.0  # B's travel time, NOT A's 1000s
    assert bus_c["headway_s"] == 300.0 - 200.0  # since B's own depart time


def test_prev_features_never_use_a_later_bus():
    df = pd.DataFrame(
        [
            {**BASE_SEGMENT, "vehicle": "A", "depart_time_s": 100.0, "travel_time_s": 40.0},
            {**BASE_SEGMENT, "vehicle": "B", "depart_time_s": 200.0, "travel_time_s": 9999.0},
            {**BASE_SEGMENT, "vehicle": "C", "depart_time_s": 300.0, "travel_time_s": 50.0},
        ]
    )
    out = compute_recent_bus_features(df, prev_n_buses=3, recent_window_s=10_000)
    bus_a = out[out["vehicle"] == "A"].iloc[0]
    # A is the first bus of the day on this segment: no earlier bus exists at all,
    # regardless of what B/C (which depart later) look like.
    assert bus_a["has_prev_bus"] == 0
    assert pd.isna(bus_a["prev_bus_tt_s"])
    assert bus_a["n_buses_recent"] == 0


def test_split_day_ranges_are_ordered_and_non_overlapping():
    params = load_params()
    featurize_params = params["featurize"]
    train_lo, train_hi = featurize_params["train_days"]
    val_lo, val_hi = featurize_params["val_days"]
    test_lo, test_hi = featurize_params["test_days"]

    assert train_lo <= train_hi < val_lo <= val_hi < test_lo <= test_hi


def test_typical_values_for_eval_split_come_only_from_train():
    train = pd.DataFrame(
        [
            {**BASE_SEGMENT, "segment_id": "425_up_1", "hour": 8, "travel_time_s": 100.0},
            {**BASE_SEGMENT, "segment_id": "425_up_1", "hour": 8, "travel_time_s": 100.0},
        ]
    )
    val = pd.DataFrame(
        [
            {**BASE_SEGMENT, "segment_id": "425_up_1", "hour": 8, "travel_time_s": 999.0},
        ]
    )
    tables = fit_historical_tables(train)
    hour_vals, segment_vals = apply_historical_tables(val, tables)
    # the val row's own (very different) travel_time_s must not influence the
    # looked-up typical value - it should be exactly the train median.
    assert hour_vals.iloc[0] == 100.0
    assert segment_vals.iloc[0] == 100.0


def test_target_and_delay_not_in_feature_list():
    assert "travel_time_s" not in FEATURES
    assert "delay_s" not in FEATURES
    assert "trip_id" not in FEATURES
    assert "day_index" not in FEATURES


def test_unseen_vehicle_becomes_other():
    train = pd.DataFrame({"vehicle": ["V1"] * 10 + ["V2"] * 2})
    kept = fit_rare_vehicle_set(train, min_count=5)
    assert "V1" in kept
    assert "V2" not in kept  # rare, below min_count

    val = pd.DataFrame({"vehicle": ["V1", "V2", "UNSEEN_VEHICLE"]})
    grouped = apply_vehicle_grouping(val, kept)
    assert grouped.tolist() == ["V1", OTHER_LABEL, OTHER_LABEL]
