import pandas as pd

from src.data.clean import assign_trips, find_bad_trips


def _rows(records):
    """Build a minimal stop-events frame sorted the way assign_trips expects."""
    df = pd.DataFrame(records)
    return df.sort_values(["route", "direction", "day_index", "vehicle", "row_order"]).reset_index(drop=True)


def test_trip_splits_on_node_reset():
    df = _rows(
        [
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 1, "time_s": 100, "row_order": 0},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 2, "time_s": 200, "row_order": 1},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 3, "time_s": 300, "row_order": 2},
            # node resets -> new trip
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 1, "time_s": 400, "row_order": 3},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 2, "time_s": 500, "row_order": 4},
        ]
    )
    out, _ = assign_trips(df, trip_gap_minutes=30)
    trip_ids = out.sort_values("row_order")["trip_id"].tolist()
    assert trip_ids[:3] == [trip_ids[0]] * 3
    assert trip_ids[3] != trip_ids[0]
    assert trip_ids[3] == trip_ids[4]


def test_trip_splits_on_long_gap():
    df = _rows(
        [
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 1, "time_s": 100, "row_order": 0},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 2, "time_s": 200, "row_order": 1},
            # gap of 40 minutes > trip_gap_minutes=30 -> new trip
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 3, "time_s": 200 + 40 * 60, "row_order": 2},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 4, "time_s": 200 + 41 * 60, "row_order": 3},
        ]
    )
    out, _ = assign_trips(df, trip_gap_minutes=30)
    trip_ids = out.sort_values("row_order")["trip_id"].tolist()
    assert trip_ids[0] == trip_ids[1]
    assert trip_ids[2] == trip_ids[3]
    assert trip_ids[0] != trip_ids[2]


def test_midnight_crossing_is_fixed_not_split():
    df = _rows(
        [
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 1, "time_s": 86000, "row_order": 0},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 2, "time_s": 86300, "row_order": 1},
            # raw time jumps back by > 12h: midnight wrap, not a real gap
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 3, "time_s": 120, "row_order": 2},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 4, "time_s": 420, "row_order": 3},
        ]
    )
    out, midnight_trips_fixed = assign_trips(df, trip_gap_minutes=30)
    out = out.sort_values("row_order")
    assert out["trip_id"].nunique() == 1
    assert midnight_trips_fixed == 1
    # corrected time_s should be strictly increasing across the midnight wrap
    assert out["time_s"].is_monotonic_increasing
    assert out["time_s"].iloc[-1] > 86400


def test_trip_with_backward_time_is_dropped():
    df = _rows(
        [
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 1, "time_s": 100, "row_order": 0},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 2, "time_s": 200, "row_order": 1},
            # small backward jump (not a midnight wrap, doesn't trigger a new trip
            # or a node decrease) -> whole trip should be dropped
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 3, "time_s": 150, "row_order": 2},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 4, "time_s": 250, "row_order": 3},
            {"route": "425", "direction": "up", "day_index": 1, "vehicle": "V1", "node": 5, "time_s": 300, "row_order": 4},
        ]
    )
    out, _ = assign_trips(df, trip_gap_minutes=30)
    assert out["trip_id"].nunique() == 1  # no node decrease, no >30min gap -> stays one trip

    bad = find_bad_trips(out, min_nodes_per_trip=2)
    trip_id = out["trip_id"].iloc[0]
    assert trip_id in bad["time_backward"]
