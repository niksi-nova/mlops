import pandas as pd

from src.data.build_segments import apply_bounds, build_segments, drop_zero_span_segments


def _stop_events(records):
    return pd.DataFrame(records)


def test_segment_travel_time_and_nodes_spanned():
    stop_events = _stop_events(
        [
            {"route": "425", "direction": "up", "vehicle": "V1", "trip_id": 1, "month": 1, "day_index": 1, "relative_weekday": 0, "node": 1, "time_s": 100},
            {"route": "425", "direction": "up", "vehicle": "V1", "trip_id": 1, "month": 1, "day_index": 1, "relative_weekday": 0, "node": 2, "time_s": 160},
            {"route": "425", "direction": "up", "vehicle": "V1", "trip_id": 1, "month": 1, "day_index": 1, "relative_weekday": 0, "node": 4, "time_s": 280},
        ]
    )
    segments = build_segments(stop_events)

    assert len(segments) == 2

    first = segments.iloc[0]
    assert first["from_node"] == 1
    assert first["to_node"] == 2
    assert first["nodes_spanned"] == 1
    assert first["travel_time_s"] == 60
    assert first["seconds_per_stop"] == 60

    second = segments.iloc[1]
    assert second["from_node"] == 2
    assert second["to_node"] == 4
    assert second["nodes_spanned"] == 2
    assert second["travel_time_s"] == 120
    assert second["seconds_per_stop"] == 60


def test_zero_span_segments_are_dropped():
    segments = pd.DataFrame({"nodes_spanned": [1, 0, 2]})
    kept = drop_zero_span_segments(segments)
    assert kept["nodes_spanned"].tolist() == [1, 2]


def test_out_of_bounds_segments_are_dropped():
    segments = pd.DataFrame(
        {
            "travel_time_s": [0.5, 30, 1800, 2000],
        }
    )
    kept = apply_bounds(segments, min_segment_seconds=1, max_segment_seconds=1800)
    assert kept["travel_time_s"].tolist() == [30, 1800]
