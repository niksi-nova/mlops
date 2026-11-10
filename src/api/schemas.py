"""Pydantic v2 request/response models for the Delhi bus travel-time API.

Field order and names mirror data/processed/feature_spec.json exactly
(categorical_features + numeric_features): route, direction, segment_id,
relative_weekday, from_node, nodes_spanned, route_position, stops_into_trip,
elapsed_trip_s, hour, minute_of_day, hour_sin, hour_cos, is_peak_am,
is_peak_pm, prev_bus_tt_s, prev3_median_tt_s, headway_s, n_buses_recent,
has_prev_bus, own_prev_segment_tt_s, own_prev_segment_ratio,
typical_segment_hour_s, typical_segment_s, te_vehicle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_ROW = {
    "route": "425",
    "direction": "down",
    "segment_id": "425_down_26",
    "relative_weekday": 6,
    "from_node": 26,
    "nodes_spanned": 1,
    "route_position": 0.6341463414634146,
    "stops_into_trip": 1,
    "elapsed_trip_s": 0.0,
    "hour": 8,
    "minute_of_day": 484,
    "hour_sin": 0.8660254037844387,
    "hour_cos": -0.49999999999999983,
    "is_peak_am": 1,
    "is_peak_pm": 0,
    "prev_bus_tt_s": 30.0,
    "prev3_median_tt_s": 30.0,
    "headway_s": 52.0,
    "n_buses_recent": 3,
    "has_prev_bus": 1,
    "own_prev_segment_tt_s": 40.0,
    "own_prev_segment_ratio": 1.0,
    "typical_segment_hour_s": 40.0,
    "typical_segment_s": 40.0,
    "te_vehicle": 144.26669174404856,
}

MAX_BATCH_SIZE = 500


class PredictionRequest(BaseModel):
    """The 25 pre-computed features the model needs for one segment."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_ROW})

    route: str = Field(..., description="Bus route identifier")
    direction: str = Field(..., description="Direction of travel along the route")
    segment_id: str = Field(..., description="Stop-to-stop segment identifier")
    relative_weekday: int = Field(..., ge=0, le=6, description="Day of week, 0-6")
    from_node: int = Field(..., ge=0, description="Origin stop/node index of the segment")
    nodes_spanned: int = Field(..., ge=1, description="Number of nodes the segment spans")
    route_position: float = Field(..., ge=0.0, le=1.0, description="Fractional position along the route, 0-1")
    stops_into_trip: int = Field(..., ge=0, description="Number of stops already made this trip")
    elapsed_trip_s: float = Field(..., ge=0.0, description="Seconds elapsed since trip start")
    hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23")
    minute_of_day: int = Field(..., ge=0, le=1439, description="Minute of day, 0-1439")
    hour_sin: float = Field(..., ge=-1.0, le=1.0, description="sin-encoded hour of day")
    hour_cos: float = Field(..., ge=-1.0, le=1.0, description="cos-encoded hour of day")
    is_peak_am: int = Field(..., ge=0, le=1, description="1 if within the AM peak window, else 0")
    is_peak_pm: int = Field(..., ge=0, le=1, description="1 if within the PM peak window, else 0")
    prev_bus_tt_s: float = Field(..., ge=0.0, description="Previous bus's travel time on this segment, seconds")
    prev3_median_tt_s: float = Field(..., ge=0.0, description="Median travel time of the last 3 buses, seconds")
    headway_s: float = Field(..., ge=0.0, description="Seconds since the previous bus on this segment")
    n_buses_recent: int = Field(..., ge=0, description="Number of buses observed recently on this segment")
    has_prev_bus: int = Field(..., ge=0, le=1, description="1 if a previous bus's data is available, else 0")
    own_prev_segment_tt_s: float = Field(..., ge=0.0, description="This bus's travel time on the previous segment, seconds")
    own_prev_segment_ratio: float = Field(..., ge=0.0, description="Ratio of this bus's previous segment time to typical")
    typical_segment_hour_s: float = Field(..., ge=0.0, description="Typical travel time for this segment at this hour, seconds")
    typical_segment_s: float = Field(..., ge=0.0, description="Typical travel time for this segment overall, seconds")
    te_vehicle: float = Field(..., description="Target-encoded vehicle effect")


class PredictionResponse(BaseModel):
    predicted_travel_time_s: float = Field(..., description="Predicted segment travel time in seconds")
    model_alias: str = Field(..., description="Registry alias the model was loaded from (e.g. 'production')")
    model_version: str = Field(..., description="Registry version of the model used")
    request_id: str = Field(..., description="Unique identifier for this prediction request")


class BatchRequest(BaseModel):
    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
