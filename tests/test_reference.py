import json
import numpy as np
import pandas as pd
from src.monitoring.reference import build_reference


def _fake_df(n=2000):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "speed": rng.normal(50, 10, n),
        "has_prev_bus": rng.integers(0, 2, n),
        "route": rng.choice(["425", "534"], n),
        "travel_time_s": rng.normal(117, 30, n),
    })


def test_reference_structure():
    df = _fake_df()
    preds = df["travel_time_s"].to_numpy() + 1.0
    ref = build_reference(df, ["speed", "has_prev_bus"], ["route"], "travel_time_s", preds)
    assert ref["n_rows"] == 2000
    assert ref["features"]["speed"]["type"] == "numeric"
    assert ref["features"]["has_prev_bus"]["type"] == "categorical"
    assert ref["features"]["route"]["type"] == "categorical"
    assert abs(sum(ref["features"]["speed"]["bin_props"]) - 1.0) < 1e-9
    assert abs(sum(ref["features"]["route"]["props"].values()) - 1.0) < 1e-9
    assert ref["target"]["name"] == "travel_time_s"
    assert "prediction" in ref
    json.dumps(ref)  # must be JSON-safe
