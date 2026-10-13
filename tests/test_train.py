"""Unit tests for src/models/train.py.

All tests use small hand-made DataFrames; no real data files are read and
no MLflow run is started.
"""

import numpy as np
import pandas as pd

from src.models.train import GlobalMedianRegressor, NaiveBaselineRegressor, clip_and_backtransform


def make_row(route: str, incident: str, min_delay: float) -> dict:
    return {"route": route, "incident": incident, "min_delay": min_delay}


# --------------------------------------------------------------------------
# NaiveBaselineRegressor fallback behaviour
# --------------------------------------------------------------------------
def test_naive_baseline_uses_group_mean_when_pair_seen_in_train():
    train = pd.DataFrame(
        [make_row("1", "Mechanical", 10.0), make_row("1", "Mechanical", 20.0)]
    )
    model = NaiveBaselineRegressor().fit(train, train["min_delay"])

    pred = model.predict(pd.DataFrame([make_row("1", "Mechanical", np.nan)]))
    assert pred[0] == 15.0


def test_naive_baseline_falls_back_to_incident_mean_for_unseen_pair():
    train = pd.DataFrame(
        [
            make_row("1", "Mechanical", 10.0),
            make_row("2", "Mechanical", 30.0),
            make_row("1", "Diversion", 100.0),
        ]
    )
    model = NaiveBaselineRegressor().fit(train, train["min_delay"])

    # route "3" + "Mechanical" was never seen together, but "Mechanical"
    # was seen (for routes 1 and 2), so it should fall back to that mean.
    pred = model.predict(pd.DataFrame([make_row("3", "Mechanical", np.nan)]))
    assert pred[0] == 20.0  # mean(10, 30)


def test_naive_baseline_falls_back_to_global_mean_for_unseen_incident():
    train = pd.DataFrame(
        [make_row("1", "Mechanical", 10.0), make_row("2", "Diversion", 30.0)]
    )
    model = NaiveBaselineRegressor().fit(train, train["min_delay"])

    pred = model.predict(pd.DataFrame([make_row("9", "NeverSeenIncident", np.nan)]))
    assert pred[0] == 20.0  # global mean of (10, 30)


# --------------------------------------------------------------------------
# predictions are never negative
# --------------------------------------------------------------------------
def test_clip_and_backtransform_never_negative():
    raw = np.array([-5.0, -0.1, 0.0, 1.0, 50.0])
    out = clip_and_backtransform(raw, log_target=False)
    assert (out >= 0).all()
    assert out[-1] == 50.0


# --------------------------------------------------------------------------
# log1p / expm1 round trip
# --------------------------------------------------------------------------
def test_log1p_expm1_round_trip_recovers_original_values():
    original = np.array([0.0, 1.0, 10.0, 137.0])
    transformed = np.log1p(original)
    recovered = clip_and_backtransform(transformed, log_target=True)
    np.testing.assert_allclose(recovered, original, rtol=1e-10)


# --------------------------------------------------------------------------
# GlobalMedianRegressor
# --------------------------------------------------------------------------
def test_global_median_predicts_constant_train_median():
    train = pd.DataFrame({"min_delay": [5.0, 10.0, 10.0, 15.0, 100.0]})
    model = GlobalMedianRegressor().fit(train, train["min_delay"])

    other_rows = pd.DataFrame({"min_delay": [0.0, 0.0, 0.0]})
    pred = model.predict(other_rows)

    assert (pred == 10.0).all()  # median of [5, 10, 10, 15, 100]
