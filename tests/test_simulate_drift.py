"""Tests for scripts/simulate_drift.py, using a small synthetic stream."""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "simulate_drift", os.path.join(ROOT, "scripts", "simulate_drift.py")
)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

PARAMS = {"window_size": 100, "seed": 42}
N_WINDOWS = 20
ONSET = int(round(sim.ONSET_FRAC * N_WINDOWS))


def make_base():
    rng = np.random.default_rng(0)
    n = PARAMS["window_size"] * N_WINDOWS
    hour = rng.integers(0, 24, n)
    has_prev = (rng.random(n) > 0.05).astype(int)
    df = pd.DataFrame({
        "hour": hour,
        "minute_of_day": hour * 60 + rng.integers(0, 60, n),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_peak_am": np.isin(hour, [7, 8, 9]).astype(int),
        "is_peak_pm": np.isin(hour, [17, 18, 19]).astype(int),
        "prev_bus_tt_s": rng.uniform(40, 200, n) * has_prev,
        "prev3_median_tt_s": rng.uniform(40, 200, n) * has_prev,
        "own_prev_segment_tt_s": rng.uniform(40, 200, n),
        "headway_s": rng.uniform(100, 900, n) * has_prev,
        "n_buses_recent": rng.integers(1, 5, n) * has_prev,
        "has_prev_bus": has_prev,
        "segment_id": rng.choice([f"s{i}" for i in range(20)], n),
        "travel_time_s": rng.uniform(20, 300, n),
    })
    df["window_id"] = np.arange(n) // PARAMS["window_size"]
    return df


def build(name):
    return sim.build_scenario(name, make_base(), N_WINDOWS, PARAMS)


def split(df):
    before = df[df["window_id"] < ONSET]
    after = df[df["window_id"] >= ONSET]
    return before, after


def test_clean_test_is_untouched_and_labelled_real():
    df, _ = build("clean_test")
    base = make_base()
    for c in base.columns:
        assert (df[c].to_numpy() == base[c].to_numpy()).all()
    assert df["data_label"].iloc[0].startswith("REAL")
    assert (df["sim_intensity"] == 0).all()


def test_perturbed_scenarios_are_labelled_simulated():
    for name in sim.SCENARIOS[1:]:
        df, _ = build(name)
        assert (df["data_label"] == "SIMULATED").all()


def test_same_seed_gives_same_result():
    a, _ = build("peak_traffic_surge")
    b, _ = build("peak_traffic_surge")
    pd.testing.assert_frame_equal(a, b)


def test_peak_surge_raises_peak_share_and_realtime_after_onset_only():
    df, _ = build("peak_traffic_surge")
    base = make_base()
    before, after = split(df)
    bbefore, bafter = split(base)
    assert before["prev_bus_tt_s"].equals(bbefore["prev_bus_tt_s"])
    assert before["is_peak_am"].equals(bbefore["is_peak_am"])
    peak = lambda d: ((d["is_peak_am"] == 1) | (d["is_peak_pm"] == 1)).mean()
    assert peak(after) > peak(bafter) + 0.1
    assert after["prev_bus_tt_s"].mean() > bafter["prev_bus_tt_s"].mean() * 1.3


def test_road_works_changes_target_only():
    df, _ = build("road_works")
    base = make_base()
    for c in ["prev_bus_tt_s", "hour", "has_prev_bus", "segment_id"]:
        assert (df[c].to_numpy() == base[c].to_numpy()).all()
    before, after = split(df)
    bbefore, bafter = split(base)
    assert before["travel_time_s"].equals(bbefore["travel_time_s"])
    assert after["travel_time_s"].mean() > bafter["travel_time_s"].mean()


def test_gps_outage_raises_share_without_prev_bus():
    df, _ = build("gps_outage")
    base = make_base()
    before, after = split(df)
    bbefore, bafter = split(base)
    assert (before["has_prev_bus"] == 0).mean() == (bbefore["has_prev_bus"] == 0).mean()
    assert (after["has_prev_bus"] == 0).mean() > (bafter["has_prev_bus"] == 0).mean() + 0.15
    no_prev = df[df["has_prev_bus"] == 0]
    assert (no_prev["prev_bus_tt_s"] == 0).all()


def test_gradual_drift_grows_to_full_strength():
    df, info = build("gradual_drift")
    inten = np.array(info["intensity_per_window"])
    assert (np.diff(inten) >= 0).all()
    assert inten[0] == 0 and inten[-1] == 1
    base = make_base()
    last = df["window_id"] == N_WINDOWS - 1
    ratio = df.loc[last, "travel_time_s"].mean() / base.loc[last, "travel_time_s"].mean()
    assert 1.25 < ratio < 1.35
