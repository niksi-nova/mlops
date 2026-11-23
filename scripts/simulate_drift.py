"""Simulate a production stream for drift demos.

Real test days 77-92 are cut into windows. Seeded perturbations are applied
on top. Every perturbed stream is labelled SIMULATED. Nothing here is real
production data and nothing is saved to disk unless --save is used.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.evaluate import load_split  # noqa: E402

SCENARIOS = [
    "clean_test",
    "peak_traffic_surge",
    "road_works",
    "gps_outage",
    "gradual_drift",
]
REALTIME_COLS = ["prev_bus_tt_s", "prev3_median_tt_s", "own_prev_segment_tt_s"]
NO_PREV_COLS = ["prev_bus_tt_s", "prev3_median_tt_s", "headway_s", "n_buses_recent"]
ONSET_FRAC = 0.4
SURGE_SCALE = 1.4          # real-time features x1.4 at full intensity
SURGE_SHIFT_PROB = 0.6     # share of off-peak rows moved into peak hours
ROAD_SEG_SHARE = 0.15      # share of segments hit by road works
ROAD_FACTOR = (1.3, 1.5)   # travel_time_s multiplier range per hit segment
OUTAGE_SHARE = 0.30        # target extra share of has_prev_bus = 0 rows
GRADUAL_TARGET_SCALE = 0.3


def load_params():
    with open(os.path.join(ROOT, "params.yaml")) as f:
        return yaml.safe_load(f)["drift"]


def load_stream(window_size):
    """Real test split, time-ordered, cut into full windows."""
    df = load_split("test")
    df = df.sort_values(["day_index", "minute_of_day", "trip_id"], kind="stable")
    df = df.reset_index(drop=True)
    n_windows = len(df) // window_size
    df = df.iloc[: n_windows * window_size].copy()
    df["window_id"] = np.arange(len(df)) // window_size
    return df, n_windows


def intensity_per_window(scenario, n_windows):
    """0 = untouched, 1 = full strength, for each window."""
    onset = int(round(ONSET_FRAC * n_windows))
    w = np.arange(n_windows)
    if scenario == "clean_test":
        return np.zeros(n_windows)
    if scenario == "gradual_drift":
        return np.clip((w - onset + 1) / max(n_windows - onset, 1), 0, 1)
    return (w >= onset).astype(float)


def scale_realtime(df, inten):
    factor = 1.0 + (SURGE_SCALE - 1.0) * inten
    for c in REALTIME_COLS:
        df[c] = df[c].to_numpy() * factor  # NaN stays NaN


def peak_traffic_surge(df, rng, inten):
    n = len(df)
    am = sorted(df.loc[df["is_peak_am"] == 1, "hour"].unique().tolist())
    pm = sorted(df.loc[df["is_peak_pm"] == 1, "hour"].unique().tolist())
    peak_hours = np.array(am + pm)
    sin_lut = np.full(24, np.nan)
    cos_lut = np.full(24, np.nan)
    for h, g in df.groupby("hour"):
        sin_lut[int(h)] = g["hour_sin"].median()
        cos_lut[int(h)] = g["hour_cos"].median()

    offpeak = (df["is_peak_am"] == 0) & (df["is_peak_pm"] == 0)
    pick = (rng.random(n) < SURGE_SHIFT_PROB * inten) & offpeak.to_numpy()
    new_h = rng.choice(peak_hours, size=n)

    hour = df["hour"].to_numpy().copy()
    mod = df["minute_of_day"].to_numpy().copy()
    hsin = df["hour_sin"].to_numpy().copy()
    hcos = df["hour_cos"].to_numpy().copy()
    pam = df["is_peak_am"].to_numpy().copy()
    ppm = df["is_peak_pm"].to_numpy().copy()

    hour[pick] = new_h[pick]
    mod[pick] = new_h[pick] * 60 + (mod[pick] % 60)
    hsin[pick] = sin_lut[new_h[pick]]
    hcos[pick] = cos_lut[new_h[pick]]
    pam[pick] = np.isin(new_h[pick], am).astype(pam.dtype)
    ppm[pick] = np.isin(new_h[pick], pm).astype(ppm.dtype)

    df["hour"], df["minute_of_day"] = hour, mod
    df["hour_sin"], df["hour_cos"] = hsin, hcos
    df["is_peak_am"], df["is_peak_pm"] = pam, ppm
    scale_realtime(df, inten)


def road_works(df, rng, inten):
    segs = np.array(sorted(df["segment_id"].unique()))
    k = max(1, int(ROAD_SEG_SHARE * len(segs)))
    chosen = rng.choice(segs, size=k, replace=False)
    factors = rng.uniform(ROAD_FACTOR[0], ROAD_FACTOR[1], size=k)
    fmap = dict(zip(chosen, factors))
    base = df["segment_id"].map(fmap).fillna(1.0).to_numpy()
    df["travel_time_s"] = df["travel_time_s"].to_numpy() * (1.0 + (base - 1.0) * inten)


def gps_outage(df, rng, inten):
    n = len(df)
    donors = np.where(df["has_prev_bus"].to_numpy() == 0)[0]
    if len(donors) == 0:
        raise ValueError("no real has_prev_bus=0 rows to copy from")
    pick = (rng.random(n) < OUTAGE_SHARE * inten) & (df["has_prev_bus"].to_numpy() == 1)
    src = rng.choice(donors, size=int(pick.sum()))
    for c in NO_PREV_COLS:
        vals = df[c].to_numpy().copy()
        vals[pick] = df[c].to_numpy()[src]
        df[c] = vals
    hp = df["has_prev_bus"].to_numpy().copy()
    hp[pick] = 0
    df["has_prev_bus"] = hp


def gradual_drift(df, rng, inten):
    scale_realtime(df, inten)
    df["travel_time_s"] = df["travel_time_s"].to_numpy() * (1.0 + GRADUAL_TARGET_SCALE * inten)


FUNCS = {
    "peak_traffic_surge": peak_traffic_surge,
    "road_works": road_works,
    "gps_outage": gps_outage,
    "gradual_drift": gradual_drift,
}


def build_scenario(name, base=None, n_windows=None, params=None):
    """Return (df, info). df has window_id, sim_intensity, scenario, data_label."""
    if name not in SCENARIOS:
        raise ValueError("unknown scenario: " + name)
    params = params or load_params()
    if base is None:
        base, n_windows = load_stream(params["window_size"])
    seed = int(params["seed"]) + SCENARIOS.index(name)
    rng = np.random.default_rng(seed)
    df = base.copy()
    per_window = intensity_per_window(name, n_windows)
    inten = per_window[df["window_id"].to_numpy()]
    if name in FUNCS:
        FUNCS[name](df, rng, inten)
    df["sim_intensity"] = inten
    df["scenario"] = name
    df["data_label"] = (
        "REAL (unperturbed test days 77-92)" if name == "clean_test" else "SIMULATED"
    )
    info = {"scenario": name, "n_windows": n_windows, "seed": seed,
            "intensity_per_window": per_window.tolist()}
    return df, info


def summarize(df, info):
    n = info["n_windows"]
    print("\n== %s  [%s]  seed=%d" % (info["scenario"], df["data_label"].iloc[0], info["seed"]))
    print("window  intensity  mean_tt_s  mean_prev_bus_tt  no_prev_share  peak_share")
    for w in sorted({0, n // 2, n - 1}):
        g = df[df["window_id"] == w]
        peak = ((g["is_peak_am"] == 1) | (g["is_peak_pm"] == 1)).mean()
        print("%6d  %9.2f  %9.1f  %16.1f  %13.3f  %10.3f" % (
            w, g["sim_intensity"].iloc[0], g["travel_time_s"].mean(),
            g["prev_bus_tt_s"].mean(), (g["has_prev_bus"] == 0).mean(), peak))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS + ["all"], default="all")
    ap.add_argument("--save", action="store_true",
                    help="write data/simulated/<scenario>.parquet (not for git)")
    args = ap.parse_args()
    params = load_params()
    base, n_windows = load_stream(params["window_size"])
    print("stream rows: %d, windows: %d x %d" % (len(base), n_windows, params["window_size"]))
    names = SCENARIOS if args.scenario == "all" else [args.scenario]
    for name in names:
        df, info = build_scenario(name, base, n_windows, params)
        summarize(df, info)
        if args.save:
            out = os.path.join(ROOT, "data", "simulated")
            os.makedirs(out, exist_ok=True)
            df.to_parquet(os.path.join(out, name + ".parquet"), index=False)


if __name__ == "__main__":
    main()
