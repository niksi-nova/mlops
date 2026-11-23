"""Run every drift scenario and write reports/drift_report.json."""
import argparse
import datetime
import importlib.util
import json
import os
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.evaluate import MODELS_DIR, load_feature_spec, load_model_index, load_split  # noqa: E402
from src.models.features import score  # noqa: E402
from src.monitoring import report as rp  # noqa: E402
from src.monitoring import status as st  # noqa: E402
from src.monitoring.performance import BASELINE_TEST_MAE, mae, mae_degradation_pct  # noqa: E402
from src.monitoring.psi import bin_proportions, psi_categorical, psi_from_props  # noqa: E402
from src.monitoring.stat_tests import chi2_categorical, ks_numeric  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "simulate_drift", os.path.join(ROOT, "scripts", "simulate_drift.py"))
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

MODEL_NAME = "lightgbm_l1"
REF_SAMPLE_ROWS = 50000
DISCLAIMER = ("Scenarios marked induced=true are SIMULATED perturbations of real "
              "test-day data (days 77-92). They are not real production drift.")


def load_params():
    return sim.load_params()


def build_context(params):
    spec = load_feature_spec()
    info = load_model_index()[MODEL_NAME]
    model = joblib.load(MODELS_DIR / info["file"])
    with open(os.path.join(ROOT, "reports", "drift_reference.json")) as f:
        ref = json.load(f)
    train = load_split("train").sample(n=REF_SAMPLE_ROWS, random_state=int(params["seed"]))
    return {"spec": spec, "model": model, "ref": ref, "ref_sample": train}


def psi_numeric_ref(fref, values):
    return psi_from_props(np.array(fref["bin_props"]),
                          bin_proportions(values, np.array(fref["bin_edges"])))


def feature_row(fname, fref, ctx, wdf, th):
    if fref["type"] == "numeric":
        v = wdf[fname].to_numpy()
        psi = psi_numeric_ref(fref, v)
        ks = ks_numeric(ctx["ref_sample"][fname].to_numpy(), v)
        flagged = st.feature_flagged(
            psi, th["psi_alert"], ks_p=ks["p_value"], ks_stat=ks["ks_stat"],
            ks_p_value=th["ks_p_value"], ks_stat_min=th["ks_stat_min"])
        ks_stat, ks_p, chi_p = ks["ks_stat"], ks["p_value"], None
    else:
        psi = psi_categorical(fref["props"], wdf[fname])
        chi_p = chi2_categorical(fref["props"], wdf[fname])["p_value"]
        flagged = st.feature_flagged(psi, th["psi_alert"])  # categorical: PSI only
        ks_stat = ks_p = None
    return {"feature": fname, "group": rp.GROUPS[fname], "type": fref["type"],
            "psi": rp.r(psi), "ks_stat": rp.r(ks_stat), "ks_p_value": rp.r(ks_p, 12),
            "chi2_p_value": rp.r(chi_p, 12),
            "status": st.feature_status(psi, th["psi_warn"], flagged)}


def run_scenario(name, base, n_windows, params, ctx, th):
    df, info = sim.build_scenario(name, base, n_windows, params)
    preds = score(MODEL_NAME, ctx["model"], df, ctx["spec"])
    y = df["travel_time_s"].to_numpy()
    ref = ctx["ref"]
    windows = []
    for w in range(n_windows):
        m = (df["window_id"] == w).to_numpy()
        wdf, wp, wy = df[m], preds[m], y[m]
        w_mae = mae(wy, wp)
        deg = mae_degradation_pct(w_mae, BASELINE_TEST_MAE)
        excluded = set(params.get("exclude_features", []))
        feats = [feature_row(f, fr, ctx, wdf, th) for f, fr in ref["features"].items() if f not in excluded]
        max_psi = max(f["psi"] for f in feats)
        t_psi = psi_numeric_ref(ref["target"], wy)
        p_psi = psi_numeric_ref(ref["prediction"], wp)
        status = st.window_status(
            max_psi, deg, t_psi, p_psi, psi_warn=th["psi_warn"], psi_alert=th["psi_alert"],
            mae_warn=th["mae_degradation_pct_warn"], mae_alert=th["mae_degradation_pct_alert"])
        d0, d1 = int(wdf["day_index"].min()), int(wdf["day_index"].max())
        windows.append({
            "window": "w%d" % (w + 1), "days": "%d-%d" % (d0, d1), "n": int(m.sum()),
            "mae": rp.r(w_mae, 4), "mae_degradation_pct": rp.r(deg, 3),
            "max_psi": rp.r(max_psi),
            "drifted_features": [f["feature"] for f in feats if f["status"] == "ALERT"],
            "target": {"mean_s": rp.r(wy.mean(), 3), "psi": rp.r(t_psi)},
            "prediction": {"mean_s": rp.r(wp.mean(), 3), "psi": rp.r(p_psi)},
            "has_prev_bus_share": rp.r(wdf["has_prev_bus"].mean(), 4),
            "status": status, "features": feats,
        })
    label, induced, origin, dtype, desc = rp.SCENARIO_META[name]
    consts = {k: v for k, v in vars(sim).items()
              if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)}
    statuses = [w["status"] for w in windows]
    return {
        "name": name, "label": label, "induced": induced, "data_origin": origin,
        "drift_type": dtype, "description": desc,
        "perturbation": None if not induced else
            {"type": name, "params": consts, "seed": info["seed"]},
        "status": rp.worst_status(statuses),
        "first_alert_window": rp.first_alert_window(windows),
        "retrain_recommended": st.has_consecutive_alerts(
            statuses, th["retrain_consecutive_alert_windows"]),
        "windows": windows, "top_drifted_features": rp.top_drifted(windows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "drift_report.json"))
    args = ap.parse_args()
    params = load_params()
    th = rp.read_thresholds(params)
    ctx = build_context(params)
    base, n_windows = sim.load_stream(params["window_days"])
    scenarios = []
    for name in sim.SCENARIOS:
        s = run_scenario(name, base, n_windows, params, ctx, th)
        scenarios.append(s)
        print("%-20s %-8s first_alert=%-5s retrain=%s" % (
            name, s["status"], s["first_alert_window"], s["retrain_recommended"]))
    ref = ctx["ref"]
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "simulated": True, "disclaimer": DISCLAIMER,
        "reference": {"dataset_version": ref["dataset_version"], "split": ref["split"],
                      "n_rows": ref["n_rows"], "model_id": MODEL_NAME,
                      "model_test_mae_s": BASELINE_TEST_MAE},
        "thresholds": th, "scenarios": scenarios, "summary": rp.build_summary(scenarios),
    }
    rp.validate_report(report, n_features=25 - len(params.get("exclude_features", [])))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("summary:", report["summary"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()