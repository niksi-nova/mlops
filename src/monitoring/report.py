"""Build helpers and validator for reports/drift_report.json."""
from __future__ import annotations

RANK = {"OK": 0, "WARNING": 1, "ALERT": 2}
STATUSES = set(RANK)

GROUPS = {
    "prev_bus_tt_s": "real-time", "prev3_median_tt_s": "real-time",
    "headway_s": "real-time", "n_buses_recent": "real-time",
    "has_prev_bus": "real-time", "own_prev_segment_tt_s": "real-time",
    "own_prev_segment_ratio": "real-time",
    "typical_segment_hour_s": "historical", "typical_segment_s": "historical",
    "te_vehicle": "historical",
    "hour": "time-of-day", "minute_of_day": "time-of-day",
    "hour_sin": "time-of-day", "hour_cos": "time-of-day",
    "is_peak_am": "time-of-day", "is_peak_pm": "time-of-day",
    "relative_weekday": "time-of-day",
    "from_node": "position", "nodes_spanned": "position",
    "route_position": "position", "stops_into_trip": "position",
    "elapsed_trip_s": "position", "route": "position",
    "direction": "position", "segment_id": "position",
}

THRESHOLD_KEYS = [
    "psi_warn", "psi_alert", "ks_p_value", "ks_stat_min",
    "mae_degradation_pct_warn", "mae_degradation_pct_alert",
    "retrain_consecutive_alert_windows",
]

REAL_ORIGIN = "real test days 77-92, unmodified"
SIM_ORIGIN = "real test days 77-92 with SIMULATED perturbation"
SCENARIO_META = {
    "clean_test": ("REAL (no perturbation)", False, REAL_ORIGIN, "none",
                   "Unmodified test days; the control, which must not alert."),
    "peak_traffic_surge": ("SIMULATED", True, SIM_ORIGIN, "covariate",
                           "Real-time features scaled up and the time-of-day mix shifted toward peak hours."),
    "road_works": ("SIMULATED", True, SIM_ORIGIN, "concept",
                   "A share of road segments gets slower, so past travel times no longer predict current ones."),
    "gps_outage": ("SIMULATED", True, SIM_ORIGIN, "data_quality",
                   "Previous-bus information goes missing for many trips, as if GPS feeds dropped out."),
    "gradual_drift": ("SIMULATED", True, SIM_ORIGIN, "gradual",
                      "A small shift that grows window by window after the onset."),
}

WINDOW_KEYS = ["window", "days", "n", "mae", "mae_degradation_pct", "max_psi",
               "drifted_features", "target", "prediction", "has_prev_bus_share",
               "status", "features"]
FEATURE_KEYS = ["feature", "group", "type", "psi", "ks_stat", "ks_p_value",
                "chi2_p_value", "status"]
SCENARIO_KEYS = ["name", "label", "induced", "data_origin", "drift_type",
                 "description", "perturbation", "status", "first_alert_window",
                 "retrain_recommended", "windows", "top_drifted_features"]
TOP_KEYS = ["generated_at", "simulated", "disclaimer", "reference", "thresholds",
            "scenarios", "summary"]
SUMMARY_KEYS = ["n_scenarios", "n_alert", "n_warning", "n_ok",
                "false_positive_check", "retrain_recommended"]
REFERENCE_KEYS = ["dataset_version", "split", "n_rows", "model_id", "model_test_mae_s"]


def r(x, nd=6):
    return None if x is None else round(float(x), nd)


def read_thresholds(params: dict) -> dict:
    """Thresholds come from params.yaml only. A missing key raises KeyError."""
    return {k: params[k] for k in THRESHOLD_KEYS}


def worst_status(statuses) -> str:
    return max(statuses, key=lambda s: RANK[s], default="OK")


def first_alert_window(windows):
    for w in windows:
        if w["status"] == "ALERT":
            return w["window"]
    return None


def top_drifted(windows, k=5):
    if not windows:
        return []
    worst = max(windows, key=lambda w: (RANK[w["status"]], w["max_psi"]))
    peak = {}
    for w in windows:
        for f in w["features"]:
            if f["feature"] not in peak or f["psi"] > peak[f["feature"]][0]:
                peak[f["feature"]] = (f["psi"], w["window"])
    rows = sorted((f for f in worst["features"] if not f.get("alerting_excluded")), key=lambda f: -f["psi"])[:k]
    return [{"feature": f["feature"], "psi": f["psi"], "ks_stat": f["ks_stat"],
             "peak_window": peak[f["feature"]][1]} for f in rows]


def build_summary(scenarios) -> dict:
    sts = [s["status"] for s in scenarios]
    clean = [s for s in scenarios if s["name"] == "clean_test"]
    fp_ok = bool(clean) and clean[0]["status"] == "OK" and not clean[0]["retrain_recommended"]
    return {
        "n_scenarios": len(scenarios),
        "n_alert": sts.count("ALERT"),
        "n_warning": sts.count("WARNING"),
        "n_ok": sts.count("OK"),
        "false_positive_check": "PASS" if fp_ok else "FAIL",
        "retrain_recommended": any(s["retrain_recommended"] for s in scenarios),
    }


def _need(obj, keys, where, errs):
    if not isinstance(obj, dict):
        errs.append("%s: not an object" % where)
        return False
    for k in keys:
        if k not in obj:
            errs.append("%s: missing key '%s'" % (where, k))
    return all(k in obj for k in keys)


def validate_report(rep: dict, n_features: int = 25) -> None:
    """Raise ValueError listing every problem found."""
    errs = []
    if _need(rep, TOP_KEYS, "report", errs):
        _need(rep["reference"], REFERENCE_KEYS, "reference", errs)
        _need(rep["thresholds"], THRESHOLD_KEYS, "thresholds", errs)
        _need(rep["summary"], SUMMARY_KEYS, "summary", errs)
        for s in rep["scenarios"]:
            sn = "scenario %s" % s.get("name", "?")
            if not _need(s, SCENARIO_KEYS, sn, errs):
                continue
            if s["status"] not in STATUSES:
                errs.append("%s: bad status %r" % (sn, s["status"]))
            for w in s["windows"]:
                wn = "%s/%s" % (sn, w.get("window", "?"))
                if not _need(w, WINDOW_KEYS, wn, errs):
                    continue
                _need(w["target"], ["mean_s", "psi"], wn + " target", errs)
                _need(w["prediction"], ["mean_s", "psi"], wn + " prediction", errs)
                if w["status"] not in STATUSES:
                    errs.append("%s: bad status %r" % (wn, w["status"]))
                if len(w["features"]) != n_features:
                    errs.append("%s: %d features, expected %d" % (wn, len(w["features"]), n_features))
                for f in w["features"]:
                    if not _need(f, FEATURE_KEYS, "%s/%s" % (wn, f.get("feature", "?")), errs):
                        continue
                    if f["status"] not in STATUSES:
                        errs.append("%s/%s: bad status %r" % (wn, f["feature"], f["status"]))
                    if f["type"] == "numeric" and f["chi2_p_value"] is not None:
                        errs.append("%s/%s: numeric needs chi2_p_value null" % (wn, f["feature"]))
                    if f["type"] == "categorical" and f["feature"] != "relative_weekday" and (f["ks_stat"] is not None or f["ks_p_value"] is not None):
                        errs.append("%s/%s: categorical needs ks fields null" % (wn, f["feature"]))
    if errs:
        raise ValueError("drift_report invalid:\n" + "\n".join(errs[:25]))
