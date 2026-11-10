"""Build the small JSON data files the demo website reads.

Reads only repo artifacts (reports/*.json, params.yaml, dvc.yaml,
data/processed/feature_spec.json, data/processed/*.parquet, and the
trained model) and writes deterministic, size-bounded JSON to
src/api/static/data/. Never embeds vehicle identifiers or raw dataset rows
beyond the presets/trip examples/scatter sample explicitly called for.

This script needs training-time dependencies (pandas, lightgbm, joblib)
that are NOT in the slim Docker image - it is not shipped; only its JSON
output is.

Usage:
    uv run python -m scripts.build_site_data
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = PROJECT_ROOT / "src" / "api" / "static" / "data"
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

RNG = np.random.RandomState(42)


def _git_short_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _round(x, n=3):
    if isinstance(x, float):
        return round(x, n)
    return x


def _clean_row(row: dict, feature_cols: list[str]) -> dict:
    """Keep only the 25 model features (+ optionally travel_time_s), drop
    anything else (trip_id, day_index, vehicle, delay_s, ...)."""
    out = {k: row[k] for k in feature_cols}
    for k, v in out.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
    return out


def build_meta(feature_spec: dict) -> dict:
    return {
        "snapshot_built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        "git_commit": _git_short_sha(),
        "dataset_version": feature_spec.get("params", {}).get("dataset_version", "delhi-data-v2"),
        "n_features": len(feature_spec["features"]),
        "categorical_features": feature_spec["categorical_features"],
        "numeric_features": feature_spec["numeric_features"],
    }


# Structural/positional features: constant for a given segment, don't vary
# meaningfully by hour - storing them once per segment instead of per-hour
# keeps routes.json from ballooning.
STATIC_LOOKUP_COLS = [
    "route_position", "nodes_spanned", "stops_into_trip", "elapsed_trip_s",
    "typical_segment_s", "te_vehicle",
]
# Genuinely hour-varying: real-time conditions and the hour-specific typical.
HOURLY_LOOKUP_COLS = [
    "prev_bus_tt_s", "prev3_median_tt_s", "headway_s", "n_buses_recent",
    "has_prev_bus", "own_prev_segment_tt_s", "own_prev_segment_ratio",
    "typical_segment_hour_s",
]


def build_routes_and_lookup(test_df: pd.DataFrame, feature_spec: dict) -> tuple[dict, dict[str, dict]]:
    """Returns (routes_meta, {per-route-direction lookup file key: payload}).

    routes_meta is small (just stop counts/segment ids) and always loaded;
    each route+direction's full numeric lookup is a separate file, fetched
    only once the user actually picks that route+direction - keeps any
    single page load well under ~400KB of JSON."""
    routes_meta: dict[str, Any] = {}
    lookup_files: dict[str, dict] = {}
    for route in sorted(test_df["route"].unique()):
        routes_meta[route] = {}
        for direction in sorted(test_df[test_df.route == route]["direction"].unique()):
            sub = test_df[(test_df.route == route) & (test_df.direction == direction)]
            n_stops = int(sub["from_node"].max()) + 1
            segments_meta = {}
            segments_lookup = {}
            for from_node in sorted(sub["from_node"].unique()):
                seg_rows = sub[sub.from_node == from_node]
                segment_id = seg_rows["segment_id"].iloc[0]
                segments_meta[str(int(from_node))] = {"segment_id": segment_id, "n_rows": int(len(seg_rows))}

                static = {c: _round(float(seg_rows[c].median()), 2) for c in STATIC_LOOKUP_COLS}
                by_hour = {}
                for hour, hour_rows in seg_rows.groupby("hour"):
                    by_hour[str(int(hour))] = {c: _round(float(hour_rows[c].median()), 1) for c in HOURLY_LOOKUP_COLS}
                overall = {c: _round(float(seg_rows[c].median()), 1) for c in HOURLY_LOOKUP_COLS}
                segments_lookup[str(int(from_node))] = {"static": static, "overall": overall, "by_hour": by_hour}

            routes_meta[route][direction] = {"n_stops": n_stops, "n_segments": len(segments_meta), "segments": segments_meta}
            lookup_files[f"lookup_{route}_{direction}.json"] = segments_lookup
    return routes_meta, lookup_files


PRESET_SPECS = [
    ("am_peak_with_prev_bus", lambda d: (d.is_peak_am == 1) & (d.has_prev_bus == 1)),
    ("pm_peak", lambda d: d.is_peak_pm == 1),
    ("off_peak", lambda d: (d.is_peak_am == 0) & (d.is_peak_pm == 0)),
    ("cold_start", lambda d: d.has_prev_bus == 0),
    ("long_segment", lambda d: d.nodes_spanned > 1),
    ("short_segment", lambda d: d.nodes_spanned == 1),
]


def build_presets(test_df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    presets = []
    used_idx = set()
    for name, mask_fn in PRESET_SPECS:
        candidates = test_df[mask_fn(test_df)]
        candidates = candidates[~candidates.index.isin(used_idx)]
        if len(candidates) == 0:
            continue
        row = candidates.sample(1, random_state=42).iloc[0]
        used_idx.add(row.name)
        presets.append(
            {
                "name": name,
                "label": name.replace("_", " ").title(),
                "features": _clean_row(row.to_dict(), feature_cols),
                "actual_travel_time_s": _round(float(row["travel_time_s"])),
            }
        )
    return presets


def build_trip_examples(test_df: pd.DataFrame, feature_cols: list[str], n_trips: int = 4) -> list[dict]:
    """A few whole real trips (consecutive segments of one trip_id)."""
    trip_counts = test_df.groupby("trip_id").size().sort_values(ascending=False)
    chosen = trip_counts.head(40).sample(min(n_trips, len(trip_counts)), random_state=7).index.tolist()
    trips = []
    for trip_id in chosen:
        trip_df = test_df[test_df.trip_id == trip_id].sort_values("from_node")
        route = trip_df["route"].iloc[0]
        direction = trip_df["direction"].iloc[0]
        segments = [
            {**_clean_row(r.to_dict(), feature_cols), "actual_travel_time_s": _round(float(r["travel_time_s"]))}
            for _, r in trip_df.iterrows()
        ]
        trips.append(
            {
                "trip_id": str(trip_id),
                "route": route,
                "direction": direction,
                "n_segments": len(segments),
                "segments": segments,
            }
        )
    return trips


def build_data_overview(
    train_df: pd.DataFrame, cleaning: dict, segments: dict, validation: dict, featurize: dict
) -> dict:
    tt = train_df["travel_time_s"]
    hist_counts, hist_edges = np.histogram(np.log1p(tt.clip(upper=tt.quantile(0.99))), bins=30)

    by_hour = train_df.groupby("hour")["travel_time_s"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    by_hour.columns = ["median", "q25", "q75"]

    heatmap = (
        train_df.groupby(["hour", "relative_weekday"])["travel_time_s"]
        .median()
        .reset_index()
        .pivot(index="relative_weekday", columns="hour", values="travel_time_s")
        .round(1)
    )

    headway_bins = pd.qcut(train_df["headway_s"], 10, duplicates="drop")
    headway_vs_tt = train_df.groupby(headway_bins, observed=True)["travel_time_s"].median()
    headway_vs_tt_x = [float(iv.mid) for iv in headway_vs_tt.index]

    has_prev = train_df[train_df.has_prev_bus == 1]
    prev_bins = pd.qcut(has_prev["prev_bus_tt_s"], 10, duplicates="drop")
    prev_vs_own = has_prev.groupby(prev_bins, observed=True)["travel_time_s"].median()
    prev_vs_own_x = [float(iv.mid) for iv in prev_vs_own.index]

    return {
        "histogram": {
            "log1p_counts": hist_counts.tolist(),
            "log1p_bin_edges": [round(float(e), 3) for e in hist_edges],
        },
        "by_hour": {str(int(h)): {"median": round(r["median"], 1), "q25": round(r["q25"], 1), "q75": round(r["q75"], 1)} for h, r in by_hour.iterrows()},
        "heatmap_hours": [int(h) for h in heatmap.columns],
        "heatmap_hour_weekday": {
            str(int(wd)): {
                str(int(h)): (None if pd.isna(v) else round(float(v), 1))
                for h, v in zip(heatmap.columns, row)
            }
            for wd, row in heatmap.iterrows()
        },
        "headway_vs_traveltime": {"x_headway_s": headway_vs_tt_x, "y_median_tt_s": [round(float(v), 1) for v in headway_vs_tt.values]},
        "prev_vs_own": {"x_prev_tt_s": prev_vs_own_x, "y_median_tt_s": [round(float(v), 1) for v in prev_vs_own.values]},
        "splits": {
            "train": featurize["rows"]["train"],
            "val": featurize["rows"]["val"],
            "test": featurize["rows"]["test"],
            "day_range": featurize["day_range"],
        },
        "cleaning": cleaning,
        "segments_report": segments,
        "validation": validation,
    }


def build_models(train_metrics: dict, test_metrics: dict, quality_gate: dict, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    model_names = list(train_metrics["models"].keys())
    comparison = []
    for name in model_names:
        val = train_metrics["models"][name]
        test = test_metrics["models"].get(name, {})
        comparison.append(
            {
                "model": name,
                "val_mae": round(val["val_mae"], 2),
                "test_mae": round(test.get("mae", val["val_mae"]), 2),
                "drift_pct": round((test.get("mae", val["val_mae"]) - val["val_mae"]) / val["val_mae"] * 100, 1),
            }
        )

    best = test_metrics["best_model"]
    best_baseline = test_metrics["best_baseline"]
    typical_baseline_mae = test_metrics["models"]["typical_baseline"]["mae"]
    recent_baseline_mae = test_metrics["models"]["recent_buses_baseline"]["mae"]
    best_mae = test_metrics["models"][best]["mae"]

    model_path = MODELS_DIR / f"{best}.joblib"
    feature_importance = []
    if model_path.exists():
        model = joblib.load(model_path)
        try:
            gains = model.booster_.feature_importance(importance_type="gain")
            names = model.booster_.feature_name()
            groups = {
                "route": "position", "direction": "position", "segment_id": "position", "relative_weekday": "time",
                "from_node": "position", "nodes_spanned": "position", "route_position": "position",
                "stops_into_trip": "position", "elapsed_trip_s": "position",
                "hour": "time", "minute_of_day": "time", "hour_sin": "time", "hour_cos": "time",
                "is_peak_am": "time", "is_peak_pm": "time",
                "prev_bus_tt_s": "realtime", "prev3_median_tt_s": "realtime", "headway_s": "realtime",
                "n_buses_recent": "realtime", "has_prev_bus": "realtime",
                "own_prev_segment_tt_s": "realtime", "own_prev_segment_ratio": "realtime",
                "typical_segment_hour_s": "historical", "typical_segment_s": "historical", "te_vehicle": "historical",
            }
            order = np.argsort(gains)[::-1][:20]
            feature_importance = [
                {"feature": names[i], "gain": round(float(gains[i]), 1), "group": groups.get(names[i], "other")}
                for i in order
            ]
        except Exception:
            feature_importance = []

        sample = test_df.sample(min(2000, len(test_df)), random_state=42)
        from src.models.evaluate import predict as model_predict

        cats = ["route", "direction", "segment_id", "relative_weekday"]
        preds = model_predict(best, model, sample, feature_cols, cats)
        actual = sample["travel_time_s"].to_numpy()
        residuals = preds - actual
        res_counts, res_edges = np.histogram(residuals.clip(-200, 200), bins=30)
        scatter = {
            "actual": [round(float(v), 1) for v in actual],
            "predicted": [round(float(v), 1) for v in preds],
        }
        residual_hist = {"counts": res_counts.tolist(), "bin_edges": [round(float(e), 1) for e in res_edges]}
    else:
        scatter = {"actual": [], "predicted": []}
        residual_hist = {"counts": [], "bin_edges": []}

    return {
        "comparison": comparison,
        "best_model": best,
        "best_baseline": best_baseline,
        "success_target_pct": test_metrics["success_target_pct"],
        "improvement_vs_typical_pct": round((typical_baseline_mae - best_mae) / typical_baseline_mae * 100, 1),
        "improvement_vs_recent_pct": round((recent_baseline_mae - best_mae) / recent_baseline_mae * 100, 1),
        "feature_importance": feature_importance,
        "scatter_sample": scatter,
        "residual_histogram": residual_hist,
        "mae_by_hour": quality_gate["hourly_slices"],
        "mae_by_has_prev_bus": quality_gate["has_prev_bus_slices"],
        "test_mae_best_model": round(best_mae, 2),
        "generalisation_gap_pct": round(test_metrics["generalisation_gap_pct"], 2),
    }


def _count_test_functions() -> int:
    import re

    count = 0
    for path in (PROJECT_ROOT / "tests").glob("*.py"):
        count += len(re.findall(r"^def test_", path.read_text(), flags=re.MULTILINE))
    return count


def build_pipeline(quality_gate: dict, model_registry: dict, approvals: list[dict], dvc_yaml: dict, test_count: int) -> dict:
    valid_approvals = [a for a in approvals if a.get("status") == "APPROVED"]
    latest_approval = valid_approvals[-1] if valid_approvals else None
    return {
        "stages": list(dvc_yaml.get("stages", {}).keys()),
        "quality_gate": {
            "passed": quality_gate["passed"],
            "checks": quality_gate["checks"],
        },
        "model_passport": {
            "alias": "production",
            "version": model_registry["active_aliases"].get("production"),
            "model_id": model_registry["model_id"],
            "git_commit": model_registry["metadata"]["git_commit"],
            "collection": model_registry["collection"],
            "workspace": model_registry["workspace"],
            "approved_by": latest_approval["approved_by"] if latest_approval else None,
            "approved_at": latest_approval["timestamp"] if latest_approval else None,
        },
        "test_count": test_count,
    }


def build_golden_time_features(peak_am: list[int], peak_pm: list[int]) -> list[dict]:
    """For all 24 hours, at minute 0 of that hour - a golden table for the
    JS/Python cross-check test (mirrors add_time_of_day_features exactly)."""
    rows = []
    for hour in range(24):
        seconds_of_day = hour * 3600
        minute_of_day = seconds_of_day // 60
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        is_peak_am = int(peak_am[0] <= hour <= peak_am[1])
        is_peak_pm = int(peak_pm[0] <= hour <= peak_pm[1])
        rows.append(
            {
                "hour": hour,
                "minute_of_day": int(minute_of_day),
                "hour_sin": round(float(hour_sin), 10),
                "hour_cos": round(float(hour_cos), 10),
                "is_peak_am": is_peak_am,
                "is_peak_pm": is_peak_pm,
            }
        )
    return rows


def main() -> int:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    feature_spec = _load_json(PROCESSED / "feature_spec.json")
    feature_cols = feature_spec["categorical_features"] + feature_spec["numeric_features"]
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text())
    dvc_yaml = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text())

    test_df = pd.read_parquet(PROCESSED / "test.parquet")
    train_df = pd.read_parquet(PROCESSED / "train.parquet")

    train_metrics = _load_json(REPORTS / "train_metrics.json")
    test_metrics = _load_json(REPORTS / "test_metrics.json")
    quality_gate = _load_json(REPORTS / "quality_gate.json")
    model_registry = _load_json(REPORTS / "model_registry.json")
    approvals = _load_json(REPORTS / "production_approvals.json")
    cleaning = _load_json(REPORTS / "cleaning_report.json")
    segments_report = _load_json(REPORTS / "segments_report.json")
    validation = _load_json(REPORTS / "validation_report.json")
    featurize_report = _load_json(REPORTS / "featurize_report.json")

    routes_meta, lookup_files = build_routes_and_lookup(test_df, feature_spec)

    outputs = {
        "meta.json": build_meta(feature_spec),
        "routes.json": routes_meta,
        **lookup_files,
        "presets.json": build_presets(test_df, feature_cols),
        "trips.json": build_trip_examples(test_df, feature_cols),
        "data_overview.json": build_data_overview(train_df, cleaning, segments_report, validation, featurize_report),
        "models.json": build_models(train_metrics, test_metrics, quality_gate, test_df, feature_cols),
        "pipeline.json": build_pipeline(
            quality_gate, model_registry, approvals, dvc_yaml, test_count=_count_test_functions()
        ),
        "golden_time_features.json": build_golden_time_features(
            params["featurize"]["peak_am"], params["featurize"]["peak_pm"]
        ),
    }

    total_bytes = 0
    for filename, payload in outputs.items():
        path = DATA_OUT / filename
        text = json.dumps(payload, indent=None, separators=(",", ":"))
        path.write_text(text)
        size = len(text.encode())
        total_bytes += size
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({size / 1024:.1f} KB)")

    print(f"total: {total_bytes / 1024:.1f} KB ({total_bytes / 1024 / 1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
