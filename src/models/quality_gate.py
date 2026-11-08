"""Model Quality Gate for Delhi Bus Travel-Time Delay Prediction.

Evaluates whether the candidate model (lightgbm_l1) selected on validation
and tested on held-out test data satisfies production deployment criteria
defined in params.yaml (quality_gate section):
  1. Performance improvements over BOTH baselines (typical & recent buses >= 10%)
  2. Generalization gap between val and test (<= 10.0%)
  3. Absolute test MAE boundary (<= 35.0 seconds)
  4. Accuracy within 60 seconds (>= 85.0%)
  5. Slice checks across hours of the day (insufficient data handling for low-volume slices)
  6. Slice checks for cold-start (has_prev_bus = 0) and warm (has_prev_bus = 1)
  7. Prediction sanity (finite, positive, plausible range [1, 1800] seconds)
  8. Model artifact size (<= 100.0 MB)
  9. Single-row warm inference latency (median <= 50.0 ms)

Writes:
  - reports/quality_gate.json

Usage:
  uv run python -m src.models.quality_gate
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import wandb

from src.models.evaluate import predict
from src.models.tracking import DATASET_VERSION, ensure_wandb_login, lineage_config, load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_METRICS_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"
TRAIN_METRICS_PATH = PROJECT_ROOT / "reports" / "train_metrics.json"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_l1.joblib"
FEATURE_SPEC_PATH = PROJECT_ROOT / "data" / "processed" / "feature_spec.json"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "quality_gate.json"


def load_test_metrics() -> dict[str, Any]:
    with open(TEST_METRICS_PATH) as f:
        return json.load(f)


def measure_model_size_mb(model_path: Path = MODEL_PATH) -> tuple[int, float]:
    """Return model file size in bytes and megabytes (base-2)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_bytes, float(size_mb)


def measure_inference_latency_ms(
    model, feature_spec: dict[str, Any], test_df: pd.DataFrame, n_warmup: int = 10, n_iter: int = 30
) -> dict[str, float]:
    """Measure warm single-row inference latency using exact expected feature order."""
    expected_columns = feature_spec["categorical_features"] + feature_spec["numeric_features"]
    cats = feature_spec["categorical_features"]
    single_row = test_df.head(1).copy()

    # Warm-up phase
    for _ in range(n_warmup):
        _ = predict("lightgbm_l1", model, single_row, expected_columns, cats)

    # Measurement phase
    durations: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = predict("lightgbm_l1", model, single_row, expected_columns, cats)
        durations.append((time.perf_counter() - t0) * 1000.0)

    return {
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "min_ms": float(np.min(durations)),
    }


def verify_prediction_sanity(
    model, feature_spec: dict[str, Any], test_df: pd.DataFrame, min_s: float, max_s: float
) -> dict[str, Any]:
    """Verify sample predictions are non-null, finite, positive, and within domain limits."""
    expected_columns = feature_spec["categorical_features"] + feature_spec["numeric_features"]
    cats = feature_spec["categorical_features"]
    sample = test_df.head(200).copy()

    preds = predict("lightgbm_l1", model, sample, expected_columns, cats)

    no_nan = bool(not np.isnan(preds).any())
    no_inf = bool(not np.isinf(preds).any())
    all_positive = bool((preds > 0).all())
    within_bounds = bool((preds >= min_s).all() and (preds <= max_s).all())

    passed = no_nan and no_inf and all_positive and within_bounds
    return {
        "passed": passed,
        "sample_size": len(preds),
        "no_nan": no_nan,
        "no_inf": no_inf,
        "all_positive": all_positive,
        "within_bounds": within_bounds,
        "min_pred_s": float(np.min(preds)),
        "max_pred_s": float(np.max(preds)),
    }


def evaluate_quality_gate(
    test_metrics: dict[str, Any],
    gate_params: dict[str, Any],
    model_size_mb: float,
    latency_info: dict[str, float],
    sanity_info: dict[str, Any],
) -> dict[str, Any]:
    """Pure logic function evaluating all quality checks.

    Returns comprehensive quality gate report dictionary.
    """
    best_model = test_metrics["best_model"]
    best_metrics = test_metrics["models"][best_model]
    overall_test_mae = float(test_metrics["test_mae_best_model"])

    # 1. Baseline comparisons
    models_dict = test_metrics["models"]
    typical_mae = float(models_dict["typical_baseline"]["mae"])
    recent_mae = float(models_dict["recent_buses_baseline"]["mae"])

    impr_over_typical = float((typical_mae - overall_test_mae) / typical_mae * 100.0)
    impr_over_recent = float((recent_mae - overall_test_mae) / recent_mae * 100.0)
    min_impr = float(gate_params["min_improvement_pct_over_baseline"])

    # 2. Generalization gap & test boundaries
    gen_gap = float(test_metrics["generalisation_gap_pct"])
    max_gen_gap = float(gate_params["max_generalisation_gap_pct"])

    max_test_mae = float(gate_params["max_test_mae"])
    within_60s = float(best_metrics["within_60s_pct"])
    min_within_60s = float(gate_params["min_within_60s_pct"])

    # 3. Model size & latency thresholds
    max_model_size = float(gate_params.get("max_model_size_mb", 100.0))
    max_latency = float(gate_params.get("max_inference_latency_ms", 50.0))

    # Core checks dictionary
    checks = {
        "improvement_over_typical_baseline": {
            "metric": "improvement_pct_over_typical_baseline",
            "value": impr_over_typical,
            "threshold": min_impr,
            "operator": ">=",
            "passed": bool(impr_over_typical >= min_impr),
        },
        "improvement_over_recent_buses_baseline": {
            "metric": "improvement_pct_over_recent_buses_baseline",
            "value": impr_over_recent,
            "threshold": min_impr,
            "operator": ">=",
            "passed": bool(impr_over_recent >= min_impr),
        },
        "max_test_mae": {
            "metric": "test_mae_best_model",
            "value": overall_test_mae,
            "threshold": max_test_mae,
            "operator": "<=",
            "passed": bool(overall_test_mae <= max_test_mae),
        },
        "generalisation_gap": {
            "metric": "generalisation_gap_pct",
            "value": gen_gap,
            "threshold": max_gen_gap,
            "operator": "<=",
            "passed": bool(gen_gap <= max_gen_gap),
        },
        "within_60s_accuracy": {
            "metric": "within_60s_pct",
            "value": within_60s,
            "threshold": min_within_60s,
            "operator": ">=",
            "passed": bool(within_60s >= min_within_60s),
        },
        "model_artifact_size": {
            "metric": "model_size_mb",
            "value": model_size_mb,
            "threshold": max_model_size,
            "operator": "<=",
            "passed": bool(model_size_mb <= max_model_size),
        },
        "single_row_inference_latency": {
            "metric": "median_latency_ms",
            "value": latency_info["median_ms"],
            "threshold": max_latency,
            "operator": "<=",
            "passed": bool(latency_info["median_ms"] <= max_latency),
        },
        "prediction_sanity": {
            "metric": "sanity_checks_passed",
            "value": 1.0 if sanity_info["passed"] else 0.0,
            "threshold": 1.0,
            "operator": "==",
            "passed": bool(sanity_info["passed"]),
        },
    }

    # 4. Slice-based evaluations
    min_slice_samples = int(gate_params.get("min_slice_samples", 100))
    max_slice_ratio = float(gate_params.get("max_slice_mae_ratio", 1.8))

    error_analysis = test_metrics.get("error_analysis", {})
    warnings: list[str] = []
    failure_reasons: list[str] = []

    # Per-hour slice checks
    hourly_slices: list[dict[str, Any]] = []
    hourly_passed = True
    for item in error_analysis.get("mae_by_hour", []):
        hour = item["hour"]
        n = item["n"]
        mae = float(item["mae"])
        ratio = float(mae / overall_test_mae)

        if n < min_slice_samples:
            status = "INSUFFICIENT_DATA"
            warnings.append(
                f"Hour {hour} has only {n} samples (< {min_slice_samples} threshold); "
                f"marked as INSUFFICIENT_DATA and excluded from gate failure."
            )
        else:
            slice_pass = bool(ratio <= max_slice_ratio)
            status = "PASS" if slice_pass else "FAIL"
            if not slice_pass:
                hourly_passed = False
                failure_reasons.append(
                    f"Hour {hour} MAE ({mae:.2f}s) ratio to overall ({ratio:.2f}) exceeded {max_slice_ratio:.2f}"
                )

        hourly_slices.append({
            "hour": hour,
            "n": n,
            "mae": mae,
            "ratio_to_overall": ratio,
            "status": status,
        })

    # has_prev_bus slice checks
    prev_bus_slices: list[dict[str, Any]] = []
    prev_bus_passed = True
    for item in error_analysis.get("mae_by_has_prev_bus", []):
        flag = item["has_prev_bus"]
        n = item["n"]
        mae = float(item["mae"])
        ratio = float(mae / overall_test_mae)

        if n < min_slice_samples:
            status = "INSUFFICIENT_DATA"
            warnings.append(
                f"has_prev_bus={flag} has only {n} samples (< {min_slice_samples} threshold); marked as INSUFFICIENT_DATA."
            )
        else:
            slice_pass = bool(ratio <= max_slice_ratio)
            status = "PASS" if slice_pass else "FAIL"
            if not slice_pass:
                prev_bus_passed = False
                failure_reasons.append(
                    f"has_prev_bus={flag} MAE ({mae:.2f}s) ratio ({ratio:.2f}) exceeded {max_slice_ratio:.2f}"
                )

        prev_bus_slices.append({
            "has_prev_bus": flag,
            "n": n,
            "mae": mae,
            "ratio_to_overall": ratio,
            "status": status,
        })

    # Collect any core failures
    for name, c in checks.items():
        if not c["passed"]:
            failure_reasons.append(
                f"Check '{name}' failed: value {c['value']:.2f} {c['operator']} {c['threshold']:.2f}"
            )

    all_core_passed = all(c["passed"] for c in checks.values())
    overall_passed = bool(all_core_passed and hourly_passed and prev_bus_passed)

    return {
        "passed": overall_passed,
        "model_id": best_model,
        "best_baseline": test_metrics.get("best_baseline", "recent_buses_baseline"),
        "lineage": lineage_config(),
        "dataset_version": DATASET_VERSION,
        "evaluation_metrics": {
            "test_mae": overall_test_mae,
            "val_mae": float(test_metrics["val_mae_best_model"]),
            "generalisation_gap_pct": gen_gap,
            "within_60s_pct": within_60s,
            "improvement_pct_over_typical": impr_over_typical,
            "improvement_pct_over_recent_buses": impr_over_recent,
        },
        "threshold_config": gate_params,
        "checks": checks,
        "hourly_slices": hourly_slices,
        "has_prev_bus_slices": prev_bus_slices,
        "model_artifact": {
            "file": MODEL_PATH.name,
            "size_mb": model_size_mb,
            "max_model_size_mb": max_model_size,
            "passed": bool(model_size_mb <= max_model_size),
        },
        "inference_latency": {
            **latency_info,
            "threshold_ms": max_latency,
            "passed": bool(latency_info["median_ms"] <= max_latency),
            "measurement_method": "30-iteration warm single-row inference timing via time.perf_counter()",
        },
        "prediction_sanity": sanity_info,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
    }


def main() -> int:
    params = load_params()
    gate_params = params.get("quality_gate", {})
    if not gate_params:
        raise ValueError("Missing 'quality_gate' section in params.yaml")

    test_metrics = load_test_metrics()
    _, model_size_mb = measure_model_size_mb()

    # Load model and specs for live latency and sanity validation
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_SPEC_PATH) as f:
        feature_spec = json.load(f)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    min_s = float(params.get("segments", {}).get("min_segment_seconds", 1.0))
    max_s = float(params.get("segments", {}).get("max_segment_seconds", 1800.0))

    sanity_info = verify_prediction_sanity(model, feature_spec, test_df, min_s, max_s)
    latency_info = measure_inference_latency_ms(model, feature_spec, test_df)

    result = evaluate_quality_gate(test_metrics, gate_params, model_size_mb, latency_info, sanity_info)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # Log to W&B if available
    train_params = params.get("train", {})
    project = train_params.get("wandb_project", "delhi-bus-delay")

    try:
        logged_in = ensure_wandb_login()
        if logged_in:
            run = wandb.init(
                project=project,
                name="quality-gate",
                job_type="quality-gate",
                config={**gate_params, **lineage_config()},
            )
            run.summary["quality_gate_passed"] = result["passed"]
            for name, check in result["checks"].items():
                run.summary[f"gate_{name}_passed"] = check["passed"]
                run.summary[f"gate_{name}_value"] = check["value"]
                run.summary[f"gate_{name}_threshold"] = check["threshold"]

            run.summary["inference_latency_median_ms"] = latency_info["median_ms"]
            run.summary["model_size_mb"] = model_size_mb

            artifact = wandb.Artifact(name="delhi-quality-gate", type="evaluation-report")
            artifact.add_file(str(REPORT_PATH))
            run.log_artifact(artifact)
            run.finish()
        else:
            print("Notice: W&B login not available; quality gate report saved locally only.")
    except Exception as e:
        print(f"Notice: W&B logging for quality gate skipped: {e}")

    print("\nQuality Gate Status:")
    for name, c in result["checks"].items():
        symbol = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"  {symbol} {name}: {c['value']:.2f} {c['operator']} {c['threshold']:.2f} ({'PASSED' if c['passed'] else 'FAILED'})")

    print(f"\nSlices evaluated: {len(result['hourly_slices'])} hourly, {len(result['has_prev_bus_slices'])} previous-bus.")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  [WARN] {w}")

    if not result["passed"]:
        print("\nERROR: Quality gate checks failed:", file=sys.stderr)
        for r in result["failure_reasons"]:
            print(f"  - {r}", file=sys.stderr)
        return 1

    print("\nAll quality gate checks passed! Model is approved for staging promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
