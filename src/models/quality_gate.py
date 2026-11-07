"""Quality gate stage for Delhi bus delay prediction models.

Evaluates whether the candidate model selected on validation and tested on
the held-out test split meets production deployment criteria defined in
params.yaml (quality_gate section):
  1. Minimum MAE improvement over the best baseline (e.g. >= 10.0%)
  2. Maximum test MAE threshold (e.g. <= 35.0 seconds)
  3. Maximum generalisation gap between validation and test (e.g. <= 10.0%)
  4. Minimum percentage of predictions accurate within 60s (e.g. >= 85.0%)

Reads:
  - reports/test_metrics.json
  - params.yaml (quality_gate section)

Writes:
  - reports/quality_gate.json

Usage:
    uv run python -m src.models.quality_gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import wandb

from src.models.tracking import ensure_wandb_login, lineage_config, load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_METRICS_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "quality_gate.json"


def load_test_metrics() -> dict[str, Any]:
    with open(TEST_METRICS_PATH) as f:
        return json.load(f)


def evaluate_quality_gate(test_metrics: dict[str, Any], gate_params: dict[str, Any]) -> dict[str, Any]:
    """Pure function evaluating metric thresholds against test results.

    Returns a dict with overall 'passed' status and detailed checks.
    """
    best_model = test_metrics["best_model"]
    best_metrics = test_metrics["models"][best_model]

    improvement = float(test_metrics["improvement_pct_over_best_baseline"])
    min_improvement = float(gate_params["min_improvement_pct_over_baseline"])

    test_mae = float(test_metrics["test_mae_best_model"])
    max_test_mae = float(gate_params["max_test_mae"])

    gen_gap = float(test_metrics["generalisation_gap_pct"])
    max_gen_gap = float(gate_params["max_generalisation_gap_pct"])

    within_60s = float(best_metrics["within_60s_pct"])
    min_within_60s = float(gate_params["min_within_60s_pct"])

    checks = {
        "baseline_improvement": {
            "metric": "improvement_pct_over_best_baseline",
            "value": improvement,
            "threshold": min_improvement,
            "operator": ">=",
            "passed": bool(improvement >= min_improvement),
        },
        "max_test_mae": {
            "metric": "test_mae_best_model",
            "value": test_mae,
            "threshold": max_test_mae,
            "operator": "<=",
            "passed": bool(test_mae <= max_test_mae),
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
    }

    all_passed = all(c["passed"] for c in checks.values())

    return {
        "passed": bool(all_passed),
        "best_model": best_model,
        "best_baseline": test_metrics.get("best_baseline", "unknown"),
        "checks": checks,
    }


def main() -> int:
    params = load_params()
    gate_params = params.get("quality_gate", {})
    if not gate_params:
        raise ValueError("Missing 'quality_gate' section in params.yaml")

    test_metrics = load_test_metrics()
    result = evaluate_quality_gate(test_metrics, gate_params)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # Log to W&B if available
    train_params = params.get("train", {})
    project = train_params.get("wandb_project", "delhi-bus-delay")

    try:
        ensure_wandb_login()
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

        artifact = wandb.Artifact(name="delhi-quality-gate", type="evaluation-report")
        artifact.add_file(str(REPORT_PATH))
        run.log_artifact(artifact)
        run.finish()
    except Exception as e:
        print(f"Warning: W&B logging for quality gate skipped: {e}")

    print("\nQuality Gate Status:")
    for name, c in result["checks"].items():
        symbol = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"  {symbol} {name}: {c['value']:.2f} {c['operator']} {c['threshold']:.2f} ({'PASSED' if c['passed'] else 'FAILED'})")

    if not result["passed"]:
        print("\nERROR: Quality gate checks failed. Model is not approved for production.", file=sys.stderr)
        return 1

    print("\nAll quality gate checks passed! Model is approved for production deployment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
