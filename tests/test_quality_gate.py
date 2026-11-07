"""Unit tests for src/models/quality_gate.py.

Tests quality gate logic against synthetic metric dictionaries; no file I/O
or network calls required.
"""

from src.models.quality_gate import evaluate_quality_gate


def _dummy_test_metrics(
    improvement: float = 19.9,
    test_mae: float = 30.6,
    gen_gap: float = 4.3,
    within_60s: float = 89.3,
):
    return {
        "best_model": "lightgbm_l1",
        "best_baseline": "recent_buses_baseline",
        "improvement_pct_over_best_baseline": improvement,
        "test_mae_best_model": test_mae,
        "generalisation_gap_pct": gen_gap,
        "models": {
            "lightgbm_l1": {
                "mae": test_mae,
                "within_60s_pct": within_60s,
            }
        },
    }


def _dummy_gate_params():
    return {
        "min_improvement_pct_over_baseline": 10.0,
        "max_test_mae": 35.0,
        "max_generalisation_gap_pct": 10.0,
        "min_within_60s_pct": 85.0,
    }


def test_evaluate_quality_gate_all_pass():
    result = evaluate_quality_gate(_dummy_test_metrics(), _dummy_gate_params())
    assert result["passed"] is True
    assert result["best_model"] == "lightgbm_l1"
    for check in result["checks"].values():
        assert check["passed"] is True


def test_evaluate_quality_gate_fails_on_high_mae():
    # test_mae = 40.0 exceeds threshold 35.0
    metrics = _dummy_test_metrics(test_mae=40.0)
    result = evaluate_quality_gate(metrics, _dummy_gate_params())
    assert result["passed"] is False
    assert result["checks"]["max_test_mae"]["passed"] is False
    assert result["checks"]["baseline_improvement"]["passed"] is True


def test_evaluate_quality_gate_fails_on_low_improvement():
    # improvement = 5.0% is below threshold 10.0%
    metrics = _dummy_test_metrics(improvement=5.0)
    result = evaluate_quality_gate(metrics, _dummy_gate_params())
    assert result["passed"] is False
    assert result["checks"]["baseline_improvement"]["passed"] is False


def test_evaluate_quality_gate_fails_on_high_generalisation_gap():
    # gen_gap = 15.0% exceeds threshold 10.0%
    metrics = _dummy_test_metrics(gen_gap=15.0)
    result = evaluate_quality_gate(metrics, _dummy_gate_params())
    assert result["passed"] is False
    assert result["checks"]["generalisation_gap"]["passed"] is False


def test_evaluate_quality_gate_fails_on_low_within_60s():
    # within_60s = 80.0% is below threshold 85.0%
    metrics = _dummy_test_metrics(within_60s=80.0)
    result = evaluate_quality_gate(metrics, _dummy_gate_params())
    assert result["passed"] is False
    assert result["checks"]["within_60s_accuracy"]["passed"] is False

