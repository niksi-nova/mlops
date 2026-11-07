"""Unit tests for src/models/quality_gate.py.

Tests quality gate logic against synthetic metric dictionaries; no external I/O
or network calls required.
"""

from src.models.quality_gate import evaluate_quality_gate


def _dummy_test_metrics(
    typical_mae: float = 41.29,
    recent_mae: float = 38.22,
    test_mae: float = 30.60,
    val_mae: float = 29.33,
    gen_gap: float = 4.32,
    within_60s: float = 89.35,
    hourly_slices: list[dict] | None = None,
    prev_bus_slices: list[dict] | None = None,
):
    if hourly_slices is None:
        hourly_slices = [
            {"hour": 4, "mae": 73.7, "n": 1},      # Insufficient data
            {"hour": 8, "mae": 32.0, "n": 6000},   # Normal pass
            {"hour": 18, "mae": 43.1, "n": 8000},  # Ratio ~1.41 <= 1.8 pass
        ]
    if prev_bus_slices is None:
        prev_bus_slices = [
            {"has_prev_bus": 0, "mae": 51.2, "n": 5788},   # Ratio ~1.67 <= 1.8 pass
            {"has_prev_bus": 1, "mae": 29.8, "n": 147917}, # Ratio ~0.97 <= 1.8 pass
        ]

    return {
        "best_model": "lightgbm_l1",
        "best_baseline": "recent_buses_baseline",
        "test_mae_best_model": test_mae,
        "val_mae_best_model": val_mae,
        "generalisation_gap_pct": gen_gap,
        "models": {
            "lightgbm_l1": {"mae": test_mae, "within_60s_pct": within_60s},
            "typical_baseline": {"mae": typical_mae},
            "recent_buses_baseline": {"mae": recent_mae},
        },
        "error_analysis": {
            "mae_by_hour": hourly_slices,
            "mae_by_has_prev_bus": prev_bus_slices,
        },
    }


def _dummy_gate_params():
    return {
        "min_improvement_pct_over_baseline": 10.0,
        "max_test_mae": 35.0,
        "max_generalisation_gap_pct": 10.0,
        "min_within_60s_pct": 85.0,
        "max_model_size_mb": 100.0,
        "max_inference_latency_ms": 50.0,
        "min_slice_samples": 100,
        "max_slice_mae_ratio": 1.8,
    }


def _dummy_latency():
    return {"median_ms": 16.25, "p95_ms": 17.21, "min_ms": 15.54}


def _dummy_sanity(passed: bool = True):
    return {"passed": passed, "no_nan": True, "no_inf": True, "all_positive": True, "within_bounds": True}


def test_evaluate_quality_gate_all_pass():
    result = evaluate_quality_gate(
        _dummy_test_metrics(), _dummy_gate_params(), model_size_mb=66.8, latency_info=_dummy_latency(), sanity_info=_dummy_sanity()
    )
    assert result["passed"] is True
    assert result["model_id"] == "lightgbm_l1"
    for check in result["checks"].values():
        assert check["passed"] is True

    # Confirm hour 4 was handled as INSUFFICIENT_DATA and didn't fail gate
    h4 = next(h for h in result["hourly_slices"] if h["hour"] == 4)
    assert h4["status"] == "INSUFFICIENT_DATA"


def test_evaluate_quality_gate_fails_on_high_mae():
    # test_mae = 40.0 exceeds threshold 35.0
    metrics = _dummy_test_metrics(test_mae=40.0)
    result = evaluate_quality_gate(
        metrics, _dummy_gate_params(), model_size_mb=66.8, latency_info=_dummy_latency(), sanity_info=_dummy_sanity()
    )
    assert result["passed"] is False
    assert result["checks"]["max_test_mae"]["passed"] is False


def test_evaluate_quality_gate_fails_on_weak_baseline_improvement():
    # Model barely beats recent buses (38.0 vs 38.22 = 0.5% improvement < 10%)
    metrics = _dummy_test_metrics(recent_mae=38.22, test_mae=38.0)
    result = evaluate_quality_gate(
        metrics, _dummy_gate_params(), model_size_mb=66.8, latency_info=_dummy_latency(), sanity_info=_dummy_sanity()
    )
    assert result["passed"] is False
    assert result["checks"]["improvement_over_recent_buses_baseline"]["passed"] is False


def test_evaluate_quality_gate_fails_on_high_generalisation_gap():
    # gen_gap = 15.0% exceeds threshold 10.0%
    metrics = _dummy_test_metrics(gen_gap=15.0)
    result = evaluate_quality_gate(
        metrics, _dummy_gate_params(), model_size_mb=66.8, latency_info=_dummy_latency(), sanity_info=_dummy_sanity()
    )
    assert result["passed"] is False
    assert result["checks"]["generalisation_gap"]["passed"] is False


def test_evaluate_quality_gate_fails_on_low_within_60s():
    metrics = _dummy_test_metrics(within_60s=80.0)
    result = evaluate_quality_gate(
        metrics, _dummy_gate_params(), model_size_mb=66.8, latency_info=_dummy_latency(), sanity_info=_dummy_sanity()
    )
    assert result["passed"] is False
    assert result["checks"]["within_60s_accuracy"]["passed"] is False


def test_evaluate_quality_gate_fails_on_oversized_model():
    result = evaluate_quality_gate(
        _dummy_test_metrics(), _dummy_gate_params(), model_size_mb=150.0, latency_info=_dummy_latency(), sanity_info=_dummy_sanity()
    )
    assert result["passed"] is False
    assert result["checks"]["model_artifact_size"]["passed"] is False


def test_evaluate_quality_gate_fails_on_high_latency():
    latency = {"median_ms": 65.0, "p95_ms": 75.0, "min_ms": 60.0}
    result = evaluate_quality_gate(
        _dummy_test_metrics(), _dummy_gate_params(), model_size_mb=66.8, latency_info=latency, sanity_info=_dummy_sanity()
    )
    assert result["passed"] is False
    assert result["checks"]["single_row_inference_latency"]["passed"] is False


def test_evaluate_quality_gate_fails_on_bad_slice_ratio():
    bad_hour = [
        {"hour": 18, "mae": 75.0, "n": 5000},  # ratio 75 / 30.6 = 2.45 > 1.8
    ]
    metrics = _dummy_test_metrics(hourly_slices=bad_hour)
    result = evaluate_quality_gate(
        metrics, _dummy_gate_params(), model_size_mb=66.8, latency_info=_dummy_latency(), sanity_info=_dummy_sanity()
    )
    assert result["passed"] is False
    assert any("Hour 18" in r for r in result["failure_reasons"])


def test_evaluate_quality_gate_fails_on_sanity():
    result = evaluate_quality_gate(
        _dummy_test_metrics(), _dummy_gate_params(), model_size_mb=66.8, latency_info=_dummy_latency(), sanity_info=_dummy_sanity(passed=False)
    )
    assert result["passed"] is False
    assert result["checks"]["prediction_sanity"]["passed"] is False
