"""Performance drift: compare MAE on a window against the model's baseline (test) MAE."""
import numpy as np

BASELINE_TEST_MAE = 30.60  # seconds, final LightGBM L1 model on test days 77-92


def mae(y_true, y_pred) -> float:
    """Mean absolute error in seconds."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_degradation_pct(window_mae: float, baseline_mae: float = BASELINE_TEST_MAE) -> float:
    """How many percent worse the window MAE is than the baseline (negative = better)."""
    return float((window_mae - baseline_mae) / baseline_mae * 100.0)


def performance_status(degradation_pct: float, alert_pct: float) -> str:
    """ALERT if more than alert_pct worse; WARNING from half of it; otherwise OK."""
    if degradation_pct > alert_pct:
        return "ALERT"
    if degradation_pct > alert_pct / 2:
        return "WARNING"
    return "OK"
