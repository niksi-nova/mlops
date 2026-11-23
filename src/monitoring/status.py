"""Drift status rules (Anika's schema). All thresholds are inclusive (>=)."""
from __future__ import annotations

OK = "OK"
WARNING = "WARNING"
ALERT = "ALERT"


def feature_flagged(
    psi: float,
    psi_alert: float,
    *,
    ks_p: float | None = None,
    ks_stat: float | None = None,
    ks_p_value: float = 0.01,
    ks_stat_min: float = 0.1,
) -> bool:
    """Flag on PSI >= psi_alert, or on KS p-value plus effect size.

    Categorical features pass no KS values, so they are flagged on PSI only.
    """
    if psi >= psi_alert:
        return True
    if ks_p is None or ks_stat is None:
        return False
    return ks_p < ks_p_value and ks_stat >= ks_stat_min


def feature_status(psi: float, psi_warn: float, flagged: bool) -> str:
    """ALERT if flagged, WARNING if PSI >= psi_warn, otherwise OK."""
    if flagged:
        return ALERT
    if psi >= psi_warn:
        return WARNING
    return OK


def mae_status(degradation_pct: float, warn_pct: float, alert_pct: float) -> str:
    """ALERT at >= alert_pct, WARNING at >= warn_pct, otherwise OK."""
    if degradation_pct >= alert_pct:
        return ALERT
    if degradation_pct >= warn_pct:
        return WARNING
    return OK


def should_retrain(window_statuses: list[str], consecutive: int) -> bool:
    """True if the most recent `consecutive` windows are all ALERT."""
    if consecutive < 1 or len(window_statuses) < consecutive:
        return False
    return all(s == ALERT for s in window_statuses[-consecutive:])

def window_status(
    max_psi: float,
    degradation_pct: float,
    target_psi: float,
    prediction_psi: float,
    *,
    psi_warn: float,
    psi_alert: float,
    mae_warn: float,
    mae_alert: float,
) -> str:
    """Window ALERT/WARNING/OK using Anika's rules (all inclusive)."""
    if (
        max_psi >= psi_alert
        or degradation_pct >= mae_alert
        or target_psi >= psi_alert
        or prediction_psi >= psi_alert
    ):
        return ALERT
    if max_psi >= psi_warn or degradation_pct >= mae_warn:
        return WARNING
    return OK


def has_consecutive_alerts(window_statuses: list[str], consecutive: int) -> bool:
    """True if ALERT appears in `consecutive` windows in a row, anywhere."""
    if consecutive < 1:
        return False
    run = 0
    for s in window_statuses:
        run = run + 1 if s == ALERT else 0
        if run >= consecutive:
            return True
    return False