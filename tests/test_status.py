from src.monitoring.status import (
    feature_flagged, feature_status, mae_status, should_retrain,
)


def test_psi_alert_is_inclusive():
    assert feature_flagged(0.25, 0.25) is True
    assert feature_flagged(0.2499, 0.25) is False


def test_ks_needs_p_value_and_effect_size():
    assert feature_flagged(0.0, 0.25, ks_p=0.001, ks_stat=0.1) is True
    assert feature_flagged(0.0, 0.25, ks_p=0.001, ks_stat=0.0999) is False
    assert feature_flagged(0.0, 0.25, ks_p=0.5, ks_stat=0.5) is False


def test_categorical_flagged_on_psi_only():
    assert feature_flagged(0.05, 0.25) is False


def test_feature_status_levels():
    assert feature_status(0.30, 0.1, True) == "ALERT"
    assert feature_status(0.10, 0.1, False) == "WARNING"
    assert feature_status(0.0999, 0.1, False) == "OK"


def test_mae_status_boundaries():
    assert mae_status(15.0, 7.5, 15.0) == "ALERT"
    assert mae_status(14.99, 7.5, 15.0) == "WARNING"
    assert mae_status(7.5, 7.5, 15.0) == "WARNING"
    assert mae_status(7.4999, 7.5, 15.0) == "OK"


def test_should_retrain_needs_consecutive_alerts():
    assert should_retrain(["OK", "ALERT", "ALERT"], 2) is True
    assert should_retrain(["ALERT", "OK", "ALERT"], 2) is False
    assert should_retrain(["ALERT"], 2) is False