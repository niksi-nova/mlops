import numpy as np
from src.monitoring.psi import psi_numeric, psi_categorical, psi_status


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    a = rng.normal(100, 20, 50000)
    b = rng.normal(100, 20, 50000)
    assert psi_numeric(a, b) < 0.1


def test_shifted_distribution_alerts():
    rng = np.random.default_rng(0)
    a = rng.normal(100, 20, 50000)
    b = rng.normal(140, 20, 50000)
    assert psi_numeric(a, b) > 0.25


def test_categorical_shift():
    ref = {"0": 0.9, "1": 0.1}
    same = [0] * 90 + [1] * 10
    shifted = [0] * 50 + [1] * 50
    assert psi_categorical(ref, same) < 0.1
    assert psi_categorical(ref, shifted) > 0.25


def test_status_levels():
    assert psi_status(0.05, 0.1, 0.25) == "OK"
    assert psi_status(0.15, 0.1, 0.25) == "WARNING"
    assert psi_status(0.30, 0.1, 0.25) == "ALERT"
