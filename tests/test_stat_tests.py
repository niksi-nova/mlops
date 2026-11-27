import numpy as np
from src.monitoring.stat_tests import ks_numeric, chi2_categorical, is_drifted


def test_ks_same_distribution_not_drifted():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 20, 5000)
    b = rng.normal(100, 20, 5000)
    assert not is_drifted(ks_numeric(a, b)["p_value"], 0.01)


def test_ks_shifted_distribution_drifted():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 20, 5000)
    b = rng.normal(115, 20, 5000)
    assert is_drifted(ks_numeric(a, b)["p_value"], 0.01)


def test_chi2_categorical():
    ref = {"0": 0.9, "1": 0.1}
    same = [0] * 900 + [1] * 100
    shifted = [0] * 500 + [1] * 500
    assert not is_drifted(chi2_categorical(ref, same)["p_value"], 0.01)
    assert is_drifted(chi2_categorical(ref, shifted)["p_value"], 0.01)
