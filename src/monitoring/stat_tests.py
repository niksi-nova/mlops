"""KS and chi-square drift tests."""
import numpy as np
import pandas as pd
from scipy import stats


def ks_numeric(reference: np.ndarray, new: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov test. Returns the KS statistic and p-value."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(new, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    res = stats.ks_2samp(ref, cur)
    return {"ks_stat": float(res.statistic), "p_value": float(res.pvalue)}


def chi2_categorical(ref_props: dict, new_values) -> dict:
    """Chi-square test of new category counts against training proportions."""
    new = pd.Series(new_values).astype(str)
    n = len(new)
    cats = sorted(set(ref_props) | set(new.unique()))
    observed = np.array([(new == c).sum() for c in cats], dtype=float)
    expected = np.array([ref_props.get(c, 0.0) for c in cats], dtype=float) * n
    expected = np.clip(expected, 1e-6, None)
    expected = expected * (observed.sum() / expected.sum())  # totals must match
    res = stats.chisquare(observed, expected)
    return {"chi2_stat": float(res.statistic), "p_value": float(res.pvalue)}


def is_drifted(p_value: float, p_threshold: float) -> bool:
    """Drift if the p-value is below the threshold from params.yaml."""
    return p_value < p_threshold
