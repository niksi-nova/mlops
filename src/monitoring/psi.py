"""Population Stability Index (PSI): compares a reference distribution to a new one."""
import numpy as np

EPS = 1e-6  # avoids log(0) and division by zero for empty bins


def make_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Inner bin edges from reference quantiles (each bin holds ~equal share of training data)."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    return edges[1:-1]  # drop min and max so outer bins are open-ended


def bin_proportions(values: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    """Share of values falling in each bin. Outer bins are open-ended."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    full = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    counts, _ = np.histogram(v, bins=full)
    return counts / max(counts.sum(), 1)


def psi_from_props(ref_props: np.ndarray, new_props: np.ndarray) -> float:
    """PSI = sum((new - ref) * ln(new / ref)) over bins."""
    r = np.clip(np.asarray(ref_props, dtype=float), EPS, None)
    n = np.clip(np.asarray(new_props, dtype=float), EPS, None)
    return float(np.sum((n - r) * np.log(n / r)))


def psi_numeric(reference: np.ndarray, new: np.ndarray, n_bins: int = 10) -> float:
    """PSI between a reference sample and a new sample of a numeric feature."""
    edges = make_bins(reference, n_bins)
    return psi_from_props(bin_proportions(reference, edges), bin_proportions(new, edges))


def psi_categorical(ref_props: dict, new_values) -> float:
    """PSI for a categorical/binary feature. ref_props maps category -> training share."""
    import pandas as pd
    new_props = pd.Series(new_values).astype(str).value_counts(normalize=True)
    cats = sorted(set(ref_props) | set(new_props.index))
    r = np.array([ref_props.get(c, 0.0) for c in cats])
    n = np.array([float(new_props.get(c, 0.0)) for c in cats])
    return psi_from_props(r, n)


def psi_status(value: float, warn: float, alert: float) -> str:
    """Turn a PSI number into OK / WARNING / ALERT using thresholds from params.yaml."""
    if value > alert:
        return "ALERT"
    if value >= warn:
        return "WARNING"
    return "OK"
