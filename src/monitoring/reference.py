"""Build the drift reference snapshot from TRAINING data (days 1-61)."""
import json
import numpy as np
import pandas as pd
from src.monitoring.psi import make_bins, bin_proportions

# 0/1 flags that the feature spec lists as numeric; binning them by quantile collapses.
BINARY_FLAGS = ["is_peak_am", "is_peak_pm", "has_prev_bus"]


def numeric_ref(x, n_bins: int = 10) -> dict:
    """Quantile bins and summary stats for one numeric column."""
    arr = np.asarray(x, dtype=float)
    missing = float(np.isnan(arr).mean())
    arr = arr[~np.isnan(arr)]
    edges = make_bins(arr, n_bins)
    return {
        "type": "numeric",
        "bin_edges": [float(e) for e in edges],
        "bin_props": [float(p) for p in bin_proportions(arr, edges)],
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "quantiles": {str(q): float(np.quantile(arr, q / 100)) for q in (5, 25, 50, 75, 95)},
        "missing_rate": missing,
    }


def categorical_ref(x) -> dict:
    """Category proportions for one categorical or binary column."""
    s = pd.Series(x)
    props = s.astype(str).value_counts(normalize=True)
    return {
        "type": "categorical",
        "props": {str(k): float(v) for k, v in props.items()},
        "missing_rate": float(s.isna().mean()),
    }


def build_reference(df, numeric, categorical, target, preds, n_bins=10) -> dict:
    """Reference snapshot: every feature, the target and the model predictions."""
    features = {}
    for c in numeric:
        if c in BINARY_FLAGS:
            features[c] = categorical_ref(df[c])
        else:
            features[c] = numeric_ref(df[c], n_bins)
    for c in categorical:
        features[c] = categorical_ref(df[c])
    return {
        "n_rows": int(len(df)),
        "n_bins": int(n_bins),
        "features": features,
        "target": {"name": target, **numeric_ref(df[target], n_bins)},
        "prediction": numeric_ref(preds, n_bins),
    }


def save_reference(ref: dict, path: str = "reports/drift_reference.json") -> None:
    with open(path, "w") as f:
        json.dump(ref, f, indent=2)
