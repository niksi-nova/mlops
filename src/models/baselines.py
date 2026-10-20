"""Baseline regressors for the Delhi bus travel-time model.

Kept in their own importable module (not src/models/train.py, which is run
as __main__ via `python -m src.models.train`) so that `joblib.dump`d models
can be `joblib.load`d back later - a class whose __module__ is "__main__"
at pickle time cannot be unpickled from a different entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class GlobalMedianRegressor(BaseEstimator, RegressorMixin):
    """Predicts the train median target value for every row, regardless of
    features — the simplest possible baseline."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GlobalMedianRegressor":
        self.median_ = float(np.median(np.asarray(y)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.median_)


class ColumnLookupRegressor(BaseEstimator, RegressorMixin):
    """Predicts an already-computed feature column directly (a lookup
    baseline, not fit on the target) - e.g. `typical_segment_hour_s` or
    `prev3_median_tt_s`, which are themselves historical/recent-conditions
    estimates built in the featurize stage."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnLookupRegressor":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)
