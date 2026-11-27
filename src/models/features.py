"""The ONE place that decides the model's input column order.

LightGBM was trained with the categorical features first, then the numeric
ones. feature_spec["features"] lists them in a different order, so never use
it to build model input. Everything (evaluate, drift, demo) calls this.
"""
import numpy as np
import pandas as pd

from src.models.evaluate import predict


def model_feature_columns(model, spec: dict) -> list[str]:
    """Column order the model was trained with."""
    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())
    return list(spec["categorical_features"]) + list(spec["numeric_features"])


def score(model_name: str, model, df: pd.DataFrame, spec: dict) -> np.ndarray:
    """Predictions in seconds, with the input columns in the correct order."""
    cols = model_feature_columns(model, spec)
    return predict(model_name, model, df, cols, spec["categorical_features"])
