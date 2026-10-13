"""Train and compare baseline models for TTC bus delay prediction.

Reads:
  - data/processed/train.parquet
  - data/processed/val.parquet  (used only for evaluation / early stopping)
  - data/processed/feature_spec.json
  - params.yaml (train section)

Trains four models, each logged as its own MLflow run in the
`train.experiment_name` experiment: naive_baseline, ridge, random_forest,
lightgbm. All are evaluated on the validation split only (test.parquet is
never loaded here).

Writes:
  - models/best_model.joblib        (the run with the lowest val_mae)
  - reports/train_metrics.json      (every model's val metrics + the winner)

Usage:
    uv run python -m src.models.train
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.models.tracking import configure_tracking, run_tags

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SPEC_PATH = PROCESSED_DIR / "feature_spec.json"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "train_metrics.json"

TARGET = "min_delay"


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_feature_spec() -> dict:
    """Load data/processed/feature_spec.json."""
    with open(FEATURE_SPEC_PATH) as f:
        return json.load(f)


def load_split(name: str) -> pd.DataFrame:
    """Load one of train.parquet / val.parquet from data/processed."""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


# --------------------------------------------------------------------------
# naive_baseline
# --------------------------------------------------------------------------
class NaiveBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predicts the train mean min_delay for the row's (route, incident)
    pair, falling back to the incident mean, then the global mean, for
    pairs/incidents never seen in training."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBaselineRegressor":
        data = X[["route", "incident"]].copy()
        data["_y"] = np.asarray(y)
        self.group_means_ = data.groupby(["route", "incident"])["_y"].mean()
        self.incident_means_ = data.groupby("incident")["_y"].mean()
        self.global_mean_ = float(data["_y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        def lookup(row: pd.Series) -> float:
            key = (row["route"], row["incident"])
            if key in self.group_means_.index:
                return float(self.group_means_.loc[key])
            if row["incident"] in self.incident_means_.index:
                return float(self.incident_means_.loc[row["incident"]])
            return self.global_mean_

        return X[["route", "incident"]].apply(lookup, axis=1).to_numpy()


# --------------------------------------------------------------------------
# global_median
# --------------------------------------------------------------------------
class GlobalMedianRegressor(BaseEstimator, RegressorMixin):
    """Predicts the train median min_delay for every row, regardless of
    features — the simplest possible baseline."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GlobalMedianRegressor":
        self.median_ = float(np.median(np.asarray(y)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.median_)


# --------------------------------------------------------------------------
# metrics / plotting helpers
# --------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """mae/rmse/r2/median_ae for y_pred vs y_true, keyed as f"{prefix}_<metric>"."""
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def plot_feature_importance(names: list[str], importances: np.ndarray, title: str, out_path: Path) -> None:
    """Horizontal bar chart of the top-20 features by importance, saved as a PNG."""
    order = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="steelblue")
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def clip_and_backtransform(raw_pred: np.ndarray, log_target: bool) -> np.ndarray:
    """expm1 the prediction if it was trained in log1p space, then clip at 0."""
    pred = np.expm1(raw_pred) if log_target else raw_pred
    return np.clip(pred, 0, None)


# --------------------------------------------------------------------------
# model-specific training
# --------------------------------------------------------------------------
def train_naive_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[Any, np.ndarray, np.ndarray, float]:
    model = NaiveBaselineRegressor()
    start = time.perf_counter()
    model.fit(train_df, train_df[TARGET])
    elapsed = time.perf_counter() - start
    train_pred = model.predict(train_df)
    val_pred = model.predict(val_df)
    return model, train_pred, val_pred, elapsed


def train_global_median(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[Any, np.ndarray, np.ndarray, float]:
    model = GlobalMedianRegressor()
    start = time.perf_counter()
    model.fit(train_df, train_df[TARGET])
    elapsed = time.perf_counter() - start
    train_pred = model.predict(train_df)
    val_pred = model.predict(val_df)
    return model, train_pred, val_pred, elapsed


def train_ridge(
    train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_features: list[str],
    categorical_features: list[str], ridge_params: dict, log_target: bool,
) -> tuple[Pipeline, np.ndarray, np.ndarray, float]:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )
    model = Pipeline([("preprocess", preprocessor), ("model", Ridge(**ridge_params))])

    y_train = np.log1p(train_df[TARGET]) if log_target else train_df[TARGET]

    start = time.perf_counter()
    model.fit(train_df, y_train)
    elapsed = time.perf_counter() - start

    train_pred = clip_and_backtransform(model.predict(train_df), log_target)
    val_pred = clip_and_backtransform(model.predict(val_df), log_target)
    return model, train_pred, val_pred, elapsed


def train_random_forest(
    train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_features: list[str],
    categorical_features: list[str], rf_params: dict, log_target: bool, random_state: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray, float, list[str], np.ndarray]:
    preprocessor = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)),
        ]
    )

    y_train = np.log1p(train_df[TARGET]) if log_target else train_df[TARGET]

    start = time.perf_counter()
    model.fit(train_df, y_train)
    elapsed = time.perf_counter() - start

    train_pred = clip_and_backtransform(model.predict(train_df), log_target)
    val_pred = clip_and_backtransform(model.predict(val_df), log_target)

    feature_names = categorical_features + numeric_features
    importances = model.named_steps["model"].feature_importances_
    return model, train_pred, val_pred, elapsed, feature_names, importances


def train_lightgbm(
    train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_features: list[str],
    categorical_features: list[str], lgbm_params: dict, log_target: bool, random_state: int,
) -> tuple[lgb.LGBMRegressor, np.ndarray, np.ndarray, float, list[str], np.ndarray]:
    feature_columns = categorical_features + numeric_features

    def prep(df: pd.DataFrame, train_categories: dict[str, pd.Index] | None = None) -> pd.DataFrame:
        X = df[feature_columns].copy()
        for col in categorical_features:
            if train_categories is None:
                X[col] = X[col].astype("category")
            else:
                X[col] = pd.Categorical(X[col], categories=train_categories[col])
        return X

    X_train = prep(train_df)
    train_categories = {col: X_train[col].cat.categories for col in categorical_features}
    X_val = prep(val_df, train_categories)

    y_train = np.log1p(train_df[TARGET]) if log_target else train_df[TARGET]
    y_val = np.log1p(val_df[TARGET]) if log_target else val_df[TARGET]

    params = dict(lgbm_params)
    early_stopping_rounds = params.pop("early_stopping_rounds")

    model = lgb.LGBMRegressor(**params, random_state=random_state, n_jobs=-1, verbosity=-1)

    start = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    elapsed = time.perf_counter() - start

    train_pred = clip_and_backtransform(model.predict(X_train), log_target)
    val_pred = clip_and_backtransform(model.predict(X_val), log_target)

    return model, train_pred, val_pred, elapsed, feature_columns, model.feature_importances_


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    params = load_params()
    train_params = params["train"]
    feature_spec = load_feature_spec()
    numeric_features = feature_spec["numeric_features"]
    categorical_features = feature_spec["categorical_features"]
    log_target = train_params["log_target"]
    random_state = train_params["random_state"]

    train_df = load_split("train")
    val_df = load_split("val")

    configure_tracking()
    mlflow.set_experiment(train_params["experiment_name"])
    tags = run_tags()

    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- naive_baseline -----------------------------------------------
    with mlflow.start_run(run_name="naive_baseline"):
        mlflow.set_tags(tags)
        model, train_pred, val_pred, elapsed = train_naive_baseline(train_df, val_df)
        metrics = regression_metrics(val_df[TARGET], val_pred, "val")
        metrics["train_mae"] = float(mean_absolute_error(train_df[TARGET], train_pred))
        metrics["training_time_seconds"] = elapsed
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(FEATURE_SPEC_PATH))
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        results["naive_baseline"] = metrics
        fitted_models["naive_baseline"] = model
        print(f"naive_baseline: val_mae={metrics['val_mae']:.3f}")

    # --- global_median --------------------------------------------------
    with mlflow.start_run(run_name="global_median"):
        mlflow.set_tags(tags)
        model, train_pred, val_pred, elapsed = train_global_median(train_df, val_df)
        metrics = regression_metrics(val_df[TARGET], val_pred, "val")
        metrics["train_mae"] = float(mean_absolute_error(train_df[TARGET], train_pred))
        metrics["training_time_seconds"] = elapsed
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(FEATURE_SPEC_PATH))
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        results["global_median"] = metrics
        fitted_models["global_median"] = model
        print(f"global_median: val_mae={metrics['val_mae']:.3f}")

    # --- ridge -----------------------------------------------------------
    with mlflow.start_run(run_name="ridge"):
        mlflow.set_tags(tags)
        mlflow.log_params(train_params["ridge"])
        mlflow.log_param("log_target", log_target)
        model, train_pred, val_pred, elapsed = train_ridge(
            train_df, val_df, numeric_features, categorical_features, train_params["ridge"], log_target
        )
        metrics = regression_metrics(val_df[TARGET], val_pred, "val")
        metrics["train_mae"] = float(mean_absolute_error(train_df[TARGET], train_pred))
        metrics["training_time_seconds"] = elapsed
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(FEATURE_SPEC_PATH))
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        results["ridge"] = metrics
        fitted_models["ridge"] = model
        print(f"ridge: val_mae={metrics['val_mae']:.3f}")

    # --- random_forest -----------------------------------------------------
    with mlflow.start_run(run_name="random_forest"):
        mlflow.set_tags(tags)
        mlflow.log_params(train_params["random_forest"])
        mlflow.log_param("log_target", log_target)
        model, train_pred, val_pred, elapsed, feat_names, importances = train_random_forest(
            train_df, val_df, numeric_features, categorical_features,
            train_params["random_forest"], log_target, random_state,
        )
        metrics = regression_metrics(val_df[TARGET], val_pred, "val")
        metrics["train_mae"] = float(mean_absolute_error(train_df[TARGET], train_pred))
        metrics["training_time_seconds"] = elapsed
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(FEATURE_SPEC_PATH))
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

        fi_path = MODELS_DIR / "random_forest_feature_importance.png"
        plot_feature_importance(feat_names, importances, "Random forest — top 20 feature importances", fi_path)
        mlflow.log_artifact(str(fi_path))

        results["random_forest"] = metrics
        fitted_models["random_forest"] = model
        print(f"random_forest: val_mae={metrics['val_mae']:.3f}")

    # --- lightgbm ------------------------------------------------------
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tags(tags)
        mlflow.log_params(train_params["lightgbm"])
        mlflow.log_param("log_target", log_target)
        model, train_pred, val_pred, elapsed, feat_names, importances = train_lightgbm(
            train_df, val_df, numeric_features, categorical_features,
            train_params["lightgbm"], log_target, random_state,
        )
        metrics = regression_metrics(val_df[TARGET], val_pred, "val")
        metrics["train_mae"] = float(mean_absolute_error(train_df[TARGET], train_pred))
        metrics["training_time_seconds"] = elapsed
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(FEATURE_SPEC_PATH))
        mlflow.lightgbm.log_model(model, name="model")

        fi_path = MODELS_DIR / "lightgbm_feature_importance.png"
        plot_feature_importance(feat_names, importances, "LightGBM — top 20 feature importances", fi_path)
        mlflow.log_artifact(str(fi_path))

        results["lightgbm"] = metrics
        fitted_models["lightgbm"] = model
        print(f"lightgbm: val_mae={metrics['val_mae']:.3f}")

    # --- lightgbm_l1 (L1/MAE objective, trained on the raw target) -----
    with mlflow.start_run(run_name="lightgbm_l1"):
        mlflow.set_tags(tags)
        mlflow.log_params(train_params["lightgbm_l1"])
        mlflow.log_param("log_target", False)
        model, train_pred, val_pred, elapsed, feat_names, importances = train_lightgbm(
            train_df, val_df, numeric_features, categorical_features,
            train_params["lightgbm_l1"], False, random_state,
        )
        metrics = regression_metrics(val_df[TARGET], val_pred, "val")
        metrics["train_mae"] = float(mean_absolute_error(train_df[TARGET], train_pred))
        metrics["training_time_seconds"] = elapsed
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(FEATURE_SPEC_PATH))
        mlflow.lightgbm.log_model(model, name="model")

        fi_path = MODELS_DIR / "lightgbm_l1_feature_importance.png"
        plot_feature_importance(feat_names, importances, "LightGBM (L1) — top 20 feature importances", fi_path)
        mlflow.log_artifact(str(fi_path))

        results["lightgbm_l1"] = metrics
        fitted_models["lightgbm_l1"] = model
        print(f"lightgbm_l1: val_mae={metrics['val_mae']:.3f}")

    # --- pick the best model, write reports -----------------------------
    best_name = min(results, key=lambda name: results[name]["val_mae"])
    best_model = fitted_models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Best model: {best_name} -> {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    naive_mae = results["naive_baseline"]["val_mae"]
    global_median_mae = results["global_median"]["val_mae"]
    best_mae = results[best_name]["val_mae"]
    mae_improvement_pct_over_naive = float((naive_mae - best_mae) / naive_mae * 100)
    mae_improvement_pct_over_global_median = float((global_median_mae - best_mae) / global_median_mae * 100)

    report = {
        "models": results,
        "best_model": best_name,
        "mae_improvement_pct_over_naive": mae_improvement_pct_over_naive,
        "mae_improvement_pct_over_global_median": mae_improvement_pct_over_global_median,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
