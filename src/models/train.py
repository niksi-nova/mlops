"""Train and compare baseline models for Delhi bus segment travel-time
prediction.

Reads:
  - data/processed/train.parquet
  - data/processed/val.parquet   (used only for evaluation / early stopping)
  - data/processed/feature_spec.json
  - params.yaml (train section)

Trains seven models, each logged as its own W&B run (job_type "train",
group "baselines-v1") in the train.wandb_project project: global_median,
typical_baseline, recent_buses_baseline, ridge, random_forest, lightgbm_l2,
lightgbm_l1. All are evaluated on the validation split only (test.parquet
is never loaded here).

Writes:
  - models/<model_name>.joblib      (every trained model)
  - models/best_model.joblib        (the run with the lowest val_mae)
  - models/model_index.json         (name -> file, val_mae, W&B run id/url)
  - reports/train_metrics.json      (every model's val metrics + the winner)

Usage:
    uv run python -m src.models.train
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
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

from src.models.baselines import ColumnLookupRegressor, GlobalMedianRegressor
from src.models.tracking import load_params, start_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SPEC_PATH = PROCESSED_DIR / "feature_spec.json"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "train_metrics.json"

GROUP = "baselines-v1"
MIN_PREDICTION_S = 1.0
SAMPLE_TABLE_SIZE = 500
TOP_N_FEATURES = 20


def load_feature_spec() -> dict:
    with open(FEATURE_SPEC_PATH) as f:
        return json.load(f)


def load_split(name: str) -> pd.DataFrame:
    """Load one of train.parquet / val.parquet from data/processed."""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


# --------------------------------------------------------------------------
# metrics / plotting / prediction helpers
# --------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """mae/rmse/r2/median_ae/within_30s_pct/within_60s_pct, keyed as
    f"{prefix}_<metric>"."""
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_median_ae": float(median_absolute_error(y_true, y_pred)),
        f"{prefix}_within_30s_pct": float((abs_err <= 30).mean() * 100),
        f"{prefix}_within_60s_pct": float((abs_err <= 60).mean() * 100),
    }


def clip_and_backtransform(raw_pred: np.ndarray, log_target: bool) -> np.ndarray:
    """expm1 the prediction if it was trained in log1p space, then clip at
    >= 1 second (a segment can't take zero or negative time)."""
    pred = np.expm1(raw_pred) if log_target else raw_pred
    return np.clip(pred, MIN_PREDICTION_S, None)


def plot_feature_importance(names: list[str], importances: np.ndarray, title: str, out_path: Path) -> None:
    """Horizontal bar chart of the top-N features by importance, saved as a PNG."""
    order = np.argsort(importances)[::-1][:TOP_N_FEATURES]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="steelblue")
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_prediction_sample_table(val_df: pd.DataFrame, val_pred: np.ndarray, target: str, random_state: int) -> "wandb.Table":
    n = min(SAMPLE_TABLE_SIZE, len(val_df))
    sample_idx = val_df.sample(n=n, random_state=random_state).index.to_numpy()
    sample = pd.DataFrame(
        {
            "actual": val_df.loc[sample_idx, target].to_numpy(),
            "predicted": val_pred[sample_idx],
            "segment_id": val_df.loc[sample_idx, "segment_id"].to_numpy(),
            "hour": val_df.loc[sample_idx, "hour"].to_numpy(),
        }
    )
    return wandb.Table(dataframe=sample)


def log_model_artifact(run, model_name: str, model_path: Path, metadata: dict[str, float]) -> None:
    artifact = wandb.Artifact(name=f"delhi-{model_name}", type="model", metadata=metadata)
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)


def use_feature_spec_artifact(run) -> None:
    """Log feature_spec.json as a dataset-spec Artifact and mark it as used
    by this run, so W&B's lineage graph shows which feature spec fed it."""
    artifact = wandb.Artifact(name="delhi-feature-spec", type="dataset-spec")
    artifact.add_file(str(FEATURE_SPEC_PATH))
    run.use_artifact(artifact)


# --------------------------------------------------------------------------
# model-specific training
# --------------------------------------------------------------------------
def train_global_median(train_df: pd.DataFrame, val_df: pd.DataFrame, target: str):
    model = GlobalMedianRegressor()
    start = time.perf_counter()
    model.fit(train_df, train_df[target])
    elapsed = time.perf_counter() - start
    train_pred = clip_and_backtransform(model.predict(train_df), log_target=False)
    val_pred = clip_and_backtransform(model.predict(val_df), log_target=False)
    return model, train_pred, val_pred, elapsed


def train_column_lookup(train_df: pd.DataFrame, val_df: pd.DataFrame, column: str):
    model = ColumnLookupRegressor(column=column)
    start = time.perf_counter()
    model.fit(train_df)
    elapsed = time.perf_counter() - start
    train_pred = clip_and_backtransform(model.predict(train_df), log_target=False)
    val_pred = clip_and_backtransform(model.predict(val_df), log_target=False)
    return model, train_pred, val_pred, elapsed


def train_ridge(
    train_df: pd.DataFrame, val_df: pd.DataFrame, target: str, numeric_features: list[str],
    categorical_features: list[str], ridge_params: dict,
):
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )
    model = Pipeline([("preprocess", preprocessor), ("model", Ridge(**ridge_params))])

    y_train = np.log1p(train_df[target])

    start = time.perf_counter()
    model.fit(train_df, y_train)
    elapsed = time.perf_counter() - start

    train_pred = clip_and_backtransform(model.predict(train_df), log_target=True)
    val_pred = clip_and_backtransform(model.predict(val_df), log_target=True)
    return model, train_pred, val_pred, elapsed


def train_random_forest(
    train_df: pd.DataFrame, val_df: pd.DataFrame, target: str, numeric_features: list[str],
    categorical_features: list[str], rf_params: dict, random_state: int,
):
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

    y_train = np.log1p(train_df[target])

    start = time.perf_counter()
    model.fit(train_df, y_train)
    elapsed = time.perf_counter() - start

    train_pred = clip_and_backtransform(model.predict(train_df), log_target=True)
    val_pred = clip_and_backtransform(model.predict(val_df), log_target=True)

    feature_names = categorical_features + numeric_features
    importances = model.named_steps["model"].feature_importances_
    return model, train_pred, val_pred, elapsed, feature_names, importances


def prep_lightgbm_features(
    df: pd.DataFrame, feature_columns: list[str], categorical_features: list[str],
    train_categories: dict[str, pd.Index] | None = None,
) -> pd.DataFrame:
    """Select LightGBM's feature columns and cast categoricals to pandas
    'category' dtype (LightGBM's native categorical handling). Pass the
    TRAIN split's fitted `train_categories` (from this same function's
    return value on train) when preparing val/test, so category codes line
    up with what the model was trained on."""
    X = df[feature_columns].copy()
    for col in categorical_features:
        if train_categories is None:
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.Categorical(X[col], categories=train_categories[col])
    return X


def train_lightgbm(
    train_df: pd.DataFrame, val_df: pd.DataFrame, target: str, numeric_features: list[str],
    categorical_features: list[str], lgbm_params: dict, log_target: bool, random_state: int,
):
    feature_columns = categorical_features + numeric_features

    X_train = prep_lightgbm_features(train_df, feature_columns, categorical_features)
    train_categories = {col: X_train[col].cat.categories for col in categorical_features}
    X_val = prep_lightgbm_features(val_df, feature_columns, categorical_features, train_categories)

    y_train = np.log1p(train_df[target]) if log_target else train_df[target]
    y_val = np.log1p(val_df[target]) if log_target else val_df[target]

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
# orchestration
# --------------------------------------------------------------------------
def run_and_log(
    name: str,
    config: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    random_state: int,
    fit_fn,
    feature_importance: tuple[list[str], np.ndarray] | None = None,
) -> tuple[Any, dict[str, float]]:
    """Run one model's training + W&B logging. `fit_fn` returns either
    (model, train_pred, val_pred, elapsed) or that plus (feat_names,
    importances) for tree models."""
    run = start_run(name=name, job_type="train", config=config, group=GROUP)

    result = fit_fn()
    if len(result) == 6:
        model, train_pred, val_pred, elapsed, feat_names, importances = result
    else:
        model, train_pred, val_pred, elapsed = result
        feat_names = importances = None

    metrics = regression_metrics(val_df[target], val_pred, "val")
    metrics["train_mae"] = float(mean_absolute_error(train_df[target], train_pred))
    metrics["training_seconds"] = elapsed
    wandb.log(metrics)

    if feat_names is not None:
        fi_path = MODELS_DIR / f"{name}_feature_importance.png"
        plot_feature_importance(feat_names, importances, f"{name} — top {TOP_N_FEATURES} feature importances", fi_path)
        wandb.log({"feature_importance": wandb.Image(str(fi_path))})

    table = make_prediction_sample_table(val_df, val_pred, target, random_state)
    wandb.log({"val_predictions_sample": table})

    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    log_model_artifact(run, name, model_path, metrics)
    use_feature_spec_artifact(run)

    print(f"{name}: val_mae={metrics['val_mae']:.3f}")
    run.finish()
    return model, metrics


def select_best_baseline(results: dict[str, dict[str, float]]) -> str:
    """The better (lower val_mae) of typical_baseline and recent_buses_baseline."""
    baseline_names = ["typical_baseline", "recent_buses_baseline"]
    return min(baseline_names, key=lambda name: results[name]["val_mae"])


def main() -> int:
    params = load_params()
    train_params = params["train"]
    feature_spec = load_feature_spec()
    numeric_features = feature_spec["numeric_features"]
    categorical_features = feature_spec["categorical_features"]
    target = train_params["target"]
    random_state = train_params["random_state"]

    train_df = load_split("train")
    val_df = load_split("val")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fitted_models: dict[str, Any] = {}
    results: dict[str, dict[str, float]] = {}

    fitted_models["global_median"], results["global_median"] = run_and_log(
        "global_median", {}, train_df, val_df, target, random_state,
        lambda: train_global_median(train_df, val_df, target),
    )

    fitted_models["typical_baseline"], results["typical_baseline"] = run_and_log(
        "typical_baseline", {"column": "typical_segment_hour_s"}, train_df, val_df, target, random_state,
        lambda: train_column_lookup(train_df, val_df, "typical_segment_hour_s"),
    )

    fitted_models["recent_buses_baseline"], results["recent_buses_baseline"] = run_and_log(
        "recent_buses_baseline", {"column": "prev3_median_tt_s"}, train_df, val_df, target, random_state,
        lambda: train_column_lookup(train_df, val_df, "prev3_median_tt_s"),
    )

    ridge_params = train_params["ridge"]
    fitted_models["ridge"], results["ridge"] = run_and_log(
        "ridge", {**ridge_params, "log_target": True}, train_df, val_df, target, random_state,
        lambda: train_ridge(train_df, val_df, target, numeric_features, categorical_features, ridge_params),
    )

    rf_params = train_params["random_forest"]
    fitted_models["random_forest"], results["random_forest"] = run_and_log(
        "random_forest", {**rf_params, "log_target": True}, train_df, val_df, target, random_state,
        lambda: train_random_forest(
            train_df, val_df, target, numeric_features, categorical_features, rf_params, random_state
        ),
    )

    lgbm_l2_params = train_params["lightgbm_l2"]
    fitted_models["lightgbm_l2"], results["lightgbm_l2"] = run_and_log(
        "lightgbm_l2", {**lgbm_l2_params, "log_target": True}, train_df, val_df, target, random_state,
        lambda: train_lightgbm(
            train_df, val_df, target, numeric_features, categorical_features, lgbm_l2_params, True, random_state
        ),
    )

    lgbm_l1_params = train_params["lightgbm_l1"]
    fitted_models["lightgbm_l1"], results["lightgbm_l1"] = run_and_log(
        "lightgbm_l1", {**lgbm_l1_params, "log_target": False}, train_df, val_df, target, random_state,
        lambda: train_lightgbm(
            train_df, val_df, target, numeric_features, categorical_features, lgbm_l1_params, False, random_state
        ),
    )

    # --- save model index, pick the best, write the report -----------------
    model_index = {
        name: {"file": f"{name}.joblib", "val_mae": results[name]["val_mae"]} for name in fitted_models
    }
    (MODELS_DIR / "model_index.json").write_text(json.dumps(model_index, indent=2))

    best_name = min(results, key=lambda name: results[name]["val_mae"])
    joblib.dump(fitted_models[best_name], BEST_MODEL_PATH)
    print(f"Best model: {best_name} -> {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    best_baseline_name = select_best_baseline(results)

    global_median_mae = results["global_median"]["val_mae"]
    best_baseline_mae = results[best_baseline_name]["val_mae"]
    best_mae = results[best_name]["val_mae"]

    improvement_pct_over_best_baseline = float((best_baseline_mae - best_mae) / best_baseline_mae * 100)
    improvement_pct_over_global_median = float((global_median_mae - best_mae) / global_median_mae * 100)

    success_target_pct = train_params["success_target_pct"]
    target_met = bool(improvement_pct_over_best_baseline >= success_target_pct)

    report = {
        "models": results,
        "best_model": best_name,
        "best_baseline": best_baseline_name,
        "improvement_pct_over_best_baseline": improvement_pct_over_best_baseline,
        "improvement_pct_over_global_median": improvement_pct_over_global_median,
        "success_target_pct": success_target_pct,
        "target_met": target_met,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
