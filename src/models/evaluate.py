"""Final evaluation of all trained TTC delay models on the held-out TEST
split (year 2024). This is the ONLY script that loads test.parquet — no
model hyperparameters or features should be changed based on what this
script finds.

Reads:
  - data/processed/test.parquet
  - data/processed/train.parquet   (categorical columns only, to rebuild
                                     the LightGBM category encoding used
                                     at training time)
  - data/processed/feature_spec.json
  - models/model_index.json + models/<name>.joblib for every model
  - reports/train_metrics.json     (validation metrics, for the val-vs-test
                                     gap and the naive/global_median
                                     comparison)
  - params.yaml (train section, for the log_target flag per model)

Writes:
  - reports/test_metrics.json
  - reports/figures/eval/*.png

Logs one MLflow run, "final-evaluation-test-2024", with all test metrics
and every eval plot as an artifact.

Usage:
    uv run python -m src.models.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

from src.models.train import (
    MODELS_DIR,
    PARAMS_PATH,
    TARGET,
    clip_and_backtransform,
    plot_feature_importance,
    prep_lightgbm_features,
)
from src.models.tracking import configure_tracking, run_tags

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SPEC_PATH = PROCESSED_DIR / "feature_spec.json"
MODEL_INDEX_PATH = MODELS_DIR / "model_index.json"
TRAIN_METRICS_PATH = PROJECT_ROOT / "reports" / "train_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "eval"

SUCCESS_TARGET_PCT = 15
LIGHTGBM_MODELS = {"lightgbm", "lightgbm_l1"}
BUCKET_EDGES = [0, 5, 10, 20, 30, 60]
BUCKET_LABELS = ["0-5", "5-10", "10-20", "20-30", "30-60", "60+"]


def load_params() -> dict:
    """Load params.yaml from the project root."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_feature_spec() -> dict:
    """Load data/processed/feature_spec.json."""
    with open(FEATURE_SPEC_PATH) as f:
        return json.load(f)


def load_model_index() -> dict:
    """Load models/model_index.json."""
    with open(MODEL_INDEX_PATH) as f:
        return json.load(f)


def load_train_metrics() -> dict:
    """Load reports/train_metrics.json (validation-time results)."""
    with open(TRAIN_METRICS_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# bucketing / within-N metric
# --------------------------------------------------------------------------
def bucket_delay(value: float) -> str:
    """Bucket a delay (minutes) into one of BUCKET_LABELS: half-open bins
    [0,5), [5,10), [10,20), [20,30), [30,60), [60, inf)."""
    for edge, label in zip(BUCKET_EDGES[1:], BUCKET_LABELS[:-1]):
        if value < edge:
            return label
    return BUCKET_LABELS[-1]


def within_n_minutes_pct(y_true: np.ndarray, y_pred: np.ndarray, n: float) -> float:
    """% of predictions within `n` minutes of the actual delay."""
    return float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= n).mean() * 100)


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------
def predict_with_model(
    name: str, model, df: pd.DataFrame, feature_columns: list[str],
    categorical_features: list[str], train_categories: dict[str, pd.Index], log_target: bool,
) -> np.ndarray:
    """Predict on `df` with the right input shape per model family, then
    back-transform/clip to original-scale minutes."""
    if name in LIGHTGBM_MODELS:
        X = prep_lightgbm_features(df, feature_columns, categorical_features, train_categories)
    else:
        X = df
    raw_pred = model.predict(X)
    return clip_and_backtransform(raw_pred, log_target)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """mae/rmse/r2/median_ae + within-5/10-minute shares."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "within_5min_pct": within_n_minutes_pct(y_true, y_pred, 5),
        "within_10min_pct": within_n_minutes_pct(y_true, y_pred, 10),
    }


def mae_by_group(df: pd.DataFrame, group_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict]:
    """{group_value: {mae, count}} for every distinct value of `group_col`."""
    work = df[[group_col]].copy()
    work["abs_error"] = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    grouped = work.groupby(group_col)["abs_error"].agg(["mean", "count"])
    return {
        str(key): {"mae": float(row["mean"]), "count": int(row["count"])}
        for key, row in grouped.iterrows()
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_model_comparison(test_results: dict[str, dict], out_path: Path) -> None:
    names = list(test_results.keys())
    maes = [test_results[name]["mae"] for name in names]
    order = np.argsort(maes)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order], [maes[i] for i in order], color="steelblue")
    ax.set_xlabel("test MAE (minutes)")
    ax.set_title("Model comparison — test MAE (2024), all 6 models")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(np.log1p(y_true), np.log1p(y_pred), gridsize=50, cmap="viridis", mincnt=1)
    lims = [0, max(np.log1p(y_true).max(), np.log1p(y_pred).max())]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("log1p(actual delay, minutes)")
    ax.set_ylabel("log1p(predicted delay, minutes)")
    ax.set_title("Predicted vs actual — best model (log1p scale)")
    ax.legend()
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=80, color="indianred")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("residual (predicted - actual), minutes")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution — best model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mae_by_bucket(by_bucket: dict[str, dict], out_path: Path) -> None:
    labels = [label for label in BUCKET_LABELS if label in by_bucket]
    maes = [by_bucket[label]["mae"] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, maes, color="darkorange")
    ax.set_xlabel("actual delay bucket (minutes)")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("MAE by actual delay bucket — best model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mae_by_incident_type(by_incident: dict[str, dict], out_path: Path) -> None:
    items = sorted(by_incident.items(), key=lambda kv: kv[1]["count"], reverse=True)[:10]
    labels = [k for k, _ in items]
    maes = [v["mae"] for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(maes)
    ax.barh([labels[i] for i in order], [maes[i] for i in order], color="slateblue")
    ax.set_xlabel("MAE (minutes)")
    ax.set_title("MAE by incident type — top 10 by count — best model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mae_by_month(by_month: dict[str, dict], out_path: Path) -> None:
    months = sorted(by_month.keys(), key=int)
    maes = [by_month[m]["mae"] for m in months]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(months, maes, color="teal")
    ax.set_xlabel("month of 2024")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("MAE by month of 2024 — best model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def best_model_feature_importance(name: str, model, categorical_features: list[str], numeric_features: list[str]) -> tuple[list[str], np.ndarray] | None:
    """(names, importances) for the best model, or None if it has no
    meaningful feature-importance concept (naive_baseline, global_median)."""
    if name in LIGHTGBM_MODELS:
        return categorical_features + numeric_features, model.feature_importances_
    if name == "random_forest":
        return categorical_features + numeric_features, model.named_steps["model"].feature_importances_
    if name == "ridge":
        encoder = model.named_steps["preprocess"].named_transformers_["cat"]
        names = list(encoder.get_feature_names_out(categorical_features)) + numeric_features
        return names, np.abs(model.named_steps["model"].coef_)
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    params = load_params()
    train_params = params["train"]
    log_target_default = train_params["log_target"]
    feature_spec = load_feature_spec()
    numeric_features = feature_spec["numeric_features"]
    categorical_features = feature_spec["categorical_features"]
    feature_columns = categorical_features + numeric_features

    model_index = load_model_index()
    train_metrics = load_train_metrics()
    best_name = train_metrics["best_model"]

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    train_cat_df = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=categorical_features)
    train_categories = {
        col: train_cat_df[col].astype("category").cat.categories for col in categorical_features
    }
    y_true = test_df[TARGET].to_numpy()

    log_target_by_model = {
        "naive_baseline": False,
        "global_median": False,
        "ridge": log_target_default,
        "random_forest": log_target_default,
        "lightgbm": log_target_default,
        "lightgbm_l1": False,
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- every model's test metrics --------------------------------------
    test_results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, info in model_index.items():
        model = joblib.load(MODELS_DIR / info["file"])
        y_pred = predict_with_model(
            name, model, test_df, feature_columns, categorical_features,
            train_categories, log_target_by_model[name],
        )
        predictions[name] = y_pred
        test_results[name] = test_metrics_for(y_true, y_pred)
        print(f"{name}: test_mae={test_results[name]['mae']:.3f}")

    best_pred = predictions[best_name]
    best_model = joblib.load(MODELS_DIR / model_index[best_name]["file"])

    # --- error analysis on the best model ---------------------------------
    buckets = test_df[TARGET].map(bucket_delay)
    by_bucket = mae_by_group(pd.DataFrame({"bucket": buckets}), "bucket", y_true, best_pred)

    by_incident_all = mae_by_group(test_df, "incident", y_true, best_pred)
    top_incidents = test_df["incident"].value_counts().head(10).index
    by_incident = {k: v for k, v in by_incident_all.items() if k in top_incidents}

    by_hour = mae_by_group(test_df.assign(hour=test_df["hour"]), "hour", y_true, best_pred)
    weekday_weekend = test_df["is_weekend"].map({0: "weekday", 1: "weekend"})
    by_weekday_weekend = mae_by_group(pd.DataFrame({"g": weekday_weekend}), "g", y_true, best_pred)
    by_month = mae_by_group(test_df.assign(month=test_df["month"]), "month", y_true, best_pred)

    location_group = test_df["location"].map(lambda v: "OTHER" if v == "OTHER" else "known")
    by_location = mae_by_group(pd.DataFrame({"g": location_group}), "g", y_true, best_pred)

    val_mae_best = train_metrics["models"][best_name]["val_mae"]
    test_mae_best = test_results[best_name]["mae"]
    generalisation_gap = test_mae_best - val_mae_best

    error_analysis = {
        "by_delay_bucket": by_bucket,
        "by_incident_type_top10": by_incident,
        "by_hour": by_hour,
        "by_weekday_weekend": by_weekday_weekend,
        "by_month_2024": by_month,
        "by_location_other_vs_known": by_location,
        "val_mae": val_mae_best,
        "test_mae": test_mae_best,
        "generalisation_gap": generalisation_gap,
    }

    # --- plots ------------------------------------------------------------
    plot_model_comparison(test_results, FIGURES_DIR / "01_model_comparison.png")
    plot_predicted_vs_actual(y_true, best_pred, FIGURES_DIR / "02_predicted_vs_actual.png")
    plot_residual_distribution(y_true, best_pred, FIGURES_DIR / "03_residual_distribution.png")
    plot_mae_by_bucket(by_bucket, FIGURES_DIR / "04_mae_by_bucket.png")
    plot_mae_by_incident_type(by_incident, FIGURES_DIR / "05_mae_by_incident_type.png")
    plot_mae_by_month(by_month, FIGURES_DIR / "06_mae_by_month.png")

    fi = best_model_feature_importance(best_name, best_model, categorical_features, numeric_features)
    fi_path = FIGURES_DIR / "07_best_model_feature_importance.png"
    if fi is not None:
        names, importances = fi
        plot_feature_importance(names, importances, f"{best_name} — top 20 feature importances (test eval)", fi_path)
    print(f"Saved eval figures to {FIGURES_DIR.relative_to(PROJECT_ROOT)}")

    # --- success criteria --------------------------------------------------
    naive_test_mae = test_results["naive_baseline"]["mae"]
    global_median_test_mae = test_results["global_median"]["mae"]
    improvement_pct_over_naive_baseline = float((naive_test_mae - test_mae_best) / naive_test_mae * 100)
    improvement_pct_over_global_median = float((global_median_test_mae - test_mae_best) / global_median_test_mae * 100)
    target_met = improvement_pct_over_naive_baseline >= SUCCESS_TARGET_PCT

    report = {
        "models": test_results,
        "best_model": best_name,
        "error_analysis": error_analysis,
        "success_target_pct": SUCCESS_TARGET_PCT,
        "improvement_pct_over_naive_baseline": improvement_pct_over_naive_baseline,
        "improvement_pct_over_global_median": improvement_pct_over_global_median,
        "target_met": target_met,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # --- mlflow run ---------------------------------------------------------
    configure_tracking()
    mlflow.set_experiment(train_params["experiment_name"])
    tags = dict(run_tags())
    tags["best_model_run_id"] = model_index[best_name]["run_id"]

    with mlflow.start_run(run_name="final-evaluation-test-2024"):
        mlflow.set_tags(tags)
        flat_metrics = {}
        for name, metrics in test_results.items():
            for metric_name, value in metrics.items():
                flat_metrics[f"{name}_{metric_name}"] = value
        flat_metrics["generalisation_gap"] = generalisation_gap
        flat_metrics["improvement_pct_over_naive_baseline"] = improvement_pct_over_naive_baseline
        flat_metrics["improvement_pct_over_global_median"] = improvement_pct_over_global_median
        mlflow.log_metrics(flat_metrics)
        mlflow.log_param("target_met", target_met)
        mlflow.log_artifact(str(REPORT_PATH))
        for png_path in sorted(FIGURES_DIR.glob("*.png")):
            mlflow.log_artifact(str(png_path), artifact_path="eval_figures")
        print("Logged final-evaluation-test-2024 run to MLflow")

    return 0


if __name__ == "__main__":
    sys.exit(main())
