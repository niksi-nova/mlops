"""Final evaluation of all trained Delhi bus travel-time models on the
held-out TEST split (day_index 77-92). This is the ONLY script that loads
test.parquet - no feature or model choice should be changed based on what
this script finds. The "best model" and "best baseline" are the ones
already selected on VALIDATION (reports/train_metrics.json); this script
only evaluates them on test, it never re-selects based on test.

Reads:
  - data/processed/test.parquet
  - data/processed/val.parquet     (only for the days-62-92 daily MAE view)
  - data/processed/feature_spec.json
  - models/model_index.json + models/<name>.joblib for every model
  - reports/train_metrics.json     (val metrics, best_model, best_baseline)
  - params.yaml (train/featurize sections)

Writes:
  - reports/test_metrics.json
  - reports/figures/eval/*.png

Usage:
    uv run python -m src.models.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from src.models.train import (
    clip_and_backtransform,
    plot_feature_importance,
    prep_lightgbm_features,
    regression_metrics,
)
from src.models.tracking import lineage_config, load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SPEC_PATH = PROCESSED_DIR / "feature_spec.json"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_INDEX_PATH = MODELS_DIR / "model_index.json"
TRAIN_METRICS_PATH = PROJECT_ROOT / "reports" / "train_metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "eval"

LIGHTGBM_MODELS = {"lightgbm_l2", "lightgbm_l1"}
LOG_TARGET_MODELS = {"ridge", "random_forest", "lightgbm_l2"}
TRAVEL_TIME_BUCKETS = [(0, 60, "0-60"), (60, 120, "60-120"), (120, 300, "120-300"), (300, 600, "300-600")]
DELAY_THRESHOLD_S = 60.0
TOP_N_WORST_SEGMENTS = 10


def load_feature_spec() -> dict:
    with open(FEATURE_SPEC_PATH) as f:
        return json.load(f)


def load_model_index() -> dict:
    with open(MODEL_INDEX_PATH) as f:
        return json.load(f)


def load_train_metrics() -> dict:
    with open(TRAIN_METRICS_PATH) as f:
        return json.load(f)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def load_all_models(model_index: dict) -> dict:
    return {name: joblib.load(MODELS_DIR / info["file"]) for name, info in model_index.items()}


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------
def predict(model_name: str, model, df: pd.DataFrame, feature_columns: list[str], categorical_features: list[str]) -> np.ndarray:
    """Predict with any of the 7 trained models, replicating each one's
    training-time preprocessing/back-transform. LightGBM realigns pandas
    'category' codes against what it saw at training time automatically
    (via its stored pandas_categorical), so no train-split categories are
    needed here - verified empirically to be bit-identical to passing them."""
    log_target = model_name in LOG_TARGET_MODELS
    if model_name in LIGHTGBM_MODELS:
        X = prep_lightgbm_features(df, feature_columns, categorical_features)
        raw = model.predict(X)
    else:
        raw = model.predict(df)
    return clip_and_backtransform(raw, log_target)


# --------------------------------------------------------------------------
# bucket / within-N-seconds helpers
# --------------------------------------------------------------------------
def bucket_travel_time(value: float) -> str:
    """0-60, 60-120, 120-300, 300-600, 600+ (left-inclusive, right-exclusive;
    600 and above -> "600+")."""
    for lo, hi, label in TRAVEL_TIME_BUCKETS:
        if lo <= value < hi:
            return label
    return "600+"


def within_n_seconds_pct(y_true: np.ndarray, y_pred: np.ndarray, threshold_s: float) -> float:
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float((abs_err <= threshold_s).mean() * 100)


# --------------------------------------------------------------------------
# error-analysis tables (best model only)
# --------------------------------------------------------------------------
def mae_by(df: pd.DataFrame, group_col: str | list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_col)["abs_error"]
        .agg(mae="mean", n="size")
        .reset_index()
        .sort_values("mae", ascending=False)
    )


def worst_segments(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return mae_by(df, "segment_id").head(n)


def mae_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df["travel_time_s"].apply(bucket_travel_time)
    order = [label for _, _, label in TRAVEL_TIME_BUCKETS] + ["600+"]
    out = mae_by(df, "bucket").set_index("bucket").reindex(order).reset_index()
    return out


def delay_view(df: pd.DataFrame, typical_pred: np.ndarray, recent_pred: np.ndarray, threshold_s: float) -> dict:
    is_delayed = df["delay_s"] > threshold_s
    share = float(is_delayed.mean())
    delayed = df[is_delayed]
    return {
        "share_of_rows_delayed_gt_60s": share,
        "best_model_mae_on_delayed_rows": float(delayed["abs_error"].mean()) if len(delayed) else None,
        "typical_baseline_mae_on_delayed_rows": float(
            np.abs(delayed["travel_time_s"].to_numpy() - typical_pred[is_delayed.to_numpy()]).mean()
        )
        if len(delayed)
        else None,
        "recent_buses_baseline_mae_on_delayed_rows": float(
            np.abs(delayed["travel_time_s"].to_numpy() - recent_pred[is_delayed.to_numpy()]).mean()
        )
        if len(delayed)
        else None,
    }


def compute_daily_mae(
    val_df: pd.DataFrame, val_pred: np.ndarray, test_df: pd.DataFrame, test_pred: np.ndarray, target: str
) -> pd.DataFrame:
    """Day-by-day MAE for day_index 62-92, labelling each day_index by
    which split it actually came from (val_df's days -> "val", test_df's
    days -> "test") - the model was selected on val, so these must stay
    visually/labelwise distinct."""
    val_abs = np.abs(val_df[target].to_numpy() - val_pred)
    test_abs = np.abs(test_df[target].to_numpy() - test_pred)
    val_part = pd.DataFrame({"day_index": val_df["day_index"].to_numpy(), "abs_error": val_abs, "split": "val"})
    test_part = pd.DataFrame({"day_index": test_df["day_index"].to_numpy(), "abs_error": test_abs, "split": "test"})
    combined = pd.concat([val_part, test_part], ignore_index=True)
    daily = combined.groupby(["day_index", "split"])["abs_error"].mean().reset_index().rename(columns={"abs_error": "mae"})
    return daily.sort_values("day_index").reset_index(drop=True)


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_test_mae_by_model(results: dict[str, dict[str, float]], out_path: Path) -> None:
    names = list(results.keys())
    maes = [results[n]["mae"] for n in names]
    order = np.argsort(maes)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([names[i] for i in order], [maes[i] for i in order], color="steelblue")
    ax.set_ylabel("test MAE (s)")
    ax.set_title("Test MAE by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lim = max(y_true.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], color="crimson", linewidth=0.8, linestyle="--")
    ax.set_xlabel("actual travel_time_s")
    ax.set_ylabel("predicted travel_time_s")
    ax.set_title("Predicted vs actual (best model, test)")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(residuals, bins=150, color="darkorange")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("predicted - actual (s)")
    ax.set_title("Residual distribution (best model, test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mae_by_hour(mae_by_hour_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = mae_by_hour_df.sort_values("hour")
    ax.plot(ordered["hour"], ordered["mae"], marker="o", color="steelblue")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Test MAE by hour of day (best model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mae_by_bucket(mae_by_bucket_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(mae_by_bucket_df["bucket"], mae_by_bucket_df["mae"], color="seagreen")
    ax.set_xlabel("travel_time_s bucket")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Test MAE by travel-time bucket (best model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_daily_mae(daily_df: pd.DataFrame, val_days: list[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for split, g in daily_df.groupby("split"):
        g = g.sort_values("day_index")
        ax.plot(g["day_index"], g["mae"], marker="o", markersize=3, label=split)
    ax.axvline(val_days[1] + 0.5, color="gray", linestyle="--", linewidth=0.8, label="val/test boundary")
    ax.set_xlabel("day_index")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Daily MAE, day_index 62-92 (best model)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    params = load_params()
    train_params = params["train"]
    featurize_params = params["featurize"]
    target = train_params["target"]

    feature_spec = load_feature_spec()
    numeric_features = feature_spec["numeric_features"]
    categorical_features = feature_spec["categorical_features"]
    feature_columns = categorical_features + numeric_features

    model_index = load_model_index()
    models = load_all_models(model_index)
    train_metrics = load_train_metrics()
    best_model_name = train_metrics["best_model"]
    best_baseline_name = train_metrics["best_baseline"]

    test_df = load_split("test")
    val_df = load_split("val")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- a. every model's test metrics ----------------------------------
    results: dict[str, dict[str, float]] = {}
    test_preds: dict[str, np.ndarray] = {}
    for name, model in models.items():
        pred = predict(name, model, test_df, feature_columns, categorical_features)
        test_preds[name] = pred
        m = regression_metrics(test_df[target], pred, "test")
        results[name] = {
            "mae": m["test_mae"],
            "rmse": m["test_rmse"],
            "r2": m["test_r2"],
            "median_ae": m["test_median_ae"],
            "within_30s_pct": m["test_within_30s_pct"],
            "within_60s_pct": m["test_within_60s_pct"],
        }
        print(f"{name}: test_mae={results[name]['mae']:.3f}")

    best_pred = test_preds[best_model_name]
    best_model = models[best_model_name]

    eval_df = test_df.copy()
    eval_df["prediction"] = best_pred
    eval_df["abs_error"] = np.abs(eval_df[target] - eval_df["prediction"])

    # --- b. error analysis for the best model ----------------------------
    mae_by_hour_df = mae_by(eval_df, "hour")
    mae_by_route_direction_df = mae_by(eval_df, ["route", "direction"])
    worst_segments_df = worst_segments(eval_df, TOP_N_WORST_SEGMENTS)

    multi_node = eval_df["nodes_spanned"] > 1
    mae_by_nodes_spanned_df = mae_by(eval_df.assign(nodes_spanned_gt_1=multi_node), "nodes_spanned_gt_1")
    mae_by_has_prev_bus_df = mae_by(eval_df, "has_prev_bus")
    mae_by_bucket_df = mae_by_bucket(eval_df)

    delay = delay_view(eval_df, test_preds["typical_baseline"], test_preds["recent_buses_baseline"], DELAY_THRESHOLD_S)

    daily_mae_df = compute_daily_mae(
        val_df, predict(best_model_name, best_model, val_df, feature_columns, categorical_features),
        test_df, best_pred, target,
    )

    val_mae_best = train_metrics["models"][best_model_name]["val_mae"]
    test_mae_best = results[best_model_name]["mae"]
    generalisation_gap_pct = float((test_mae_best - val_mae_best) / val_mae_best * 100)

    # --- c. plots ----------------------------------------------------------
    plot_test_mae_by_model(results, FIG_DIR / "01_test_mae_by_model.png")
    plot_predicted_vs_actual(eval_df[target].to_numpy(), best_pred, FIG_DIR / "02_predicted_vs_actual.png")
    plot_residual_distribution(eval_df[target].to_numpy(), best_pred, FIG_DIR / "03_residual_distribution.png")
    plot_mae_by_hour(mae_by_hour_df, FIG_DIR / "04_mae_by_hour.png")
    plot_mae_by_bucket(mae_by_bucket_df, FIG_DIR / "05_mae_by_travel_time_bucket.png")
    plot_daily_mae(daily_mae_df, featurize_params["val_days"], FIG_DIR / "06_daily_mae_val_test.png")

    if best_model_name in LIGHTGBM_MODELS:
        gain_importances = np.array(best_model.booster_.feature_importance(importance_type="gain"))
        gain_names = best_model.booster_.feature_name()
        plot_feature_importance(
            gain_names, gain_importances,
            f"{best_model_name} — top 20 feature importances (gain)",
            FIG_DIR / "07_best_model_feature_importance.png",
        )

    # --- d. test_metrics.json ----------------------------------------------
    global_median_mae = results["global_median"]["mae"]
    best_baseline_mae = results[best_baseline_name]["mae"]
    improvement_pct_over_best_baseline = float((best_baseline_mae - test_mae_best) / best_baseline_mae * 100)
    improvement_pct_over_global_median = float((global_median_mae - test_mae_best) / global_median_mae * 100)
    success_target_pct = train_params["success_target_pct"]
    target_met = bool(improvement_pct_over_best_baseline >= success_target_pct)

    report = {
        "models": results,
        "best_model": best_model_name,
        "best_baseline": best_baseline_name,
        "improvement_pct_over_best_baseline": improvement_pct_over_best_baseline,
        "improvement_pct_over_global_median": improvement_pct_over_global_median,
        "success_target_pct": success_target_pct,
        "target_met": target_met,
        "val_mae_best_model": val_mae_best,
        "test_mae_best_model": test_mae_best,
        "generalisation_gap_pct": generalisation_gap_pct,
        "error_analysis": {
            "mae_by_hour": mae_by_hour_df.to_dict(orient="records"),
            "mae_by_route_direction": mae_by_route_direction_df.to_dict(orient="records"),
            "worst_segments": worst_segments_df.to_dict(orient="records"),
            "mae_by_nodes_spanned_gt_1": mae_by_nodes_spanned_df.to_dict(orient="records"),
            "mae_by_has_prev_bus": mae_by_has_prev_bus_df.to_dict(orient="records"),
            "mae_by_travel_time_bucket": mae_by_bucket_df.to_dict(orient="records"),
            "delay_view_gt_60s": delay,
            "daily_mae": daily_mae_df.to_dict(orient="records"),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # --- e. W&B run ----------------------------------------------------------
    run = wandb.init(
        project=train_params["wandb_project"],
        name="final-evaluation-test",
        job_type="evaluate",
        config=lineage_config(),
    )
    run.use_artifact(f"delhi-{best_model_name}:latest", type="model")

    run.summary.update(
        {
            "best_model": best_model_name,
            "best_baseline": best_baseline_name,
            "test_mae_best_model": test_mae_best,
            "val_mae_best_model": val_mae_best,
            "generalisation_gap_pct": generalisation_gap_pct,
            "improvement_pct_over_best_baseline": improvement_pct_over_best_baseline,
            "improvement_pct_over_global_median": improvement_pct_over_global_median,
            "target_met": target_met,
            **{f"test_{name}_mae": results[name]["mae"] for name in results},
        }
    )

    for fig_path in sorted(FIG_DIR.glob("*.png")):
        wandb.log({fig_path.stem: wandb.Image(str(fig_path))})

    wandb.log({"mae_by_hour": wandb.Table(dataframe=mae_by_hour_df)})
    wandb.log({"daily_mae": wandb.Table(dataframe=daily_mae_df)})

    run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
