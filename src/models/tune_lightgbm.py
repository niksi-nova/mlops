"""Small LightGBM hyperparameter sweep (NOT a DVC stage — run ad hoc).

Trains 8 LightGBM configurations (num_leaves x learning_rate x
min_child_samples), each logged as a child run under one parent run named
"lightgbm-sweep" in the train.experiment_name experiment. Prints a table
of configs sorted by val_mae.

Usage:
    uv run python -m src.models.tune_lightgbm
"""

from __future__ import annotations

import sys
import mlflow
import pandas as pd

from src.models.train import (
    TARGET,
    load_feature_spec,
    load_params,
    load_split,
    mean_absolute_error,
    train_lightgbm,
)
from src.models.tracking import configure_tracking, run_tags

# 8 hand-picked configs spanning all three option values for num_leaves
# (31/63/127), learning_rate (0.03/0.05/0.1) and min_child_samples
# (20/50/100) — a small diagonal sweep rather than the full 27-combo grid.
SWEEP_CONFIGS = [
    {"num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
    {"num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 100},
    {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 100},
    {"num_leaves": 63, "learning_rate": 0.1, "min_child_samples": 20},
    {"num_leaves": 127, "learning_rate": 0.03, "min_child_samples": 100},
    {"num_leaves": 127, "learning_rate": 0.1, "min_child_samples": 50},
]
SWEEP_SIZE = len(SWEEP_CONFIGS)


def build_configs() -> list[dict]:
    """The 8 LightGBM configs for this sweep."""
    return list(SWEEP_CONFIGS)


def main() -> int:
    params = load_params()
    train_params = params["train"]
    feature_spec = load_feature_spec()
    numeric_features = feature_spec["numeric_features"]
    categorical_features = feature_spec["categorical_features"]
    log_target = train_params["log_target"]
    random_state = train_params["random_state"]
    base_lgbm_params = train_params["lightgbm"]

    train_df = load_split("train")
    val_df = load_split("val")

    configure_tracking()
    mlflow.set_experiment(train_params["experiment_name"])
    tags = run_tags()
    tags["mlflow.source.name"] = "src/models/tune_lightgbm.py"

    rows = []
    with mlflow.start_run(run_name="lightgbm-sweep") as parent_run:
        mlflow.set_tags(tags)
        for i, overrides in enumerate(build_configs()):
            lgbm_params = dict(base_lgbm_params)
            lgbm_params.update(overrides)

            with mlflow.start_run(run_name=f"lightgbm-sweep-{i}", nested=True):
                mlflow.set_tags(tags)
                mlflow.log_params(lgbm_params)
                mlflow.log_param("log_target", log_target)

                model, _, val_pred, elapsed, _, _ = train_lightgbm(
                    train_df, val_df, numeric_features, categorical_features,
                    lgbm_params, log_target, random_state,
                )
                val_mae = float(mean_absolute_error(val_df[TARGET], val_pred))
                mlflow.log_metric("val_mae", val_mae)
                mlflow.log_metric("training_time_seconds", elapsed)

                rows.append({**overrides, "val_mae": val_mae, "training_time_seconds": elapsed})
                print(f"  [{i + 1}/{SWEEP_SIZE}] {overrides} -> val_mae={val_mae:.3f}")

        best_val_mae = min(row["val_mae"] for row in rows)
        mlflow.log_metric("best_val_mae", best_val_mae)

    table = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    print()
    print(table.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
