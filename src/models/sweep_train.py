"""Single lightgbm_l1 training run for a W&B sweep
(sweeps/lightgbm_l1_sweep.yaml). NOT run directly or as a DVC stage - a
`wandb agent` launches this script once per trial (as a module, per the
sweep's `command:`), with the sweep's hyperparameters injected into
wandb.config. Any hyperparameter the sweep doesn't set falls back to
params.yaml's train.lightgbm_l1 defaults, so this also runs standalone
(e.g. for local testing with WANDB_MODE=disabled) without a sweep agent.

Usage (after creating the sweep):
    uv run wandb sweep sweeps/lightgbm_l1_sweep.yaml
    uv run wandb agent <sweep_id>

Local smoke test (no sweep, no network):
    WANDB_MODE=disabled uv run python -m src.models.sweep_train
"""

from __future__ import annotations

import lightgbm as lgb
import wandb

from src.models.train import (
    clip_and_backtransform,
    load_feature_spec,
    load_split,
    prep_lightgbm_features,
    regression_metrics,
)
from src.models.tracking import ensure_wandb_login, lineage_config, load_params

LGBM_FIXED_PARAMS = ["n_estimators", "early_stopping_rounds", "objective"]


def merged_lightgbm_params(defaults: dict, sweep_overrides: dict) -> dict:
    """lightgbm_l1 params from params.yaml, with any key the sweep actually
    set (wandb.config) taking precedence - so a standalone run (no sweep
    agent, empty wandb.config) still gets sane defaults."""
    return {**defaults, **sweep_overrides}


def main() -> int:
    ensure_wandb_login()
    params = load_params()
    train_params = params["train"]
    project = train_params["wandb_project"]
    target = train_params["target"]
    random_state = train_params["random_state"]

    run = wandb.init(project=project, job_type="sweep")
    wandb.config.update(lineage_config())

    merged = merged_lightgbm_params(train_params["lightgbm_l1"], dict(wandb.config))
    wandb.config.update(merged)

    feature_spec = load_feature_spec()
    numeric_features = feature_spec["numeric_features"]
    categorical_features = feature_spec["categorical_features"]
    feature_columns = categorical_features + numeric_features

    train_df = load_split("train")
    val_df = load_split("val")

    X_train = prep_lightgbm_features(train_df, feature_columns, categorical_features)
    train_categories = {col: X_train[col].cat.categories for col in categorical_features}
    X_val = prep_lightgbm_features(val_df, feature_columns, categorical_features, train_categories)

    y_train = train_df[target]
    y_val = val_df[target]

    lgbm_kwargs = {k: v for k, v in merged.items() if k not in LGBM_FIXED_PARAMS}
    model = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=merged["n_estimators"],
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        **lgbm_kwargs,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=merged["early_stopping_rounds"], verbose=False)],
    )

    val_pred = clip_and_backtransform(model.predict(X_val), log_target=False)
    metrics = regression_metrics(y_val, val_pred, "val")
    wandb.log(metrics)
    print(
        f"val_mae={metrics['val_mae']:.3f} val_rmse={metrics['val_rmse']:.3f} "
        f"val_median_ae={metrics['val_median_ae']:.3f}"
    )

    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
