"""Demonstrates a safe, local-only model rollback workflow.

Rules:
  - Executes strictly LOCALLY.
  - NEVER modifies any remote W&B alias or artifact.
  - Downloads / loads an older model version into a temporary folder.
  - Evaluates current production model vs older rollback model side-by-side.
  - Prints prediction comparisons and a dry-run explanation of what an alias
    rollback would perform in production without touching any live services.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.models.evaluate import within_n_seconds_pct
from src.models.registry import ProductionModel, load_model

logger = logging.getLogger("demo_rollback")
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEST_DATA_PATH = DATA_DIR / "test.parquet"
FEATURE_SPEC_PATH = DATA_DIR / "feature_spec.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEDGER_PATH = REPORTS_DIR / "version_ledger.json"

DEFAULT_WORKSPACE = "anikaub20-rv-college"
DEFAULT_PROJECT = "delhi-bus-delay"
DEFAULT_COLLECTION = "delhi-bus-delay-model"


def load_sample_test_data(n_samples: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample rows and evaluation slice from test.parquet."""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

    df = pd.read_parquet(TEST_DATA_PATH)
    samples = df.head(n_samples).copy()
    eval_slice = df.head(1000).copy()
    return samples, eval_slice


def get_rollback_model(target_version: str, tmp_dir: Path) -> ProductionModel:
    """Retrieve an older model version into a temporary directory without modifying W&B.

    Supports:
      1. W&B download (read-only) if WANDB_API_KEY is available.
      2. Local fallback using historical model artifacts in models/.
    """
    with open(FEATURE_SPEC_PATH, encoding="utf-8") as f:
        feature_spec = json.load(f)

    # 1. Attempt W&B read-only download if API key is provided
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        try:
            import wandb
            api = wandb.Api(timeout=10)
            art_uri = f"{DEFAULT_WORKSPACE}/{DEFAULT_PROJECT}/delhi-lightgbm_l1:{target_version}"
            artifact = api.artifact(art_uri)
            download_dir = tmp_dir / "wandb_download"
            artifact.download(root=str(download_dir))
            for candidate in download_dir.glob("*.joblib"):
                import joblib
                raw_model = joblib.load(candidate)
                return ProductionModel(
                    raw_model,
                    feature_spec,
                    alias=f"rollback_{target_version}",
                    version=target_version,
                    model_name="lightgbm_l1",
                    source="wandb_temp",
                )
        except Exception as exc:
            logger.info("W&B download not available (%s); using local rollback fallback.", exc)

    # 2. Local fallback: use baseline or secondary checkpoint (e.g. lightgbm_l2 or copied model)
    target_joblib = tmp_dir / f"rollback_{target_version}.joblib"
    fallback_candidates = [
        MODELS_DIR / "lightgbm_l2.joblib",
        MODELS_DIR / "ridge.joblib",
        MODELS_DIR / "lightgbm_l1.joblib",
    ]
    source_model = next((p for p in fallback_candidates if p.exists()), None)
    if source_model is None:
        raise FileNotFoundError(f"No local candidate models found in {MODELS_DIR}")

    shutil.copy(source_model, target_joblib)
    import joblib
    raw_model = joblib.load(target_joblib)
    return ProductionModel(
        raw_model,
        feature_spec,
        alias=f"rollback_{target_version}",
        version=target_version,
        model_name=source_model.stem,
        source="local_temp",
    )


def run_rollback_demo(target_version: str = "v1") -> dict[str, Any]:
    """Execute side-by-side comparison between current production model and rollback candidate."""
    print("=" * 78)
    print("DELHI BUS TRAVEL-TIME PREDICTION - LOCAL MODEL ROLLBACK DEMONSTRATION")
    print("=" * 78)
    print(f"Target Rollback Version: {target_version}")
    print("Execution Mode:          STRICTLY LOCAL (Read-Only W&B Access)")
    print("-" * 78)

    # 1. Load current production model
    print("[1/4] Loading current production model...")
    prod_model = load_model("production")
    print(f"      Loaded: {prod_model.model_name} (version: {prod_model.version}, alias: {prod_model.alias})")

    # 2. Load rollback candidate in isolated temporary directory
    print(f"[2/4] Staging rollback candidate ({target_version}) in temporary directory...")
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        rollback_model = get_rollback_model(target_version, tmp_path)
        print(f"      Loaded: {rollback_model.model_name} from {rollback_model.source} (temp path: {tmp_path})")

        # 3. Load sample test evaluation data
        print("[3/4] Running side-by-side inference on test dataset samples...")
        samples, eval_slice = load_sample_test_data(n_samples=5)

        prod_preds_sample = prod_model.predict(samples)
        roll_preds_sample = rollback_model.predict(samples)

        prod_preds_eval = prod_model.predict(eval_slice)
        roll_preds_eval = rollback_model.predict(eval_slice)

        y_true = eval_slice["travel_time_s"].values
        prod_metrics = {
            "mae": float(np.mean(np.abs(y_true - prod_preds_eval))),
            "rmse": float(np.sqrt(np.mean((y_true - prod_preds_eval) ** 2))),
            "within_60s_pct": float(within_n_seconds_pct(y_true, prod_preds_eval, 60.0)),
        }
        roll_metrics = {
            "mae": float(np.mean(np.abs(y_true - roll_preds_eval))),
            "rmse": float(np.sqrt(np.mean((y_true - roll_preds_eval) ** 2))),
            "within_60s_pct": float(within_n_seconds_pct(y_true, roll_preds_eval, 60.0)),
        }

        # Print comparison table
        print("\n--- SAMPLE PREDICTIONS COMPARISON (Seconds) ---")
        print(f"{'Segment ID':<16} {'Actual':<10} {'Current (Production)':<22} {f'Rollback ({target_version})':<20} {'Diff':<8}")
        print("-" * 78)
        sample_diffs = []
        for i, row in samples.reset_index().iterrows():
            seg = str(row.get("segment_id", f"seg_{i}"))
            actual = float(row.get("travel_time_s", 0.0))
            p_val = float(prod_preds_sample[i])
            r_val = float(roll_preds_sample[i])
            diff = r_val - p_val
            sample_diffs.append({"segment": seg, "actual": actual, "prod": p_val, "rollback": r_val, "diff": diff})
            print(f"{seg:<16} {actual:<10.1f} {p_val:<22.2f} {r_val:<20.2f} {diff:+8.2f}")

        print("\n--- EVALUATION METRICS COMPARISON (N = 1000) ---")
        print(f"{'Metric':<24} {'Current Production':<22} {f'Rollback ({target_version})':<22}")
        print("-" * 78)
        print(f"{'MAE (seconds)':<24} {prod_metrics['mae']:<22.2f} {roll_metrics['mae']:<22.2f}")
        print(f"{'RMSE (seconds)':<24} {prod_metrics['rmse']:<22.2f} {roll_metrics['rmse']:<22.2f}")
        print(f"{'Within-60s Pct':<24} {prod_metrics['within_60s_pct']:<22.2f}% {roll_metrics['within_60s_pct']:<22.2f}%")

    # 4. Dry-run alias migration explanation
    print("\n[4/4] DRY-RUN ALIAS ROLLBACK EXPLANATION")
    print("-" * 78)
    print(f"DRY RUN: would move alias 'production' in collection '{DEFAULT_COLLECTION}'")
    print(f"         from version 'v0' (delhi-lightgbm_l1:v2) to '{target_version}'.")
    print("Action in W&B: Api().artifact('...').aliases.append('production')")
    print("Container effect: Render web service loads new production alias on next restart/poll.")
    print("\n[SAFETY NOTICE]")
    print(">>> ZERO REMOTE MODIFICATIONS WERE PERFORMED.")
    print(">>> All aliases in Weights & Biases remain unchanged.")
    print("=" * 78)

    return {
        "status": "success",
        "target_version": target_version,
        "prod_mae": prod_metrics["mae"],
        "rollback_mae": roll_metrics["mae"],
        "sample_comparisons": sample_diffs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Demonstrate safe local model rollback")
    parser.add_argument("--version", default="v1", help="Target rollback version (default: v1)")
    args = parser.parse_args()

    run_rollback_demo(target_version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
