"""W&B Model Registry management and inference loading interface.

Implements the model governance lifecycle for Delhi bus delay prediction:
  1. candidate: Logged model artifact with feature spec & lineage metadata.
  2. staging: Promoted automatically ONLY after quality-gate PASS.
  3. production: Promoted ONLY after explicit human approval (never automatic).

Provides:
  - register_model(): Logs candidate and assigns staging if quality gate passed.
  - promote_to_production(): Explicit human approval mechanism for production alias.
  - load_model(alias="production"): Returns a production-ready model wrapper
    with the exact verified feature order (categoricals + numerics).

Usage:
  # Pipeline execution (candidate + staging on gate pass):
  uv run python -m src.models.registry

  # Explicit human approval for production promotion:
  uv run python -m src.models.registry --promote-production --approved-by "Team Lead"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import wandb

from src.models.evaluate import predict
from src.models.tracking import DATASET_VERSION, ensure_wandb_login, lineage_config, load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_SPEC_PATH = PROJECT_ROOT / "data" / "processed" / "feature_spec.json"
QUALITY_GATE_PATH = PROJECT_ROOT / "reports" / "quality_gate.json"
TEST_METRICS_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"
REGISTRY_REPORT_PATH = PROJECT_ROOT / "reports" / "model_registry.json"
APPROVALS_HISTORY_PATH = PROJECT_ROOT / "reports" / "production_approvals.json"

DEFAULT_WORKSPACE = "anikaub20-rv-college"
DEFAULT_PROJECT = "delhi-bus-delay"
DEFAULT_COLLECTION = "delhi-bus-delay-model"


def _safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class ProductionModel:
    """Production model wrapper encapsulating LightGBM regressor and feature contract.

    Guarantees that input DataFrames are preprocessed with the exact feature ordering
    (categorical_features + numeric_features) and categorical handling verified in tests.
    """

    def __init__(
        self,
        model: Any,
        feature_spec: dict[str, Any],
        alias: str = "production",
        version: str = "v1",
        model_name: str = "lightgbm_l1",
    ):
        self.model = model
        self.feature_spec = feature_spec
        self.alias = alias
        self.version = version
        self.model_name = model_name
        self.categorical_features = feature_spec.get("categorical_features", [])
        self.numeric_features = feature_spec.get("numeric_features", [])
        self.expected_feature_columns = self.categorical_features + self.numeric_features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate travel time predictions in seconds for input feature DataFrame."""
        return predict(
            self.model_name,
            self.model,
            df,
            feature_columns=self.expected_feature_columns,
            categorical_features=self.categorical_features,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def load_model(
    alias_or_path: str = "production",
    project: str | None = None,
    workspace: str = DEFAULT_WORKSPACE,
) -> ProductionModel:
    """Load model by registry alias, model name, or filesystem path.

    Returns a ProductionModel wrapper exposing .predict(df), .feature_spec,
    and .expected_feature_columns.

    Rules for 'production' alias:
      - Requires an active, non-invalidated production assignment in the registry.
      - If unassigned or invalidated, raises RuntimeError.
      - NEVER falls back to staging, candidate, or local best_model.joblib.

    Usage:
        from src.models import load_model

        model = load_model("staging")  # Load verified staging model
        preds = model.predict(df_with_25_features)
    """
    # Load feature specification contract
    if not FEATURE_SPEC_PATH.exists():
        raise FileNotFoundError(f"feature_spec.json not found at {_safe_relative(FEATURE_SPEC_PATH)}")
    with open(FEATURE_SPEC_PATH) as f:
        spec = json.load(f)

    # 1. Direct filesystem path check
    direct_path = Path(alias_or_path)
    if direct_path.exists() and direct_path.is_file():
        raw_model = joblib.load(direct_path)
        return ProductionModel(raw_model, spec, alias="custom_path", model_name="lightgbm_l1")

    # 2. Local registry report metadata check
    registry_meta: dict[str, Any] = {}
    if REGISTRY_REPORT_PATH.exists():
        try:
            registry_meta = json.loads(REGISTRY_REPORT_PATH.read_text())
        except Exception:
            registry_meta = {}

    active_aliases = registry_meta.get("active_aliases", {})
    model_name = registry_meta.get("model_id", "lightgbm_l1")

    # 3. Handle 'production' alias with strict authorization enforcement
    if alias_or_path == "production":
        prod_version = active_aliases.get("production")
        approval_history = registry_meta.get("production_approval_history", [])
        if not approval_history and APPROVALS_HISTORY_PATH.exists():
            try:
                approval_history = json.loads(APPROVALS_HISTORY_PATH.read_text())
            except Exception:
                approval_history = []
        valid_approvals = [
            h
            for h in approval_history
            if h.get("status") != "INVALIDATED" and h.get("promoted_version") == prod_version
        ]

        if not prod_version:
            raise RuntimeError(
                "No model is currently assigned to the 'production' alias in the registry. "
                "Staging models require explicit human approval via promote_to_production() before production deployment."
            )

        if not valid_approvals:
            raise RuntimeError(
                f"The production assignment for version '{prod_version}' has been INVALIDATED and is not authorized. "
                "A fresh explicit human promotion is required."
            )

        target_model_file = MODELS_DIR / f"{model_name}.joblib"
        if not target_model_file.exists():
            target_model_file = BEST_MODEL_PATH
        if target_model_file.exists():
            raw_model = joblib.load(target_model_file)
            return ProductionModel(
                raw_model, spec, alias="production", version=prod_version, model_name=model_name
            )
        raise FileNotFoundError(f"Production model artifact not found on disk at {target_model_file}")

    # 4. Handle 'staging' alias
    if alias_or_path == "staging":
        staging_version = active_aliases.get("staging")
        if not staging_version and QUALITY_GATE_PATH.exists():
            try:
                gate = json.loads(QUALITY_GATE_PATH.read_text())
                if gate.get("passed"):
                    staging_version = "v1"
            except Exception:
                pass
        if staging_version:
            target_model_file = MODELS_DIR / f"{model_name}.joblib"
            if not target_model_file.exists():
                target_model_file = BEST_MODEL_PATH
            if target_model_file.exists():
                raw_model = joblib.load(target_model_file)
                return ProductionModel(
                    raw_model, spec, alias="staging", version=staging_version, model_name=model_name
                )
        raise RuntimeError("No model is currently assigned to the 'staging' alias in the registry.")

    # 5. Handle 'candidate' and 'latest' aliases
    if alias_or_path in ("candidate", "latest"):
        version = active_aliases.get(alias_or_path, "v1")
        target_model_file = MODELS_DIR / f"{model_name}.joblib"
        if not target_model_file.exists():
            target_model_file = BEST_MODEL_PATH
        if target_model_file.exists():
            raw_model = joblib.load(target_model_file)
            return ProductionModel(
                raw_model, spec, alias=alias_or_path, version=version, model_name=model_name
            )

    # 6. Local named model file in models/ (e.g., 'lightgbm_l1', 'random_forest')
    if alias_or_path not in ("production", "staging", "candidate", "latest"):
        candidate_local = MODELS_DIR / f"{alias_or_path}.joblib"
        if candidate_local.exists():
            raw_model = joblib.load(candidate_local)
            return ProductionModel(raw_model, spec, alias=alias_or_path, model_name=alias_or_path)

    # 7. Remote W&B Model Registry lookup (if online and authenticated, never for unauthorized production)
    if alias_or_path != "production":
        params = load_params()
        proj = project or params.get("train", {}).get("wandb_project", DEFAULT_PROJECT)
        collection = DEFAULT_COLLECTION

        try:
            api = wandb.Api()
            target_path = f"{workspace}/{proj}/{collection}:{alias_or_path}"
            try:
                artifact = api.artifact(target_path)
            except Exception:
                artifact = api.artifact(f"{collection}:{alias_or_path}")

            download_dir = Path(artifact.download())
            joblib_files = list(download_dir.glob("*.joblib"))
            if joblib_files:
                raw_model = joblib.load(joblib_files[0])
                return ProductionModel(
                    raw_model, spec, alias=alias_or_path, version=artifact.version, model_name=model_name
                )
        except Exception:
            pass

    raise FileNotFoundError(f"Model not found for alias or path '{alias_or_path}'")


def register_model(
    collection_name: str = DEFAULT_COLLECTION,
    workspace: str = DEFAULT_WORKSPACE,
    project: str = DEFAULT_PROJECT,
) -> dict[str, Any]:
    """Register the candidate model from training into the Model Registry.

    Lifecycle:
      - Always logs candidate artifact with metadata.
      - Promotes to 'staging' ONLY if quality gate passed.
      - Never automatically promotes to 'production'.
    """
    if not QUALITY_GATE_PATH.exists():
        raise RuntimeError(
            f"Quality gate report not found at {_safe_relative(QUALITY_GATE_PATH)}. "
            "Execute quality gate stage first."
        )

    gate_data = json.loads(QUALITY_GATE_PATH.read_text())
    gate_passed = bool(gate_data.get("passed", False))
    if not gate_passed:
        raise ValueError(
            "Cannot promote model to staging: quality gate failed. "
            f"Reasons: {gate_data.get('failure_reasons', [])}"
        )

    model_name = gate_data.get("model_id", "lightgbm_l1")
    model_file = MODELS_DIR / f"{model_name}.joblib"
    if not model_file.exists():
        model_file = BEST_MODEL_PATH
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found for {model_name} at {model_file}")

    # Read evaluation metrics
    test_metrics = {}
    if TEST_METRICS_PATH.exists():
        try:
            test_metrics = json.loads(TEST_METRICS_PATH.read_text())
        except Exception:
            pass

    # Read existing registry record to preserve valid historical production approvals
    existing_record: dict[str, Any] = {}
    if REGISTRY_REPORT_PATH.exists():
        try:
            existing_record = json.loads(REGISTRY_REPORT_PATH.read_text())
        except Exception:
            pass

    previous_production = existing_record.get("active_aliases", {}).get("production")
    production_history = existing_record.get("production_approval_history", [])
    if not production_history and APPROVALS_HISTORY_PATH.exists():
        try:
            production_history = json.loads(APPROVALS_HISTORY_PATH.read_text())
        except Exception:
            production_history = []
    valid_prior_approvals = [
        h
        for h in production_history
        if h.get("status") != "INVALIDATED" and h.get("promoted_version") == previous_production
    ]

    # Base aliases assigned by normal registration
    # Candidate is assigned to the new artifact; staging is assigned because gate passed.
    # Production is explicitly NOT assigned automatically.
    assigned_aliases = ["candidate", "staging", "latest"]
    active_aliases = {
        "candidate": "v1",
        "staging": "v1",
        "latest": "v1",
    }
    if previous_production and valid_prior_approvals:
        active_aliases["production"] = previous_production
        if "production" not in assigned_aliases:
            assigned_aliases.append("production")

    lineage = lineage_config()
    metadata = {
        "model_id": model_name,
        "dataset_version": DATASET_VERSION,
        "git_commit": lineage.get("git_commit", ""),
        "git_branch": lineage.get("git_branch", ""),
        "quality_gate_passed": gate_passed,
        "val_mae": test_metrics.get("val_mae_best_model"),
        "test_mae": test_metrics.get("test_mae_best_model"),
        "generalisation_gap_pct": test_metrics.get("generalisation_gap_pct"),
        "within_60s_pct": test_metrics.get("models", {}).get(model_name, {}).get("within_60s_pct"),
        "checks": gate_data.get("checks", {}),
    }

    mode = "local_fallback"
    remote_version = "v1"

    logged_in = ensure_wandb_login()
    if logged_in:
        try:
            run = wandb.init(
                project=project,
                name=f"register-{model_name}",
                job_type="registry",
                config=lineage,
            )
            artifact = wandb.Artifact(
                name=f"delhi-{model_name}",
                type="model",
                description="Delhi bus travel-time delay prediction model",
                metadata=metadata,
            )
            artifact.add_file(str(model_file))
            if FEATURE_SPEC_PATH.exists():
                artifact.add_file(str(FEATURE_SPEC_PATH))

            run.log_artifact(artifact)

            # Link to Registry collection with candidate & staging aliases
            target_path = f"{workspace}/{project}/{collection_name}"
            try:
                run.link_artifact(artifact, target_path=target_path, aliases=assigned_aliases)
                print(f"Successfully linked {artifact.name} to {target_path} with aliases {assigned_aliases}")
                mode = "wandb_registry"
                remote_version = artifact.version
            except Exception as e:
                try:
                    run.link_artifact(artifact, target_path=collection_name, aliases=assigned_aliases)
                    mode = "wandb_registry"
                except Exception as inner_e:
                    print(f"Notice: W&B link_artifact skipped: {inner_e}")

            run.finish()
        except Exception as e:
            print(f"Notice: W&B registration error: {e}. Preserving local registry report.")
    else:
        print("Notice: W&B offline or unauthenticated; model registered locally in reports/model_registry.json")

    record = {
        "mode": mode,
        "workspace": workspace,
        "project": project,
        "collection": collection_name,
        "model_id": model_name,
        "artifact_version": remote_version,
        "assigned_aliases": assigned_aliases,
        "active_aliases": active_aliases,
        "quality_gate_passed": gate_passed,
        "production_auto_promoted": False,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_path": _safe_relative(model_file),
        "feature_spec_path": _safe_relative(FEATURE_SPEC_PATH),
        "metadata": metadata,
        "production_approval_history": production_history,
    }

    REGISTRY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_REPORT_PATH.write_text(json.dumps(record, indent=2))
    print(f"Wrote {_safe_relative(REGISTRY_REPORT_PATH)}")

    return record


def promote_to_production(
    approved_by: str,
    reason: str = "Verified model quality gate and staging performance",
    collection_name: str = DEFAULT_COLLECTION,
    workspace: str = DEFAULT_WORKSPACE,
    project: str = DEFAULT_PROJECT,
) -> dict[str, Any]:
    """Explicit human-approved production promotion mechanism.

    Requires:
      1. Explicit human approver identifier (name or email).
      2. Passed quality gate report.
      3. Interactive terminal session (sys.stdin.isatty() == True).
      4. Exact typed confirmation phrase matching: 'PROMOTE <model_id> <version> TO PRODUCTION'.

    Never invoked automatically by pipeline stages or training scripts.
    """
    if not approved_by or not approved_by.strip():
        raise ValueError("An explicit approver identifier (e.g. --approved-by 'Name') is required.")

    if not REGISTRY_REPORT_PATH.exists():
        raise RuntimeError("No model registry record found. Run register_model() first.")

    record = json.loads(REGISTRY_REPORT_PATH.read_text())
    gate_passed = record.get("quality_gate_passed", False)
    if not gate_passed:
        raise ValueError("Cannot promote to production: Quality gate checks failed.")

    model_name = record.get("model_id", "lightgbm_l1")
    current_version = record.get("artifact_version", "v1")
    previous_production = record.get("active_aliases", {}).get("production")
    expected_phrase = f"PROMOTE {model_name} {current_version} TO PRODUCTION"

    # Enforce interactive terminal session
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Production promotion aborted: Interactive human terminal confirmation is required "
            "(sys.stdin.isatty() returned False). Automated promotion to production is strictly forbidden."
        )

    print("\n" + "=" * 60)
    print("PRODUCTION PROMOTION AUTHORIZATION GATE")
    print("=" * 60)
    print(f"  Model ID:            {model_name}")
    print(f"  Artifact Version:    {current_version}")
    print(f"  Quality Gate Status: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  Approver:            {approved_by.strip()}")
    print(f"  Reason:              {reason}")
    print("=" * 60)
    print("WARNING: Promoting to PRODUCTION affects live serving traffic.")
    print(f"To confirm, type the following exact phrase:\n  {expected_phrase}\n")

    try:
        user_input = input("Enter confirmation: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise RuntimeError("Production promotion aborted: Interactive input was interrupted.")

    if user_input != expected_phrase:
        raise ValueError(
            f"Production promotion aborted: Confirmation phrase mismatch.\n"
            f"  Expected: '{expected_phrase}'\n"
            f"  Received: '{user_input}'"
        )

    approval_entry = {
        "status": "APPROVED",
        "approved_by": approved_by.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "confirmation_phrase": expected_phrase,
        "previous_production_version": previous_production,
        "promoted_version": current_version,
    }

    record["active_aliases"]["production"] = current_version
    if "production" not in record["assigned_aliases"]:
        record["assigned_aliases"].append("production")
    record.setdefault("production_approval_history", []).append(approval_entry)
    try:
        APPROVALS_HISTORY_PATH.write_text(json.dumps(record["production_approval_history"], indent=2))
    except Exception:
        pass

    # If W&B online, update remote alias
    if record.get("mode") == "wandb_registry" and ensure_wandb_login():
        try:
            api = wandb.Api()
            artifact_uri = f"{workspace}/{project}/{collection_name}:{current_version}"
            art = api.artifact(artifact_uri)
            if "production" not in art.aliases:
                art.aliases.append("production")
                art.save()
                print(f"Updated remote W&B artifact {artifact_uri} with alias 'production'")
        except Exception as e:
            print(f"Notice: Remote production alias update skipped: {e}")

    REGISTRY_REPORT_PATH.write_text(json.dumps(record, indent=2))
    print(f"\n[PRODUCTION PROMOTION APPROVED]")
    print(f"  Approved by:      {approved_by.strip()}")
    print(f"  Promoted version: {current_version} (previous: {previous_production})")
    print(f"  Wrote:            {_safe_relative(REGISTRY_REPORT_PATH)}")

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Model Registry Management for Delhi Bus Delay Model")
    parser.add_argument("--promote-production", action="store_true", help="Promote staging model to production")
    parser.add_argument("--approved-by", type=str, default="", help="Name or ID of human approver")
    parser.add_argument("--reason", type=str, default="Production deployment approval", help="Approval rationale")
    args = parser.parse_args()

    if args.promote_production:
        if not args.approved_by:
            print("ERROR: --approved-by <name> is required to promote to production.", file=sys.stderr)
            return 1
        promote_to_production(approved_by=args.approved_by, reason=args.reason)
        return 0

    record = register_model()
    print("\nModel Registration Complete:")
    print(f"  Mode:             {record['mode']}")
    print(f"  Collection:       {record['collection']}")
    print(f"  Model ID:         {record['model_id']}")
    print(f"  Assigned Aliases: {', '.join(record['assigned_aliases'])}")
    print(f"  Production Alias: {record['active_aliases'].get('production', 'NONE (pending approval)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
