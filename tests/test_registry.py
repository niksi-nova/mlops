"""Unit tests for Model Registry management and model loading interface.

Tests candidate logging, staging promotion gated on quality checks, strict human-only
production promotion requirements (interactive terminal check, exact confirmation phrase),
rejection of invalidated approvals, and loading error handling without requiring live
W&B credentials or network access.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.baselines import GlobalMedianRegressor
from src.models.registry import (
    DEFAULT_COLLECTION,
    ProductionModel,
    load_model,
    main,
    promote_to_production,
    register_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(autouse=True)
def isolate_approval_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Isolate approval history reads and writes to a temporary path for every test.

    Guarantees that test executions (even those exercising interactive promotion)
    never read or modify the real tracked reports/production_approvals.json file.
    """
    import src.models.registry as registry_module

    temporary_path = tmp_path / "production_approvals.json"
    temporary_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        registry_module,
        "APPROVALS_HISTORY_PATH",
        temporary_path,
    )


@pytest.fixture
def dummy_spec(tmp_path: Path) -> dict[str, Any]:
    spec = {
        "features": ["feat1", "cat1"],
        "numeric_features": ["feat1"],
        "categorical_features": ["cat1"],
    }
    spec_path = tmp_path / "feature_spec.json"
    spec_path.write_text(json.dumps(spec))
    return spec


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> Path:
    dummy_model = GlobalMedianRegressor()
    model_path = tmp_path / "lightgbm_l1.joblib"
    joblib.dump(dummy_model, model_path)
    return model_path


def test_load_model_from_direct_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dummy_model = GlobalMedianRegressor()
    model_file = tmp_path / "dummy.joblib"
    joblib.dump(dummy_model, model_file)

    spec_path = tmp_path / "feature_spec.json"
    spec_path.write_text(json.dumps({"categorical_features": ["c1"], "numeric_features": ["n1"], "features": ["c1", "n1"]}))
    monkeypatch.setattr("src.models.registry.FEATURE_SPEC_PATH", spec_path)

    loaded = load_model(str(model_file))
    assert isinstance(loaded, ProductionModel)
    assert isinstance(loaded.model, GlobalMedianRegressor)
    assert loaded.alias == "custom_path"


def test_load_model_staging_succeeds():
    """Verify load_model('staging') loads the quality-gate-approved staging model."""
    loaded = load_model("staging")
    assert isinstance(loaded, ProductionModel)
    assert loaded.alias == "staging"
    assert len(loaded.expected_feature_columns) == 25
    assert loaded.expected_feature_columns == loaded.categorical_features + loaded.numeric_features


def test_load_model_production_raises_error_when_unassigned():
    """Verify load_model('production') strictly fails when production is not actively approved."""
    with pytest.raises(RuntimeError, match="No model is currently assigned to the 'production' alias"):
        load_model("production")


def test_invalidated_approval_history_cannot_activate_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that a historical approval marked INVALIDATED cannot be loaded as production."""
    fake_report = tmp_path / "model_registry.json"
    invalidated_record = {
        "mode": "local_fallback",
        "model_id": "lightgbm_l1",
        "artifact_version": "v1",
        "assigned_aliases": ["candidate", "staging", "latest"],
        "active_aliases": {"candidate": "v1", "staging": "v1", "production": "v1"},
        "production_approval_history": [
            {
                "status": "INVALIDATED",
                "approved_by": "Asrith Alapati",
                "timestamp": "2026-11-07T13:07:36.011233+00:00",
                "invalidated_at": "2026-11-07T13:48:00+00:00",
                "invalidation_reason": "Automated promotion executed by assistant without authorization",
                "promoted_version": "v1",
            }
        ],
    }
    fake_report.write_text(json.dumps(invalidated_record))
    monkeypatch.setattr("src.models.registry.REGISTRY_REPORT_PATH", fake_report)

    with pytest.raises(RuntimeError, match="has been INVALIDATED and is not authorized"):
        load_model("production")


def test_register_model_fails_when_gate_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_path = tmp_path / "nonexistent_quality_gate.json"
    monkeypatch.setattr("src.models.registry.QUALITY_GATE_PATH", fake_path)

    with pytest.raises(RuntimeError, match="Quality gate report not found"):
        register_model()


def test_register_model_fails_when_gate_not_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_path = tmp_path / "failed_gate.json"
    fake_path.write_text(json.dumps({"passed": False, "failure_reasons": ["MAE too high"]}))
    monkeypatch.setattr("src.models.registry.QUALITY_GATE_PATH", fake_path)

    with pytest.raises(ValueError, match="Cannot promote model to staging"):
        register_model()


def test_normal_registration_never_assigns_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify normal registration assigns candidate & staging, but NEVER production."""
    fake_gate = tmp_path / "passed_gate.json"
    fake_gate.write_text(json.dumps({"passed": True, "model_id": "lightgbm_l1"}))
    monkeypatch.setattr("src.models.registry.QUALITY_GATE_PATH", fake_gate)

    fake_report = tmp_path / "model_registry.json"
    monkeypatch.setattr("src.models.registry.REGISTRY_REPORT_PATH", fake_report)

    dummy_model = GlobalMedianRegressor()
    dummy_model_path = tmp_path / "lightgbm_l1.joblib"
    joblib.dump(dummy_model, dummy_model_path)
    monkeypatch.setattr("src.models.registry.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.registry.BEST_MODEL_PATH", dummy_model_path)

    spec_path = tmp_path / "feature_spec.json"
    spec_path.write_text(json.dumps({"categorical_features": ["c1"], "numeric_features": ["n1"], "features": ["c1", "n1"]}))
    monkeypatch.setattr("src.models.registry.FEATURE_SPEC_PATH", spec_path)

    # Mock W&B calls
    class FakeArtifact:
        def __init__(self, name, type, description="", metadata=None):
            self.name = name
            self.type = type
            self.description = description
            self.metadata = metadata or {}
            self.version = "v1"

        def add_file(self, path):
            pass

    class FakeRun:
        def __init__(self):
            self.linked: list[tuple] = []

        def log_artifact(self, art):
            pass

        def link_artifact(self, art, target_path, aliases):
            self.linked.append((art, target_path, aliases))

        def finish(self):
            pass

    fake_run = FakeRun()
    monkeypatch.setattr("src.models.registry.wandb.Artifact", FakeArtifact)
    monkeypatch.setattr("src.models.registry.wandb.init", lambda **kwargs: fake_run)
    monkeypatch.setattr("src.models.registry.ensure_wandb_login", lambda: True)
    monkeypatch.setattr("src.models.registry.lineage_config", lambda: {"git_commit": "abc1234", "git_branch": "main"})

    record = register_model(collection_name="delhi-bus-delay-model")

    assert record["collection"] == "delhi-bus-delay-model"
    assert record["model_id"] == "lightgbm_l1"
    # Candidate and staging assigned; production NOT assigned
    assert "candidate" in record["assigned_aliases"]
    assert "staging" in record["assigned_aliases"]
    assert "latest" in record["assigned_aliases"]
    assert "production" not in record["assigned_aliases"]
    assert record["production_auto_promoted"] is False
    assert record["active_aliases"].get("production") is None


def test_quality_gate_pass_permits_staging_not_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that passing the quality gate permits staging alias, but never production."""
    fake_gate = tmp_path / "passed_gate.json"
    fake_gate.write_text(json.dumps({"passed": True, "model_id": "lightgbm_l1"}))
    monkeypatch.setattr("src.models.registry.QUALITY_GATE_PATH", fake_gate)

    fake_report = tmp_path / "model_registry.json"
    monkeypatch.setattr("src.models.registry.REGISTRY_REPORT_PATH", fake_report)

    dummy_model = GlobalMedianRegressor()
    dummy_model_path = tmp_path / "lightgbm_l1.joblib"
    joblib.dump(dummy_model, dummy_model_path)
    monkeypatch.setattr("src.models.registry.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.registry.BEST_MODEL_PATH", dummy_model_path)

    spec_path = tmp_path / "feature_spec.json"
    spec_path.write_text(json.dumps({"categorical_features": ["c1"], "numeric_features": ["n1"], "features": ["c1", "n1"]}))
    monkeypatch.setattr("src.models.registry.FEATURE_SPEC_PATH", spec_path)
    monkeypatch.setattr("src.models.registry.ensure_wandb_login", lambda: False)

    record = register_model()
    assert record["active_aliases"]["staging"] == "v1"
    assert "production" not in record["active_aliases"]


def test_promote_to_production_requires_approver():
    """Verify that promote_to_production rejects missing or empty approver string."""
    with pytest.raises(ValueError, match="approver identifier"):
        promote_to_production(approved_by="")
    with pytest.raises(ValueError, match="approver identifier"):
        promote_to_production(approved_by="   ")


def test_promote_to_production_fails_without_interactive_terminal(monkeypatch: pytest.MonkeyPatch):
    """Verify promote_to_production aborts if not in an interactive terminal."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(RuntimeError, match="Interactive human terminal confirmation is required"):
        promote_to_production(approved_by="Human Lead")


def test_promote_to_production_fails_with_wrong_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify promote_to_production aborts if typed confirmation string does not match exactly."""
    fake_report = tmp_path / "model_registry.json"
    initial_record = {
        "mode": "local_fallback",
        "collection": "delhi-bus-delay-model",
        "model_id": "lightgbm_l1",
        "artifact_version": "v1",
        "assigned_aliases": ["candidate", "staging", "latest"],
        "active_aliases": {"candidate": "v1", "staging": "v1", "latest": "v1"},
        "quality_gate_passed": True,
    }
    fake_report.write_text(json.dumps(initial_record))
    monkeypatch.setattr("src.models.registry.REGISTRY_REPORT_PATH", fake_report)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes promote please")

    with pytest.raises(ValueError, match="Confirmation phrase mismatch"):
        promote_to_production(approved_by="Human Lead")

    # Confirm registry file was NOT modified to include production
    updated_record = json.loads(fake_report.read_text())
    assert "production" not in updated_record["active_aliases"]


def test_promote_to_production_succeeds_with_correct_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify explicit interactive confirmation successfully promotes to production and audits."""
    fake_report = tmp_path / "model_registry.json"
    initial_record = {
        "mode": "local_fallback",
        "collection": "delhi-bus-delay-model",
        "model_id": "lightgbm_l1",
        "artifact_version": "v1",
        "assigned_aliases": ["candidate", "staging", "latest"],
        "active_aliases": {"candidate": "v1", "staging": "v1", "latest": "v1"},
        "quality_gate_passed": True,
    }
    fake_report.write_text(json.dumps(initial_record))
    monkeypatch.setattr("src.models.registry.REGISTRY_REPORT_PATH", fake_report)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "PROMOTE lightgbm_l1 v1 TO PRODUCTION")

    updated = promote_to_production(approved_by="Authorized Lead", reason="Approved after review")

    assert updated["active_aliases"]["production"] == "v1"
    assert "production" in updated["assigned_aliases"]
    assert len(updated["production_approval_history"]) == 1
    approval = updated["production_approval_history"][0]
    assert approval["status"] == "APPROVED"
    assert approval["approved_by"] == "Authorized Lead"
    assert approval["confirmation_phrase"] == "PROMOTE lightgbm_l1 v1 TO PRODUCTION"


def test_normal_dvc_pipeline_cannot_promote_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify standard CLI pipeline execution (uv run python -m src.models.registry) never promotes production."""
    fake_gate = tmp_path / "quality_gate.json"
    fake_gate.write_text(json.dumps({"passed": True, "model_id": "lightgbm_l1"}))
    monkeypatch.setattr("src.models.registry.QUALITY_GATE_PATH", fake_gate)

    fake_report = tmp_path / "model_registry.json"
    monkeypatch.setattr("src.models.registry.REGISTRY_REPORT_PATH", fake_report)

    dummy_model = GlobalMedianRegressor()
    dummy_model_path = tmp_path / "lightgbm_l1.joblib"
    joblib.dump(dummy_model, dummy_model_path)
    monkeypatch.setattr("src.models.registry.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.models.registry.BEST_MODEL_PATH", dummy_model_path)

    spec_path = tmp_path / "feature_spec.json"
    spec_path.write_text(json.dumps({"categorical_features": ["c1"], "numeric_features": ["n1"], "features": ["c1", "n1"]}))
    monkeypatch.setattr("src.models.registry.FEATURE_SPEC_PATH", spec_path)
    monkeypatch.setattr("src.models.registry.ensure_wandb_login", lambda: False)

    monkeypatch.setattr(sys, "argv", ["src.models.registry"])
    rc = main()
    assert rc == 0

    record = json.loads(fake_report.read_text())
    assert "production" not in record["active_aliases"]
    assert record["production_auto_promoted"] is False


def test_prediction_equivalence_and_wrapper_contract():
    """Verify loaded staging ProductionModel wraps predictions faithfully and exposes expected properties."""
    from src.models.evaluate import predict as real_predict

    # Use actual staging model and feature spec from project
    model = load_model("staging")
    assert isinstance(model, ProductionModel)
    assert len(model.expected_feature_columns) == 25
    assert model.expected_feature_columns == model.categorical_features + model.numeric_features

    # Load 5 sample rows from test dataset
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet").head(5)

    # 1. Prediction via wrapper method
    preds_wrapper = model.predict(df)

    # 2. Prediction via raw evaluate.predict function
    preds_direct = real_predict(
        "lightgbm_l1",
        model.model,
        df,
        feature_columns=model.expected_feature_columns,
        categorical_features=model.categorical_features,
    )

    np.testing.assert_array_equal(preds_wrapper, preds_direct)
    assert len(preds_wrapper) == 5
    assert not np.isnan(preds_wrapper).any()


def test_real_tracked_approval_file_remains_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regression test proving that real tracked approval reports are never modified during tests."""
    from src.models.registry import APPROVALS_HISTORY_PATH

    real_approvals_path = PROJECT_ROOT / "reports" / "production_approvals.json"
    real_approvals_content = real_approvals_path.read_text(encoding="utf-8") if real_approvals_path.exists() else None

    real_registry_path = PROJECT_ROOT / "reports" / "model_registry.json"
    real_registry_content = real_registry_path.read_text(encoding="utf-8") if real_registry_path.exists() else None

    # Simulate an interactive promotion attempt in test environment
    fake_report = tmp_path / "model_registry.json"
    fake_report.write_text(
        json.dumps({
            "mode": "local_fallback",
            "model_id": "lightgbm_l1",
            "artifact_version": "v1",
            "assigned_aliases": ["candidate", "staging", "latest"],
            "active_aliases": {"candidate": "v1", "staging": "v1", "latest": "v1"},
            "quality_gate_passed": True,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.models.registry.REGISTRY_REPORT_PATH", fake_report)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "PROMOTE lightgbm_l1 v1 TO PRODUCTION")

    # Run promotion inside test
    promote_to_production(approved_by="Test Human", reason="Regression test for path isolation")

    # The temporary isolated file must receive the update
    temp_approvals = json.loads(APPROVALS_HISTORY_PATH.read_text(encoding="utf-8"))
    assert len(temp_approvals) == 1
    assert temp_approvals[0]["approved_by"] == "Test Human"

    # The real tracked approval files must be byte-for-byte unchanged
    if real_approvals_content is not None:
        assert real_approvals_path.read_text(encoding="utf-8") == real_approvals_content
    else:
        assert not real_approvals_path.exists()

    if real_registry_content is not None:
        assert real_registry_path.read_text(encoding="utf-8") == real_registry_content
