"""Unit tests for scripts/build_version_ledger.py.

Guarantees exact schema compliance, offline fallback without live network or
W&B dependencies, and 5-layer provenance tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.build_version_ledger import (
    build_version_ledger,
    fetch_wandb_ledger_data,
    get_dvc_md5,
    get_git_info,
    get_local_model_versions,
    get_params_md5,
    main,
)


def test_build_version_ledger_schema():
    """Verify version ledger produces exact dictionary structure expected by dashboard."""
    ledger = build_version_ledger()

    # 1. Top-level keys
    assert "generated_at" in ledger
    assert "layers" in ledger
    assert "model_versions" in ledger
    assert "registry" in ledger
    assert isinstance(ledger["model_versions"], list)
    assert isinstance(ledger["registry"], dict)

    # 2. Layers schema
    layers = ledger["layers"]
    assert "code" in layers
    assert "data" in layers
    assert "model" in layers
    assert "image" in layers
    assert "config" in layers

    assert "git_commit" in layers["code"]
    assert "branch" in layers["code"]
    assert layers["data"]["dataset_version"] == "delhi-data-v2"
    assert "dvc_md5" in layers["data"]
    assert layers["model"]["artifact"] == "delhi-lightgbm_l1"
    assert layers["image"]["registry"] == "ghcr.io/niksi-nova/mlops"
    assert "latest" in layers["image"]["tags"]
    assert "params_md5" in layers["config"]

    # 3. Model versions list
    versions = ledger["model_versions"]
    assert len(versions) >= 3
    ver_names = [v["version"] for v in versions]
    assert "v2" in ver_names
    assert "v1" in ver_names
    assert "v0" in ver_names

    for v in versions:
        assert "artifact" in v
        assert "version" in v
        assert "aliases" in v
        assert "digest" in v
        assert "created_at" in v
        assert "test_mae" in v
        assert isinstance(v["test_mae"], (int, float))
        assert "git_commit" in v
        assert "dataset_version" in v

    # 4. Registry collection
    reg = ledger["registry"]
    assert reg["collection"] == "delhi-bus-delay-model"
    assert isinstance(reg["versions"], list)
    assert len(reg["versions"]) >= 1
    v0_entry = reg["versions"][0]
    assert v0_entry["version"] == "v0"
    assert "production" in v0_entry["aliases"]
    assert "delhi-lightgbm_l1" in v0_entry["source"]


def test_version_ledger_offline_fallback(monkeypatch: pytest.MonkeyPatch):
    """Verify ledger operates offline without WANDB_API_KEY and sets source='local_reports'."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert fetch_wandb_ledger_data("any", "any", "any") is None

    ledger = build_version_ledger()
    assert ledger.get("source") == "local_reports"

    # Differences across versions are documented without invention
    versions = {v["version"]: v for v in ledger["model_versions"]}
    assert versions["v2"]["test_mae"] < versions["v0"]["test_mae"]
    assert versions["v2"]["git_commit"] != versions["v0"]["git_commit"]


def test_get_helpers_return_valid_strings():
    """Verify helper functions return non-empty strings without raising."""
    full_c, short_c, branch = get_git_info()
    assert len(full_c) > 0
    assert len(short_c) > 0
    assert len(branch) > 0

    dvc_hash = get_dvc_md5()
    assert len(dvc_hash) == 32

    params_hash = get_params_md5()
    assert len(params_hash) == 32


def test_build_version_ledger_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify scripts.build_version_ledger CLI writes valid JSON and exits with 0."""
    fake_output = tmp_path / "reports" / "version_ledger.json"
    fake_output.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.build_version_ledger.OUTPUT_PATH", fake_output)
    monkeypatch.setattr("scripts.build_version_ledger.REPORTS_DIR", fake_output.parent)

    rc = main()
    assert rc == 0
    assert fake_output.exists()

    content = json.loads(fake_output.read_text(encoding="utf-8"))
    assert content["layers"]["data"]["dataset_version"] == "delhi-data-v2"
