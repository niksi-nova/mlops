"""Verify the served dashboard and its embedded evidence snapshot."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from scripts.build_site_data import build_dashboard
from src.api.main import app
from src.monitoring.report import validate_report

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/api/static"


def test_dashboard_page_and_navigation():
    client = TestClient(app)
    response = client.get("/static/dashboard.html")
    assert response.status_code == 200
    assert "<noscript>" in response.text
    for page in STATIC.glob("*.html"):
        assert 'href="/static/dashboard.html"' in page.read_text()
    for asset in ["css/dashboard.css", "js/dashboard.js", "data/dashboard.json"]:
        assert client.get("/static/" + asset).status_code == 200


def test_dashboard_build_is_deterministic_and_matches_evidence():
    snapshot = json.loads((STATIC / "data/dashboard.json").read_text())
    assert json.dumps(build_dashboard()) == json.dumps(build_dashboard())
    assert snapshot == build_dashboard()
    validate_report(snapshot["drift_report"])
    assert set(snapshot["version_ledger"]["layers"]) == {"code", "data", "model", "image", "config"}
    assert snapshot["version_ledger"]["registry"]["versions"]
    for v in snapshot["version_ledger"]["model_versions"]:
        assert {"artifact", "version", "aliases", "digest"} <= v.keys()
    for c in snapshot["governance_status"]["controls"]:
        assert {"id", "name", "risk", "evidence", "status"} <= c.keys()
        assert c["status"] in {"PASS", "FAIL", "MANUAL", "PLANNED"}
    for c in snapshot["quality_gate"]["checks"].values():
        assert {"value", "operator", "passed", "threshold"} <= c.keys()
    assert all("mae" in m for m in snapshot["test_metrics"]["models"].values())
    assert snapshot["train_metrics"]["models"].keys() == snapshot["test_metrics"]["models"].keys()
    assert all("promoted_version" in row for row in snapshot["production_approvals"])
    assert "production" in snapshot["model_registry"]["active_aliases"]
    assert len(snapshot["stages"]) == 12
    for row in snapshot["stages"]:
        assert row["status"] and (ROOT / row["evidence"]).is_file()


def test_whole_site_build_repeats_byte_for_byte(tmp_path, monkeypatch):
    from scripts import build_site_data as builder
    import pytest

    required = ["train.parquet", "test.parquet", "feature_spec.json"]
    if not all((builder.PROCESSED / name).is_file() for name in required):
        pytest.skip("Full site build requires the DVC feature spec and parquet splits")
    monkeypatch.setattr(builder, "DATA_OUT", tmp_path)
    assert builder.main() == 0
    first = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert "dashboard.json" in first
    assert builder.main() == 0
    assert first == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
