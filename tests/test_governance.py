"""Unit tests for scripts/check_governance.py.

Guarantees schema compliance of reports/governance_status.json, dynamic control
evaluation, valid status categorization (PASS, FAIL, MANUAL, PLANNED), and basic
secrets scanning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.check_governance import (
    check_governance,
    evaluate_controls,
    main,
    scan_for_secrets,
)


def test_check_governance_schema():
    """Verify check_governance produces exact dictionary structure expected by dashboard."""
    report = check_governance()

    assert "generated_at" in report
    assert "controls" in report
    assert "summary" in report

    summary = report["summary"]
    assert "pass" in summary
    assert "fail" in summary
    assert "manual" in summary
    assert "planned" in summary

    allowed_statuses = {"PASS", "FAIL", "MANUAL", "PLANNED"}
    controls = report["controls"]
    assert len(controls) >= 8

    for c in controls:
        assert "id" in c
        assert "name" in c
        assert "risk" in c
        assert "evidence" in c
        assert "status" in c
        assert "detail" in c
        assert c["status"] in allowed_statuses


def test_check_governance_summary_matches_controls():
    """Verify summary counts match exact breakdown of control statuses."""
    report = check_governance()
    controls = report["controls"]
    summary = report["summary"]

    pass_count = sum(1 for c in controls if c["status"] == "PASS")
    fail_count = sum(1 for c in controls if c["status"] == "FAIL")
    manual_count = sum(1 for c in controls if c["status"] == "MANUAL")
    planned_count = sum(1 for c in controls if c["status"] == "PLANNED")

    assert summary["pass"] == pass_count
    assert summary["fail"] == fail_count
    assert summary["manual"] == manual_count
    assert summary["planned"] == planned_count
    assert sum(summary.values()) == len(controls)


def test_secrets_scanner_detects_credentials(tmp_path: Path):
    """Verify regex scanner catches unredacted API key patterns and passes clean files."""
    test_file = tmp_path / "leak.py"
    test_file.write_text('WANDB_API_KEY = "abc123def456abc123def456abc12345"\n', encoding="utf-8")

    clean, msg = scan_for_secrets(tmp_path)
    assert not clean
    assert "leak.py" in msg

    # Clean file test
    test_file.write_text('WANDB_API_KEY = os.environ.get("WANDB_API_KEY")\n', encoding="utf-8")
    clean, msg = scan_for_secrets(tmp_path)
    assert clean
    assert "zero unredacted keys detected" in msg


def test_check_governance_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify scripts.check_governance CLI writes valid JSON and exits appropriately."""
    fake_output = tmp_path / "reports" / "governance_status.json"
    fake_output.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.check_governance.OUTPUT_PATH", fake_output)
    monkeypatch.setattr("scripts.check_governance.REPORTS_DIR", fake_output.parent)

    rc = main()
    assert rc == 0
    assert fake_output.exists()

    content = json.loads(fake_output.read_text(encoding="utf-8"))
    assert content["summary"]["pass"] >= 7
