"""Check the shipped demo even when CI has no model or parquet files."""
import json
from pathlib import Path

from src.monitoring.report import GROUPS, build_summary, validate_report


def test_shipped_drift_report():
    reports = Path(__file__).resolve().parents[1] / "reports"
    report = json.loads((reports / "drift_report.json").read_text())
    reference = json.loads((reports / "drift_reference.json").read_text())
    validate_report(report)
    assert report["simulated"] is True
    assert report["reference"]["n_rows"] == reference["n_rows"] == 606967
    assert report["summary"] == build_summary(report["scenarios"])
    assert report["summary"]["false_positive_check"] == "PASS"
    for scenario in report["scenarios"]:
        assert scenario["windows"]
        if scenario["induced"]:
            assert scenario["label"] == "SIMULATED"
            assert {"type", "params", "seed"} <= scenario["perturbation"].keys()
        for window in scenario["windows"]:
            features = {row["feature"]: row for row in window["features"]}
            assert features.keys() == reference["features"].keys() == GROUPS.keys()
            assert {name for name, row in features.items() if row["alerting_excluded"]} == {
                "relative_weekday", "te_vehicle"
            }
            for name in ("relative_weekday", "te_vehicle"):
                assert features[name]["ks_stat"] is not None
                assert features[name]["psi"] >= 0
