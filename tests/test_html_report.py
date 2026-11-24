import json
import os

from src.monitoring.html_report import render_html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_render_html_contains_simulated_and_scenarios():
    with open(os.path.join(ROOT, "reports", "drift_report.json")) as f:
        rep = json.load(f)
    page = render_html(rep)
    assert "SIMULATED" in page
    for s in rep["scenarios"]:
        assert s["name"] in page or s["label"] in page
