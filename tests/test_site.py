"""Tests for the demo website (src/api/static/) - no network, no browser.

Covers: every page route returns 200, every static asset a page links to
actually exists, every data JSON file parses and has the keys the pages
read, the embedded presets/trip rows pass PredictionRequest validation,
the numbers shown on /models match reports/test_metrics.json, and the
JavaScript time-feature formulas match Python for all 24 hours.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "src" / "api" / "static"
DATA_DIR = STATIC_DIR / "data"

PAGES = ["/", "/trip", "/data", "/models", "/pipeline", "/about"]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", PAGES)
def test_page_returns_200(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


_ASSET_PATTERN = re.compile(r'(?:href|src)="(/static/[^"]+)"')


@pytest.mark.parametrize("path", PAGES)
def test_every_linked_static_asset_exists(path):
    html_file = {
        "/": "index.html", "/trip": "trip.html", "/data": "data.html",
        "/models": "models.html", "/pipeline": "pipeline.html", "/about": "about.html",
    }[path]
    html = (STATIC_DIR / html_file).read_text()
    links = _ASSET_PATTERN.findall(html)
    assert links, f"{html_file} links no /static/ assets at all"
    for link in links:
        asset_path = STATIC_DIR / link.removeprefix("/static/")
        assert asset_path.exists(), f"{html_file} links missing asset {link}"


DATA_FILE_EXPECTED_KEYS = {
    "meta.json": {"snapshot_built", "git_commit", "numeric_features", "categorical_features"},
    "routes.json": set(),  # dynamic top-level keys (route numbers)
    "presets.json": set(),  # top-level is a list
    "trips.json": set(),  # top-level is a list
    "data_overview.json": {"histogram", "by_hour", "heatmap_hours", "heatmap_hour_weekday", "splits", "cleaning"},
    "models.json": {"comparison", "best_model", "feature_importance", "scatter_sample", "mae_by_hour"},
    "pipeline.json": {"stages", "quality_gate", "model_passport", "test_count"},
    "golden_time_features.json": set(),  # top-level is a list
}


@pytest.mark.parametrize("filename,expected_keys", DATA_FILE_EXPECTED_KEYS.items())
def test_data_file_parses_and_has_expected_keys(filename, expected_keys):
    path = DATA_DIR / filename
    assert path.exists(), f"missing {filename}"
    payload = json.loads(path.read_text())
    if expected_keys:
        assert expected_keys.issubset(payload.keys())


def test_route_lookup_files_exist_for_every_route_direction():
    routes = json.loads((DATA_DIR / "routes.json").read_text())
    for route, directions in routes.items():
        for direction in directions:
            lookup_path = DATA_DIR / f"lookup_{route}_{direction}.json"
            assert lookup_path.exists(), f"missing lookup file for {route}/{direction}"
            lookup = json.loads(lookup_path.read_text())
            assert lookup, f"lookup file for {route}/{direction} is empty"


def test_presets_have_all_25_features_and_validate():
    presets = json.loads((DATA_DIR / "presets.json").read_text())
    assert len(presets) >= 4
    for preset in presets:
        features = preset["features"]
        assert len(features) == 25
        PredictionRequest(**features)  # raises if invalid
        assert isinstance(preset["actual_travel_time_s"], (int, float))


def test_trip_segments_have_all_25_features_and_validate():
    trips = json.loads((DATA_DIR / "trips.json").read_text())
    assert len(trips) >= 2
    for trip in trips:
        assert len(trip["segments"]) >= 2
        for seg in trip["segments"]:
            features = {k: v for k, v in seg.items() if k != "actual_travel_time_s"}
            assert len(features) == 25
            PredictionRequest(**features)


def test_models_page_numbers_match_test_metrics_json():
    models_data = json.loads((DATA_DIR / "models.json").read_text())
    test_metrics = json.loads((PROJECT_ROOT / "reports" / "test_metrics.json").read_text())

    assert models_data["best_model"] == test_metrics["best_model"]
    assert models_data["best_baseline"] == test_metrics["best_baseline"]
    assert models_data["test_mae_best_model"] == pytest.approx(test_metrics["test_mae_best_model"], abs=0.01)
    assert models_data["generalisation_gap_pct"] == pytest.approx(test_metrics["generalisation_gap_pct"], abs=0.01)

    for row in models_data["comparison"]:
        expected_test_mae = test_metrics["models"][row["model"]]["mae"]
        assert row["test_mae"] == pytest.approx(expected_test_mae, abs=0.01)


def test_pipeline_quality_gate_matches_report():
    pipeline_data = json.loads((DATA_DIR / "pipeline.json").read_text())
    quality_gate = json.loads((PROJECT_ROOT / "reports" / "quality_gate.json").read_text())
    assert pipeline_data["quality_gate"]["passed"] == quality_gate["passed"]
    assert pipeline_data["quality_gate"]["checks"].keys() == quality_gate["checks"].keys()


def test_golden_time_features_table_has_all_24_hours():
    golden = json.loads((DATA_DIR / "golden_time_features.json").read_text())
    assert len(golden) == 24
    hours = {row["hour"] for row in golden}
    assert hours == set(range(24))
    for row in golden:
        assert row["minute_of_day"] == row["hour"] * 60


def test_javascript_time_features_match_python_golden_table():
    """Runs static/js/time_features.js under Node and checks every one of
    the 24 golden rows (computed by build_site_data.py, mirroring
    src/features/time_features.py exactly) comes out identical in JS."""
    js_path = STATIC_DIR / "js" / "time_features.js"
    golden_path = DATA_DIR / "golden_time_features.json"

    node_script = f"""
    const fs = require('fs');
    eval(fs.readFileSync({str(js_path)!r}, 'utf8'));
    const golden = JSON.parse(fs.readFileSync({str(golden_path)!r}, 'utf8'));
    let failures = [];
    for (const row of golden) {{
      const got = deriveTimeFeatures(row.minute_of_day);
      for (const key of ['hour', 'minute_of_day', 'is_peak_am', 'is_peak_pm']) {{
        if (got[key] !== row[key]) failures.push(`hour=${{row.hour}} ${{key}}: got ${{got[key]}} expected ${{row[key]}}`);
      }}
      for (const key of ['hour_sin', 'hour_cos']) {{
        if (Math.abs(got[key] - row[key]) > 1e-8) failures.push(`hour=${{row.hour}} ${{key}}: got ${{got[key]}} expected ${{row[key]}}`);
      }}
    }}
    if (failures.length) {{
      console.error(failures.join('\\n'));
      process.exit(1);
    }}
    console.log('all 24 hours match');
    """
    result = subprocess.run(["node", "-e", node_script], capture_output=True, text=True)
    assert result.returncode == 0, f"JS/Python time-feature mismatch:\n{result.stdout}\n{result.stderr}"


def test_counter_page_links_time_features_script():
    html = (STATIC_DIR / "index.html").read_text()
    assert "time_features.js" in html
