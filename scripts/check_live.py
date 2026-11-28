"""Smoke-test the deployed Delhi demo and its prediction endpoint."""
import argparse
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://delhi-bus-delay-api.onrender.com"
DASHBOARD_KEYS = {
    "stages", "train_metrics", "test_metrics", "quality_gate", "drift_report",
    "version_ledger", "governance_status",
}
SITE_PAGES = ("/", "/trip", "/data", "/models", "/pipeline", "/about")

# Example in docs/api.md.
EXAMPLE = {
    'route': '425',
    'direction': 'down',
    'segment_id': '425_down_26',
    'relative_weekday': 6,
    'from_node': 26,
    'nodes_spanned': 1,
    'route_position': 0.6341463414634146,
    'stops_into_trip': 1,
    'elapsed_trip_s': 0.0,
    'hour': 8,
    'minute_of_day': 484,
    'hour_sin': 0.8660254037844387,
    'hour_cos': -0.49999999999999983,
    'is_peak_am': 1,
    'is_peak_pm': 0,
    'prev_bus_tt_s': 30.0,
    'prev3_median_tt_s': 30.0,
    'headway_s': 52.0,
    'n_buses_recent': 3,
    'has_prev_bus': 1,
    'own_prev_segment_tt_s': 40.0,
    'own_prev_segment_ratio': 1.0,
    'typical_segment_hour_s': 40.0,
    'typical_segment_s': 40.0,
    'te_vehicle': 144.26669174404856,
}


def fetch(base_url, path, *, payload=None, wake_timeout=0):
    """Retry transient responses on the initial health request only."""
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        base_url.rstrip("/") + path, data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    deadline = time.monotonic() + wake_timeout
    delay = 2
    while True:
        remaining = deadline - time.monotonic()
        try:
            with urlopen(request, timeout=min(10, max(0.1, remaining)) if wake_timeout else 10) as response:
                status, content = response.status, response.read()
            if not wake_timeout or status != 200 or json_field(content, "model_loaded") is True:
                return status, content
        except HTTPError as error:
            status, content = error.code, b""
            error.close()
            if status not in {429, 500, 502, 503, 504}:
                return status, content
        except (URLError, OSError):
            status, content = None, b""

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return status, content
        time.sleep(min(delay, remaining))
        if time.monotonic() >= deadline:
            return status, content
        delay = min(delay * 2, 10)


def json_object(body):
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def json_field(body, key):
    return json_object(body).get(key)


def nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def model_info_ok(body):
    info = json_object(body)
    return (info.get("alias") == "production"
            and nonempty_string(info.get("source"))
            and nonempty_string(info.get("git_commit")))


def prediction_ok(body):
    value = json_field(body, "predicted_travel_time_s")
    return type(value) in (int, float) and 0 < value < math.inf


def check_live(base_url, *, wake_timeout=120):
    """Return one row per endpoint without printing bodies or request errors."""
    checks = [
        ("GET", "/health", lambda body: json_field(body, "model_loaded") is True, "model_loaded must be true"),
        ("GET", "/model-info", model_info_ok, "source, production alias and git commit required"),
        ("GET", "/static/dashboard.html", lambda body: b"<title>Pipeline dashboard" in body, "dashboard title missing"),
        ("GET", "/static/data/dashboard.json", lambda body: DASHBOARD_KEYS <= json_object(body).keys(), "missing dashboard keys or invalid JSON"),
        *(("GET", page, None, "") for page in SITE_PAGES),
        ("POST", "/predict", prediction_ok, "positive finite predicted_travel_time_s required"),
    ]
    rows = []
    for method, path, validate, error in checks:
        status, body = fetch(
            base_url, path, payload=EXAMPLE if method == "POST" else None,
            wake_timeout=wake_timeout if path == "/health" else 0,
        )
        if status != 200:
            rows.append((method, path, status, False, "Expected HTTP 200"))
        elif validate is not None and not validate(body):
            rows.append((method, path, status, False, error))
        else:
            rows.append((method, path, status, True, ""))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    rows = check_live(args.base_url)
    print(f"{'Check':<38} {'HTTP':<6} {'Result':<6} Detail", flush=True)
    for method, path, status, passed, detail in rows:
        outcome = "PASS" if passed else "FAIL"
        print(f"{method + ' ' + path:<38} {str(status or '—'):<6} {outcome:<6} {detail}", flush=True)
    return int(not all(row[3] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
