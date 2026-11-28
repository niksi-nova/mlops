"""Exercise the smoke CLI with mocked urllib responses."""
import io
import json
from urllib.error import HTTPError

from scripts import check_live
from src.api.schemas import PredictionRequest


class Response(io.BytesIO):
    status = 200


def test_smoke_passes_all_documented_routes(monkeypatch, capsys):
    responses = {
        "/health": b'{"model_loaded":true}',
        "/model-info": b'{"alias":"production","source":"wandb","git_commit":"abc123"}',
        "/static/dashboard.html": b"<title>Pipeline dashboard</title>",
        "/static/data/dashboard.json": json.dumps(dict.fromkeys(check_live.DASHBOARD_KEYS)).encode(),
        "/predict": b'{"predicted_travel_time_s":30.7}',
    }
    paths = []

    def open_response(request, timeout):
        path = request.full_url.removeprefix("http://example.test")
        paths.append(path)
        if path == "/predict":
            assert request.get_method() == "POST"
            PredictionRequest.model_validate(json.loads(request.data))
        else:
            assert request.get_method() == "GET"
        return Response(responses.get(path, b""))

    monkeypatch.setattr(check_live, "urlopen", open_response)
    assert check_live.main(["--base-url", "http://example.test/"]) == 0
    assert set(paths) == {*responses, *check_live.SITE_PAGES}
    assert capsys.readouterr().out.count("PASS") == 11


def test_smoke_continues_on_404_and_invalid_payloads(monkeypatch, capsys):
    def open_response(request, timeout):
        path = request.full_url.removeprefix(check_live.BASE_URL)
        if path == "/static/dashboard.html":
            raise HTTPError(request.full_url, 404, "not found", {}, io.BytesIO())
        return Response({
            "/health": b'{"model_loaded":true}',
            "/model-info": b'{"alias":"staging","source":"wandb","git_commit":"abc123"}',
            "/static/data/dashboard.json": b"invalid",
            "/predict": b'{"predicted_travel_time_s":true}',
        }.get(path, b""))

    monkeypatch.setattr(check_live, "urlopen", open_response)
    assert check_live.main([]) == 1
    output = capsys.readouterr().out
    assert output.count("FAIL") == 4
    assert "404" in output
    assert "POST /predict" in output


def test_wake_backoff_stops_at_deadline(monkeypatch):
    now = [0]
    delays = []

    def sleep(seconds):
        delays.append(seconds)
        now[0] += seconds

    def unavailable(request, timeout):
        raise HTTPError(request.full_url, 503, "starting", {}, io.BytesIO())

    monkeypatch.setattr(check_live.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(check_live.time, "sleep", sleep)
    monkeypatch.setattr(check_live, "urlopen", unavailable)
    status, _ = check_live.fetch("http://example.test", "/health", wake_timeout=9)
    assert status == 503
    assert delays == [2, 4, 3]
