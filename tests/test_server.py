"""HTTP surface: health, phases, the console, and the SSE stream."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import server
from app.agent import RunResult
from app.schemas import Phase
from app.telemetry import RunTracker, Update


@pytest.fixture
def client():
    return TestClient(server.app)


# --------------------------------------------------------------------- health


def test_health_reports_ready_when_preflight_is_clean(client, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda repo=None: [])
    body = client.get("/api/health").json()
    assert body == {"ready": True, "problems": []}


def test_health_reports_problems(client, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda repo=None: ["docker is not running"])
    body = client.get("/api/health").json()
    assert body["ready"] is False
    assert "docker is not running" in body["problems"]


# --------------------------------------------------------------------- phases


def test_phases_lists_every_stage_in_order(client):
    ids = [p["id"] for p in client.get("/api/phases").json()["phases"]]
    assert ids == [p.value for p in Phase]
    assert ids[0] == "requirements"
    assert ids[-1] == "final"


def test_console_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SE-Agent" in response.text


# ------------------------------------------------------------------ validation


def test_empty_request_is_rejected(client):
    assert client.post("/api/run", json={"request": ""}).status_code == 422


def test_iteration_budget_above_the_ceiling_is_rejected(client):
    response = client.post("/api/run", json={"request": "do it", "max_iterations": 99})
    assert response.status_code == 422


# ---------------------------------------------------------------------- stream


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events = []
    for chunk in text.split("\n\n"):
        if not chunk.strip():
            continue
        name, payload = None, None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name is not None:
            events.append((name, payload))
    return events


def test_preflight_failure_is_streamed_not_raised(client, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda repo=None: ["no API key"])
    response = client.post("/api/run", json={"request": "add a thing"})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[0][0] == "error"
    assert events[0][1]["fatal"] is True
    assert "no API key" in events[0][1]["text"]


def test_updates_and_result_are_streamed(client, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda repo=None: [])

    tracker = RunTracker()
    tracker.tool_calls = 4

    async def fake_stream(request, repo=None, limit=None):
        yield Update(kind="phase", text="Requirements", phase=Phase.REQUIREMENTS)
        yield Update(kind="tool", text="read_file(path=a.py)", phase=Phase.ANALYSIS)
        yield Update(kind="tests", text="6 passed, 0 failed", phase=Phase.TESTING)
        yield RunResult(
            session_id="run-abc",
            state={"tests_passed": True, "review_approved": True, "iteration": 2},
            tracker=tracker,
        )

    monkeypatch.setattr(server, "stream_run", fake_stream)

    response = client.post("/api/run", json={"request": "add a thing"})
    events = parse_sse(response.text)
    kinds = [name for name, _ in events]
    assert kinds == ["phase", "tool", "tests", "result"]

    result = events[-1][1]
    assert result["succeeded"] is True
    assert result["session_id"] == "run-abc"
    assert result["iterations"] == 2
    assert result["tool_calls"] == 4


def test_a_failed_run_is_reported_as_not_succeeded(client, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda repo=None: [])

    async def fake_stream(request, repo=None, limit=None):
        yield RunResult(
            session_id="run-x",
            state={"tests_passed": True, "review_approved": False},
            tracker=RunTracker(),
        )

    monkeypatch.setattr(server, "stream_run", fake_stream)
    result = parse_sse(client.post("/api/run", json={"request": "x"}).text)[-1][1]
    assert result["succeeded"] is False
    assert result["tests_passed"] is True
    assert result["review_approved"] is False


def test_a_crash_mid_stream_becomes_an_error_event(client, monkeypatch):
    monkeypatch.setattr(server, "preflight", lambda repo=None: [])

    async def fake_stream(request, repo=None, limit=None):
        yield Update(kind="phase", text="Requirements", phase=Phase.REQUIREMENTS)
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(server, "stream_run", fake_stream)
    events = parse_sse(client.post("/api/run", json={"request": "x"}).text)
    assert events[-1][0] == "error"
    assert "provider exploded" in events[-1][1]["text"]
