"""Transport behaviour of the NVIDIA client: retries, errors, parsing."""

from __future__ import annotations

import json

import pytest
import requests

from app.llm.config import NVIDIAConfig
from app.llm.nvidia_kimi import (
    NVIDIAAuthError,
    NVIDIAError,
    NVIDIAKimiClient,
    NVIDIARateLimitError,
    NVIDIAResponseError,
    NVIDIAServerError,
    NVIDIATimeoutError,
    _parse_tool_calls,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Keep retry tests instant."""
    monkeypatch.setattr("app.llm.nvidia_kimi.time.sleep", lambda _s: None)


@pytest.fixture
def client():
    return NVIDIAKimiClient(
        NVIDIAConfig(api_key="nvapi-secret-key", max_retries=2, timeout=5)
    )


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def completion_body(content="hello", tool_calls=None, finish="stop"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                    "reasoning_content": "PRIVATE REASONING MUST NOT LEAK",
                },
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


MESSAGES = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------------ happy path


def test_chat_returns_assistant_text(monkeypatch, client):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(200, completion_body("hi there"))
    )
    assert client.chat(MESSAGES) == "hi there"


def test_reasoning_content_is_never_returned(monkeypatch, client):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(200, completion_body("answer"))
    )
    completion = client.complete(MESSAGES)
    assert completion.text == "answer"
    assert "PRIVATE REASONING" not in completion.text
    assert "PRIVATE REASONING" not in json.dumps(completion.usage)


def test_usage_is_captured(monkeypatch, client):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(200, completion_body())
    )
    assert client.complete(MESSAGES).usage["total_tokens"] == 15


def test_payload_carries_required_fields(monkeypatch, client):
    seen = {}

    def capture(url, headers=None, json=None, timeout=None, **kw):
        seen.update({"url": url, "headers": headers, "payload": json})
        return FakeResponse(200, completion_body())

    monkeypatch.setattr(requests, "post", capture)
    client.chat(MESSAGES)

    assert seen["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer nvapi-secret-key"
    payload = seen["payload"]
    assert payload["model"] == "moonshotai/kimi-k3"
    assert payload["reasoning_effort"] == "max"
    assert payload["stream"] is False
    assert payload["seed"] == 0


# ---------------------------------------------------------------- tool calling


def test_tool_calls_are_parsed(monkeypatch, client):
    calls = [
        {
            "id": "read_file:0",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "app.py"}'},
        }
    ]
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse(200, completion_body(None, calls, "tool_calls")),
    )
    completion = client.complete(MESSAGES, tools=[{"type": "function"}])
    assert completion.has_tool_calls
    assert completion.tool_calls[0].name == "read_file"
    assert completion.tool_calls[0].arguments == {"path": "app.py"}


def test_malformed_tool_arguments_do_not_crash():
    calls = [{"id": "x", "function": {"name": "f", "arguments": "{not json"}}]
    parsed = _parse_tool_calls(calls)
    assert parsed[0].arguments["__raw_arguments__"] == "{not json"


def test_tools_omitted_from_payload_when_absent(monkeypatch, client):
    seen = {}
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, json=None, **k: (
            seen.update(payload=json),
            FakeResponse(200, completion_body()),
        )[1],
    )
    client.chat(MESSAGES)
    assert "tools" not in seen["payload"]


# --------------------------------------------------------------------- errors


def test_auth_error_is_not_retried(monkeypatch, client):
    attempts = []

    def post(*a, **k):
        attempts.append(1)
        return FakeResponse(401, text="invalid key")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(NVIDIAAuthError, match="rejected the API key"):
        client.chat(MESSAGES)
    assert len(attempts) == 1, "auth failures must not consume retries"


def test_api_key_is_redacted_from_error_text(monkeypatch, client):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse(401, text="bad token nvapi-secret-key rejected"),
    )
    with pytest.raises(NVIDIAAuthError) as exc:
        client.chat(MESSAGES)
    assert "nvapi-secret-key" not in str(exc.value)
    assert "***" in str(exc.value)


def test_rate_limit_is_retried_then_raised(monkeypatch, client):
    attempts = []

    def post(*a, **k):
        attempts.append(1)
        return FakeResponse(429, text="slow down", headers={"Retry-After": "0"})

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(NVIDIARateLimitError):
        client.chat(MESSAGES)
    assert len(attempts) == 3, "1 initial attempt + 2 retries"


def test_server_error_recovers_on_retry(monkeypatch, client):
    responses = [
        FakeResponse(503, text="unavailable"),
        FakeResponse(200, completion_body("recovered")),
    ]
    monkeypatch.setattr(requests, "post", lambda *a, **k: responses.pop(0))
    assert client.chat(MESSAGES) == "recovered"


def test_server_error_surfaces_after_retries(monkeypatch, client):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(500, text="boom"))
    with pytest.raises(NVIDIAServerError):
        client.chat(MESSAGES)


def test_timeout_is_retried_then_raised(monkeypatch, client):
    def post(*a, **k):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(NVIDIATimeoutError):
        client.chat(MESSAGES)


def test_connection_error_is_retried_then_raised(monkeypatch, client):
    def post(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(NVIDIAError, match="Could not reach"):
        client.chat(MESSAGES)


def test_non_json_body_is_reported(monkeypatch, client):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(200, None, text="<html>502</html>")
    )
    with pytest.raises(NVIDIAResponseError, match="non-JSON"):
        client.chat(MESSAGES)


def test_empty_choices_is_reported(monkeypatch, client):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(200, {"choices": []})
    )
    with pytest.raises(NVIDIAResponseError, match="Malformed"):
        client.chat(MESSAGES)


def test_content_free_completion_is_reported(monkeypatch, client):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResponse(200, completion_body(None, None, "length")),
    )
    with pytest.raises(NVIDIAResponseError, match="neither content nor tool calls"):
        client.chat(MESSAGES)


def test_unknown_model_404_is_not_retried(monkeypatch, client):
    attempts = []

    def post(*a, **k):
        attempts.append(1)
        return FakeResponse(404, text="model not found")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(NVIDIAResponseError, match="not found"):
        client.chat(MESSAGES)
    assert len(attempts) == 1


# ------------------------------------------------------------------ streaming


def test_sse_deltas_are_concatenated(client):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}',
        "data: [DONE]",
    ]
    assert "".join(client._iter_sse_text(iter(lines))) == "Hello"


def test_malformed_sse_lines_are_skipped(client):
    lines = ["data: {broken", 'data: {"choices":[{"delta":{"content":"ok"}}]}']
    assert "".join(client._iter_sse_text(iter(lines))) == "ok"
