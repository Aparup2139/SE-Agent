"""Gemini adapter: schema legality, retry policy, transport resilience."""

from __future__ import annotations

import asyncio

import pytest
from google.adk.models.google_llm import _ResourceExhaustedError
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, ConfigDict

from app.llm import gemini_model
from app.llm.gemini_model import SEAgentGemini, build_gemini


class Inner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class Outer(BaseModel):
    """Shaped like the workflow models: strict, with a nested list."""

    model_config = ConfigDict(extra="forbid")
    summary: str
    items: list[Inner]


def _leftovers(node) -> list[str]:
    """Every place ``additionalProperties`` still survives in a schema."""
    found = []
    if isinstance(node, dict):
        for key in ("additionalProperties", "additional_properties"):
            if key in node:
                found.append(key)
        for value in node.values():
            found.extend(_leftovers(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_leftovers(item))
    elif isinstance(node, types.Schema):
        if node.additional_properties is not None:
            found.append("additional_properties")
        for field in gemini_model._NESTED_SCHEMA_FIELDS:
            found.extend(_leftovers(getattr(node, field, None)))
    return found


# --------------------------------------------------------------- schema legality


def test_a_strict_model_would_otherwise_carry_the_illegal_field():
    """Guard the premise: Pydantic really does emit it, nested too."""
    assert _leftovers(Outer.model_json_schema())


def test_a_model_class_is_converted_and_stripped():
    request = LlmRequest(
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Outer
        )
    )
    gemini_model._sanitize(request)
    assert _leftovers(request.config.response_schema) == []


def test_tool_parameter_schemas_are_stripped_too():
    parameters = types.Schema(
        type=types.Type.OBJECT,
        additional_properties=False,
        properties={"nested": types.Schema(type=types.Type.OBJECT, additional_properties=False)},
    )
    request = LlmRequest(
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name="set_model_response", parameters=parameters)
                    ]
                )
            ]
        )
    )
    gemini_model._sanitize(request)
    declaration = request.config.tools[0].function_declarations[0]
    assert _leftovers(declaration.parameters) == []


def test_a_request_without_a_config_is_left_alone():
    request = LlmRequest()
    request.config = None
    gemini_model._sanitize(request)  # must not raise


# ------------------------------------------------------------------ retry policy


def test_demand_spikes_are_waited_out():
    """503 is transient and worth a retry; 429 is handled by failover."""
    model = build_gemini("coder")
    codes = model.retry_options.http_status_codes
    assert 503 in codes and 504 in codes
    assert model.retry_options.attempts > 1


# ------------------------------------------------------- per-agent model choice


def test_agents_are_spread_across_models(monkeypatch):
    """The free tier caps requests per day *per model*, so one model per
    agent is what makes a whole run fit at all."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    agents = ["requirements", "architecture", "coder", "tester", "reviewer"]
    chosen = {gemini_model.model_for(name) for name in agents}
    assert len(chosen) > 1
    assert chosen <= set(gemini_model.FREE_TIER_MODELS)


def test_an_agent_keeps_its_model_across_runs(monkeypatch):
    """Order-independent, so a failed agent can be reproduced."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    assert gemini_model.model_for("coder") == gemini_model.model_for("coder")


def test_a_pinned_model_overrides_the_spread(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    assert gemini_model.model_for("coder") == "gemini-3.8-flash"
    assert gemini_model.model_for("tester") == "gemini-3.8-flash"


def test_the_model_list_is_configurable(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", " alpha , beta ")
    assert gemini_model.model_for("coder") in {"alpha", "beta"}


# ------------------------------------------------------------ transport failures


def _response(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


@pytest.mark.asyncio
async def test_a_dns_failure_is_retried(monkeypatch):
    """A DNS blip mid-run must not end the run.

    aiohttp's ClientConnectorDNSError is an OSError, and genai only retries
    *httpx* transport errors, so nothing below us retries this.
    """
    monkeypatch.setattr(gemini_model.asyncio, "sleep", _no_sleep)
    calls = []

    async def flaky(self, llm_request, stream=False):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("getaddrinfo failed")
        yield _response("ok")

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", flaky)
    model = build_gemini("coder")
    out = [r async for r in model.generate_content_async(LlmRequest())]

    assert len(calls) == 3
    assert out[0].content.parts[0].text == "ok"


@pytest.mark.asyncio
async def test_a_turn_that_already_yielded_is_never_replayed(monkeypatch):
    """Retrying after output would duplicate what the caller already got."""
    monkeypatch.setattr(gemini_model.asyncio, "sleep", _no_sleep)

    async def fails_late(self, llm_request, stream=False):
        yield _response("first")
        raise OSError("connection reset")

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", fails_late)
    model = build_gemini("coder")

    seen = []
    with pytest.raises(OSError):
        async for response in model.generate_content_async(LlmRequest()):
            seen.append(response)
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_permanent_transport_failure_still_raises(monkeypatch):
    monkeypatch.setattr(gemini_model.asyncio, "sleep", _no_sleep)
    calls = []

    async def always_fails(self, llm_request, stream=False):
        calls.append(1)
        raise OSError("getaddrinfo failed")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", always_fails)
    model = build_gemini("coder")

    with pytest.raises(OSError):
        async for _ in model.generate_content_async(LlmRequest()):
            pass
    assert len(calls) == gemini_model.TRANSPORT_ATTEMPTS


async def _no_sleep(_seconds: float) -> None:
    """Keep the backoff out of the test's wall clock."""


def test_sanity_the_model_class_is_a_gemini():
    assert issubclass(SEAgentGemini, gemini_model.Gemini)


def test_asyncio_is_still_the_module_the_tests_patch():
    """The sleep patch above is only meaningful while this holds."""
    assert asyncio is gemini_model.asyncio


# ----------------------------------------------------------- quota failover


@pytest.fixture(autouse=True)
def _forget_exhausted_models():
    """Exhaustion is process-wide state; no test may leak it into another."""
    gemini_model._EXHAUSTED.clear()
    yield
    gemini_model._EXHAUSTED.clear()


@pytest.mark.asyncio
async def test_a_spent_daily_quota_fails_over_to_another_model(monkeypatch):
    """20 requests/day/model means waiting cannot help -- only switching."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")
    used = []

    async def quota_bound(self, llm_request, stream=False):
        used.append(self.model)
        if self.model == "first":
            raise _quota_error(PER_DAY)
        yield _response("ok")

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", quota_bound)
    model = build_gemini("coder")
    model.model = "first"
    out = [r async for r in model.generate_content_async(LlmRequest())]

    assert used == ["first", "second"]
    assert out[0].content.parts[0].text == "ok"
    assert "first" in gemini_model._EXHAUSTED


@pytest.mark.asyncio
async def test_failover_rewrites_the_model_the_request_carries(monkeypatch):
    """ADK sends llm_request.model, so switching self.model alone is a no-op."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")
    seen = []

    async def quota_bound(self, llm_request, stream=False):
        seen.append(llm_request.model)
        if llm_request.model == "first":
            raise _quota_error(PER_DAY)
        yield _response("ok")

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", quota_bound)
    model = build_gemini("coder")
    model.model = "first"
    request = LlmRequest(model="first")
    [r async for r in model.generate_content_async(request)]

    assert seen == ["first", "second"]
    assert request.model == "second"


@pytest.mark.asyncio
async def test_exhausting_every_model_raises_rather_than_looping(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")
    calls = []

    async def always_spent(self, llm_request, stream=False):
        calls.append(self.model)
        raise _quota_error(PER_DAY)
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", always_spent)
    model = build_gemini("coder")

    with pytest.raises(ClientError):
        async for _ in model.generate_content_async(LlmRequest()):
            pass
    assert len(calls) == 2  # each model tried exactly once


@pytest.mark.asyncio
async def test_a_non_quota_client_error_is_not_failed_over(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "first,second")

    async def bad_request(self, llm_request, stream=False):
        raise ClientError(400, {"error": {"code": 400, "message": "bad schema"}})
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", bad_request)
    model = build_gemini("coder")

    with pytest.raises(ClientError) as exc:
        async for _ in model.generate_content_async(LlmRequest()):
            pass
    assert exc.value.code == 400
    assert not gemini_model._EXHAUSTED


def test_a_spent_model_is_not_handed_to_the_next_agent(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")
    gemini_model._EXHAUSTED.add("first")
    assert gemini_model.model_for("coder") == "second"
    assert gemini_model.model_for("tester") == "second"


def test_waiting_out_a_429_is_kept_only_when_one_model_is_pinned(monkeypatch):
    """With nowhere to fail over to, the retry has to absorb the 429 itself."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    assert 429 in build_gemini("coder").retry_options.http_status_codes

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")
    assert 429 not in build_gemini("coder").retry_options.http_status_codes


# ------------------------------------------- per-minute vs per-day 429s


def _quota_error(quota_id: str, retry_delay: str = "39s") -> ClientError:
    inner = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": quota_id}],
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay,
                    },
                ],
            }
        },
    )
    return _ResourceExhaustedError(inner)


PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
PER_MINUTE = "GenerateContentInputTokensPerModelPerMinute-FreeTier"


def test_a_daily_cap_is_read_as_spent():
    spent, delay = gemini_model._quota_verdict(_quota_error(PER_DAY))
    assert spent is True
    assert delay == 39.0


def test_a_per_minute_window_is_not_read_as_spent():
    spent, delay = gemini_model._quota_verdict(_quota_error(PER_MINUTE, "12s"))
    assert spent is False
    assert delay == 12.0


def test_an_unparseable_429_is_treated_as_a_transient_window():
    """Better to wait once than to blacklist a healthy model."""
    spent, delay = gemini_model._quota_verdict(ClientError(429, {"error": {"code": 429}}))
    assert spent is False
    assert delay == 0.0


@pytest.mark.asyncio
async def test_a_per_minute_limit_waits_on_the_same_model(monkeypatch):
    """This is what wrongly ate every model on a brand-new project.

    Several agents run in parallel, so one burst trips the per-minute token
    window; treating that as exhaustion blacklisted the whole list in
    seconds while the quota was in fact fine.
    """
    slept = []

    async def record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(gemini_model.asyncio, "sleep", record)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")
    used = []

    async def throttled(self, llm_request, stream=False):
        used.append(self.model)
        if len(used) == 1:
            raise _quota_error(PER_MINUTE, "12s")
        yield _response("ok")

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", throttled)
    model = build_gemini("coder")
    model.model = "first"
    out = [r async for r in model.generate_content_async(LlmRequest())]

    assert used == ["first", "first"]  # same model, after a wait
    assert slept == [12.0]  # the delay the server named
    assert not gemini_model._EXHAUSTED  # nothing blacklisted
    assert out[0].content.parts[0].text == "ok"


@pytest.mark.asyncio
async def test_a_relentless_per_minute_limit_eventually_fails_over(monkeypatch):
    async def record(seconds):
        pass

    monkeypatch.setattr(gemini_model.asyncio, "sleep", record)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")
    used = []

    async def throttled(self, llm_request, stream=False):
        used.append(self.model)
        if self.model == "first":
            raise _quota_error(PER_MINUTE, "5s")
        yield _response("ok")

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", throttled)
    model = build_gemini("coder")
    model.model = "first"
    [r async for r in model.generate_content_async(LlmRequest())]

    waits = gemini_model.MINUTE_WINDOW_WAITS
    assert used == ["first"] * (waits + 1) + ["second"]


@pytest.mark.asyncio
async def test_a_named_delay_is_capped(monkeypatch):
    """A server asking for an hour must not hang the run for one."""
    slept = []

    async def record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(gemini_model.asyncio, "sleep", record)
    monkeypatch.setenv("GEMINI_MODELS", "first,second")

    async def throttled(self, llm_request, stream=False):
        if not slept:
            raise _quota_error(PER_MINUTE, "3600s")
        yield _response("ok")

    monkeypatch.setattr(gemini_model.Gemini, "generate_content_async", throttled)
    model = build_gemini("coder")
    [r async for r in model.generate_content_async(LlmRequest())]

    assert slept == [gemini_model.MAX_MINUTE_WAIT]


# --------------------------------------------------------- project identity


def test_the_project_is_named_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_PROJECT", "projects/613559642633")
    monkeypatch.setenv("GEMINI_PROJECT_NUMBER", "613559642633")
    # The number is already in the name, so it is not repeated.
    assert gemini_model.project_label() == "projects/613559642633"

    monkeypatch.setenv("GEMINI_PROJECT", "cadence-prod")
    assert gemini_model.project_label() == "cadence-prod (613559642633)"


def test_an_unnamed_project_still_reads_sensibly(monkeypatch):
    monkeypatch.delenv("GEMINI_PROJECT", raising=False)
    monkeypatch.delenv("GEMINI_PROJECT_NUMBER", raising=False)
    assert gemini_model.project_label() == "unidentified project"

    monkeypatch.setenv("GEMINI_PROJECT_NUMBER", "613559642633")
    assert gemini_model.project_label() == "613559642633"


def test_the_model_list_is_read_fresh_every_call(monkeypatch):
    """Editing .env must be enough -- no caching between calls."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "alpha")
    assert gemini_model._candidates() == ("alpha",)
    monkeypatch.setenv("GEMINI_MODELS", "beta,gamma")
    assert gemini_model._candidates() == ("beta", "gamma")
