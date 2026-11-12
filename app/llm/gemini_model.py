"""ADK's native Gemini, adapted to this project's schemas.

Two things have to be fixed up before a Gemini turn can carry SE-Agent's
structured output:

* **Retries.** ``google-genai`` never retries unless told to (no
  ``retry_options`` means "never retry"), so a single 429 -- routine on the
  free tier -- or one 503 demand spike would end a run mid-phase. A run is
  dozens of calls across eight phases; it has to ride those out.
* **Schema dialect.** Every workflow model derives from ``SEModel``, which
  sets ``extra="forbid"``, so Pydantic emits ``additionalProperties: false``.
  ADK carries that into ``types.Schema.additional_properties``, and the
  Gemini REST API rejects the field outright::

      400 INVALID_ARGUMENT ... Unknown name "additional_properties" at
      'generation_config.response_schema': Cannot find field.

  Stripping it here rather than loosening ``SEModel`` keeps strict
  validation everywhere else, including the OpenAI-compatible providers
  where the field is both legal and useful.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import zlib
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel

logger = logging.getLogger(__name__)


RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=6,
    initial_delay=2.0,
    max_delay=120.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)


RETRY_OPTIONS_NO_429 = RETRY_OPTIONS.model_copy(
    update={"http_status_codes": [408, 500, 502, 503, 504]}
)


TRANSPORT_ATTEMPTS = 4
_PER_DAY_MARKER = "PerDay"
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")

# How many times one turn may wait out a per-minute window before giving up on
# this model and failing over anyway.
MINUTE_WINDOW_WAITS = 3
MAX_MINUTE_WAIT = 90.0

# Ordered cheapest-capability first; every entry was verified callable on a
# new free key, unlike 2.5 Flash which Google now refuses to new users.
FREE_TIER_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
)


def _quota_verdict(exc: ClientError) -> tuple[bool, float]:
    """Read a 429 as ``(spent_for_the_day, seconds_to_wait)``.

    The structure of the error payload has changed shape between versions, so
    this reads the stringified error rather than indexing into it: a missed
    marker costs one needless failover, an exception here would cost the run.
    """
    text = f"{getattr(exc, 'details', '') or ''} {exc}"
    match = _RETRY_DELAY.search(text)
    delay = float(match.group(1)) if match else 0.0
    return _PER_DAY_MARKER in text, delay


# Models found out of quota during this process, so the next agent to need
# one does not have to rediscover it.
_EXHAUSTED: set[str] = set()


def _candidates() -> tuple[str, ...]:
    pinned = (os.environ.get("GEMINI_MODEL") or "").strip()
    if pinned:
        return (pinned,)
    configured = tuple(
        name.strip() for name in (os.environ.get("GEMINI_MODELS") or "").split(",") if name.strip()
    )
    return configured or FREE_TIER_MODELS


def project_label() -> str:

    name = (os.environ.get("GEMINI_PROJECT") or "").strip()
    number = (os.environ.get("GEMINI_PROJECT_NUMBER") or "").strip()
    if name and number and number not in name:
        return f"{name} ({number})"
    return name or number or "unidentified project"


def model_for(agent: str) -> str:
    """Which model one agent runs on.

    Each agent is mapped to its own model so the run draws on several daily
    quotas rather than exhausting one. The mapping is by content hash, not
    call order, so an agent keeps the same model across runs and a failure
    stays reproducible. Models already found spent are skipped.
    """
    candidates = _candidates()
    fresh = tuple(name for name in candidates if name not in _EXHAUSTED) or candidates
    return fresh[zlib.crc32(agent.encode()) % len(fresh)]


_NESTED_SCHEMA_FIELDS = (
    "items",
    "not_",
    "any_of",
    "one_of",
    "all_of",
    "properties",
    "defs",
)


def _strip_additional_properties(node: Any) -> None:
    """Recursively clear ``additionalProperties`` from a schema of any shape.

    ``response_schema`` reaches here as a JSON-schema dict, a ``types.Schema``,
    or a nest of both, so this walks whatever it is given rather than assuming
    one representation.
    """
    if isinstance(node, dict):
        node.pop("additionalProperties", None)
        node.pop("additional_properties", None)
        for value in node.values():
            _strip_additional_properties(value)
    elif isinstance(node, list):
        for item in node:
            _strip_additional_properties(item)
    elif isinstance(node, types.Schema):
        node.additional_properties = None
        for field in _NESTED_SCHEMA_FIELDS:
            _strip_additional_properties(getattr(node, field, None))


def _legal_schema(schema: Any) -> Any:
    """One schema, made legal for Gemini -- converting a model class if needed.

    ADK passes the Pydantic class itself, which genai only turns into a schema
    at request time, so there is nothing to strip until it is converted here.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    _strip_additional_properties(schema)
    return schema


def _sanitize(request: LlmRequest) -> None:
    """Make every schema in one request legal for the Gemini API."""
    config = request.config
    if config is None:
        return

    if config.response_schema is not None:
        config.response_schema = _legal_schema(config.response_schema)
    for tool in config.tools or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            declaration.parameters = _legal_schema(declaration.parameters)
            declaration.response = _legal_schema(declaration.response)


class SEAgentGemini(Gemini):
    """Gemini with retries, legal schemas, and failover off a spent quota."""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        _sanitize(llm_request)
        transport_left = TRANSPORT_ATTEMPTS
        minute_waits_left = MINUTE_WINDOW_WAITS
        while True:
            produced = False
            try:
                async for response in super().generate_content_async(llm_request, stream=stream):
                    produced = True
                    yield response
                return
            except (TimeoutError, OSError) as exc:
                transport_left -= 1
                if produced or transport_left <= 0:
                    raise
                delay = 2.0 * 2 ** (TRANSPORT_ATTEMPTS - transport_left - 1)
                logger.warning(
                    "gemini transport failure (%s: %s); retrying in %.0fs",
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except ClientError as exc:
                if produced or exc.code != 429:
                    raise
                spent_for_the_day, retry_after = _quota_verdict(exc)

                if not spent_for_the_day and minute_waits_left > 0:
                    minute_waits_left -= 1
                    wait = min(max(retry_after, 2.0), MAX_MINUTE_WAIT)
                    logger.warning(
                        "%s hit a per-minute limit; waiting %.0fs (%d more)",
                        self.model,
                        wait,
                        minute_waits_left,
                    )
                    await asyncio.sleep(wait)
                    continue

                replacement = _fail_over(self.model)
                if replacement is None:
                    raise
                logger.warning(
                    "%s is out of quota for the day; this agent continues on %s",
                    self.model,
                    replacement,
                )
                self.model = replacement
                llm_request.model = replacement
                minute_waits_left = MINUTE_WINDOW_WAITS


def _fail_over(spent: str) -> str | None:
    """Mark ``spent`` exhausted and name the next model to try, if any.

    The free tier's cap is per model per day, so a spent quota is not
    something to wait out -- the same request succeeds immediately against
    another model. Exhaustion is remembered for the process so each model is
    only discovered spent once, rather than once per agent.
    """
    _EXHAUSTED.add(spent)
    for candidate in _candidates():
        if candidate not in _EXHAUSTED:
            return candidate
    logger.error(
        "every configured Gemini model is out of quota for the day in %s: %s. "
        "Free-tier quota is counted per project and resets at midnight "
        "Pacific; set GEMINI_MODELS to other models, or use another project.",
        project_label(),
        ", ".join(sorted(_EXHAUSTED)),
    )
    return None


def build_gemini(agent: str) -> SEAgentGemini:
    """The Gemini model one named agent runs on."""
    model = model_for(agent)
    logger.info("agent %s runs on %s (%s)", agent, model, project_label())
    retry = RETRY_OPTIONS if len(_candidates()) == 1 else RETRY_OPTIONS_NO_429
    return SEAgentGemini(model=model, retry_options=retry)
