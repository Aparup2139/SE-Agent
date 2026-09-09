"""The single NVIDIA NIM access point for SE-Agent.

Every LLM call in the system funnels through :class:`NVIDIAKimiClient`. No agent
may talk to ``integrate.api.nvidia.com`` directly -- that keeps retries, error
taxonomy, timeouts and key hygiene in exactly one place.

Two layers are exposed:

* :meth:`NVIDIAKimiClient.complete` / :meth:`acomplete` return a structured
  :class:`Completion` (text + tool calls + usage). The ADK adapter uses these.
* :meth:`NVIDIAKimiClient.chat` / :meth:`achat` return plain assistant text, for
  simple prompt-in/text-out calls.

Kimi K3 is a reasoning model and NIM returns its private reasoning in a
``reasoning_content`` field. It is deliberately dropped everywhere below: no
chain-of-thought ever reaches an agent, the session state, or the UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import requests

from .config import NVIDIAConfig, get_config

logger = logging.getLogger(__name__)

# Status codes worth a second attempt. 401/403/404/422 are configuration
# problems -- retrying them just burns time and rate-limit budget.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Rate limits need a slower schedule than ordinary transient errors: a NIM
# throttling window outlasts a 1s/2s/4s backoff, so those retries would be spent
# for nothing. 5s/10s/20s/40s/60s gives the window a chance to reset.
RATE_LIMIT_BASE_DELAY = 5.0
RATE_LIMIT_MAX_DELAY = 60.0


class NVIDIAError(RuntimeError):
    """Base class for every NVIDIA backend failure."""


class NVIDIAAuthError(NVIDIAError):
    """401/403 -- bad or unauthorized API key. Never retried."""


class NVIDIARateLimitError(NVIDIAError):
    """429 -- retried with backoff, then surfaced."""


class NVIDIAServerError(NVIDIAError):
    """5xx -- retried with backoff, then surfaced."""


class NVIDIATimeoutError(NVIDIAError):
    """Read/connect timeout after exhausting retries."""


class NVIDIAResponseError(NVIDIAError):
    """The HTTP call succeeded but the body was not a usable completion."""


@dataclass(slots=True)
class ToolCall:
    """One function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Completion:
    """A single assistant turn, normalized away from the wire format."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _redact(text: str, config: NVIDIAConfig) -> str:
    """Strip the API key out of anything headed for a log or an exception."""
    if not text:
        return text
    return text.replace(config.api_key, "***")


def _backoff(attempt: int) -> float:
    return min(2.0**attempt, 30.0) + random.uniform(0, 0.5)


def _retry_after_seconds(
    headers: Any, attempt: int, status: int | None = None
) -> float:
    """Honour a server-provided Retry-After, else back off exponentially.

    NIM does not always send Retry-After with a 429, and the ordinary backoff
    (about 1s, 2s, 4s) is far too quick to outlast a rate-limit window -- it
    just spends the retry budget and fails. Rate limits therefore get their own,
    much longer schedule.
    """
    try:
        raw = headers.get("Retry-After")
    except AttributeError:  # pragma: no cover - defensive
        raw = None
    if raw:
        try:
            return min(float(raw), 60.0)
        except (TypeError, ValueError):
            pass
    if status == 429:
        return min(RATE_LIMIT_BASE_DELAY * (2**attempt), RATE_LIMIT_MAX_DELAY) + (
            random.uniform(0, 1.0)
        )
    return _backoff(attempt)


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Normalize OpenAI-style ``tool_calls`` into :class:`ToolCall` objects."""
    calls: list[ToolCall] = []
    for index, item in enumerate(raw or []):
        function = (item or {}).get("function") or {}
        name = function.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                # A malformed argument blob is the model's error, not a transport
                # error. Surface it to the caller as a string so the agent loop
                # can feed the failure back rather than crashing the run.
                parsed = {"__raw_arguments__": arguments}
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            parsed = {}
        calls.append(
            ToolCall(
                id=item.get("id") or f"{name}:{index}",
                name=name,
                arguments=parsed,
            )
        )
    return calls


class NVIDIAKimiClient:
    """Chat-completions client for ``moonshotai/kimi-k3`` on NVIDIA NIM."""

    def __init__(self, config: NVIDIAConfig | None = None) -> None:
        self.config = config or get_config()

    # ---------------------------------------------------------------- helpers

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def model(self) -> str:
        return self.config.model

    def _headers(self, stream: bool) -> dict[str, str]:
        headers = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }
        # Keyless local providers (Ollama) need no bearer token.
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": list(messages),
            "model": self.config.model,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "seed": self.config.seed if seed is None else seed,
            "stream": stream,
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "reasoning_effort": (
                self.config.reasoning_effort
                if reasoning_effort is None
                else reasoning_effort
            ),
        }
        if tools:
            payload["tools"] = list(tools)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)
        return payload

    def _raise_for_status(self, status: int, body: str) -> None:
        """Map an HTTP status onto the exception taxonomy above."""
        detail = _redact(body, self.config)[:500]
        if status in (401, 403):
            raise NVIDIAAuthError(
                f"NVIDIA rejected the API key (HTTP {status}). Check NVIDIA_API_KEY "
                f"in your .env. Response: {detail}"
            )
        if status == 404:
            raise NVIDIAResponseError(
                f"Model or endpoint not found (HTTP 404) for model "
                f"{self.config.model!r} at {self.config.base_url!r}: {detail}"
            )
        if status == 429:
            raise NVIDIARateLimitError(f"NVIDIA rate limit (HTTP 429): {detail}")
        if status >= 500:
            raise NVIDIAServerError(f"NVIDIA server error (HTTP {status}): {detail}")
        if status >= 400:
            raise NVIDIAResponseError(f"NVIDIA request failed (HTTP {status}): {detail}")

    def _to_completion(self, data: dict[str, Any]) -> Completion:
        """Convert a chat-completions body into a :class:`Completion`.

        ``reasoning_content`` is intentionally never read.
        """
        try:
            choices = data["choices"]
            if not choices:
                raise KeyError("choices")
            choice = choices[0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise NVIDIAResponseError(
                f"Malformed completion response: {json.dumps(data)[:400]}"
            ) from exc

        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        content = message.get("content") or ""
        if not content and not tool_calls:
            raise NVIDIAResponseError(
                "Completion contained neither content nor tool calls "
                f"(finish_reason={choice.get('finish_reason')!r})"
            )
        return Completion(
            text=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
        )

    @staticmethod
    def _iter_sse_text(lines: Iterator[str]) -> Iterator[str]:
        """Yield visible content deltas from an OpenAI-style SSE stream."""
        for line in lines:
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece

    # ----------------------------------------------------------- sync request

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with bounded retries; returns the decoded JSON body."""
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    self.config.chat_completions_url,
                    headers=self._headers(False),
                    json=payload,
                    timeout=self.config.timeout,
                )
            except requests.Timeout:
                last_error = NVIDIATimeoutError(
                    f"NVIDIA request timed out after {self.config.timeout}s"
                )
            except requests.ConnectionError as exc:
                last_error = NVIDIAError(
                    f"Could not reach {self.config.base_url}: "
                    f"{_redact(str(exc), self.config)}"
                )
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise NVIDIAResponseError(
                            "NVIDIA returned a non-JSON body: "
                            f"{_redact(response.text, self.config)[:400]}"
                        ) from exc

                # Configuration errors raise immediately -- retrying is pointless.
                if response.status_code not in RETRYABLE_STATUS:
                    self._raise_for_status(response.status_code, response.text)

                try:
                    self._raise_for_status(response.status_code, response.text)
                except NVIDIAError as exc:
                    last_error = exc

                if attempt < self.config.max_retries:
                    delay = _retry_after_seconds(
                        response.headers, attempt, response.status_code
                    )
                    logger.warning(
                        "NVIDIA HTTP %s; retrying in %.1fs (attempt %d/%d)",
                        response.status_code,
                        delay,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise last_error

            if attempt < self.config.max_retries:
                delay = _backoff(attempt)
                logger.warning(
                    "NVIDIA transport failure; retrying in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt + 1,
                    self.config.max_retries,
                    last_error,
                )
                time.sleep(delay)
                continue
            raise last_error

        raise last_error or NVIDIAError("NVIDIA request failed for an unknown reason")

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Completion:
        """Run one chat completion and return text plus any tool calls."""
        return self._to_completion(
            self._post(
                self._payload(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                    stream=False,
                    extra=extra_body,
                )
            )
        )

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Run one chat completion and return the assistant text.

        With ``stream=True`` the SSE deltas are consumed and concatenated, so the
        return type stays a plain string; use :meth:`stream_chat` to observe
        tokens as they arrive.
        """
        if stream:
            return "".join(
                self.stream_chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body,
                )
            )
        return self.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        ).text

    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield assistant text deltas as they arrive over SSE."""
        payload = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            stream=True,
            extra=extra_body,
        )
        try:
            with requests.post(
                self.config.chat_completions_url,
                headers=self._headers(True),
                json=payload,
                timeout=self.config.timeout,
                stream=True,
            ) as response:
                if response.status_code >= 400:
                    self._raise_for_status(response.status_code, response.text)
                yield from self._iter_sse_text(response.iter_lines(decode_unicode=True))
        except requests.Timeout as exc:
            raise NVIDIATimeoutError(
                f"NVIDIA stream timed out after {self.config.timeout}s"
            ) from exc
        except requests.ConnectionError as exc:
            raise NVIDIAError(
                f"Could not reach {self.config.base_url}: "
                f"{_redact(str(exc), self.config)}"
            ) from exc

    # ---------------------------------------------------------- async request

    async def _apost(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Async POST with bounded retries; returns the decoded JSON body."""
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(
                        self.config.chat_completions_url,
                        headers=self._headers(False),
                        json=payload,
                    )
                except httpx.TimeoutException:
                    last_error = NVIDIATimeoutError(
                        f"NVIDIA request timed out after {self.config.timeout}s"
                    )
                except httpx.TransportError as exc:
                    last_error = NVIDIAError(
                        f"Could not reach {self.config.base_url}: "
                        f"{_redact(str(exc), self.config)}"
                    )
                else:
                    if response.status_code < 400:
                        try:
                            return response.json()
                        except ValueError as exc:
                            raise NVIDIAResponseError(
                                "NVIDIA returned a non-JSON body: "
                                f"{_redact(response.text, self.config)[:400]}"
                            ) from exc

                    if response.status_code not in RETRYABLE_STATUS:
                        self._raise_for_status(response.status_code, response.text)

                    try:
                        self._raise_for_status(response.status_code, response.text)
                    except NVIDIAError as exc:
                        last_error = exc

                    if attempt < self.config.max_retries:
                        delay = _retry_after_seconds(
                        response.headers, attempt, response.status_code
                    )
                        logger.warning(
                            "NVIDIA HTTP %s; retrying in %.1fs (attempt %d/%d)",
                            response.status_code,
                            delay,
                            attempt + 1,
                            self.config.max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise last_error

                if attempt < self.config.max_retries:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise last_error

        raise last_error or NVIDIAError("NVIDIA request failed for an unknown reason")

    async def acomplete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Completion:
        """Async counterpart to :meth:`complete`; used by the ADK adapter."""
        return self._to_completion(
            await self._apost(
                self._payload(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                    stream=False,
                    extra=extra_body,
                )
            )
        )

    async def achat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Async counterpart to :meth:`chat`."""
        completion = await self.acomplete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )
        return completion.text

    async def astream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Async SSE streaming of assistant text deltas."""
        payload = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            stream=True,
            extra=extra_body,
        )
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream(
                "POST",
                self.config.chat_completions_url,
                headers=self._headers(True),
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    self._raise_for_status(response.status_code, body)
                async for line in response.aiter_lines():
                    for piece in self._iter_sse_text(iter([line])):
                        yield piece


_default_client: NVIDIAKimiClient | None = None


def get_client() -> NVIDIAKimiClient:
    """Process-wide singleton client."""
    global _default_client
    if _default_client is None:
        _default_client = NVIDIAKimiClient()
    return _default_client
