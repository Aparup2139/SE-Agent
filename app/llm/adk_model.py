"""ADK ``BaseLlm`` adapter for Kimi K3 on NVIDIA NIM.

This is a *provider* adapter, not an orchestration layer. It translates between
ADK's ``LlmRequest``/``LlmResponse`` (google-genai types) and the OpenAI-style
chat-completions wire format that NVIDIA NIM speaks, delegating every HTTP call
to :class:`~app.llm.nvidia_kimi.NVIDIAKimiClient`. Everything above it -- agents,
runners, sessions, events, workflows -- stays native ADK.

Registering the class with ``LLMRegistry`` means an agent can be declared the
ordinary ADK way::

    LlmAgent(model="moonshotai/kimi-k3", ...)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.genai import types
from pydantic import BaseModel

from .config import DEFAULT_MODEL
from .nvidia_kimi import Completion, NVIDIAError, NVIDIAKimiClient, get_client

logger = logging.getLogger(__name__)

# NIM reports OpenAI finish reasons; ADK wants genai FinishReason values.
_FINISH_REASONS = {
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "content_filter": types.FinishReason.SAFETY,
}


# --------------------------------------------------------------------- schema


def _schema_to_json(schema: Any) -> dict[str, Any]:
    """Convert whatever ADK hands us into a plain JSON Schema dict.

    Four shapes turn up in practice: ``None``, an already-plain dict, a genai
    ``Schema`` instance (from a tool's function declaration), and a Pydantic
    model *class* (from ``LlmAgent(output_schema=...)``, which ADK passes as the
    type itself rather than an instance).

    genai also spells types in upper case (``"OBJECT"``); JSON Schema and the
    OpenAI function-calling format want lower case.
    """
    if schema is None:
        return {"type": "object", "properties": {}}

    if isinstance(schema, dict):
        raw = schema
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        # A Pydantic model class: ask Pydantic for the schema directly.
        raw = schema.model_json_schema()
    elif isinstance(schema, BaseModel):
        raw = schema.model_dump(exclude_none=True, by_alias=True)
    else:
        # Anything else with a dict-like dump (a genai Schema, typically).
        dump = getattr(schema, "model_dump", None)
        if callable(dump):
            raw = dump(exclude_none=True, by_alias=True)
        else:
            logger.warning("unrecognized schema type %r; sending an open object", type(schema))
            return {"type": "object", "properties": {}}

    return _normalize_schema(raw)


def _normalize_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            out["type"] = value.lower()
        elif key in ("properties", "defs", "$defs"):
            out["$defs" if key in ("defs", "$defs") else key] = {
                name: _normalize_schema(sub) for name, sub in (value or {}).items()
            }
        elif key == "ref":
            out["$ref"] = value
        elif key in ("items", "any_of", "anyOf"):
            out["anyOf" if key in ("any_of", "anyOf") else "items"] = _normalize_schema(
                value
            )
        else:
            out[key] = _normalize_schema(value)
    return out


def _tools_to_openai(config: types.GenerateContentConfig) -> list[dict[str, Any]]:
    """Flatten ADK's ``tools`` into OpenAI function-tool declarations."""
    tools: list[dict[str, Any]] = []
    for tool in config.tools or []:
        declarations = getattr(tool, "function_declarations", None) or []
        for declaration in declarations:
            parameters = (
                declaration.parameters_json_schema
                if getattr(declaration, "parameters_json_schema", None)
                else _schema_to_json(declaration.parameters)
            )
            if isinstance(parameters, dict):
                parameters = _normalize_schema(parameters)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": parameters
                        or {"type": "object", "properties": {}},
                    },
                }
            )
    return tools


def _response_format(
    config: types.GenerateContentConfig,
) -> dict[str, Any] | None:
    """Map ADK structured-output settings onto NIM's ``response_format``."""
    schema = getattr(config, "response_json_schema", None)
    if not schema and config.response_schema is not None:
        schema = _schema_to_json(config.response_schema)
    if schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": False,
                "schema": _normalize_schema(schema),
            },
        }
    if config.response_mime_type == "application/json":
        return {"type": "json_object"}
    return None


# ------------------------------------------------------------------- contents


def _system_text(config: types.GenerateContentConfig) -> str:
    instruction = config.system_instruction
    if not instruction:
        return ""
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, types.Content):
        return "\n".join(p.text for p in instruction.parts or [] if p.text)
    if isinstance(instruction, list):
        chunks = []
        for item in instruction:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, types.Part) and item.text:
                chunks.append(item.text)
        return "\n".join(chunks)
    return str(instruction)


def _stringify_response(response: Any) -> str:
    """Render a tool result as the string body of a ``role="tool"`` message."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    try:
        return json.dumps(response, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(response)


def _contents_to_messages(contents: list[types.Content]) -> list[dict[str, Any]]:
    """Convert ADK conversation contents into OpenAI chat messages.

    ADK carries tool results as ``function_response`` parts on a *user* Content;
    the OpenAI format needs them as separate ``role="tool"`` messages keyed by
    ``tool_call_id``, so a single Content can fan out into several messages.
    """
    messages: list[dict[str, Any]] = []

    for content in contents:
        role = "assistant" if content.role == "model" else "user"
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for part in content.parts or []:
            # Never round-trip the model's private reasoning back to the API.
            if getattr(part, "thought", None):
                continue
            if part.text:
                text_chunks.append(part.text)
            elif part.function_call:
                call = part.function_call
                tool_calls.append(
                    {
                        "id": call.id or call.name,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.args or {}, ensure_ascii=False, default=str
                            ),
                        },
                    }
                )
            elif part.function_response:
                fr = part.function_response
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": fr.id or fr.name,
                        "content": _stringify_response(fr.response),
                    }
                )

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_chunks) or None,
                    "tool_calls": tool_calls,
                }
            )
        elif text_chunks:
            messages.append({"role": role, "content": "\n".join(text_chunks)})

        messages.extend(tool_results)

    return messages


def _completion_to_parts(completion: Completion) -> list[types.Part]:
    parts: list[types.Part] = []
    if completion.text:
        parts.append(types.Part(text=completion.text))
    for call in completion.tool_calls:
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id=call.id, name=call.name, args=call.arguments
                )
            )
        )
    return parts


def _usage(completion: Completion) -> types.GenerateContentResponseUsageMetadata | None:
    if not completion.usage:
        return None
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=completion.usage.get("prompt_tokens"),
        candidates_token_count=completion.usage.get("completion_tokens"),
        total_token_count=completion.usage.get("total_tokens"),
    )


# --------------------------------------------------------------------- model


class KimiK3Llm(BaseLlm):
    """ADK model backed by ``moonshotai/kimi-k3`` on NVIDIA NIM."""

    model: str = DEFAULT_MODEL

    @staticmethod
    def supported_models() -> list[str]:
        """Regexes ADK's registry matches agent ``model`` strings against."""
        return [r"moonshotai/.*", r"kimi.*", r"nvidia_nim/.*"]

    @property
    def _client(self) -> NVIDIAKimiClient:
        # Resolved lazily so importing the module never requires an API key.
        return get_client()

    def _build_call(self, llm_request: LlmRequest) -> dict[str, Any]:
        """Translate an ``LlmRequest`` into kwargs for the NVIDIA client."""
        config = llm_request.config or types.GenerateContentConfig()

        messages: list[dict[str, Any]] = []
        system = _system_text(config)
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_contents_to_messages(llm_request.contents or []))
        if not messages or all(m["role"] == "system" for m in messages):
            # NIM requires at least one non-system turn.
            messages.append({"role": "user", "content": "Proceed."})

        return {
            "messages": messages,
            "tools": _tools_to_openai(config) or None,
            "response_format": _response_format(config),
            "max_tokens": config.max_output_tokens,
            "temperature": config.temperature,
            "seed": config.seed,
        }

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Produce one or more ``LlmResponse`` objects for an ADK turn."""
        call = self._build_call(llm_request)

        try:
            # Tool-calling turns are never streamed: partial function-call
            # arguments cannot be executed, so a single complete response is
            # both simpler and safer than reassembling deltas.
            if stream and not call["tools"]:
                accumulated: list[str] = []
                async for delta in self._client.astream_chat(
                    call["messages"],
                    max_tokens=call["max_tokens"],
                    temperature=call["temperature"],
                    seed=call["seed"],
                ):
                    accumulated.append(delta)
                    yield LlmResponse(
                        content=types.Content(
                            role="model", parts=[types.Part(text=delta)]
                        ),
                        partial=True,
                    )
                text = "".join(accumulated)
                yield LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part(text=text)]
                    ),
                    turn_complete=True,
                    finish_reason=types.FinishReason.STOP,
                    model_version=self.model,
                )
                return

            completion = await self._client.acomplete(**call)
        except NVIDIAError as exc:
            # Surface provider failures as ADK error responses so the runner can
            # emit an event instead of tearing down the whole workflow.
            logger.error("Kimi K3 call failed: %s", exc)
            yield LlmResponse(
                error_code=type(exc).__name__,
                error_message=str(exc),
                turn_complete=True,
            )
            return

        parts = _completion_to_parts(completion)
        if not parts:
            parts = [types.Part(text="")]

        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            finish_reason=_FINISH_REASONS.get(
                completion.finish_reason or "stop", types.FinishReason.STOP
            ),
            usage_metadata=_usage(completion),
            model_version=self.model,
            turn_complete=True,
        )


LLMRegistry.register(KimiK3Llm)
