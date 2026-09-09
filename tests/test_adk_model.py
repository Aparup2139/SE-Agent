"""ADK <-> NVIDIA translation: contents, tools, schemas, responses."""

from __future__ import annotations

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.genai import types

from app.llm.adk_model import (
    KimiK3Llm,
    _contents_to_messages,
    _normalize_schema,
    _response_format,
    _schema_to_json,
    _system_text,
    _tools_to_openai,
)
from app.llm.nvidia_kimi import Completion, ToolCall

# ------------------------------------------------------------------- registry


def test_kimi_is_resolvable_by_model_string():
    assert LLMRegistry.resolve("moonshotai/kimi-k3") is KimiK3Llm


def test_default_model_is_kimi_k3():
    assert KimiK3Llm().model == "moonshotai/kimi-k3"


def test_ollama_model_name_is_also_resolvable():
    # SE-Agent's only custom model class must serve whichever provider is
    # configured, so a non-NVIDIA model shape (Ollama's "name:tag") resolves
    # too, without needing a code change to switch providers.
    assert LLMRegistry.resolve("qwen3:4b") is KimiK3Llm


# --------------------------------------------------------------------- schema


def test_genai_schema_types_are_lowercased():
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={"path": types.Schema(type=types.Type.STRING)},
        required=["path"],
    )
    result = _schema_to_json(schema)
    assert result["type"] == "object"
    assert result["properties"]["path"]["type"] == "string"
    assert result["required"] == ["path"]


def test_nested_array_schema_is_normalized():
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "files": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))
        },
    )
    files = _schema_to_json(schema)["properties"]["files"]
    assert files["type"] == "array"
    assert files["items"]["type"] == "string"


def test_none_schema_becomes_empty_object():
    assert _schema_to_json(None) == {"type": "object", "properties": {}}


def test_ref_and_defs_are_renamed_to_json_schema_spelling():
    normalized = _normalize_schema({"defs": {"A": {"type": "STRING"}}, "ref": "#/defs/A"})
    assert "$defs" in normalized
    assert normalized["$ref"] == "#/defs/A"
    assert normalized["$defs"]["A"]["type"] == "string"


# ---------------------------------------------------------------------- tools


def test_function_declarations_become_openai_tools():
    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="read_file",
                        description="Read a file",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={"path": types.Schema(type=types.Type.STRING)},
                        ),
                    )
                ]
            )
        ]
    )
    tools = _tools_to_openai(config)
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "read_file"
    assert tools[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"


def test_no_tools_yields_empty_list():
    assert _tools_to_openai(types.GenerateContentConfig()) == []


# ----------------------------------------------------------- system + contents


def test_system_instruction_accepts_plain_string():
    config = types.GenerateContentConfig(system_instruction="Be terse.")
    assert _system_text(config) == "Be terse."


def test_system_instruction_accepts_content():
    config = types.GenerateContentConfig(
        system_instruction=types.Content(parts=[types.Part(text="Be terse.")])
    )
    assert _system_text(config) == "Be terse."


def test_user_and_model_turns_map_to_openai_roles():
    contents = [
        types.Content(role="user", parts=[types.Part(text="hi")]),
        types.Content(role="model", parts=[types.Part(text="hello")]),
    ]
    assert _contents_to_messages(contents) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_function_call_becomes_assistant_tool_calls():
    contents = [
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="c1", name="read_file", args={"path": "a.py"}
                    )
                )
            ],
        )
    ]
    message = _contents_to_messages(contents)[0]
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == "c1"
    assert message["tool_calls"][0]["function"]["name"] == "read_file"
    assert '"path"' in message["tool_calls"][0]["function"]["arguments"]


def test_function_response_becomes_tool_message():
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="c1", name="read_file", response={"content": "x = 1"}
                    )
                )
            ],
        )
    ]
    messages = _contents_to_messages(contents)
    assert messages == [{"role": "tool", "tool_call_id": "c1", "content": '{"content": "x = 1"}'}]


def test_tool_call_and_result_round_trip_preserves_ids():
    contents = [
        types.Content(role="user", parts=[types.Part(text="read it")]),
        types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(id="c9", name="f", args={}))],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="c9", name="f", response={"ok": True}
                    )
                )
            ],
        ),
    ]
    messages = _contents_to_messages(contents)
    assert messages[1]["tool_calls"][0]["id"] == "c9"
    assert messages[2]["tool_call_id"] == "c9"


def test_thought_parts_are_never_sent_back():
    contents = [
        types.Content(
            role="model",
            parts=[
                types.Part(text="secret reasoning", thought=True),
                types.Part(text="visible answer"),
            ],
        )
    ]
    messages = _contents_to_messages(contents)
    assert messages[0]["content"] == "visible answer"
    assert "secret reasoning" not in messages[0]["content"]


# ------------------------------------------------------------ structured output


def test_response_schema_becomes_json_schema_format():
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            properties={"summary": types.Schema(type=types.Type.STRING)},
        ),
    )
    fmt = _response_format(config)
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["properties"]["summary"]["type"] == "string"


def test_json_mime_without_schema_uses_json_object():
    config = types.GenerateContentConfig(response_mime_type="application/json")
    assert _response_format(config) == {"type": "json_object"}


def test_plain_text_needs_no_response_format():
    assert _response_format(types.GenerateContentConfig()) is None


# ------------------------------------------------------------- request build


def test_empty_conversation_gets_a_user_turn():
    """NIM rejects a request whose only message is a system prompt."""
    request = LlmRequest(
        contents=[],
        config=types.GenerateContentConfig(system_instruction="Do the thing."),
    )
    messages = KimiK3Llm()._build_call(request)["messages"]
    assert messages[0]["role"] == "system"
    assert any(m["role"] == "user" for m in messages)


def test_generation_settings_are_forwarded():
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=512, seed=7),
    )
    call = KimiK3Llm()._build_call(request)
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 512
    assert call["seed"] == 7


def _tool_config(*names: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(name=name, description="d") for name in names
                ]
            )
        ]
    )


def test_tool_choice_is_forced_when_set_model_response_is_offered():
    # A weak model can answer in plain text instead of calling the tool that
    # carries structured output -- ADK's own validation for that path has no
    # retry and crashes the whole run (see set_model_response_tool.py vs.
    # llm_agent.py's unguarded __maybe_save_output_to_state). Forcing a tool
    # call whenever this tool is offered keeps every turn on ADK's
    # retry-capable path instead.
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
        config=_tool_config("read_file", "set_model_response"),
    )
    call = KimiK3Llm()._build_call(request)
    assert call["tool_choice"] == "required"


def test_tool_choice_is_untouched_for_ordinary_tools():
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
        config=_tool_config("read_file"),
    )
    call = KimiK3Llm()._build_call(request)
    assert call.get("tool_choice") is None


def test_tool_choice_is_untouched_with_no_tools():
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )
    call = KimiK3Llm()._build_call(request)
    assert call.get("tool_choice") is None


# ---------------------------------------------------------------- responses


@pytest.mark.asyncio
async def test_text_completion_becomes_llm_response(monkeypatch):
    model = KimiK3Llm()

    async def fake(**kwargs):
        return Completion(text="done", finish_reason="stop")

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="go")])],
        config=types.GenerateContentConfig(),
    )
    responses = [r async for r in model.generate_content_async(request)]
    assert len(responses) == 1
    assert responses[0].content.parts[0].text == "done"
    assert responses[0].content.role == "model"


@pytest.mark.asyncio
async def test_tool_call_completion_becomes_function_call_part(monkeypatch):
    model = KimiK3Llm()

    async def fake(**kwargs):
        return Completion(
            tool_calls=[ToolCall(id="c1", name="run_tests", arguments={"path": "tests"})],
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="test it")])],
        config=types.GenerateContentConfig(),
    )
    responses = [r async for r in model.generate_content_async(request)]
    call = responses[0].content.parts[0].function_call
    assert call.name == "run_tests"
    assert call.args == {"path": "tests"}


@pytest.mark.asyncio
async def test_provider_failure_becomes_error_response(monkeypatch):
    from app.llm.nvidia_kimi import NVIDIAAuthError

    model = KimiK3Llm()

    async def fake(**kwargs):
        raise NVIDIAAuthError("bad key")

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="go")])],
        config=types.GenerateContentConfig(),
    )
    responses = [r async for r in model.generate_content_async(request)]
    assert responses[0].error_code == "NVIDIAAuthError"
    assert "bad key" in responses[0].error_message


# ------------------------------------------- structured-output guardrail


def _schema_request() -> LlmRequest:
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="go")])],
        config=_tool_config("read_file", "set_model_response"),
    )


@pytest.mark.asyncio
async def test_plain_text_reply_is_retried_until_a_tool_call_lands(monkeypatch):
    # A weak model can answer in plain text instead of calling
    # set_model_response, which ADK cannot recover from on its own (see
    # llm_agent.py's unguarded __maybe_save_output_to_state). This is our own
    # safety net: keep nudging the model until it actually calls the tool.
    model = KimiK3Llm()
    attempts = []

    async def fake(messages, **kwargs):
        attempts.append(messages)
        if len(attempts) < 3:
            return Completion(text="here is my answer in prose", finish_reason="stop")
        return Completion(
            tool_calls=[ToolCall(id="c1", name="set_model_response", arguments={})],
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    responses = [r async for r in model.generate_content_async(_schema_request())]

    assert len(attempts) == 3
    call = responses[0].content.parts[0].function_call
    assert call.name == "set_model_response"


@pytest.mark.asyncio
async def test_giving_up_becomes_a_recoverable_error_not_a_crash(monkeypatch):
    model = KimiK3Llm()

    async def always_plain_text(messages, **kwargs):
        return Completion(text="still prose", finish_reason="stop")

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(always_plain_text))
    responses = [r async for r in model.generate_content_async(_schema_request())]

    assert responses[-1].error_code == "StructuredOutputError"


@pytest.mark.asyncio
async def test_a_real_tool_call_on_the_first_try_is_never_retried(monkeypatch):
    model = KimiK3Llm()
    attempts = []

    async def fake(messages, **kwargs):
        attempts.append(messages)
        return Completion(
            tool_calls=[ToolCall(id="c1", name="set_model_response", arguments={})],
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    [r async for r in model.generate_content_async(_schema_request())]

    assert len(attempts) == 1


# ---------------------------------------------- output_schema as a model class


def test_pydantic_model_class_is_accepted_as_a_response_schema():
    """ADK passes LlmAgent(output_schema=Model) as the class, not an instance."""
    from app.schemas import Requirements

    result = _schema_to_json(Requirements)
    assert result["type"] == "object"
    assert "summary" in result["properties"]


def test_response_format_built_from_a_model_class():
    from app.schemas import Requirements

    config = types.GenerateContentConfig(response_schema=Requirements)
    fmt = _response_format(config)
    assert fmt["type"] == "json_schema"
    assert "summary" in fmt["json_schema"]["schema"]["properties"]


def test_nested_model_refs_survive_conversion():
    """Requirements nests AcceptanceCriterion, so $defs/$ref must be preserved."""
    from app.schemas import Requirements

    result = _schema_to_json(Requirements)
    assert "$defs" in result
    assert "AcceptanceCriterion" in result["$defs"]


def test_unrecognized_schema_degrades_to_an_open_object():
    assert _schema_to_json(object()) == {"type": "object", "properties": {}}


# ------------------------------------------- recovering a described tool call


def _schema_request_with_fields() -> LlmRequest:
    """A ``set_model_response`` declared with real parameters, as ADK does."""
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="go")])],
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="set_model_response",
                            description="d",
                            parameters=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "summary": types.Schema(type=types.Type.STRING),
                                    "risks": types.Schema(type=types.Type.STRING),
                                },
                            ),
                        )
                    ]
                )
            ]
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        # The wrapper spellings a model reaches for, and the bare payload.
        '{"tool": "set_model_response", "arguments": {"summary": "s"}}',
        '{"name": "set_model_response", "arguments": {"summary": "s"}}',
        '{"function": {"name": "set_model_response", "args": {"summary": "s"}}}',
        '{"summary": "s"}',
        'Sure!\n```json\n{"tool": "set_model_response", "args": {"summary": "s"}}\n```',
        # Arguments handed back re-encoded as a string.
        '{"name": "set_model_response", "arguments": "{\\"summary\\": \\"s\\"}"}',
    ],
)
@pytest.mark.asyncio
async def test_a_described_tool_call_is_recovered_not_retried(monkeypatch, text):
    """The payload is right, only the channel is wrong -- re-asking is futile.

    A small model routinely puts the call in ``content`` as JSON. It believes
    it complied, so nudging it produces the same answer again; read the call
    out of the text instead.
    """
    model = KimiK3Llm()
    attempts = []

    async def fake(messages, **kwargs):
        attempts.append(messages)
        return Completion(text=text, finish_reason="stop")

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    responses = [r async for r in model.generate_content_async(_schema_request_with_fields())]

    assert len(attempts) == 1  # recovered on the spot, never re-asked
    call = responses[0].content.parts[0].function_call
    assert call.name == "set_model_response"
    assert call.args == {"summary": "s"}


@pytest.mark.asyncio
async def test_prose_with_unrelated_json_is_not_mistaken_for_a_call(monkeypatch):
    """Stray JSON must not become a fabricated call with garbage arguments."""
    model = KimiK3Llm()

    async def fake(messages, **kwargs):
        return Completion(text='I would set {"unrelated": 1} on the config.', finish_reason="stop")

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    responses = [r async for r in model.generate_content_async(_schema_request_with_fields())]

    assert responses[-1].error_code == "StructuredOutputError"


@pytest.mark.asyncio
async def test_a_call_naming_a_different_tool_is_not_recovered(monkeypatch):
    model = KimiK3Llm()

    async def fake(messages, **kwargs):
        return Completion(
            text='{"tool": "read_file", "arguments": {"path": "x"}}',
            finish_reason="stop",
        )

    monkeypatch.setattr(type(model._client), "acomplete", staticmethod(fake))
    responses = [r async for r in model.generate_content_async(_schema_request_with_fields())]

    assert responses[-1].error_code == "StructuredOutputError"
