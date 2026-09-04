# Progress log

Append-only. Newest last.

## Phase 1 -- Provider layer and ADK bridge

**Goal:** get from an empty directory to a real ADK agent running on Kimi K3.

### Verification performed

| Check | Result |
| --- | --- |
| `python --version` | 3.13.5 |
| `google-adk` version | **2.8.0** (was 1.28.0 globally; pinned in `.venv/`) |
| Docker | 28.3.0 |
| Live NIM chat completion | 200 OK, returned `SE-AGENT-OK` |
| Live NIM function calling | 200 OK, `finish_reason=tool_calls` |
| ADK agent + tool + runner | tool call -> result -> answer |
| `pytest -q` | 54 passed |

### What was built

- `app/llm/config.py` -- validated settings, key hygiene, actionable startup
  error when `NVIDIA_API_KEY` is missing.
- `app/llm/nvidia_kimi.py` -- the only HTTP path to NVIDIA NIM. Sync and async,
  streaming and non-streaming, bounded retries with `Retry-After` support, and
  a typed error taxonomy (`auth` / `rate limit` / `server` / `timeout` /
  `response`).
- `app/llm/adk_model.py` -- `KimiK3Llm(BaseLlm)`, registered with
  `LLMRegistry`. Translates genai `Content`/`Part`/`Schema` to and from the
  OpenAI-style wire format NIM speaks.

### Notes worth remembering

- ADK 2.8.0's `LlmRequest.config.tools` holds genai `Tool` objects whose
  `Schema.type` is upper case (`"OBJECT"`). OpenAI function schemas need lower
  case, hence `_normalize_schema`.
- ADK represents a tool *result* as a `function_response` part on a **user**
  `Content`. OpenAI needs a separate `role="tool"` message, so one ADK Content
  can fan out into several wire messages.
- `google.adk.models.llm_request` emits an experimental-feature warning for
  JSON-schema function declarations; suppressed in `pyproject.toml`.
- Reading ADK source on Windows needs `PYTHONIOENCODING=utf-8` -- the default
  cp1252 console codec chokes on arrows in docstrings.

### Result

Phase 1 complete. Foundation verified live, not just typechecked.
