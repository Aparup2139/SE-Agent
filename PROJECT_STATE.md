# PROJECT_STATE

Single source of truth for resuming work. Keep it short.

## Current phase

Phase 1 complete (provider + ADK bridge verified live). Starting Phase 2:
structured schemas and repository tools.

## Completed work

- **Environment verified.** Python 3.13.5, `google-adk==2.8.0` installed into
  `.venv/`, Docker 28.3.0 present, git repository initialized.
- **Centralized NVIDIA layer** (`app/llm/`). `config.py` validates settings and
  keeps the API key out of `repr`/`str`/logs; `nvidia_kimi.py` is the sole HTTP
  path to NIM, with bounded retries, a full error taxonomy, and SSE streaming.
- **Live Kimi K3 request verified** against
  `https://integrate.api.nvidia.com/v1/chat/completions`.
- **Native ADK adapter** (`app/llm/adk_model.py`). `KimiK3Llm` subclasses
  `BaseLlm` and is registered with `LLMRegistry`, so agents are declared the
  ordinary way: `LlmAgent(model="moonshotai/kimi-k3", ...)`.
- **End-to-end ADK loop verified live**: agent -> tool call -> tool result ->
  final answer, through `InMemoryRunner`.
- **54 unit tests passing.**

## Important decisions

1. **`BaseLlm` adapter, not LiteLLM.** ADK 2.8.0 has no native NVIDIA NIM
   provider. A LiteLLM route would have worked but would have opened a second
   path to the API, breaking the "centralize NVIDIA access" rule. Subclassing
   `BaseLlm` keeps every HTTP call inside `NVIDIAKimiClient` while leaving
   agents, runner, sessions, events and workflows as stock ADK.
2. **Kimi K3 supports native function calling on NIM** -- confirmed live, so
   ADK tools work without a text-parsing fallback.
3. **`reasoning_content` is dropped everywhere.** NIM returns Kimi's private
   chain-of-thought in that field; the client reads only `content`, and
   `thought` parts are stripped before being sent back. No reasoning reaches
   agents, state, or the UI.
4. **Tool-calling turns are never streamed.** Partial function-call arguments
   are not executable, so those turns return one complete response.
5. **Provider errors become `LlmResponse` errors,** not exceptions, so a failed
   call surfaces as an ADK event instead of killing the workflow.
6. **The `.env` key was normalized.** It arrived as `Nvidia_API_key="Bearer
   nvapi-..."`; the loader now strips a `Bearer ` prefix and quotes, and reads
   several historical spellings.

## Files changed

```
app/llm/{__init__,config,nvidia_kimi,adk_model}.py
tests/{conftest,test_llm_config,test_nvidia_client,test_adk_model}.py
pyproject.toml  .gitignore  .env.example  PROJECT_STATE.md  progress.md
```

## Tests / status

`python -m pytest -q` -> **54 passed**. No integration tests written yet
(live calls verified manually).

## Blockers

None.

## Next actions

1. Pydantic schemas in `app/schemas/` (`Requirements`, `RepositoryAnalysis`,
   `RiskAnalysis`, `ArchitectureDesign`, `ImplementationPlan`, `TestReport`,
   `CodeReview`, `FinalReport`).
2. Repository tools in `app/tools/` with path-traversal guards and output caps.
3. Docker sandbox in `sandbox/` for real test execution.
4. Agents, then the graph workflow and the dynamic coding loop.
