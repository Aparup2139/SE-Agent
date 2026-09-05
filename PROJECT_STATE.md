# PROJECT_STATE

Single source of truth for resuming work. Keep it short.

## Current phase

Phase 3: the system is complete and wired end to end. Verifying a full run
against a real target repository.

## Completed work

**Phase 1 -- provider layer and ADK bridge**
- Python 3.13.5, `google-adk==2.8.0` in `.venv/`, Docker 28.3.0, git initialized.
- `app/llm/`: the only path to NVIDIA NIM. Validated config with key hygiene,
  bounded retries, a typed error taxonomy, sync/async, streaming.
- `app/llm/adk_model.py`: `KimiK3Llm(BaseLlm)`, registered with `LLMRegistry`,
  so agents are declared as `LlmAgent(model="moonshotai/kimi-k3", ...)`.

**Phase 2 -- schemas, tools, agents, workflow**
- `app/schemas/`: typed contract between stages with enforced size limits.
- `app/tools/`: path-confined repository tools; secrets and binaries refused.
- `app/tools/sandbox.py` + `sandbox/Dockerfile`: Docker execution with no
  network, read-only root, dropped capabilities, allowlisted environment.
- `app/agents/`: ten agents, all Kimi K3, exposed as factories.
- `app/workflows/graph.py`: ADK 2.8.0 `Workflow` graph with explicit routing.

**Phase 3 -- entry point, telemetry, compaction**
- `app/agent.py`: runner, session/artifact/memory services, preflight, CLI.
- `app/telemetry.py`: ADK events -> phase checklist and progress updates.
- Context compaction enabled via `App(events_compaction_config=...)`, with the
  summarizer given an explicit `KimiK3Llm`.
- `README.md`, `evaluation/fixtures/textstats/` (a real target project).

## Important decisions

1. **`BaseLlm` adapter, not LiteLLM.** LiteLLM would have opened a second path
   to the API, breaking centralization. Subclassing `BaseLlm` keeps every HTTP
   call in `NVIDIAKimiClient` while leaving everything above it stock ADK.
2. **Native `Workflow`, not `SequentialAgent`/`LoopAgent`.** ADK 2.8.0
   deprecates those in favour of `Workflow`, which is also the API that
   provides explicit routing and real cycles. Verified that cycles, `ctx.route`
   and `ctx.state` all work before committing to it.
3. **The tester has no model.** A model asked to report test results can
   fabricate a pass silently. `TesterAgent` is a deterministic `BaseAgent` whose
   `TestReport` comes from a process exit code and stdout.
4. **Agents are factories, not singletons.** ADK allows one parent per agent
   instance, and the architect appears twice in the graph.
5. **Routing reads structured state.** `tests_passed` and
   `CodeReview.routing_scope` decide the next node. An unparseable review routes
   to "changes required", never to approval.
6. **Compaction runs on Kimi K3.** The summarizer defaults to its own model;
   passing `KimiK3Llm()` explicitly closes the one gap where another model could
   have entered the system.
7. **`reasoning_content` is dropped everywhere** -- provider, state, telemetry.

## Files changed

```
app/{agent,telemetry}.py
app/llm/{__init__,config,nvidia_kimi,adk_model}.py
app/agents/{__init__,common,requirements,repository,risk,architecture,
            planner,coder,coordinator,debugger,reviewer,tester}.py
app/tools/{__init__,workspace,repo_tools,sandbox}.py
app/schemas/{__init__,workflow}.py
app/workflows/{__init__,graph}.py
sandbox/Dockerfile   evaluation/fixtures/textstats/
tests/{conftest,test_llm_config,test_nvidia_client,test_adk_model,
       test_schemas,test_tools,test_sandbox,test_workflow,test_telemetry}.py
README.md  pyproject.toml  .gitignore  .env.example
```

## Tests / status

`pytest -q` -> **204 passed, 1 skipped** (symlink test needs Windows privileges),
plus 2 live Docker tests that pass when the daemon is up.

Verified live: NIM chat completion, NIM function calling, a full ADK agent loop,
Docker sandbox execution, and that `NVIDIA_API_KEY` does not reach the sandbox.

## Blockers

None. Docker Desktop must be running for the sandbox; `preflight()` reports it
clearly when it is not.

## Next actions

1. Finish the end-to-end run on `evaluation/fixtures/textstats` and record the
   result in `progress.md`.
2. Build out `evaluation/` into a small measured suite (task success, test
   success, iterations, tool calls, latency) against a single-agent baseline.
3. Optional: a frontend over `stream_run`, which already yields exactly the
   updates a UI needs.
