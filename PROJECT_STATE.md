# PROJECT_STATE

Single source of truth for resuming work. Keep it short.

## Current phase

The system is complete and running end to end. A full run against
`evaluation/fixtures/textstats` is in progress; component-level verification is
done.

## Completed work

**Phase 1 -- provider layer and ADK bridge**
- Python 3.13.5, `google-adk==2.8.0` in `.venv/`, git initialized.
- `app/llm/` is the only path to NVIDIA NIM: validated config with key hygiene,
  bounded retries, a typed error taxonomy, sync/async, streaming.
- `app/llm/adk_model.py`: `KimiK3Llm(BaseLlm)` registered with `LLMRegistry`.

**Phase 2 -- schemas, tools, agents, workflow**
- `app/schemas/`: the typed contract between stages, with enforced size limits.
- `app/tools/`: path-confined repository tools; secrets and binaries refused.
- `sandbox/local.sh` + `app/tools/sandbox.py`: the only execution path. Tiers
  itself bwrap > unshare > none, probing each by running it. A run refuses to
  start on `none` unless `SE_AGENT_SANDBOX_ALLOW_UNSAFE=1`.
  `SE_AGENT_SANDBOX_WSL` reaches a real tier from Windows.
- `app/agents/`: ten agents, all Kimi K3 except the model-free tester.
- `app/workflows/graph.py`: ADK 2.8.0 `Workflow` graph with explicit routing.

**Phase 3 -- entry point, telemetry, compaction, server, evaluation**
- `app/agent.py` (runner, services, preflight, CLI), `app/telemetry.py`.
- Context compaction via `App(events_compaction_config=...)`, summarizer pinned
  to `KimiK3Llm`.
- `app/server.py`: FastAPI SSE stream plus a single-file console.
- `evaluation/`: fixture project, three tasks, harness with sandboxed
  verification and a single-agent baseline.

## Important decisions

1. **`BaseLlm` adapter, not LiteLLM** -- LiteLLM would have opened a second path
   to the API, breaking centralization.
2. **Native `Workflow`, not `SequentialAgent`/`LoopAgent`** -- 2.8.0 deprecates
   those, and `Workflow` is what provides explicit routing and real cycles.
3. **The tester has no model** -- its `TestReport` comes from a process exit
   code and stdout, so a model cannot fabricate a pass.
3b. **Docker was removed entirely.** One execution path is easier to reason
    about than two; bwrap reproduces the container's guarantees where a real
    boundary is available, and where none is, the run refuses to start rather
    than pretending.
4. **Agents are factories** -- ADK allows one parent per instance.
5. **Routing reads structured state**, and an unparseable review routes to
   "changes required", never to approval.
6. **A green suite with no diff is not a pass** -- otherwise a failed
   implementation step lets the pre-existing tests carry the run to review.
7. **Compaction runs on Kimi K3** -- the summarizer would otherwise default to
   its own model.
8. **Test logs go to artifacts**, not session state.
9. **Exit 97 means the sandbox failed**, never that tests failed -- otherwise a
   broken sandbox reads as a red suite and the debugger chases a phantom bug.
10. **The script backend reports what it actually isolates** through
    `preflight()`, so nothing claims containment it does not have.

## Tests / status

`pytest -q` -> **263 passed, 1 skipped** (symlink test needs Windows
privileges). Lint clean.

Verified live against the provider: NIM chat completion, NIM function calling, a
full ADK agent loop, sandboxed execution, that `NVIDIA_API_KEY` does not
reach the sandbox, structured output round-tripping through
`set_model_response`, and the tester catching a deliberately introduced bug
with the exact failing assertion.

Verified without the provider: `tests/test_graph_integration.py` runs the whole
graph -- real Runner, real routing, real tester -- against a stub model, so
every branch (green/approved, retry-to-exhaustion, no-diff, sandbox failure,
minor -> coder, major -> architect, budget accounting) is pinned regardless of
quota.

## Operational notes

- **Latency dominates.** A single model turn takes ~111s at
  `reasoning_effort=max` and ~58s at `low` (Kimi K3 accepts only those two).
  A full run is 20+ turns. Set `NVIDIA_REASONING_EFFORT=low` for speed.
- **Free NIM tiers rate-limit hard.** 429s are retried on a 5s/10s/20s/40s/60s
  schedule; a run that exhausts the budget reports failure rather than faking
  success.
- `preflight()` names the active isolation tier, and blocks a run that would
  have none unless that was accepted explicitly.

## Blockers

None.

## Next actions

1. Record the completed end-to-end run in `progress.md`.
2. Run `python -m evaluation.run_eval --baseline` once quota allows, and record
   the comparison.
3. Optional: swap `InMemorySessionService` for the database-backed one
   (`pip install "google-adk[db]"`) so runs survive a restart.
