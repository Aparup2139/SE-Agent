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

## Phase 2 -- schemas, tools, agents, workflow

**Goal:** everything between a feature request and a reviewed change.

### What was built

- `app/schemas/workflow.py` -- the typed contract between stages, with enforced
  length and count limits. Two properties carry the weight: `TestReport.is_green`
  (false unless tests actually ran and all passed) and `CodeReview.routing_scope`
  (derived from the findings, so routing follows the review's content).
- `app/tools/` -- `workspace.py` holds the safety primitives every tool goes
  through; `repo_tools.py` the read/search/git/edit tools; `sandbox.py` the
  Docker executor and the pytest output parser.
- `app/agents/` -- ten agents. Nine are Kimi K3 with role-specific prompts; the
  tester has no model at all.
- `app/workflows/graph.py` -- the graph.

### Notes worth remembering

- **ADK 2.8.0 deprecates `SequentialAgent`, `ParallelAgent` and `LoopAgent`** in
  favour of `Workflow`. The deprecation warning surfaced during testing. Since
  `Workflow` is also the API that provides explicit routing and real cycles --
  exactly what this design needs -- the workflow was rebuilt on it.
- Before committing to `Workflow`, a throwaway probe confirmed that `ctx.state`,
  `ctx.route` and **back-edges (cycles)** all work as hoped.
- **ADK allows an agent instance exactly one parent.** The first assembly failed
  with `architecture_agent already has a parent agent`. Agents became factories.
- ADK reconciles `output_schema` with `tools` itself, via an injected
  `set_model_response` tool, so an agent can have both.
- The sandbox image must be built before tests can run; `preflight()` says so.

### Result

182 tests passing, including live Docker isolation checks.

## Phase 3 -- entry point, telemetry, compaction, evaluation

### What was built

- `app/agent.py` (runner, services, preflight, CLI), `app/telemetry.py`
  (events -> phase checklist), context compaction, `evaluation/`.

### Two bugs the first live run found

1. **`output_schema` arrives as a class, not an instance.** ADK passes
   `LlmAgent(output_schema=Requirements)` through as the Pydantic *class*, so
   `_schema_to_json` called an unbound `model_dump` and the first node died with
   `BaseModel.model_dump() missing 1 required positional argument: 'self'`.
   It now handles all four shapes ADK produces.
2. **Timeout and retry budget were both too small.** 120s is not enough for
   `reasoning_effort=max`, and three quick retries cannot outlast a rate-limit
   window. Now 300s and 5 retries, with 429 on its own slower schedule
   (5s/10s/20s/40s/60s) since NIM does not reliably send `Retry-After`.

### The end-to-end run

The full workflow ran against `evaluation/fixtures/textstats`. The NVIDIA
account was rate-limited throughout, so **every model call returned HTTP 429**
and no code was written. What that run still proves is worth recording:

- All eight phases were traversed in the correct order.
- Routing worked: implement -> test -> review -> back to implement, then the
  iteration budget was spent and the graph terminated at the report.
- The iteration ceiling held: `--max-iterations 2` produced exactly two
  implementation attempts before routing to `exhausted`.
- **The tester ran twice against real Docker and reported 6 passed** -- it needs
  no model, which is exactly why it was built that way.
- The run exited 1 and produced no report. It did not claim success.

### The gap that run exposed

The tester reported a green suite while the coder had never run, and the graph
routed that to review. A green suite with an unchanged working tree is the
pre-existing suite, and proves nothing about the requested feature.
`decide_after_tests` now requires a non-empty diff before treating a pass as a
pass, and a git failure counts as "no evidence of change" rather than evidence.

### Still outstanding

A full successful run needs NIM quota. The system is verified at every level
that does not depend on it.

### Result

212 tests passing.

## Phase 4 -- server, security, and what the live runs established

### What was built

- `app/server.py` -- FastAPI SSE stream over the same `stream_run` the CLI uses,
  plus a single-file console so the server is usable without a build step.
- `SE_AGENT_MAX_CONCURRENCY` -- an escape hatch for tight provider quotas. The
  analysis fan-out doubles the instantaneous request rate, which on the free NIM
  tier costs more in 429 retries than the parallelism saves.

### Security review of the server commit

An automated review flagged the first server commit, correctly. `POST /api/run`
started an agent that writes files and runs commands, accepted an arbitrary host
path as its target, and required no authentication. On any non-loopback
interface that is remote code execution.

Fixed with two controls, neither defaulting to open:

- `SE_AGENT_API_TOKEN`, compared with `hmac.compare_digest`. With no token set,
  only loopback callers are served -- binding to `0.0.0.0` must not silently
  hand out code execution.
- `SE_AGENT_REPO_ROOT` confines the target directory, resolving before the
  containment check so traversal and symlinks both fail. Unset, the `repo` field
  is refused rather than defaulting to anywhere on disk.

### Verification that does not depend on the provider

- **The tester detects a real bug.** An off-by-one was introduced into
  `word_count` in a copy of the fixture; the sandbox reported `5 passed,
  1 failed`, `is_green=False`, and the exact assertion
  (`test_word_count_counts_words - AssertionError: assert 5 == 4`). This is the
  safety-critical path, verified end to end against real Docker.
- Structured output round-trips: the repository agent returned a populated
  `RepositoryAnalysis` through ADK's `set_model_response` tool, so the
  schema/tools path works against Kimi K3 through the adapter.
- 238 tests pass; lint clean.

### The blocker

The NVIDIA account is rate-limited to the point where a full run cannot finish.
Two complete attempts were made. The first (`reasoning_effort=max`) 429'd on
every model call; the second (`low`) got Requirements and the repository
analysis through -- both producing real, correct structured output -- before the
quota was exhausted again.

Measured latency, for planning purposes: a single model turn is ~111s at `max`
and ~58s at `low`, and a full run is 20+ turns.

### What the failed runs did prove

Both runs traversed all eight phases in order, honoured the iteration ceiling
exactly, ran the tester against real Docker, terminated at the report node, and
**exited non-zero without producing a report** rather than claiming success.
The failure mode this project was built to prevent did not occur even when
every model call failed.

### Next action

Re-run `python -m app.agent --repo ... "<request>"` when quota allows, and
`python -m evaluation.run_eval --baseline` for the measured comparison.

## Sandbox backends

Docker is no longer the only way to execute repository code. `SE_AGENT_SANDBOX`
now selects between the container backend and `sandbox/local.sh`, a shell
script that applies whatever isolation the host OS actually offers.

The script was chosen over a second Python backend because it keeps every
OS-specific decision in ~130 readable lines, and because the contract it
satisfies -- `script <session-id> <timeout> <command>`, exit 97 for
infrastructure failure -- makes a future backend (OpenShell, bubblewrap, a VM)
a new script rather than new Python.

### What it isolates, honestly

| Platform | Applied |
| --- | --- |
| Everywhere | `env -i` allowlist, workspace as cwd, private HOME/TMPDIR, file-size ceiling |
| Linux / WSL | additionally `unshare -rn` network namespace, process ceiling |
| Windows | nothing further -- **not a security boundary** |

`preflight()` reports the probe verbatim, so the GUI health line states what is
actually in force rather than implying containment.

### Verified

- The project's own suite runs green through the script backend (6 passed on
  the textstats fixture).
- An off-by-one introduced into `word_count` is caught with the exact
  assertion: `FAILED test_word_count_counts_words - AssertionError: assert 5 == 4`.
- `NVIDIA_API_KEY`, `SE_AGENT_API_TOKEN` and `AWS_SECRET_ACCESS_KEY` are absent
  from the command's environment; only 16 allowlisted variables are present.
- A broken sandbox reports `execution_error` with zero failures, so it cannot
  be mistaken for a red suite.

### Two bugs found while building it

1. The clean environment discards the active virtualenv, so the command got the
   system Python without pytest. The script now prefers the workspace `.venv`,
   overridable with `SE_AGENT_SANDBOX_PATH`.
2. A Windows path contains a colon, which is also the `PATH` separator, so
   prepending `C:/...` silently produced two nonexistent entries. Everything
   bound for `PATH` now goes through `cygpath -u` first.

## Docker removed; the sandbox tiers itself

Docker is gone -- image, backend class, daemon check, marker and all. There is
one execution path now: `sandbox/local.sh`.

### Tiers

The script picks the strongest mechanism available and **probes each by running
it**, so something present but broken (a kernel with unprivileged user
namespaces disabled) demotes rather than failing every run.

| Tier | Boundary |
| --- | --- |
| `bwrap` | mount+net+pid namespaces; workspace is the only writable path |
| `unshare` | network and user namespace |
| `none` | clean environment only -- not a boundary |

### The gate

A run **refuses to start** on `none` unless `SE_AGENT_SANDBOX_ALLOW_UNSAFE=1`.
Unsandboxed execution is now a decision someone made, not a silent default, and
the preflight string keeps saying `ACCEPTED AS UNSAFE` afterwards so no UI can
imply otherwise. `SE_AGENT_SANDBOX_REQUIRE` sets a floor in the other
direction.

### Verified on this machine

- Windows, no flags: preflight **refuses**, naming all three ways forward.
- `SE_AGENT_SANDBOX_ALLOW_UNSAFE=1`: allowed, labelled `ACCEPTED AS UNSAFE`.
- `SE_AGENT_SANDBOX_WSL=Ubuntu-24.04`: preflight passes with no gate, tier
  `unshare`, commands execute in the workspace.
- Network through the full Python -> WSL -> script path: `BLOCKED` by default,
  `REACHABLE` only with `network=True`.
- `NVIDIA_API_KEY`, `SE_AGENT_API_TOKEN`: absent. 13 variables visible.

### Not verified here

`bwrap` is not installed in this WSL distribution and installing it needs sudo,
so the top tier is implemented and probed but has not been exercised. The probe
is a functional check rather than a `command -v`, so a broken bwrap demotes to
`unshare` instead of breaking runs.

### The cost of removing Docker, stated plainly

The container image carried the toolchain. Without it the sandbox uses the
host's, so reaching a real boundary from Windows means WSL -- and the project's
dependencies must then be installed inside WSL, because that is the Python that
will run the tests. There is no way around this that does not reintroduce an
image.
