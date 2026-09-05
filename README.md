# SE-Agent

A multi-agent software engineering assistant built on **Google ADK 2.8.0**, with
**Kimi K3 on NVIDIA NIM** as its only model.

Give it a feature request in plain language. It analyses the repository, designs
the change, plans it, implements it, **runs the real tests in a sandbox**, reviews
its own work independently, fixes what the review finds, and reports honestly on
what it did and did not achieve.

---

## What makes it trustworthy

Most of the design effort went into making the system unable to lie to you.

**Test results come from a process, not a prediction.** The tester agent has no
model attached. It runs the suite in the sandbox and builds its report from the
exit code and stdout. There is no path by which a model's opinion becomes a test
result. A suite that collects no tests is reported as an error, never as a pass.

**Routing is deterministic.** Where the workflow goes next is computed from
structured state -- the tester's real exit code, the review's own findings --
never by asking a model to decide. A review that fails to parse is treated as
"changes required", so a malformed response can never become an accidental
approval.

**The reviewer cannot edit.** It holds read-only tools, so its independence is
structural rather than a matter of instruction. It cannot quietly fix a problem
instead of reporting it.

**Repository code runs behind a sandbox.** By default that is a container with
no network, a read-only root filesystem, dropped capabilities, and bounded CPU,
memory and processes. A script backend is available where Docker is not; it
applies what the host OS offers and *states plainly what that is*, including
when the answer is "no boundary" -- see **Sandbox backends**.

What holds under both is the environment allowlist: `NVIDIA_API_KEY` and your
host credentials are absent from the environment the tested code sees, rather
than merely unmentioned.

**Private reasoning stays private.** Kimi K3 returns its chain-of-thought in a
separate field. It is dropped in the provider layer, never stored in state, and
never surfaced in the UI.

---

## Requirements

- Python 3.11+
- Docker (running) -- executes repository code and tests.
  A script backend is available instead; see **Sandbox backends** below.
- An NVIDIA NIM API key from <https://build.nvidia.com/>

## Setup

**1. Create the environment**

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

**2. Add your API key**

```bash
cp .env.example .env            # then edit it and set NVIDIA_API_KEY
```

`.env` is gitignored. The key is never logged, never included in an exception
message, and never passed into the sandbox.

For a substantially faster first run, also set `NVIDIA_REASONING_EFFORT=low`.

**3. Choose a sandbox backend**

Repository code is never executed unguarded, but you pick how it is guarded.

*Docker (default, a real security boundary):*

```bash
docker build -t se-agent-sandbox:latest sandbox/
```

*Or the script backend, which needs no container:*

```bash
export SE_AGENT_SANDBOX=script          # Windows PowerShell: $env:SE_AGENT_SANDBOX="script"
```

See **Sandbox backends** below for what each one actually isolates. On Windows
the script backend is not a security boundary.

**4. Confirm everything is ready**

```bash
python -c "from app.agent import preflight; print(preflight() or 'ready')"
```

This checks the key, the workspace and the selected sandbox, and prints *every*
problem at once rather than failing on the first. `ready` means a run can start.

**5. Run the tests**

```bash
pytest -q                  # 272 passed, 1 skipped
pytest -q -m "not docker"  # if the Docker daemon is not running
```

---

## Running it

### Option A -- the command line

```bash
python -m app.agent "add a /health endpoint that returns the build SHA"

python -m app.agent --repo ../my-project --max-iterations 3 \
    "the retry helper drops the Retry-After header; fix it and add a test"
```

| Flag | Meaning |
| --- | --- |
| `--repo PATH` | The repository to work in (default: the current directory) |
| `--max-iterations N` | Implement/test/review attempts (default 5, hard ceiling 10) |
| `--verbose` | Debug logging |

The process exits non-zero if the run did not succeed, so it composes in CI.

### Option B -- the web console

```bash
uvicorn app.server:app
```

Open <http://127.0.0.1:8000>.

1. The header shows a health dot. **Run stays disabled** until preflight passes,
   and the problems are printed next to it -- so a stopped Docker daemon is
   visible before you submit rather than twenty minutes into a run.
2. Type the feature request. Optionally set a repository path, the iteration
   budget, and an API token if the server has one configured.
3. Press **Run**. The response is a server-sent event stream, so progress
   appears as it happens.
4. The left rail tracks the eight phases, marking each done, active or failed,
   with live iteration, tool-call and elapsed counters.
5. The **Activity** tab is the running log: phases, tool calls, test results,
   review verdicts. The **Artifacts** tab fills in as each stage completes --
   requirements, repository analysis, risks, architecture, plan, test report,
   review findings, final report -- and later iterations replace their cards in
   place, so the panel always shows current truth.
6. A banner states the outcome. It reads *did not succeed* unless the tests
   passed **and** the review approved.

Because a run **writes files and executes code**, two controls bound what the
HTTP surface may do:

| Variable | Effect |
| --- | --- |
| `SE_AGENT_API_TOKEN` | Bearer token required on every run. Unset, the server refuses any caller that is not on loopback. |
| `SE_AGENT_REPO_ROOT` | The only directory tree a request may target. Unset, the `repo` field is refused outright and runs use the server's own directory. |
| `SE_AGENT_SANDBOX` | `docker` (default) or `script`. See **Sandbox backends**. |

Neither default is open: an unconfigured server works locally and is inert to
anyone else. **Do not expose it without setting a token.**

```bash
SE_AGENT_API_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))") \
SE_AGENT_REPO_ROOT=/srv/projects \
uvicorn app.server:app --host 0.0.0.0
```

### The API directly

```bash
curl -N -X POST http://127.0.0.1:8000/api/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SE_AGENT_API_TOKEN" \
  -d '{"request": "add a word-count helper", "max_iterations": 3}'
```

`GET /api/health` reports readiness; `GET /api/phases` lists the stages.
Stream events are `phase`, `agent`, `tool`, `tests`, `review`, `stage`,
`message`, `error`, and finally `result`.

### A note on runtime

Kimi K3 is a reasoning model and the default `reasoning_effort` is `max`, which
is slow: measured on a trivial prompt, a single model turn takes roughly 111s at
`max` and 58s at `low` (those are the only two values Kimi K3 accepts -- not
`medium`). A full run is twenty or more turns once tool calls are counted, so
expect it to take a while.

Free NIM tiers are also aggressively rate-limited; the client backs off on 429
for up to a minute per attempt, and a run that exhausts its retries reports the
failure rather than pretending to have succeeded.

---

## Sandbox backends

Repository code is never executed unguarded. Choose the backend with
`SE_AGENT_SANDBOX`:

| Value | Isolation |
| --- | --- |
| `docker` (default) | A real boundary: no network, read-only root, dropped capabilities, bounded CPU/memory/PIDs, environment allowlist. |
| `script` | `sandbox/local.sh` -- a network namespace and process ceiling on Linux and WSL; on Windows only a clean environment and a working directory. |

```bash
SE_AGENT_SANDBOX=script python -m app.agent --repo . "add a word-count helper"
```

**On Windows the script backend is not a security boundary.** The agent's code
runs as you, with your files and your network. It is appropriate for a
repository you would have run `pytest` in anyway; it is not appropriate for an
untrusted task. Read `sandbox/local.sh` -- it is short, and it states exactly
what it applies on each platform. `preflight()` reports the same, so the
console's health line never claims containment it does not have.

What holds on every platform is the environment allowlist: the command starts
from `env -i` and receives only what the script lists, so `NVIDIA_API_KEY` and
every other host secret are absent rather than merely unmentioned.

Because the clean environment also discards any active virtualenv, the script
prefers the workspace's own `.venv`; `SE_AGENT_SANDBOX_PATH` overrides that
with a directory to put first on `PATH`.

The script's contract is deliberately small, so a different mechanism is a new
script rather than new Python:

```
script <session-id> <timeout-seconds> <command>
  exit  the command's own exit code
  97    the sandbox itself failed -- never confused with a failing test suite
```

---

## Edge cases, and how to check them yourself

Every claim on this page has a test behind it. Run the whole suite with
`pytest -q`, or check one property at a time.

### The system cannot fake success

| Edge case | What must happen | Check it |
| --- | --- | --- |
| A test suite collects nothing | Reported as an error, never a pass | `pytest -q -k no_tests_collected_is_an_error` |
| The test command crashes on import | Execution error, not "0 failures" | `pytest -q -k collection_crash` |
| The sandbox itself fails to start | `execution_error`, zero failures, not a red suite | `pytest -q -k reserved_code_means_the_sandbox_failed` |
| Tests pass but nothing was implemented | Routed to failure, review never runs | `pytest -q -k green_suite_with_no_diff` |
| A review response is malformed | Treated as "changes required", never approval | `pytest -q -k malformed_review_cannot_become` |
| A review approves despite a critical finding | Rejected by the schema itself | `pytest -q -k approval_cannot_hide_a_critical` |
| Tests never pass | Budget exhausts, run still terminates at a report | `pytest -q -k retried_until_the_budget_runs_out` |
| The iteration ceiling | Spent exactly, never exceeded | `pytest -q -k budget_is_honoured_exactly` |

To watch the tester catch a real bug end to end, break a copy of the fixture and
run the suite through the sandbox:

```bash
python - <<'EOF'
import shutil, pathlib
from app.tools.sandbox import get_sandbox
from app.tools.workspace import Workspace

src = pathlib.Path("evaluation/fixtures/textstats")
dst = pathlib.Path("fixture-copy")
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)

stats = dst / "textstats" / "stats.py"
stats.write_text(
    stats.read_text().replace("len(text.split())", "len(text.split()) + 1")
)

box = get_sandbox()
box.workspace = Workspace(dst.resolve())
report = box.run_tests("python -m pytest -q")
print("green:", report.is_green, "| failed:", report.failed)
print(*report.failure_summary, sep="
")
EOF
```

Expect `green: False | failed: 1` and the exact assertion:

```
FAILED tests/test_stats.py::test_word_count_counts_words - AssertionError: assert 5 == 4
```

It runs through whichever backend `SE_AGENT_SANDBOX` selects. The fixture has no
virtualenv of its own, so with the script backend point it at one:
`SE_AGENT_SANDBOX_PATH=$PWD/.venv/Scripts` (or `.venv/bin`).

That is the whole point of the design: the failure text comes from a real pytest
process, so no agent is in a position to describe it differently.

### Secrets and the filesystem

| Edge case | What must happen | Check it |
| --- | --- | --- |
| `NVIDIA_API_KEY` reaching repository code | Absent from the sandbox environment | `pytest -q -k nvidia_key_never_reaches_the_container` |
| The same, script backend | Absent, on every platform | `pytest -q -k real_script_hides_the_nvidia_key` |
| Any other host secret | Only an explicit allowlist is forwarded | `pytest -q -k no_host_secret_is_forwarded` |
| The key in a log or exception | Redacted | `pytest -q -k api_key_is_redacted_from_error_text` |
| The key in `repr()` | Excluded from the model | `pytest -q -k api_key_never_appears_in_repr` |
| An agent reading `.env` | Refused as a secret file | `pytest -q -k reading_dotenv_is_refused` |
| An agent reading a private key | Refused | `pytest -q -k reading_a_private_key_is_refused` |
| `../../etc/passwd` | Refused before any read | `pytest -q -k parent_traversal_is_blocked` |
| A symlink pointing outside | Resolved first, then refused | `pytest -q -k symlink_out_of_the_workspace` |
| An absolute path outside the workspace | Refused | `pytest -q -k absolute_path_outside_root` |

Prove the key is absent yourself:

```bash
NVIDIA_API_KEY=nvapi-must-not-leak python -c "
from app.tools.sandbox import get_sandbox
print('leaked' if 'nvapi-must-not-leak' in get_sandbox().run('env').combined_output else 'absent')"
```

### The HTTP surface

| Edge case | What must happen | Check it |
| --- | --- | --- |
| No token supplied when one is configured | 401 | `pytest -q -k missing_token_is_rejected` |
| A wrong token | 401, compared in constant time | `pytest -q -k wrong_token_is_rejected` |
| No token configured, remote caller | 403 -- loopback only | `pytest -q -k token_free_server_serves_loopback` |
| A caller naming a repo outside the root | 403 | `pytest -q -k repo_outside_the_root_is_refused` |
| `../../` in the repo field | 403 | `pytest -q -k traversal_out_of_the_root` |
| A repo chosen with no root configured | 403 -- refused outright | `pytest -q -k refused_without_a_configured_root` |
| An iteration budget above the ceiling | 422 | `pytest -q -k budget_above_the_ceiling` |
| Preflight failure | Streamed as a fatal `error` event, not a 500 | `pytest -q -k preflight_failure_is_streamed` |
| The run crashing mid-stream | Becomes an `error` event the client sees | `pytest -q -k crash_mid_stream` |

Check the auth boundary by hand:

```bash
SE_AGENT_API_TOKEN=secret uvicorn app.server:app &
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/api/run \
  -H 'Content-Type: application/json' -d '{"request":"x"}'          # 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/api/run \
  -H 'Authorization: Bearer wrong' \
  -H 'Content-Type: application/json' -d '{"request":"x"}'          # 401
```

### The provider

| Edge case | What must happen | Check it |
| --- | --- | --- |
| Missing API key | Actionable error at startup, not a stack trace | `pytest -q -k missing_key_raises_actionable` |
| A key wrapped in quotes or `Bearer ` | Cleaned before use | `pytest -q -k plain_key_is_unchanged` |
| Rate limiting (429) | Backed off slowly, then reported honestly | `pytest -q -k rate_limit_is_retried_then_raised` |
| Repeated 429s | Backoff grows and is capped | `pytest -q -k rate_limit_backoff_grows` |
| A request that times out | Retried, then raised | `pytest -q -k timeout_is_retried_then_raised` |
| An authentication failure | **Not** retried -- retrying a bad key is pointless | `pytest -q -k auth` |
| Malformed tool arguments | Handled, not a crash | `pytest -q -k malformed_tool_arguments` |
| Malformed SSE lines while streaming | Skipped | `pytest -q -k malformed_sse_lines` |
| Chain-of-thought in the response | Dropped in the provider layer | `pytest -q -k reasoning_content_is_never_returned` |
| Chain-of-thought in an ADK event | Never sent back to the model or the UI | `pytest -q -k thought_parts_are_never_sent_back` |

### Environment problems

| Edge case | What must happen | How to reproduce |
| --- | --- | --- |
| Docker not running | Named in preflight, Run disabled in the console | Stop Docker Desktop, then `python -c "from app.agent import preflight; print(preflight())"` |
| The sandbox image not built | Preflight says exactly which command builds it | `docker image rm se-agent-sandbox:latest`, then preflight |
| No container runtime at all | Switch backends and continue | `SE_AGENT_SANDBOX=script python -c "from app.agent import preflight; print(preflight() or 'ready')"` |
| `bash` missing for the script backend | Preflight says so and names the alternative | `PATH= python -c "from app.tools.sandbox import ScriptSandbox; print(ScriptSandbox().preflight())"` |
| A runaway test suite | Killed with its children, reported as a timeout | `pytest -q -k timeout_is_reported_as_a_sandbox_error` |
| Which isolation is actually active | Stated, never assumed | `bash sandbox/local.sh --probe` |

That last one is worth running before you trust the script backend. On Linux it
prints a network namespace and process ceiling; on Windows it prints
`(no isolation boundary)`, and the console's health line says the same.

---

## How it works

```
                          START
                            |
                       requirements
                        /        \
              repository          risk        (run concurrently)
                        \        /
                       analysis_join
                            |
                       architecture <-------------------+
                            |                           | major
                         planner                        |
                            |                           |
                     +-> iteration_gate --exhausted--> report
                     |      |                            ^
                     |   implement                       |
                     |      |                            | approved
                     |    test                           |
                     |      |                            |
                     |  test_gate --passed--> review --> review_gate
                     |      |                            |
                     |   failed                       minor
                     |      |                            |
                     +-- diagnose <-----------------------+
```

The graph is ADK 2.8.0's native `Workflow`: agents are nodes, edges carry route
labels, and a node picks the next edge by setting `ctx.route`. Cycles -- fix and
retest, or re-architect after a major finding -- are expressed directly rather
than simulated with nested loop agents.

Every path back into implementation passes through `iteration_gate`, so the
iteration ceiling cannot be bypassed by a cycle, and the graph always terminates
in a report.

### The agents

| Agent | Role | Authority |
| --- | --- | --- |
| Requirements | Turns the request into checkable acceptance criteria | none |
| Repository | Maps the codebase and finds the real test command | read-only |
| Risk | Predicts dependency, security and regression problems | read-only |
| Architecture | Designs the change | read-only |
| Planner | Breaks the design into ordered, verifiable tasks | read-only |
| Coordinator | Implements, and consults specialists when stuck | read + write + execute |
| **Tester** | **Runs the real suite in the sandbox** | **execute only, no model** |
| Debugger | Diagnoses failures; deliberately cannot patch | read + execute |
| Reviewer | Independent judgement on the finished change | read-only |
| Reporter | Writes an honest account of the run | read-only |

Repository and risk analysis run in parallel because neither reads the other's
output. The coordinator can consult security and testing specialists as tools,
which returns control to it rather than handing over the run.

---

## Project layout

```
app/
├── agent.py          entry point: runner, services, CLI
├── telemetry.py      ADK events -> readable progress
├── agents/           the ten agents, as factories
├── workflows/graph.py  the graph and its routing decisions
├── llm/              the only path to NVIDIA NIM
│   ├── config.py       settings and key hygiene
│   ├── nvidia_kimi.py  HTTP client: retries, error taxonomy, streaming
│   └── adk_model.py    KimiK3Llm(BaseLlm), registered with LLMRegistry
├── tools/            repository tools and the sandbox backends
└── schemas/          the typed contract between stages
sandbox/Dockerfile    the execution image
sandbox/local.sh      the script backend
evaluation/fixtures/  sample projects to run against
```

### Why a `BaseLlm` adapter rather than LiteLLM

ADK 2.8.0 has no native NVIDIA NIM provider. Routing through LiteLLM would have
worked, but it would have opened a second path to the API alongside
`NVIDIAKimiClient`, which is exactly what "centralize NVIDIA access" rules out.

Subclassing `BaseLlm` and registering it with `LLMRegistry` keeps every HTTP
call in one place while letting agents be declared the ordinary ADK way:

```python
LlmAgent(model="moonshotai/kimi-k3", ...)
```

Everything above the provider layer -- agents, runner, sessions, events,
workflow, memory -- is stock ADK.

---

## Development

```bash
pytest -q                  # full suite
pytest -q -m "not docker"  # skip tests needing a Docker daemon
ruff check app tests
```

The suite covers path-traversal and secret-file refusal, sandbox isolation on
both backends, pytest output parsing (including the cases that must not read as
a pass), every routing decision, and the ADK/NIM translation in both directions.
`tests/test_graph_integration.py` runs the whole graph -- real runner, real
routing, real tester -- against a stub model, so every branch is covered without
provider quota. See **Edge cases** above for the property behind each test.

`PROJECT_STATE.md` holds current status; `progress.md` is the running log.
