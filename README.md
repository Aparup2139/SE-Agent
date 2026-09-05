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

**Repository code runs behind a sandbox, and the sandbox says what it is.**
`sandbox/local.sh` uses the strongest isolation the host actually provides and
reports which tier that was. **A run refuses to start with no boundary** unless
you accept that deliberately -- so "unsandboxed" is a decision someone made,
never a silent default. See **The sandbox**.

Whatever the tier, the environment is an allowlist: `NVIDIA_API_KEY` and your
host credentials are absent from what the tested code sees, rather than merely
unmentioned.

**Private reasoning stays private.** Kimi K3 returns its chain-of-thought in a
separate field. It is dropped in the provider layer, never stored in state, and
never surfaced in the UI.

---

## Requirements

- Python 3.11+
- `bash` (Git Bash on Windows) -- runs the sandbox script
- For a real isolation boundary: `bubblewrap` or `unshare` (Linux/WSL).
  See **The sandbox**; on Windows this means WSL.
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

**3. Set up isolation**

Repository code is executed, so it needs a boundary. `sandbox/local.sh` picks
the strongest one available and **a run refuses to start if that is none**.

*Linux -- the full boundary:*

```bash
sudo apt install bubblewrap        # or: dnf install bubblewrap
bash sandbox/local.sh --probe      # expect: bwrap: filesystem+network+pid namespaces
```

*Windows -- reach a Linux tier through WSL:*

```bash
export SE_AGENT_SANDBOX_WSL=Ubuntu-24.04
```

The tests then run with WSL's toolchain, so install the project's dependencies
inside that distribution too.

*Or accept no boundary, deliberately:*

```bash
export SE_AGENT_SANDBOX_ALLOW_UNSAFE=1
```

Appropriate for a repository you would have run `pytest` in anyway. Not
appropriate for an untrusted task -- the code runs as you.

**4. Confirm everything is ready**

```bash
python -c "from app.agent import preflight; print(preflight() or 'ready')"
```

This checks the key, the workspace and the selected sandbox, and prints *every*
problem at once rather than failing on the first. `ready` means a run can start.

**5. Run the tests**

```bash
pytest -q                  # 263 passed, 1 skipped
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
   and the problems are printed next to it -- so a missing isolation boundary
   is visible before you submit rather than twenty minutes into a run. The
   line names the active tier, so it never implies containment.
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
| `SE_AGENT_SANDBOX_ALLOW_UNSAFE` | Permit a run with no isolation boundary. See **The sandbox**. |
| `SE_AGENT_SANDBOX_WSL` | Run the sandbox inside a WSL distribution, to reach a real tier from Windows. |

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

## The sandbox

Every command an agent runs goes through `sandbox/local.sh`. It selects a tier
by **running** each mechanism, not by looking for the binary, so something
present but broken -- a kernel with unprivileged user namespaces disabled --
demotes rather than failing every run.

| Tier | What it gives | Where |
| --- | --- | --- |
| `bwrap` | Fresh mount namespace: the workspace is the only writable path, most of the filesystem is not visible at all. No network, own PID and IPC space. | Linux/WSL with bubblewrap |
| `unshare` | Network and user namespace. Bounds what the command can *reach*, not what it can read. | Linux/WSL |
| `none` | A clean environment and a working directory. **Not a boundary.** | Windows, bare |

```bash
bash sandbox/local.sh --probe     # what is in force here
```

`preflight()` reports the same string, so the console's health line states the
active tier rather than implying containment.

**A run refuses to start on `none`.** Set `SE_AGENT_SANDBOX_ALLOW_UNSAFE=1` to
accept it; the preflight message then ends in `ACCEPTED AS UNSAFE`, and keeps
saying so. To go the other way and refuse anything weaker than a chosen tier:

```bash
SE_AGENT_SANDBOX_REQUIRE=bwrap    # or unshare
```

### What holds at every tier

The command starts from `env -i` and receives only the variables the script
lists, so `NVIDIA_API_KEY` and every other host secret are absent rather than
merely unmentioned. It runs in the workspace, with a private `HOME` and
`TMPDIR`, a file-size ceiling, a process ceiling, and a wall-clock timeout
enforced twice -- once by the script, once by the caller, which kills the whole
process group so a runaway suite cannot outlive it.

### Windows

The namespaces are Linux's, so Windows reaches only `none` on its own.
`SE_AGENT_SANDBOX_WSL=<distro>` runs the script inside WSL, where `unshare`
works and bubblewrap can be installed.

The trade-off is real: the tests then run with **WSL's** Python, not Windows',
so the project's dependencies must exist there. That is the cost of a boundary
without a container -- there is no image carrying the toolchain any more.

### The interpreter

A clean environment also discards any active virtualenv, so the script prefers
the workspace's own `.venv`. `SE_AGENT_SANDBOX_PATH` overrides that with a
directory to put first on `PATH`.

### The contract

Deliberately small, so a different mechanism is a new script rather than new
Python. `SE_AGENT_SANDBOX_CMD` points at one:

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
| `NVIDIA_API_KEY` reaching repository code | Absent from the sandbox environment, at every tier | `pytest -q -k real_script_hides_the_nvidia_key` |
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
| No isolation available | **The run refuses to start** | `python -c "from app.agent import preflight; print(preflight())"` on a host with no namespaces |
| That refusal overridden | Allowed, but still labelled unsafe | `pytest -q -k the_gate_can_be_opened_deliberately` |
| A real tier present | No gate, no warning | `pytest -q -k a_real_tier_needs_no_gate` |
| A required tier unavailable | Refused by the script itself | `SE_AGENT_SANDBOX_REQUIRE=bwrap bash sandbox/local.sh s 30 'echo hi'; echo $?` -> 97 |
| `bash` missing | Preflight says so and names the alternative | `PATH= python -c "from app.tools.sandbox import Sandbox; print(Sandbox().preflight())"` |
| A runaway test suite | Killed with its children, reported as a timeout | `pytest -q -k timeout_is_reported_as_a_sandbox_error` |
| Which isolation is actually active | Stated, never assumed | `bash sandbox/local.sh --probe` |

Check the network boundary itself, on a host that has one:

```bash
SE_AGENT_WORKSPACE=$PWD bash sandbox/local.sh t 30 \
  'getent hosts pypi.org >/dev/null 2>&1 && echo REACHABLE || echo BLOCKED'
# BLOCKED

SE_AGENT_WORKSPACE=$PWD SE_AGENT_SANDBOX_NETWORK=1 bash sandbox/local.sh t 30 \
  'getent hosts pypi.org >/dev/null 2>&1 && echo REACHABLE || echo BLOCKED'
# REACHABLE
```

The last row is the one to run before trusting the sandbox anywhere new.

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
├── tools/            repository tools and the sandbox
└── schemas/          the typed contract between stages
sandbox/local.sh      the isolation script -- read this one
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
pytest -q                  # full suite: 263 passed, 1 skipped
ruff check app tests
```

The suite covers path-traversal and secret-file refusal, sandbox isolation and
its refusal to run unguarded, pytest output parsing (including the cases that must not read as
a pass), every routing decision, and the ADK/NIM translation in both directions.
`tests/test_graph_integration.py` runs the whole graph -- real runner, real
routing, real tester -- against a stub model, so every branch is covered without
provider quota. See **Edge cases** above for the property behind each test.

`PROJECT_STATE.md` holds current status; `progress.md` is the running log.
