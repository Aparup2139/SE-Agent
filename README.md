# SE-Agent

A multi-agent software engineering assistant built on **Google ADK 2.8.0**, with
**Kimi K3 on NVIDIA NIM** as its only model.

Give it a feature request in plain language. It analyses the repository, designs
the change, plans it, implements it, **runs the real tests in Docker**, reviews
its own work independently, fixes what the review finds, and reports honestly on
what it did and did not achieve.

---

## What makes it trustworthy

Most of the design effort went into making the system unable to lie to you.

**Test results come from a process, not a prediction.** The tester agent has no
model attached. It runs the suite in a container and builds its report from the
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

**Repository code never runs on your machine.** Everything executes in a
container with no network, a read-only root filesystem, dropped capabilities,
and bounded CPU, memory and processes. The container environment is an explicit
allowlist, so `NVIDIA_API_KEY` and your host credentials cannot reach the code
being tested.

**Private reasoning stays private.** Kimi K3 returns its chain-of-thought in a
separate field. It is dropped in the provider layer, never stored in state, and
never surfaced in the UI.

---

## Requirements

- Python 3.11+
- Docker (running) -- used to execute repository code and tests
- An NVIDIA NIM API key from <https://build.nvidia.com/>

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"

cp .env.example .env            # then add your NVIDIA_API_KEY

docker build -t se-agent-sandbox:latest sandbox/
```

`.env` is gitignored. The key is never logged, never included in an exception
message, and never passed into the sandbox.

## Usage

```bash
python -m app.agent "add a /health endpoint that returns the build SHA"

python -m app.agent --repo ../my-project --max-iterations 3 \
    "the retry helper drops the Retry-After header; fix it and add a test"
```

`--max-iterations` bounds how many implement/test/review attempts it makes
(default 5, hard ceiling 10). SE-Agent checks its prerequisites before starting
and reports every problem at once rather than failing on the first.

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
| **Tester** | **Runs the real suite in Docker** | **execute only, no model** |
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
├── tools/            repository tools and the Docker sandbox
└── schemas/          the typed contract between stages
sandbox/Dockerfile    the execution image
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

The suite covers path-traversal and secret-file refusal, sandbox isolation
flags, pytest output parsing (including the cases that must not read as a pass),
the routing decisions, and the ADK/NIM translation in both directions.

`PROJECT_STATE.md` holds current status; `progress.md` is the running log.
