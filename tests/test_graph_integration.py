"""End-to-end graph execution against a stub model.

These tests run the *real* workflow -- the real ADK Runner, the real graph, the
real routing nodes and the real tester agent -- with the model and the Docker
sandbox replaced. That isolates the orchestration, which is the part that must
be correct regardless of what any model says, and lets it be verified without
network access or provider quota.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncGenerator

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agents import common as agents_common
from app.agents import tester as tester_module
from app.schemas import ChangeScope, CodeReview, ReviewVerdict, Severity, TestReport
from app.workflows import graph as graph_module

STUB_MODEL = "stub/deterministic"


class StubLlm(BaseLlm):
    """A model that answers from a script instead of a network call.

    Each agent is answered by name. Agents with an ``output_schema`` are given
    structured output the way ADK asks for it -- through the injected
    ``set_model_response`` tool -- so the real structured-output path is
    exercised rather than bypassed.
    """

    model: str = STUB_MODEL
    # Set per-test: agent name -> payload for set_model_response.
    responses: dict = {}

    @staticmethod
    def supported_models() -> list[str]:
        return [r"stub/.*"]

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        agent = _agent_name(llm_request)
        payload = type(self).responses.get(agent, DEFAULT_RESPONSES.get(agent))

        if payload is not None and not _already_answered(llm_request):
            # ADK asks for structured output two different ways: agents that
            # also hold tools get an injected set_model_response tool, agents
            # without tools get response_schema and must answer in JSON text.
            if _has_response_tool(llm_request):
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=f"{agent}-call",
                                    name="set_model_response",
                                    args=payload,
                                )
                            )
                        ],
                    ),
                    turn_complete=True,
                )
            else:
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=json.dumps(payload))],
                    ),
                    finish_reason=types.FinishReason.STOP,
                    turn_complete=True,
                )
            return

        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=f"{agent} finished.")]
            ),
            finish_reason=types.FinishReason.STOP,
            turn_complete=True,
        )


def _has_response_tool(llm_request) -> bool:
    """Whether ADK injected the structured-output tool for this request."""
    for tool in llm_request.config.tools or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            if declaration.name == "set_model_response":
                return True
    return False


def _agent_name(llm_request) -> str:
    """Recover which agent is asking, from its system instruction."""
    instruction = llm_request.config.system_instruction or ""
    text = instruction if isinstance(instruction, str) else str(instruction)
    for name in (
        "Requirements Analyst",
        "Repository Analyst",
        "Risk and Dependency Analyst",
        "Software Architect",
        "Implementation Planner",
        "Implementation Coordinator",
        "Debugger",
        "Code Reviewer",
        "Reporter",
    ):
        if name in text:
            return name
    return "unknown"


def _already_answered(llm_request) -> bool:
    """True once this agent has produced its structured answer.

    Without this the stub would call set_model_response forever, since ADK
    feeds the tool result back and asks again.
    """
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if part.function_response and part.function_response.name == (
                "set_model_response"
            ):
                return True
    return False


DEFAULT_RESPONSES: dict = {
    "Requirements Analyst": {
        "summary": "Add the requested feature.",
        "functional": ["it works"],
        "acceptance_criteria": [
            {"id": "AC-1", "description": "works", "verification": "a test"}
        ],
    },
    "Repository Analyst": {
        "languages": ["Python"],
        "structure_summary": "a small package",
        "test_command": "pytest -q",
    },
    "Risk and Dependency Analyst": {
        "risks": [
            {
                "id": "R-1",
                "category": "regression",
                "description": "might break callers",
                "severity": "low",
                "mitigation": "run the suite",
            }
        ]
    },
    "Software Architect": {"approach": "add a function to the existing module"},
    "Implementation Planner": {
        "tasks": [
            {
                "task_id": "T-1",
                "objective": "add the function",
                "validation_criteria": "its test passes",
            }
        ]
    },
    "Reporter": {
        "feature_request": "add a feature",
        "outcome": "completed",
        "summary": "done",
    },
}


LLMRegistry.register(StubLlm)


class StubSandbox:
    """Stands in for Docker. Returns a scripted sequence of test results."""

    def __init__(self, reports: list[TestReport]):
        self.reports = list(reports)
        self.calls = 0

    def run_tests_with_output(self, command, timeout=None):
        self.calls += 1
        report = self.reports[min(self.calls - 1, len(self.reports) - 1)]
        return report, "stub pytest output"


@pytest.fixture
def stub_everything(monkeypatch):
    """Swap the model, the sandbox and git for deterministic stand-ins."""
    monkeypatch.setattr(agents_common, "MODEL", STUB_MODEL)
    StubLlm.responses = {}

    def install(reports, changed_files):
        sandbox = StubSandbox(reports)
        monkeypatch.setattr(tester_module, "get_sandbox", lambda: sandbox)
        monkeypatch.setattr(graph_module, "_changed_files", lambda: list(changed_files))
        return sandbox

    return install


async def run_graph(state: dict | None = None) -> tuple[dict, list[str]]:
    """Run the whole graph and return final state plus the node sequence."""
    runner = InMemoryRunner(agent=graph_module.build_workflow(), app_name="graph_test")
    session = await runner.session_service.create_session(
        app_name="graph_test",
        user_id="u",
        state=state or graph_module.initial_state("add a feature", limit=2),
    )
    message = types.Content(role="user", parts=[types.Part(text="add a feature")])

    visited: list[str] = []
    async for event in runner.run_async(
        user_id="u", session_id=session.id, new_message=message
    ):
        author = getattr(event, "author", None)
        if author and (not visited or visited[-1] != author):
            visited.append(author)

    refreshed = await runner.session_service.get_session(
        app_name="graph_test", user_id="u", session_id=session.id
    )
    return dict(refreshed.state), visited


def review_payload(verdict, findings=()):
    return json.loads(
        CodeReview(
            verdict=verdict, summary="stub review", findings=list(findings)
        ).model_dump_json()
    )


def finding(scope=ChangeScope.MINOR, severity=Severity.MEDIUM):
    return {
        "id": "F-1",
        "category": "correctness",
        "severity": severity.value,
        "description": "d",
        "recommendation": "r",
        "scope": scope.value,
        "file": "",
        "line": None,
    }


# --------------------------------------------------------------------- happy


@pytest.mark.asyncio
async def test_a_green_reviewed_run_reaches_the_report(stub_everything):
    """Tests pass on a real diff, review approves, the run ends at the report."""
    stub_everything(
        [TestReport(passed=7, test_count=7, exit_code=0)], ["textstats/stats.py"]
    )
    StubLlm.responses = {
        "Code Reviewer": review_payload(ReviewVerdict.APPROVED),
    }

    state, visited = await run_graph()

    assert state["tests_passed"] is True
    assert state["review_approved"] is True
    assert state["iteration"] == 1, "one iteration was enough"
    assert "report" in visited


@pytest.mark.asyncio
async def test_the_tester_result_comes_from_the_sandbox(stub_everything):
    sandbox = stub_everything(
        [TestReport(passed=3, test_count=3, exit_code=0)], ["a.py"]
    )
    StubLlm.responses = {"Code Reviewer": review_payload(ReviewVerdict.APPROVED)}

    state, _ = await run_graph()

    assert sandbox.calls >= 1, "the tester must actually invoke the sandbox"
    assert state["tests"]["passed"] == 3


# ------------------------------------------------------------------ failures


@pytest.mark.asyncio
async def test_failing_tests_are_retried_until_the_budget_runs_out(stub_everything):
    """Never-passing tests must exhaust the budget, not loop forever."""
    sandbox = stub_everything(
        [TestReport(passed=1, failed=1, test_count=2, exit_code=1)], ["a.py"]
    )

    state, visited = await run_graph()

    assert state["tests_passed"] is False
    assert state["iterations_exhausted"] is True
    assert state["iteration"] == 3, "two attempts, then the gate stops it"
    assert sandbox.calls == 2, "one test run per attempt"
    assert "report" in visited, "an exhausted run still reports"


@pytest.mark.asyncio
async def test_a_green_suite_with_no_diff_never_reaches_review(stub_everything):
    """The regression this guards: nothing implemented must not read as success."""
    stub_everything([TestReport(passed=6, test_count=6, exit_code=0)], [])
    StubLlm.responses = {"Code Reviewer": review_payload(ReviewVerdict.APPROVED)}

    state, visited = await run_graph()

    assert state["tests_passed"] is True, "the suite did pass"
    assert not state.get("review_approved"), "but it must not be approved"
    assert "review" not in visited, "the reviewer must never have been consulted"
    assert state["iterations_exhausted"] is True


@pytest.mark.asyncio
async def test_a_sandbox_failure_never_reaches_review(stub_everything):
    stub_everything(
        [TestReport(exit_code=0, execution_error="docker unreachable")], ["a.py"]
    )
    StubLlm.responses = {"Code Reviewer": review_payload(ReviewVerdict.APPROVED)}

    state, visited = await run_graph()

    assert state["tests_passed"] is False
    assert "review" not in visited


# ------------------------------------------------------------------- review


@pytest.mark.asyncio
async def test_changes_required_sends_the_run_back_to_implementation(stub_everything):
    stub_everything([TestReport(passed=5, test_count=5, exit_code=0)], ["a.py"])
    StubLlm.responses = {
        "Code Reviewer": review_payload(
            ReviewVerdict.CHANGES_REQUIRED, [finding(ChangeScope.MINOR)]
        )
    }

    state, visited = await run_graph()

    assert state["review_approved"] is False
    assert visited.count("implement") >= 2, "it went back to the engineer"
    assert state["iterations_exhausted"] is True


@pytest.mark.asyncio
async def test_a_major_finding_returns_to_the_architect(stub_everything):
    stub_everything([TestReport(passed=5, test_count=5, exit_code=0)], ["a.py"])
    StubLlm.responses = {
        "Code Reviewer": review_payload(
            ReviewVerdict.CHANGES_REQUIRED, [finding(ChangeScope.MAJOR)]
        )
    }

    state, visited = await run_graph()

    assert visited.count("architecture") >= 2, "the design was revisited"
    assert state["review_approved"] is False


@pytest.mark.asyncio
async def test_a_malformed_review_cannot_become_an_approval(stub_everything):
    """Two layers stand between a bad review and an approval.

    ADK validates structured output against the schema first, so a nonsense
    verdict never reaches session state at all -- the node fails instead. The
    `read_review` guard in the router is the second layer, covered directly in
    test_workflow.py. Either way the run must not end approved.
    """
    stub_everything([TestReport(passed=5, test_count=5, exit_code=0)], ["a.py"])
    StubLlm.responses = {"Code Reviewer": {"verdict": "PROBABLY_FINE"}}

    try:
        state, visited = await run_graph()
    except Exception:
        # ADK refused the malformed output outright. That is the first layer
        # doing its job: nothing was written, so nothing was approved.
        return

    assert not state.get("review_approved")
    assert "report" not in visited or not state.get("review_approved")


# ------------------------------------------------------------------- budget


@pytest.mark.asyncio
async def test_the_iteration_budget_is_honoured_exactly(stub_everything):
    sandbox = stub_everything(
        [TestReport(passed=1, failed=1, test_count=2, exit_code=1)], ["a.py"]
    )
    state, _ = await run_graph(graph_module.initial_state("x", limit=3))

    assert sandbox.calls == 3, "exactly the budgeted number of attempts"
    assert state["iteration"] == 4, "the gate ran once more and stopped"


@pytest.mark.asyncio
async def test_analysis_runs_before_architecture(stub_everything):
    stub_everything([TestReport(passed=1, test_count=1, exit_code=0)], ["a.py"])
    StubLlm.responses = {"Code Reviewer": review_payload(ReviewVerdict.APPROVED)}

    _, visited = await run_graph()

    for earlier, later in (
        ("requirements", "architecture"),
        ("architecture", "planner"),
        ("planner", "implement"),
        ("implement", "test"),
    ):
        assert earlier in visited and later in visited
        assert visited.index(earlier) < visited.index(later), f"{earlier} -> {later}"


@pytest.mark.asyncio
async def test_git_is_never_invoked_on_the_host_during_the_graph(
    stub_everything, monkeypatch
):
    """The stub replaces _changed_files; nothing else may shell out to git."""
    stub_everything([TestReport(passed=1, test_count=1, exit_code=0)], ["a.py"])
    StubLlm.responses = {"Code Reviewer": review_payload(ReviewVerdict.APPROVED)}

    calls = []
    real_run = subprocess.run

    def watched(args, *a, **k):
        if args and args[0] == "git":
            calls.append(args)
        return real_run(args, *a, **k)

    monkeypatch.setattr(subprocess, "run", watched)
    await run_graph()
    assert calls == []
