"""Workflow graph shape, deterministic routing, and the tester's state contract."""

from __future__ import annotations

import pytest
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agents.tester import TesterAgent, _resolve_command, _summarize
from app.schemas import (
    ChangeScope,
    CodeReview,
    ReviewFinding,
    ReviewVerdict,
    Severity,
    TestReport,
)
from app.workflows.graph import (
    DEFAULT_MAX_ITERATIONS,
    ITERATION_CEILING,
    ROUTE_APPROVED,
    ROUTE_CONTINUE,
    ROUTE_EXHAUSTED,
    ROUTE_FAILED,
    ROUTE_MAJOR,
    ROUTE_MINOR,
    ROUTE_PASSED,
    build_workflow,
    decide_after_review,
    decide_after_tests,
    decide_iteration,
    initial_state,
    max_iterations,
    read_review,
)


@pytest.fixture(scope="module")
def workflow():
    return build_workflow()


@pytest.fixture
def edges(workflow):
    """Edges as (from, route, to) triples, which is how the spec reads."""
    return {(e.from_node.name, e.route, e.to_node.name) for e in workflow.edges}


class FakeContext:
    """Minimal stand-in for the workflow Context a routing node receives."""

    def __init__(self, state=None):
        self.state = dict(state or {})
        self.route = None


# ------------------------------------------------------------------- topology


def test_workflow_starts_at_requirements(edges):
    assert ("__START__", None, "requirements") in edges


def test_repository_and_risk_analysis_fan_out_and_join(edges):
    assert ("requirements", None, "repository") in edges
    assert ("requirements", None, "risk") in edges
    assert ("repository", None, "analysis_join") in edges
    assert ("risk", None, "analysis_join") in edges
    assert ("analysis_join", None, "architecture") in edges


def test_main_path_reaches_implementation(edges):
    assert ("architecture", None, "planner") in edges
    assert ("planner", None, "iteration_gate") in edges
    assert ("iteration_gate", ROUTE_CONTINUE, "implement") in edges
    assert ("implement", None, "test") in edges


def test_failing_tests_route_to_diagnosis_then_back(edges):
    assert ("test", None, "test_gate") in edges
    assert ("test_gate", ROUTE_FAILED, "diagnose") in edges
    assert ("diagnose", None, "iteration_gate") in edges


def test_passing_tests_route_to_an_independent_review(edges):
    assert ("test_gate", ROUTE_PASSED, "review") in edges
    assert ("review", None, "review_gate") in edges


def test_review_outcomes_route_three_ways(edges):
    assert ("review_gate", ROUTE_APPROVED, "report") in edges
    assert ("review_gate", ROUTE_MINOR, "iteration_gate") in edges
    assert ("review_gate", ROUTE_MAJOR, "architecture") in edges


def test_exhausted_iterations_still_produce_a_report(edges):
    """The run must always terminate in a report, never just stop."""
    assert ("iteration_gate", ROUTE_EXHAUSTED, "report") in edges


def test_every_path_into_implementation_passes_the_iteration_gate(edges):
    """Otherwise a cycle could bypass the budget."""
    into_implement = {(f, r) for f, r, t in edges if t == "implement"}
    assert into_implement == {("iteration_gate", ROUTE_CONTINUE)}


def test_report_is_the_only_terminal_node(edges):
    sources = {f for f, _, _ in edges}
    targets = {t for _, _, t in edges}
    assert targets - sources == {"report"}


# --------------------------------------------------------------------- budget


def test_default_iteration_budget_is_five():
    assert DEFAULT_MAX_ITERATIONS == 5
    assert max_iterations() == 5


def test_budget_is_configurable(monkeypatch):
    monkeypatch.setenv("SE_AGENT_MAX_ITERATIONS", "3")
    assert max_iterations() == 3


def test_budget_is_hard_capped(monkeypatch):
    monkeypatch.setenv("SE_AGENT_MAX_ITERATIONS", "500")
    assert max_iterations() == ITERATION_CEILING


def test_budget_of_zero_is_raised_to_one(monkeypatch):
    monkeypatch.setenv("SE_AGENT_MAX_ITERATIONS", "0")
    assert max_iterations() == 1


def test_malformed_budget_falls_back(monkeypatch):
    monkeypatch.setenv("SE_AGENT_MAX_ITERATIONS", "many")
    assert max_iterations() == DEFAULT_MAX_ITERATIONS


def test_initial_state_seeds_the_run():
    state = initial_state("add a healthcheck endpoint", limit=3)
    assert state["feature_request"] == "add a healthcheck endpoint"
    assert state["iteration"] == 0
    assert state["max_iterations"] == 3


# ------------------------------------------------------------ iteration gate


@pytest.mark.asyncio
async def test_iteration_gate_counts_up_and_continues():
    ctx = FakeContext({"iteration": 0, "max_iterations": 3})
    await decide_iteration(ctx)
    assert ctx.route == ROUTE_CONTINUE
    assert ctx.state["iteration"] == 1


@pytest.mark.asyncio
async def test_iteration_gate_stops_at_the_budget():
    ctx = FakeContext({"iteration": 3, "max_iterations": 3})
    await decide_iteration(ctx)
    assert ctx.route == ROUTE_EXHAUSTED
    assert ctx.state["iterations_exhausted"] is True


@pytest.mark.asyncio
async def test_iteration_gate_allows_exactly_the_budgeted_attempts():
    """A budget of 2 must permit iterations 1 and 2, then stop."""
    state = {"iteration": 0, "max_iterations": 2}
    routes = []
    for _ in range(3):
        ctx = FakeContext(state)
        await decide_iteration(ctx)
        routes.append(ctx.route)
        state = ctx.state
    assert routes == [ROUTE_CONTINUE, ROUTE_CONTINUE, ROUTE_EXHAUSTED]


# ------------------------------------------------------------------ test gate


@pytest.mark.asyncio
async def test_test_gate_routes_a_pass_to_review():
    ctx = FakeContext({"tests_passed": True})
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_PASSED


@pytest.mark.asyncio
async def test_test_gate_routes_a_failure_to_diagnosis():
    ctx = FakeContext({"tests_passed": False, "tests": {"failed": 2, "errors": 0}})
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED


@pytest.mark.asyncio
async def test_test_gate_treats_a_sandbox_failure_as_a_failure():
    """A sandbox that could not run is never a pass."""
    ctx = FakeContext(
        {"tests_passed": False, "tests": {"execution_error": "docker unreachable"}}
    )
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED
    assert "could not run" in message


@pytest.mark.asyncio
async def test_test_gate_with_no_result_at_all_does_not_pass():
    ctx = FakeContext({})
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED


# ---------------------------------------------------------------- review gate


def _review(verdict, findings=()):
    return CodeReview(verdict=verdict, summary="s", findings=list(findings))


def _finding(scope=ChangeScope.MINOR, severity=Severity.MEDIUM):
    return ReviewFinding(
        id="F-1",
        category="correctness",
        severity=severity,
        description="d",
        recommendation="r",
        scope=scope,
    )


@pytest.mark.asyncio
async def test_review_gate_approves():
    state = {"review": _review(ReviewVerdict.APPROVED).model_dump()}
    ctx = FakeContext(state)
    await decide_after_review(ctx)
    assert ctx.route == ROUTE_APPROVED
    assert ctx.state["review_approved"] is True


@pytest.mark.asyncio
async def test_review_gate_sends_minor_findings_back_to_the_engineer():
    state = {
        "review": _review(ReviewVerdict.CHANGES_REQUIRED, [_finding()]).model_dump()
    }
    ctx = FakeContext(state)
    await decide_after_review(ctx)
    assert ctx.route == ROUTE_MINOR
    assert ctx.state["review_approved"] is False


@pytest.mark.asyncio
async def test_review_gate_sends_major_findings_back_to_the_architect():
    review = _review(
        ReviewVerdict.CHANGES_REQUIRED, [_finding(), _finding(ChangeScope.MAJOR)]
    )
    ctx = FakeContext({"review": review.model_dump()})
    await decide_after_review(ctx)
    assert ctx.route == ROUTE_MAJOR


@pytest.mark.asyncio
async def test_unparseable_review_never_approves():
    """A malformed review must not become an accidental approval."""
    ctx = FakeContext({"review": {"verdict": "PROBABLY_FINE"}})
    await decide_after_review(ctx)
    assert ctx.route == ROUTE_MINOR
    assert ctx.state["review_approved"] is False


@pytest.mark.asyncio
async def test_missing_review_never_approves():
    ctx = FakeContext({})
    await decide_after_review(ctx)
    assert ctx.route == ROUTE_MINOR


def test_read_review_returns_none_for_junk():
    assert read_review({}) is None
    assert read_review({"review": {"verdict": "MAYBE"}}) is None


def test_read_review_passes_through_a_parsed_review():
    original = _review(ReviewVerdict.APPROVED)
    assert read_review({"review": original}) is original


# ----------------------------------------------------------- agent guarantees


def test_every_model_backed_node_uses_kimi_k3(workflow):
    """No node may quietly fall back to another model."""
    seen = []

    def walk(obj, depth=0):
        if depth > 6 or obj is None:
            return
        model = getattr(obj, "model", None)
        name = getattr(obj, "name", None)
        if isinstance(model, str) and model:
            seen.append((name, model))
        for attr in ("agent", "sub_agents", "_agent"):
            child = getattr(obj, attr, None)
            if isinstance(child, list):
                for item in child:
                    walk(item, depth + 1)
            elif child is not None:
                walk(child, depth + 1)

    for edge in workflow.edges:
        walk(edge.from_node)
        walk(edge.to_node)

    assert seen, "expected at least one model-backed node"
    assert all(model == "moonshotai/kimi-k3" for _, model in seen), seen


# ------------------------------------------------------------- test command


def test_test_command_comes_from_the_repository_analysis():
    state = {"repository": {"test_command": "npm test -- --ci"}}
    assert _resolve_command(state) == "npm test -- --ci"


def test_missing_repository_analysis_falls_back_to_pytest():
    assert _resolve_command({}) == "pytest -q"


def test_blank_test_command_falls_back():
    assert _resolve_command({"repository": {"test_command": "   "}}) == "pytest -q"


# ------------------------------------------------------------------ summaries


def test_green_summary_says_so():
    report = TestReport(passed=7, test_count=7, exit_code=0, duration_seconds=1.5)
    assert "All tests passed" in _summarize(report)


def test_failure_summary_lists_the_failures():
    report = TestReport(
        passed=2,
        failed=1,
        test_count=3,
        exit_code=1,
        failure_summary=["FAILED tests/test_a.py::test_b - assert 1 == 2"],
    )
    text = _summarize(report)
    assert "Tests failed" in text
    assert "test_b" in text


def test_sandbox_failure_summary_does_not_claim_a_test_result():
    report = TestReport(execution_error="docker daemon unreachable")
    text = _summarize(report)
    assert "could not run" in text
    assert "passed" not in text


# ------------------------------------------------- tester through a real runner


class FakeSandbox:
    """Stands in for Docker so the tester's state contract can be exercised."""

    def __init__(self, report: TestReport):
        self.report = report
        self.commands: list[str] = []

    def run_tests(self, command, timeout=None):
        self.commands.append(command)
        return self.report


async def _run_tester(report: TestReport, initial: dict | None = None):
    """Run TesterAgent through a real ADK runner and return the resulting state."""
    tester = TesterAgent()
    sandbox = FakeSandbox(report)
    tester.sandbox = sandbox

    runner = InMemoryRunner(agent=tester, app_name="tester_test")
    session = await runner.session_service.create_session(
        app_name="tester_test", user_id="u", state=initial or {}
    )
    message = types.Content(role="user", parts=[types.Part(text="run the tests")])
    async for _ in runner.run_async(
        user_id="u", session_id=session.id, new_message=message
    ):
        pass

    refreshed = await runner.session_service.get_session(
        app_name="tester_test", user_id="u", session_id=session.id
    )
    return refreshed.state, sandbox


@pytest.mark.asyncio
async def test_tester_writes_a_passing_result_to_state():
    state, _ = await _run_tester(TestReport(passed=9, test_count=9, exit_code=0))
    assert state["tests_passed"] is True
    assert state["tests"]["passed"] == 9


@pytest.mark.asyncio
async def test_tester_writes_a_failing_result_to_state():
    state, _ = await _run_tester(
        TestReport(passed=1, failed=2, test_count=3, exit_code=1)
    )
    assert state["tests_passed"] is False
    assert state["tests"]["failed"] == 2


@pytest.mark.asyncio
async def test_tester_reports_a_sandbox_failure_as_not_passing():
    """The one failure mode that must never be mistaken for success."""
    state, _ = await _run_tester(
        TestReport(exit_code=0, execution_error="docker daemon unreachable")
    )
    assert state["tests_passed"] is False
    assert state["tests"]["execution_error"]


@pytest.mark.asyncio
async def test_tester_uses_the_repository_test_command():
    _, sandbox = await _run_tester(
        TestReport(passed=1, test_count=1, exit_code=0),
        {"repository": {"test_command": "go test ./..."}},
    )
    assert sandbox.commands == ["go test ./..."]
