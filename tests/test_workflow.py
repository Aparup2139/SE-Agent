"""Workflow graph shape, deterministic routing, and the tester's state contract."""

from __future__ import annotations

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agents.tester import (
    TesterAgent,
    _resolve_command,
    _summarize,
    detect_test_command,
)
from app.llm import gemini_model
from app.llm.config import get_config
from app.schemas import (
    ChangeScope,
    CodeReview,
    ReviewFinding,
    ReviewVerdict,
    RunMode,
    Severity,
    TestReport,
)
from app.tools.workspace import Workspace
from app.workflows import graph
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
    # Captured up front, so the no-op guard has something to measure against.
    assert "baseline_tree" in state


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


@pytest.fixture
def with_changes(monkeypatch):
    """Pretend the working tree has been modified."""
    monkeypatch.setattr(graph, "_changed_files", lambda state: ["textstats/stats.py"])


@pytest.fixture
def without_changes(monkeypatch):
    monkeypatch.setattr(graph, "_changed_files", lambda state: [])


@pytest.mark.asyncio
async def test_test_gate_routes_a_pass_to_review(with_changes):
    ctx = FakeContext({"tests_passed": True})
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_PASSED
    assert ctx.state["files_changed"] == ["textstats/stats.py"]


@pytest.mark.asyncio
async def test_green_tests_with_no_diff_are_not_a_pass(without_changes):
    """The pre-existing suite passing proves nothing was implemented."""
    ctx = FakeContext({"tests_passed": True})
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED
    assert "changed no files" in message


@pytest.mark.asyncio
async def test_unreadable_git_status_is_not_treated_as_a_change(monkeypatch):
    """A tool failure must not be mistaken for evidence of an implementation."""
    from app.tools import repo_tools

    monkeypatch.setattr(repo_tools, "working_tree_state", lambda: None)
    ctx = FakeContext({"tests_passed": True})
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED


def test_changed_files_reports_only_what_this_run_touched(monkeypatch):
    """The developer's own uncommitted work is not the agent's output."""
    from app.tools import repo_tools

    monkeypatch.setattr(
        repo_tools,
        "working_tree_state",
        lambda: {"ambient.py": "1/0", "agent.py": "9/2", "new.py": "untracked/12"},
    )
    baseline = {"ambient.py": "1/0", "agent.py": "3/0"}
    assert graph._changed_files({"baseline_tree": baseline}) == ["agent.py", "new.py"]


@pytest.mark.asyncio
async def test_a_dirty_tree_the_run_did_not_touch_is_still_a_no_op(monkeypatch):
    """The bug this guards: a pass on an ambient diff the run never made."""
    from app.tools import repo_tools

    tree = {"ambient.py": "42/7"}
    monkeypatch.setattr(repo_tools, "working_tree_state", lambda: dict(tree))
    ctx = FakeContext({"tests_passed": True, "baseline_tree": dict(tree)})
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED
    assert "changed no files" in message


def test_a_missing_baseline_falls_back_to_any_dirty_file(monkeypatch):
    """Weaker than a real baseline, but it never blocks a genuine change."""
    from app.tools import repo_tools

    monkeypatch.setattr(repo_tools, "working_tree_state", lambda: {"a.py": "1/0"})
    assert graph._changed_files({}) == ["a.py"]


@pytest.mark.asyncio
async def test_test_gate_routes_a_failure_to_diagnosis():
    ctx = FakeContext({"tests_passed": False, "tests": {"failed": 2, "errors": 0}})
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED


@pytest.mark.asyncio
async def test_test_gate_treats_a_sandbox_failure_as_a_failure():
    """A sandbox that could not run is never a pass."""
    ctx = FakeContext(
        {"tests_passed": False, "tests": {"execution_error": "the sandbox failed to start"}}
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
    state = {"review": _review(ReviewVerdict.CHANGES_REQUIRED, [_finding()]).model_dump()}
    ctx = FakeContext(state)
    await decide_after_review(ctx)
    assert ctx.route == ROUTE_MINOR
    assert ctx.state["review_approved"] is False


@pytest.mark.asyncio
async def test_review_gate_sends_major_findings_back_to_the_architect():
    review = _review(ReviewVerdict.CHANGES_REQUIRED, [_finding(), _finding(ChangeScope.MAJOR)])
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


def test_every_model_backed_node_uses_the_configured_model(workflow):
    """No node may quietly fall back to a model nobody configured.

    Which model that is depends on the provider -- one shared model for the
    OpenAI-compatible backends, one per agent for Gemini, whose free tier
    caps requests per model per day.
    """
    seen = []

    def walk(obj, depth=0):
        if depth > 6 or obj is None:
            return
        model = getattr(obj, "model", None)
        name = getattr(obj, "name", None)
        # A model is either a name ADK resolves or a configured instance
        # carrying its own name; both have to be accounted for.
        if isinstance(model, BaseLlm):
            model = model.model
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
    config = get_config()
    allowed = set(gemini_model._candidates()) if config.provider == "gemini" else {config.model}
    assert {model for _, model in seen} <= allowed, seen


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
    report = TestReport(execution_error="the sandbox failed to start")
    text = _summarize(report)
    assert "could not run" in text
    assert "passed" not in text


# ------------------------------------------------- tester through a real runner


class FakeSandbox:
    """Stands in for the sandbox so the tester's state contract can be exercised."""

    def __init__(self, report: TestReport, output: str = "raw pytest output"):
        self.report = report
        self.output = output
        self.commands: list[str] = []

    def run_tests_with_output(self, command, timeout=None):
        self.commands.append(command)
        return self.report, self.output


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
    async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
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
    state, _ = await _run_tester(TestReport(passed=1, failed=2, test_count=3, exit_code=1))
    assert state["tests_passed"] is False
    assert state["tests"]["failed"] == 2


@pytest.mark.asyncio
async def test_tester_reports_a_sandbox_failure_as_not_passing():
    """The one failure mode that must never be mistaken for success."""
    state, _ = await _run_tester(
        TestReport(exit_code=0, execution_error="the sandbox failed to start")
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


@pytest.mark.asyncio
async def test_full_test_log_goes_to_an_artifact_not_into_state():
    """A failing suite's log is thousands of lines; state must not carry it."""
    huge = "FAILED test_x\n" * 5000
    tester = TesterAgent()
    tester.sandbox = FakeSandbox(
        TestReport(passed=1, failed=1, test_count=2, exit_code=1), output=huge
    )

    runner = InMemoryRunner(agent=tester, app_name="artifact_test")
    session = await runner.session_service.create_session(app_name="artifact_test", user_id="u")
    message = types.Content(role="user", parts=[types.Part(text="run")])
    async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
        pass

    refreshed = await runner.session_service.get_session(
        app_name="artifact_test", user_id="u", session_id=session.id
    )
    tests = refreshed.state["tests"]
    assert tests["output_artifact"], "the log should be stored as an artifact"
    assert huge not in str(refreshed.state), "the raw log must not enter state"

    saved = await runner.artifact_service.list_artifact_keys(
        app_name="artifact_test", user_id="u", session_id=session.id
    )
    assert tests["output_artifact"] in saved


# ------------------------------------------------------------- concurrency


def test_concurrency_is_unbounded_by_default(monkeypatch):
    """Repository and risk analysis are meant to overlap."""
    monkeypatch.delenv("SE_AGENT_MAX_CONCURRENCY", raising=False)
    assert graph.max_concurrency() is None
    assert build_workflow().max_concurrency is None


def test_concurrency_can_be_serialized(monkeypatch):
    """An escape hatch for tightly rate-limited providers."""
    monkeypatch.setenv("SE_AGENT_MAX_CONCURRENCY", "1")
    assert graph.max_concurrency() == 1
    assert build_workflow().max_concurrency == 1


def test_malformed_concurrency_falls_back_to_unbounded(monkeypatch):
    monkeypatch.setenv("SE_AGENT_MAX_CONCURRENCY", "lots")
    assert graph.max_concurrency() is None


def test_zero_concurrency_is_raised_to_one(monkeypatch):
    monkeypatch.setenv("SE_AGENT_MAX_CONCURRENCY", "0")
    assert graph.max_concurrency() == 1


# ------------------------------------------------------------------ run modes


@pytest.fixture(scope="module")
def normal_edges():
    """The direct path's edges, as (from, route, to) triples."""
    workflow = build_workflow(RunMode.NORMAL)
    return {(e.from_node.name, e.route, e.to_node.name) for e in workflow.edges}


def test_normal_mode_plans_before_it_implements(normal_edges):
    """Without a plan the coordinator surveys the repository, not changes it."""
    assert ("__START__", None, "planner") in normal_edges
    assert ("planner", None, "iteration_gate") in normal_edges


def test_normal_mode_still_gates_every_route_into_implementation(normal_edges):
    """The gate is the only place the attempt budget holds."""
    assert ("iteration_gate", ROUTE_CONTINUE, "implement") in normal_edges
    assert {frm for frm, _, to in normal_edges if to == "implement"} == {
        "iteration_gate"
    }


def test_a_normal_retry_does_not_replan(normal_edges):
    """The plan is already in history; re-planning would buy nothing."""
    assert ("diagnose", None, "iteration_gate") in normal_edges
    assert {frm for frm, _, to in normal_edges if to == "planner"} == {"__START__"}


def test_normal_mode_skips_the_analysis_pipeline(normal_edges):
    reached = {to for _, _, to in normal_edges}
    assert not reached & {
        "requirements",
        "repository",
        "risk",
        "analysis_join",
        "architecture",
        "review",
        "review_gate",
    }
    # The planner is the one analysis stage the direct path keeps.
    assert "planner" in reached


def test_normal_mode_reports_a_pass_without_review(normal_edges):
    assert ("test_gate", ROUTE_PASSED, "report") in normal_edges


def test_normal_mode_can_still_exhaust_its_budget(normal_edges):
    assert ("iteration_gate", ROUTE_EXHAUSTED, "report") in normal_edges


def test_initial_state_records_the_mode():
    assert initial_state("x", mode=RunMode.NORMAL)["mode"] == "normal"
    assert initial_state("x")["mode"] == "thinking"


@pytest.mark.asyncio
async def test_no_tests_collected_is_not_a_failure_in_normal_mode(with_changes):
    """A document or a rename has no test to pass; demanding one blocks it."""
    ctx = FakeContext(
        {"mode": "normal", "tests_passed": False, "tests": {"test_count": 0}}
    )
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_PASSED
    assert ctx.state["tests_inconclusive"] is True
    assert "no tests were collected" in message


@pytest.mark.asyncio
async def test_an_empty_suite_is_an_error_but_tolerated_directly(with_changes):
    """pytest calls "0 collected" an execution error; the direct path forgives."""
    ctx = FakeContext(
        {
            "mode": "normal",
            "tests_passed": False,
            "tests": {"test_count": 0, "execution_error": "no tests were collected"},
        }
    )
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_PASSED
    assert ctx.state["tests_inconclusive"] is True


@pytest.mark.asyncio
async def test_an_empty_suite_still_fails_the_full_pipeline(with_changes):
    """A feature nobody tested is not a delivered feature."""
    ctx = FakeContext(
        {
            "tests_passed": False,
            "tests": {"test_count": 0, "execution_error": "no tests were collected"},
        }
    )
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED


@pytest.mark.asyncio
async def test_the_direct_path_still_fails_on_a_broken_sandbox(with_changes):
    """The tolerance is for an absent suite only, not a runner that died."""
    ctx = FakeContext(
        {
            "mode": "normal",
            "tests_passed": False,
            "tests": {
                "test_count": 0,
                "execution_error": "the sandbox failed to start",
            },
        }
    )
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED
    assert "could not run" in message


@pytest.mark.asyncio
async def test_no_tests_and_no_changes_is_still_a_failure(without_changes):
    """Relaxing the test demand must not relax the evidence demand."""
    ctx = FakeContext(
        {"mode": "normal", "tests_passed": False, "tests": {"test_count": 0}}
    )
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED


@pytest.mark.asyncio
async def test_an_unimportable_suite_routes_as_could_not_run(without_changes):
    """Not a test failure: the debugger must not hunt a bug that may not exist."""
    ctx = FakeContext(
        {
            "tests_passed": False,
            "tests": {"execution_error": "the test suite could not be imported"},
        }
    )
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED
    assert "could not run" in message


@pytest.mark.asyncio
async def test_exhausting_the_budget_still_records_what_changed(with_changes):
    """The other way into the report; without this it reports nothing changed."""
    ctx = FakeContext({"iteration": 3, "max_iterations": 3})
    await decide_iteration(ctx)
    assert ctx.route == ROUTE_EXHAUSTED
    assert ctx.state["files_changed"] == ["textstats/stats.py"]


# ------------------------------------------- a request already satisfied


def _plan(changes_expected):
    return {"tasks": [], "changes_expected": changes_expected}


@pytest.mark.asyncio
async def test_a_verified_no_op_is_a_pass_not_a_failure(without_changes):
    """Green tests and nothing changed is the signature of a run that did

    nothing *and* of a request already met. The repository is identical in
    both, so only the plan can tell them apart.
    """
    ctx = FakeContext({"tests_passed": True, "plan": _plan(False)})
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_PASSED
    assert ctx.state["no_change_needed"] is True
    assert ctx.state["files_changed"] == []
    assert "already in the requested state" in message


@pytest.mark.asyncio
async def test_a_plan_that_expected_changes_still_fails_on_a_no_op(without_changes):
    """The guard stays on for every ordinary run."""
    ctx = FakeContext({"tests_passed": True, "plan": _plan(True)})
    message = await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED
    assert "changed no files" in message


@pytest.mark.asyncio
async def test_a_missing_plan_keeps_the_guard_on(without_changes):
    """Absent or unreadable, the answer is that changes were expected."""
    for state in ({}, {"plan": None}, {"plan": "junk"}):
        ctx = FakeContext({"tests_passed": True, **state})
        await decide_after_tests(ctx)
        assert ctx.route == ROUTE_FAILED, state


@pytest.mark.asyncio
async def test_a_verification_plan_cannot_excuse_failing_tests(without_changes):
    """changes_expected=False forgives an absent diff, never a red suite."""
    ctx = FakeContext(
        {
            "tests_passed": False,
            "plan": _plan(False),
            "tests": {"test_count": 5, "failed": 2},
        }
    )
    await decide_after_tests(ctx)
    assert ctx.route == ROUTE_FAILED


# ------------------------------------------------- discovering the test runner

# Enough of a unittest file to be recognized as one.
CASE = """
class TestThing(unittest.TestCase):
    def test_x(self):
        pass
"""


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """A directory with no project manifest -- the case pytest cannot guess."""
    monkeypatch.setenv("SE_AGENT_TARGET_REPO", str(tmp_path))
    return tmp_path


def test_a_declared_pytest_config_is_believed(bare):
    (bare / "pyproject.toml").write_text("[tool.pytest.ini_options]", encoding="utf-8")
    (bare / "checks.py").write_text("import unittest" + CASE, encoding="utf-8")
    assert detect_test_command(Workspace(bare)) == "pytest -q"


def test_files_named_for_pytest_get_pytest(bare):
    (bare / "test_thing.py").write_text("def test_x(): pass", encoding="utf-8")
    assert detect_test_command(Workspace(bare)) == "pytest -q"


def test_a_unittest_suite_pytest_would_never_collect(bare):
    """The real gap: tester.py matches no pytest pattern, so pytest finds none."""
    (bare / "tester.py").write_text("import unittest" + CASE, encoding="utf-8")
    assert detect_test_command(Workspace(bare)) == "python -m unittest -v tester"


def test_unittest_modules_are_named_not_discovered(bare):
    """A wide discover pattern imports the whole tree, side effects and all."""
    (bare / "tester.py").write_text("import unittest" + CASE, encoding="utf-8")
    (bare / "checks.py").write_text("import unittest" + CASE, encoding="utf-8")
    command = detect_test_command(Workspace(bare))
    assert "discover" not in command
    assert command.split()[-2:] == ["checks", "tester"]


def test_an_unimportable_filename_is_not_offered_as_a_module(bare):
    """`python -m unittest my-tests` cannot work; do not produce it."""
    (bare / "my-tests.py").write_text("import unittest" + CASE, encoding="utf-8")
    assert detect_test_command(Workspace(bare)) == "pytest -q"


def test_a_javascript_project_gets_npm(bare):
    (bare / "package.json").write_text("{}", encoding="utf-8")
    assert detect_test_command(Workspace(bare)) == "npm test"


def test_an_empty_directory_falls_back_to_pytest(bare):
    """Which collects nothing, and reports that honestly."""
    assert detect_test_command(Workspace(bare)) == "pytest -q"


def test_the_repository_analysis_still_wins(bare):
    """Detection is the fallback; a manifest command outranks it."""
    (bare / "tester.py").write_text("import unittest" + CASE, encoding="utf-8")
    resolved = _resolve_command({"repository": {"test_command": "make check"}})
    assert resolved == "make check"
