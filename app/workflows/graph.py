"""The SE-Agent workflow graph.

Built on ADK 2.8.0's ``Workflow``: nodes are agents, edges carry route labels,
and a node chooses the next edge by setting ``ctx.route``. That makes the
control flow a real graph rather than a nest of sequential and loop agents --
and it means cycles (fix -> retest, review -> re-architect) are expressed
directly.

::

                          START
                            |
                       requirements
                        /        \\
              repository          risk        (fan-out, run concurrently)
                        \\        /
                       analysis_join
                            |
                       architecture <-------------------+
                            |                           | major
                         planner                         |
                            |                            |
                     +-> iteration_gate --exhausted--> report
                     |      |                            ^
                     |   implement                        |
                     |      |                             | approved
                     |    test                            |
                     |      |                             |
                     |  test_gate --passed--> review --> review_gate
                     |      |                             |
                     |   failed                        minor
                     |      |                             |
                     +-- diagnose <------------------------+

Every routing decision is made by a function node reading structured state --
the tester's real exit code, the review's own findings -- never by asking a
model where to go.
"""

from __future__ import annotations

import logging
import os

from google.adk.agents import Context
from google.adk.workflow import START, Edge, JoinNode, Workflow, node

from ..agents.architecture import build_architecture_agent
from ..agents.coordinator import build_coordinator_agent, build_report_agent
from ..agents.debugger import build_debugger_agent
from ..agents.planner import build_direct_planner_agent, build_planner_agent
from ..agents.repository import build_repository_agent
from ..agents.requirements import build_requirements_agent
from ..agents.reviewer import build_reviewer_agent
from ..agents.risk import build_risk_agent
from ..agents.tester import TesterAgent
from ..schemas import ChangeScope, CodeReview, RunMode

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 5
ITERATION_CEILING = 10

# Route labels. Named constants so an edge and the node that selects it cannot
# drift apart silently.
ROUTE_CONTINUE = "continue"
ROUTE_EXHAUSTED = "exhausted"
ROUTE_PASSED = "passed"
ROUTE_FAILED = "failed"
ROUTE_APPROVED = "approved"
ROUTE_MINOR = "minor"
ROUTE_MAJOR = "major"

# The sandbox's wording for pytest's exit code 5. A project with no suite is a
# normal thing to be asked to work in; a feature with no test is not.
_NO_TESTS = "no tests were collected"


def max_iterations() -> int:
    """Iteration ceiling, overridable by environment but always bounded."""
    raw = os.environ.get("SE_AGENT_MAX_ITERATIONS")
    try:
        value = int(raw) if raw else DEFAULT_MAX_ITERATIONS
    except ValueError:
        return DEFAULT_MAX_ITERATIONS
    return max(1, min(value, ITERATION_CEILING))


def max_concurrency() -> int | None:
    """How many nodes may run at once. ``None`` means unbounded.

    The analysis stage fans out deliberately, which doubles the instantaneous
    request rate. On a tightly rate-limited NIM tier that is the difference
    between a run completing and a run spending its retry budget on 429s, so
    setting ``SE_AGENT_MAX_CONCURRENCY=1`` serializes the graph at the cost of
    the parallelism.
    """
    raw = os.environ.get("SE_AGENT_MAX_CONCURRENCY")
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def read_review(state) -> CodeReview | None:
    """Parse the reviewer's structured output, tolerating a malformed turn.

    A model can return something that does not validate. That is a bad review,
    not a crash -- callers treat ``None`` as "changes required" so the workflow
    keeps working rather than approving by accident.
    """
    raw = state.get("review")
    if not raw:
        return None
    if isinstance(raw, CodeReview):
        return raw
    try:
        return CodeReview.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - every validation failure is the same here
        logger.warning("review output did not validate: %s", exc)
        return None


def _is_normal(state) -> bool:
    """True on the direct path. Absent mode means the full graph, as before."""
    return state.get("mode") == RunMode.NORMAL.value


def _changes_expected(state) -> bool:
    """Whether the plan said, before implementing, that files should change.

    This is the one thing the repository cannot tell you. "Green tests, nothing
    changed" is the signature of a run that silently did nothing *and* of a
    request that was already satisfied -- the code is identical either way, so
    no amount of inspection separates them. Only the request does, and the
    planner is the stage that reads the request.

    Declared up front, so it is a pre-commitment rather than an excuse: a plan
    cannot decide after a failed implementation that no change was needed.
    Absent or unreadable means expected, which keeps the guard on by default.
    """
    plan = state.get("plan")
    if isinstance(plan, dict):
        return bool(plan.get("changes_expected", True))
    return getattr(plan, "changes_expected", True)


def _tree_state() -> dict[str, str] | None:
    """Fingerprint the workspace's uncommitted changes, or ``None`` if it cannot."""
    from ..tools.repo_tools import working_tree_state

    state = working_tree_state()
    if state is None:
        logger.warning("could not read the working tree state")
    return state


def _changed_files(state) -> list[str]:
    """Files this run changed, measured against the tree it started from.

    A dirty working tree is the normal case: the developer has uncommitted work
    of their own. Comparing against the run's own baseline is what separates
    "the agent edited this" from "this was already modified when we arrived" --
    checking merely whether the tree is dirty would call every ambient edit a
    result of the run, and would never fire the no-op guard at all.

    Returns an empty list when nothing can answer. A tool failure must not be
    mistaken for evidence of a change.
    """
    current = _tree_state()
    if current is None:
        return []
    baseline = state.get("baseline_tree")
    if baseline is None:
        # No baseline: the fingerprint failed at run start but works now. Fall
        # back to "anything dirty counts" -- weaker, but never blocks a change.
        baseline = {}
    return sorted(path for path, mark in current.items() if baseline.get(path) != mark)


# ------------------------------------------------------------- routing nodes


async def decide_iteration(ctx: Context) -> str:
    """Count the attempt and stop the workflow if the budget is spent.

    Every path back into implementation passes through here, so this is the one
    place the hard limit has to hold.
    """
    limit = int(ctx.state.get("max_iterations") or max_iterations())
    iteration = int(ctx.state.get("iteration", 0)) + 1
    ctx.state["iteration"] = iteration
    ctx.state["max_iterations"] = limit

    if iteration > limit:
        ctx.state["iterations_exhausted"] = True
        # Record what the run changed before handing over to the report. This is
        # the *other* way into the reporter, and without this it arrives with no
        # changed-file list at all -- then reports "nothing was changed" about
        # work that is sitting on disk, which is worse than reporting nothing.
        ctx.state["files_changed"] = _changed_files(ctx.state)
        ctx.route = ROUTE_EXHAUSTED
        return (
            f"Iteration budget exhausted ({limit} iterations). Reporting on the "
            "work as it stands."
        )

    ctx.route = ROUTE_CONTINUE
    return f"--- Implementation iteration {iteration} of {limit} ---"


async def decide_after_tests(ctx: Context) -> str:
    """Route on the real test outcome.

    ``tests_passed`` is written by the tester from a process exit code. It is
    the only thing consulted here -- no model gets a say in whether the tests
    passed.

    A green suite is necessary but not sufficient. If this run changed no files,
    the suite that passed is the one that was already passing before the run
    started, which says nothing about the requested feature. That happens
    whenever the implementation step fails -- a provider error, an exhausted
    rate limit -- and without this check an untouched repository would sail
    through to review on the strength of its pre-existing tests.

    Measured against the baseline captured by :func:`initial_state`, not against
    a clean tree, so the guard still fires on a repository that already had
    uncommitted work in it. The exception is a plan that declared up front that
    no change was needed: a request already satisfied is answered, not failed.

    In ``RunMode.NORMAL`` a suite that collected nothing is not a failure. A
    green suite is the evidence a *feature* needs, but the direct path is also
    used for work no test covers -- writing a document, renaming a file -- and
    there the demand for a passing test is one the task cannot meet. The
    changed-file check still holds, so a run that did nothing still fails.
    """
    if ctx.state.get("tests_passed"):
        changed = _changed_files(ctx.state)
        if not changed and _changes_expected(ctx.state):
            ctx.route = ROUTE_FAILED
            return (
                "Tests passed, but this run changed no files -- nothing was "
                "implemented, so these are the pre-existing tests. Treating this "
                "as a failed iteration."
            )
        if not changed:
            # The plan said before implementing that no change was required, and
            # the suite it named is green. That is the request answered, not a
            # run that did nothing -- and because the declaration was made up
            # front it cannot be an excuse invented after failing.
            ctx.state["files_changed"] = []
            ctx.state["no_change_needed"] = True
            ctx.route = ROUTE_PASSED
            return (
                "Tests pass and the plan expected no change: the repository is "
                "already in the requested state. Nothing to implement."
            )
        ctx.state["files_changed"] = changed
        ctx.route = ROUTE_PASSED
        return (
            f"Tests passed across {len(changed)} changed file(s); sending the "
            "change for independent review."
        )

    tests = ctx.state.get("tests") or {}
    error = tests.get("execution_error") or ""

    # "No tests collected" arrives as an execution error, because for a feature
    # that is exactly what it is. On the direct path it is the ordinary state of
    # a project with no suite, so it must be checked before the error branch --
    # otherwise the leniency below can never be reached.
    collected_nothing = not tests.get("test_count") and (
        not error or _NO_TESTS in error
    )

    if error and not (_is_normal(ctx.state) and collected_nothing):
        ctx.route = ROUTE_FAILED
        return f"Tests could not run: {error}. Diagnosing."

    if _is_normal(ctx.state) and collected_nothing:
        changed = _changed_files(ctx.state)
        if not changed:
            ctx.route = ROUTE_FAILED
            return (
                "No tests were collected and this run changed no files -- there "
                "is no evidence anything was done. Treating this as a failed "
                "iteration."
            )
        ctx.state["files_changed"] = changed
        ctx.state["tests_inconclusive"] = True
        ctx.route = ROUTE_PASSED
        return (
            f"{len(changed)} file(s) changed, but no tests were collected, so "
            "nothing here is verified by a test. Reporting the change as it is."
        )

    failed = tests.get("failed", 0)
    errors = tests.get("errors", 0)
    ctx.route = ROUTE_FAILED
    return f"Tests failed ({failed} failed, {errors} errors); diagnosing."


async def decide_after_review(ctx: Context) -> str:
    """Route on the review's own findings.

    ``CodeReview.routing_scope`` is derived from the findings, so the
    destination follows from the review's content rather than from a separate
    judgement about it.
    """
    review = read_review(ctx.state)

    if review is None:
        ctx.state["review_approved"] = False
        ctx.route = ROUTE_MINOR
        return (
            "The review could not be parsed; treating it as changes required "
            "rather than risking an accidental approval."
        )

    scope = review.routing_scope

    if scope is ChangeScope.NONE:
        ctx.state["review_approved"] = True
        ctx.route = ROUTE_APPROVED
        return f"Review APPROVED. {review.summary}"

    ctx.state["review_approved"] = False

    if scope is ChangeScope.MAJOR:
        ctx.route = ROUTE_MAJOR
        blocking = [
            f.description for f in review.findings if f.scope is ChangeScope.MAJOR
        ]
        detail = "\n".join(f"  - {d}" for d in blocking[:5])
        return f"Review found architectural problems; re-designing:\n{detail}"

    ctx.route = ROUTE_MINOR
    return (
        f"Review requires {len(review.findings)} change(s); returning to the "
        "engineer."
    )


# The routing logic above is kept as plain functions so it can be tested
# directly; these wrap them as workflow nodes.
iteration_gate = node(decide_iteration, name="iteration_gate")
test_gate = node(decide_after_tests, name="test_gate")
review_gate = node(decide_after_review, name="review_gate")


# -------------------------------------------------------------- graph builder


def build_workflow(mode: RunMode = RunMode.THINKING) -> Workflow:
    """Assemble the SE-Agent graph for one run mode.

    A fresh set of agents is built per call: ADK gives an agent one parent, and
    the architect appears twice in this graph (once on the main path, once on
    the re-design cycle), so instances cannot be shared.

    The iteration budget is not baked in here -- it lives in session state, seeded
    by :func:`initial_state`, so a resumed run keeps the budget it began with.

    ``RunMode.NORMAL`` builds the direct path only, and builds *only* the agents
    that path uses -- constructing the four unreachable analysis agents would
    cost a model instance each for nothing.
    """
    if mode is RunMode.NORMAL:
        return _direct_workflow()

    requirements = node(build_requirements_agent(), name="requirements")
    repository = node(build_repository_agent(), name="repository")
    risk = node(build_risk_agent(), name="risk")
    analysis_join = JoinNode(name="analysis_join")
    architecture = node(build_architecture_agent(), name="architecture")
    planner = node(build_planner_agent(), name="planner")
    implement = node(build_coordinator_agent(), name="implement")
    test = node(TesterAgent(), name="test")
    diagnose = node(build_debugger_agent(), name="diagnose")
    review = node(build_reviewer_agent(), name="review")
    report = node(build_report_agent(), name="report")

    workflow = Workflow(
        name="se_agent",
        max_concurrency=max_concurrency(),
        description=(
            "Takes a feature request through requirements, parallel repository "
            "and risk analysis, architecture, planning, implementation with "
            "real sandboxed tests, independent review, and a final report."
        ),
        edges=[
            Edge(from_node=START, to_node=requirements),
            # Repository and risk analysis are independent, so they fan out and
            # join rather than running one after the other.
            Edge(from_node=requirements, to_node=repository),
            Edge(from_node=requirements, to_node=risk),
            Edge(from_node=repository, to_node=analysis_join),
            Edge(from_node=risk, to_node=analysis_join),
            Edge(from_node=analysis_join, to_node=architecture),
            Edge(from_node=architecture, to_node=planner),
            # Every route into implementation passes the iteration gate.
            Edge(from_node=planner, to_node=iteration_gate),
            Edge(from_node=iteration_gate, to_node=implement, route=ROUTE_CONTINUE),
            Edge(from_node=iteration_gate, to_node=report, route=ROUTE_EXHAUSTED),
            Edge(from_node=implement, to_node=test),
            Edge(from_node=test, to_node=test_gate),
            # Failing tests are diagnosed, then implementation is retried.
            Edge(from_node=test_gate, to_node=diagnose, route=ROUTE_FAILED),
            Edge(from_node=diagnose, to_node=iteration_gate),
            # Passing tests earn a review, never a pass by themselves.
            Edge(from_node=test_gate, to_node=review, route=ROUTE_PASSED),
            Edge(from_node=review, to_node=review_gate),
            Edge(from_node=review_gate, to_node=report, route=ROUTE_APPROVED),
            Edge(from_node=review_gate, to_node=iteration_gate, route=ROUTE_MINOR),
            Edge(from_node=review_gate, to_node=architecture, route=ROUTE_MAJOR),
        ],
    )
    return workflow


def _direct_workflow() -> Workflow:
    """The direct path: plan, implement, test, report.

    It keeps the two things that make a result mean anything -- a real test run
    and the changed-file check -- and drops the four analysis agents, which are
    most of a run's model calls. What it gives up is the independent review.
    """
    direct_planner = node(build_direct_planner_agent(), name="planner")
    implement = node(build_coordinator_agent(), name="implement")
    test = node(TesterAgent(), name="test")
    diagnose = node(build_debugger_agent(), name="diagnose")
    report = node(build_report_agent(), name="report")

    return Workflow(
        name="se_agent_normal",
        max_concurrency=max_concurrency(),
        description=(
            "Plans a request against the repository, implements it, runs the "
            "project's real tests, and reports what changed."
        ),
        edges=[
            # Plan first, even here. Without one the coordinator has nothing to
            # converge on and spends its calls surveying the repository instead
            # of changing it.
            Edge(from_node=START, to_node=direct_planner),
            # Then the iteration gate: the direct path still needs a bounded
            # number of attempts, and that is the only place the bound is
            # enforced. Retries re-enter here, not at the planner -- the plan is
            # in history already and re-planning would cost a call to arrive at
            # the same place.
            Edge(from_node=direct_planner, to_node=iteration_gate),
            Edge(from_node=iteration_gate, to_node=implement, route=ROUTE_CONTINUE),
            Edge(from_node=iteration_gate, to_node=report, route=ROUTE_EXHAUSTED),
            Edge(from_node=implement, to_node=test),
            Edge(from_node=test, to_node=test_gate),
            Edge(from_node=test_gate, to_node=diagnose, route=ROUTE_FAILED),
            Edge(from_node=diagnose, to_node=iteration_gate),
            # No reviewer on this path: the change goes straight to the report.
            # Skipping review is the cost of skipping the pipeline.
            Edge(from_node=test_gate, to_node=report, route=ROUTE_PASSED),
        ],
    )


def initial_state(
    feature_request: str,
    limit: int | None = None,
    mode: RunMode = RunMode.THINKING,
) -> dict:
    """Seed session state for a run.

    The iteration budget is placed in state rather than baked into the graph so
    that a resumed session keeps the budget it started with. The working-tree
    baseline is seeded here for the same reason: a resumed run must keep
    measuring against the tree its *first* iteration saw.
    """
    return {
        "feature_request": feature_request,
        "iteration": 0,
        "max_iterations": limit if limit is not None else max_iterations(),
        # The gates read this to know which guarantees apply to this run.
        "mode": mode.value,
        # Snapshot the tree before anything runs, so the no-op guard can tell
        # this run's edits from work that was already uncommitted.
        "baseline_tree": _tree_state(),
    }


root_agent = build_workflow()
