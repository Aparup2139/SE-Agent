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
from ..agents.planner import build_planner_agent
from ..agents.repository import build_repository_agent
from ..agents.requirements import build_requirements_agent
from ..agents.reviewer import build_reviewer_agent
from ..agents.risk import build_risk_agent
from ..agents.tester import TesterAgent
from ..schemas import ChangeScope, CodeReview

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


def max_iterations() -> int:
    """Iteration ceiling, overridable by environment but always bounded."""
    raw = os.environ.get("SE_AGENT_MAX_ITERATIONS")
    try:
        value = int(raw) if raw else DEFAULT_MAX_ITERATIONS
    except ValueError:
        return DEFAULT_MAX_ITERATIONS
    return max(1, min(value, ITERATION_CEILING))


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
    """
    if ctx.state.get("tests_passed"):
        ctx.route = ROUTE_PASSED
        return "Tests passed; sending the change for independent review."

    tests = ctx.state.get("tests") or {}
    if tests.get("execution_error"):
        ctx.route = ROUTE_FAILED
        return f"Tests could not run: {tests['execution_error']}. Diagnosing."

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


def build_workflow() -> Workflow:
    """Assemble the full SE-Agent graph.

    A fresh set of agents is built per call: ADK gives an agent one parent, and
    the architect appears twice in this graph (once on the main path, once on
    the re-design cycle), so instances cannot be shared.

    The iteration budget is not baked in here -- it lives in session state, seeded
    by :func:`initial_state`, so a resumed run keeps the budget it began with.
    """
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


def initial_state(feature_request: str, limit: int | None = None) -> dict:
    """Seed session state for a run.

    The iteration budget is placed in state rather than baked into the graph so
    that a resumed session keeps the budget it started with.
    """
    return {
        "feature_request": feature_request,
        "iteration": 0,
        "max_iterations": limit if limit is not None else max_iterations(),
    }


root_agent = build_workflow()
