"""Workflow schema invariants -- especially the ones routing depends on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    ChangeScope,
    CodeReview,
    ImplementationPlan,
    Phase,
    Requirements,
    ReviewFinding,
    ReviewVerdict,
    Severity,
    Task,
    TestReport,
    WorkflowState,
)

# ---------------------------------------------------------------- test report


def test_green_requires_tests_to_have_actually_run():
    """An empty suite is not a pass -- it is a missing suite."""
    assert not TestReport(exit_code=0, test_count=0).is_green


def test_green_report():
    report = TestReport(passed=12, test_count=12, exit_code=0)
    assert report.is_green and report.ran


def test_failures_are_not_green():
    assert not TestReport(passed=3, failed=1, test_count=4, exit_code=1).is_green


def test_errors_are_not_green():
    assert not TestReport(passed=3, errors=1, test_count=4, exit_code=0).is_green


def test_sandbox_failure_is_not_green_even_with_zero_exit():
    report = TestReport(
        passed=5, test_count=5, exit_code=0, execution_error="docker daemon unreachable"
    )
    assert not report.is_green
    assert not report.ran


def test_negative_counts_are_rejected():
    with pytest.raises(ValidationError):
        TestReport(passed=-1)


# --------------------------------------------------------------------- review


def finding(scope=ChangeScope.MINOR, severity=Severity.MEDIUM):
    return ReviewFinding(
        id="F-1",
        category="correctness",
        severity=severity,
        description="d",
        recommendation="r",
        scope=scope,
    )


def test_approved_review_routes_nowhere():
    review = CodeReview(verdict=ReviewVerdict.APPROVED, summary="ok")
    assert review.routing_scope is ChangeScope.NONE


def test_minor_findings_route_to_the_coder():
    review = CodeReview(
        verdict=ReviewVerdict.CHANGES_REQUIRED, summary="s", findings=[finding()]
    )
    assert review.routing_scope is ChangeScope.MINOR


def test_any_major_finding_routes_to_the_architect():
    review = CodeReview(
        verdict=ReviewVerdict.CHANGES_REQUIRED,
        summary="s",
        findings=[finding(), finding(scope=ChangeScope.MAJOR)],
    )
    assert review.routing_scope is ChangeScope.MAJOR


def test_approval_cannot_hide_a_critical_finding():
    """Guards against a model approving while still reporting a blocker."""
    with pytest.raises(ValidationError, match="cannot be APPROVED"):
        CodeReview(
            verdict=ReviewVerdict.APPROVED,
            summary="looks fine",
            findings=[finding(severity=Severity.CRITICAL)],
        )


def test_approval_tolerates_low_severity_notes():
    review = CodeReview(
        verdict=ReviewVerdict.APPROVED,
        summary="ok",
        findings=[finding(severity=Severity.LOW)],
    )
    assert review.routing_scope is ChangeScope.NONE


# ----------------------------------------------------------------------- plan


def make_task(task_id, deps=()):
    return Task(
        task_id=task_id,
        objective="o",
        validation_criteria="v",
        dependencies=list(deps),
    )


def test_plan_accepts_resolvable_dependencies():
    plan = ImplementationPlan(tasks=[make_task("T-1"), make_task("T-2", ["T-1"])])
    assert len(plan.tasks) == 2


def test_plan_rejects_dangling_dependency():
    with pytest.raises(ValidationError, match="unknown task"):
        ImplementationPlan(tasks=[make_task("T-1", ["T-99"])])


# --------------------------------------------------------------------- limits


def test_unbounded_requirement_lists_are_rejected():
    with pytest.raises(ValidationError):
        Requirements(summary="s", functional=[f"req {i}" for i in range(50)])


def test_unknown_fields_are_rejected():
    """Catches a model inventing keys instead of filling the schema."""
    with pytest.raises(ValidationError):
        Requirements(summary="s", made_up_field="x")


def test_long_text_is_rejected_rather_than_silently_stored():
    with pytest.raises(ValidationError):
        Requirements(summary="x" * 5000)


# ---------------------------------------------------------------------- state


def test_state_starts_at_requirements():
    state = WorkflowState(feature_request="add a healthcheck endpoint")
    assert state.phase is Phase.REQUIREMENTS
    assert state.iteration == 0
    assert not state.iterations_exhausted


def test_iteration_limit_is_detected():
    state = WorkflowState(feature_request="x", iteration=5, max_iterations=5)
    assert state.iterations_exhausted


def test_iteration_ceiling_is_capped():
    with pytest.raises(ValidationError):
        WorkflowState(feature_request="x", max_iterations=99)


def test_state_round_trips_through_json():
    """Session state is serialized, so every model must survive a round trip."""
    state = WorkflowState(
        feature_request="add caching",
        tests=TestReport(passed=3, test_count=3),
        review=CodeReview(verdict=ReviewVerdict.APPROVED, summary="ok"),
    )
    restored = WorkflowState.model_validate_json(state.model_dump_json())
    assert restored.tests.is_green
    assert restored.review.verdict is ReviewVerdict.APPROVED
