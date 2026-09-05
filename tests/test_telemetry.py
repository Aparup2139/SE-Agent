"""Progress tracking: phase transitions, structured results, and reasoning hygiene."""

from __future__ import annotations

from types import SimpleNamespace

from app.schemas import Phase, StageStatus
from app.telemetry import PHASE_ORDER, RunTracker, _brief_args


def event(author="", parts=None, state_delta=None, error=None):
    """Build a minimal stand-in for an ADK event."""
    content = SimpleNamespace(parts=parts) if parts is not None else None
    actions = SimpleNamespace(state_delta=state_delta)
    return SimpleNamespace(
        author=author,
        content=content,
        actions=actions,
        error_message=error,
        error_code="TestError" if error else None,
    )


def part(text=None, function_call=None, thought=None):
    return SimpleNamespace(text=text, function_call=function_call, thought=thought)


# -------------------------------------------------------------------- phases


def test_everything_starts_pending():
    tracker = RunTracker()
    assert all(s is StageStatus.PENDING for s in tracker.phase_status.values())


def test_entering_a_node_marks_its_phase_running():
    tracker = RunTracker()
    tracker.observe(event(author="requirements"))
    assert tracker.phase_status[Phase.REQUIREMENTS] is StageStatus.RUNNING


def test_advancing_completes_the_earlier_phase():
    tracker = RunTracker()
    tracker.observe(event(author="requirements"))
    tracker.observe(event(author="architecture"))
    assert tracker.phase_status[Phase.REQUIREMENTS] is StageStatus.COMPLETE
    assert tracker.phase_status[Phase.ARCHITECTURE] is StageStatus.RUNNING


def test_parallel_analysis_nodes_share_one_phase():
    tracker = RunTracker()
    updates = tracker.observe(event(author="repository"))
    updates += tracker.observe(event(author="risk"))
    phase_updates = [u for u in updates if u.kind == "phase"]
    assert len(phase_updates) == 1, "repository and risk are both the analysis phase"


def test_unknown_author_does_not_change_phase():
    tracker = RunTracker()
    tracker.observe(event(author="requirements"))
    tracker.observe(event(author="some_internal_node"))
    assert tracker.current_phase is Phase.REQUIREMENTS


def test_checklist_shows_progress():
    tracker = RunTracker()
    tracker.observe(event(author="requirements"))
    tracker.observe(event(author="planner"))
    lines = tracker.checklist()
    assert len(lines) == len(PHASE_ORDER)
    assert lines[0].startswith("[x]")
    assert any(line.startswith("[>]") for line in lines)
    assert any(line.startswith("[ ]") for line in lines)


def test_finish_closes_the_running_phase():
    tracker = RunTracker()
    tracker.observe(event(author="report"))
    tracker.finish()
    assert tracker.phase_status[Phase.FINAL] is StageStatus.COMPLETE


# ------------------------------------------------------------ structured data


def test_test_results_are_surfaced():
    tracker = RunTracker()
    updates = tracker.observe(
        event(state_delta={"tests": {"passed": 5, "failed": 2, "errors": 0}})
    )
    tests = [u for u in updates if u.kind == "tests"]
    assert tests and "5 passed" in tests[0].text and "2 failed" in tests[0].text


def test_a_sandbox_failure_is_not_reported_as_counts():
    tracker = RunTracker()
    updates = tracker.observe(
        event(state_delta={"tests": {"execution_error": "docker unreachable"}})
    )
    text = [u for u in updates if u.kind == "tests"][0].text
    assert "could not run" in text
    assert "passed" not in text


def test_review_verdict_is_surfaced():
    tracker = RunTracker()
    updates = tracker.observe(
        event(state_delta={"review": {"verdict": "CHANGES_REQUIRED", "findings": [1, 2]}})
    )
    review = [u for u in updates if u.kind == "review"][0]
    assert "CHANGES_REQUIRED" in review.text
    assert "2 finding" in review.text


def test_iteration_is_tracked():
    tracker = RunTracker()
    tracker.observe(event(state_delta={"iteration": 3}))
    assert tracker.iteration == 3


def test_malformed_iteration_is_ignored():
    tracker = RunTracker()
    tracker.observe(event(state_delta={"iteration": "three"}))
    assert tracker.iteration == 0


# -------------------------------------------------------------------- content


def test_tool_calls_are_counted_and_reported():
    tracker = RunTracker()
    call = SimpleNamespace(name="read_file", args={"path": "app/main.py"})
    updates = tracker.observe(event(author="implement", parts=[part(function_call=call)]))
    tool = [u for u in updates if u.kind == "tool"][0]
    assert "read_file" in tool.text
    assert "app/main.py" in tool.text
    assert tracker.tool_calls == 1


def test_private_reasoning_is_never_surfaced():
    """Thought parts must not reach the user, in telemetry any more than in state."""
    tracker = RunTracker()
    updates = tracker.observe(
        event(
            author="implement",
            parts=[
                part(text="INTERNAL DELIBERATION", thought=True),
                part(text="Implemented the endpoint."),
            ],
        )
    )
    texts = " ".join(u.text for u in updates)
    assert "INTERNAL DELIBERATION" not in texts
    assert "Implemented the endpoint." in texts


def test_blank_text_produces_no_update():
    tracker = RunTracker()
    updates = tracker.observe(event(author="implement", parts=[part(text="   ")]))
    assert not [u for u in updates if u.kind == "message"]


def test_events_without_content_are_safe():
    """A content-free event still advances the phase, but yields no text."""
    updates = RunTracker().observe(event(author="implement"))
    assert [u.kind for u in updates] == ["phase"]


# --------------------------------------------------------------------- errors


def test_errors_are_recorded_and_fail_the_phase():
    tracker = RunTracker()
    tracker.observe(event(author="implement"))
    updates = tracker.observe(event(author="implement", error="model unreachable"))
    assert [u for u in updates if u.kind == "error"]
    assert tracker.errors == ["TestError: model unreachable"]
    assert tracker.phase_status[Phase.CODING] is StageStatus.FAILED


# ------------------------------------------------------------------ arg brief


def test_tool_args_are_rendered_compactly():
    assert _brief_args({"path": "a.py", "line": 3}) == "path=a.py, line=3"


def test_long_tool_args_are_truncated():
    rendered = _brief_args({"content": "x" * 500})
    assert len(rendered) <= 80
    assert rendered.endswith("...")


def test_empty_args_render_empty():
    assert _brief_args({}) == ""
    assert _brief_args(None) == ""


# ------------------------------------------------------------------- stages


def test_structured_stage_output_is_surfaced_for_the_ui():
    tracker = RunTracker()
    updates = tracker.observe(
        event(state_delta={"requirements": {"summary": "add a thing"}})
    )
    stage = [u for u in updates if u.kind == "stage"]
    assert len(stage) == 1
    assert stage[0].text == "Requirements"
    assert stage[0].phase is Phase.REQUIREMENTS
    assert stage[0].data == {
        "key": "requirements",
        "value": {"summary": "add a thing"},
    }


def test_tests_and_review_are_not_duplicated_as_stages():
    """They already have their own kinds; sending both doubles the payload."""
    tracker = RunTracker()
    updates = tracker.observe(
        event(state_delta={"tests": {"passed": 1}, "review": {"verdict": "approved"}})
    )
    assert [u.kind for u in updates] == ["tests", "review"]


def test_an_empty_or_non_dict_stage_value_is_ignored():
    tracker = RunTracker()
    assert tracker.observe(event(state_delta={"plan": {}})) == []
    assert tracker.observe(event(state_delta={"plan": "oops"})) == []
