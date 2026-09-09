"""The coordinator is handed its plan, and told to do nothing else."""

from __future__ import annotations

import pytest

from app.agents import common as agents_common
from app.agents.coordinator import build_coordinator_agent, coordinator_instruction

PLAN = {
    "ordering_rationale": "validation first",
    "tasks": [
        {
            "task_id": "T-1",
            "objective": "Raise ValueError on mismatched lengths",
            "files": ["knapsack.py"],
            "dependencies": [],
            "implementation_notes": "Both guards return 0 today.",
            "validation_criteria": "test_01 passes",
        }
    ],
}


class FakeContext:
    """Stands in for ADK's ReadonlyContext, which exposes session state."""

    def __init__(self, state):
        self.state = state


def test_the_plan_is_printed_into_the_prompt():
    """State is not visible to an agent unless something puts it there."""
    text = coordinator_instruction(FakeContext({"plan": PLAN}))
    assert "T-1" in text
    assert "Raise ValueError on mismatched lengths" in text
    assert "knapsack.py" in text
    assert "Both guards return 0 today." in text
    assert "test_01 passes" in text


def test_the_prompt_forbids_wandering():
    """The failure being fixed: 20 calls of git archaeology, then giving up."""
    text = coordinator_instruction(FakeContext({"plan": PLAN}))
    assert "Change only files a task names" in text
    assert "git log" in text
    assert "Do not look for other work" in text


def test_a_missing_plan_is_reported_not_improvised():
    """A coordinator with no plan must stop, not invent a task it can do."""
    for state in ({}, {"plan": None}, {"plan": {"tasks": []}}, {"plan": "junk"}):
        text = coordinator_instruction(FakeContext(state))
        assert "No plan reached this agent" in text, state


def test_it_is_never_told_that_no_instructions_were_given():
    """It said exactly that once, with a good plan sitting in state."""
    text = coordinator_instruction(FakeContext({"plan": PLAN}))
    assert "never conclude that no instructions" in text


@pytest.mark.asyncio
async def test_adk_resolves_the_instruction_with_the_plan_in_it(monkeypatch):
    """End to end through ADK: a provider bypasses state templating, so this

    checks the prompt ADK would actually send, preamble included.
    """
    monkeypatch.setattr(agents_common, "MODEL", "stub/deterministic")
    agent = build_coordinator_agent()
    text, bypassed = await agent.canonical_instruction(FakeContext({"plan": PLAN}))
    assert bypassed is True
    assert "You are one agent in SE-Agent" in text
    assert "T-1" in text and "knapsack.py" in text


# ------------------------------------------------- the reporter's own brief


def test_the_reporter_is_told_what_was_asked():
    """It is not otherwise: state reaches an agent only if something puts it there."""
    from app.agents.coordinator import report_instruction

    text = report_instruction(
        FakeContext({"feature_request": "write binarytree.py", "plan": PLAN})
    )
    assert "## What was asked" in text
    assert "write binarytree.py" in text
    assert "T-1" in text


def test_the_reporter_is_forbidden_from_inventing_a_request():
    """Given none, one reported a previous hour's task as this run's work."""
    from app.agents.coordinator import report_instruction

    text = report_instruction(FakeContext({}))
    assert "The request did not reach this agent" in text
    assert "Do not reconstruct one from the" in text


def test_the_reporter_is_warned_off_the_repository_s_leftovers():
    """A directory can hold a previous task's files, tests and fixes."""
    from app.agents.coordinator import report_instruction

    text = report_instruction(FakeContext({"feature_request": "x", "plan": PLAN}))
    assert "Report on **this** run only" in text
    assert "none of that is this run's work" in text
