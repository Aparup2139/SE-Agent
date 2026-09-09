"""Turn ADK events into the progress a human (or a UI) can actually follow.

ADK emits a fine-grained event stream: model turns, tool calls, tool results,
state deltas. This module translates that into the coarse picture someone
watching a run wants -- which phase, which agent, which iteration, what the
tests said -- without ever surfacing private reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import Phase, RunMode, StageStatus

# Which workflow node belongs to which phase, for the checklist a UI renders.
NODE_PHASES: dict[str, Phase] = {
    "requirements": Phase.REQUIREMENTS,
    "repository": Phase.ANALYSIS,
    "risk": Phase.ANALYSIS,
    "analysis_join": Phase.ANALYSIS,
    "architecture": Phase.ARCHITECTURE,
    "planner": Phase.PLANNING,
    "iteration_gate": Phase.CODING,
    "implement": Phase.CODING,
    "coordinator_agent": Phase.CODING,
    "test": Phase.TESTING,
    "test_gate": Phase.TESTING,
    "tester_agent": Phase.TESTING,
    "diagnose": Phase.CODING,
    "debugger_agent": Phase.CODING,
    "review": Phase.REVIEW,
    "review_gate": Phase.REVIEW,
    "reviewer_agent": Phase.REVIEW,
    "report": Phase.FINAL,
    "report_agent": Phase.FINAL,
}

PHASE_ORDER = [
    Phase.REQUIREMENTS,
    Phase.ANALYSIS,
    Phase.ARCHITECTURE,
    Phase.PLANNING,
    Phase.CODING,
    Phase.TESTING,
    Phase.REVIEW,
    Phase.FINAL,
]

# The direct path visits four of them. Listing the subset explicitly (rather
# than filtering PHASE_ORDER) keeps the checklist a UI renders honest: a normal
# run shows four steps and completes four, instead of showing eight and leaving
# four pending forever.
NORMAL_PHASE_ORDER = [
    Phase.PLANNING,
    Phase.CODING,
    Phase.TESTING,
    Phase.FINAL,
]


def phase_order(mode: RunMode = RunMode.THINKING) -> list[Phase]:
    """The phases a run in this mode actually passes through."""
    return NORMAL_PHASE_ORDER if mode is RunMode.NORMAL else PHASE_ORDER


PHASE_LABELS = {
    Phase.REQUIREMENTS: "Requirements",
    Phase.ANALYSIS: "Repository & risk analysis",
    Phase.ARCHITECTURE: "Architecture",
    Phase.PLANNING: "Planning",
    Phase.CODING: "Implementation",
    Phase.TESTING: "Testing",
    Phase.REVIEW: "Review",
    Phase.FINAL: "Final report",
}


# Structured state a run produces, in the order it lands. ``tests`` and
# ``review`` are absent: they already have their own update kinds.
STAGE_KEYS: dict[str, tuple[Phase, str]] = {
    "requirements": (Phase.REQUIREMENTS, "Requirements"),
    "repository": (Phase.ANALYSIS, "Repository analysis"),
    "risks": (Phase.ANALYSIS, "Risk analysis"),
    "architecture": (Phase.ARCHITECTURE, "Architecture"),
    "plan": (Phase.PLANNING, "Implementation plan"),
    "report": (Phase.FINAL, "Final report"),
}


@dataclass
class Update:
    """One thing worth telling the user about."""

    kind: str  # phase | agent | tool | tests | review | error | message
    text: str
    phase: Phase | None = None
    agent: str = ""
    iteration: int = 0
    data: dict[str, Any] = field(default_factory=dict)


class RunTracker:
    """Folds an ADK event stream into per-phase progress.

    Kept deliberately tolerant: an event shape that is not recognized produces
    no update rather than an exception, because losing telemetry must never
    take down a run.
    """

    def __init__(self, mode: RunMode = RunMode.THINKING) -> None:
        self.mode = mode
        self.order = phase_order(mode)
        self.phase_status: dict[Phase, StageStatus] = {
            phase: StageStatus.PENDING for phase in self.order
        }
        self.current_phase: Phase | None = None
        self.current_agent: str = ""
        self.iteration: int = 0
        self.tool_calls: int = 0
        self.errors: list[str] = []

    # ------------------------------------------------------------------ phases

    def _enter_phase(self, phase: Phase) -> Update | None:
        if phase is self.current_phase:
            return None

        if phase not in self.phase_status:
            # A node from a phase this mode does not track. Report it, but do
            # not invent a checklist row for it.
            return None

        # Everything before the new phase has, by definition, been reached.
        index = self.order.index(phase)
        for earlier in self.order[:index]:
            if self.phase_status[earlier] is StageStatus.RUNNING:
                self.phase_status[earlier] = StageStatus.COMPLETE

        self.phase_status[phase] = StageStatus.RUNNING
        self.current_phase = phase
        return Update(
            kind="phase",
            text=PHASE_LABELS[phase],
            phase=phase,
            iteration=self.iteration,
        )

    def checklist(self) -> list[str]:
        """The progress display: done, running, pending."""
        marks = {
            StageStatus.COMPLETE: "[x]",
            StageStatus.RUNNING: "[>]",
            StageStatus.PENDING: "[ ]",
            StageStatus.FAILED: "[!]",
            StageStatus.SKIPPED: "[-]",
        }
        return [
            f"{marks[self.phase_status[phase]]} {PHASE_LABELS[phase]}"
            for phase in self.order
        ]

    # ------------------------------------------------------------------ events

    def observe(self, event: Any) -> list[Update]:
        """Translate one ADK event into zero or more updates."""
        updates: list[Update] = []

        author = getattr(event, "author", "") or ""
        if author and author != self.current_agent:
            self.current_agent = author
            phase = NODE_PHASES.get(author)
            if phase is not None:
                entered = self._enter_phase(phase)
                if entered:
                    updates.append(entered)

        updates.extend(self._observe_state(event))
        updates.extend(self._observe_content(event, author))

        error_message = getattr(event, "error_message", None)
        if error_message:
            code = getattr(event, "error_code", "") or "error"
            self.errors.append(f"{code}: {error_message}")
            if self.current_phase:
                self.phase_status[self.current_phase] = StageStatus.FAILED
            updates.append(
                Update(
                    kind="error",
                    text=f"{code}: {error_message}",
                    phase=self.current_phase,
                    agent=author,
                    iteration=self.iteration,
                )
            )

        return updates

    def _observe_state(self, event: Any) -> list[Update]:
        """Surface the structured results the workflow routes on."""
        actions = getattr(event, "actions", None)
        delta = getattr(actions, "state_delta", None) if actions else None
        if not delta:
            return []

        updates: list[Update] = []

        if "iteration" in delta:
            try:
                self.iteration = int(delta["iteration"])
            except (TypeError, ValueError):
                pass

        for key, (phase, label) in STAGE_KEYS.items():
            value = delta.get(key)
            if isinstance(value, dict) and value:
                updates.append(
                    Update(
                        kind="stage",
                        text=label,
                        phase=phase,
                        iteration=self.iteration,
                        data={"key": key, "value": value},
                    )
                )

        if "tests" in delta:
            tests = delta["tests"] or {}
            if tests.get("execution_error"):
                text = f"Tests could not run: {tests['execution_error']}"
            else:
                text = (
                    f"{tests.get('passed', 0)} passed, "
                    f"{tests.get('failed', 0)} failed, "
                    f"{tests.get('errors', 0)} errors"
                )
            updates.append(
                Update(
                    kind="tests",
                    text=text,
                    phase=Phase.TESTING,
                    iteration=self.iteration,
                    data=tests,
                )
            )

        if "review" in delta:
            review = delta["review"] or {}
            verdict = review.get("verdict", "?")
            findings = review.get("findings") or []
            updates.append(
                Update(
                    kind="review",
                    text=f"Review: {verdict} ({len(findings)} finding(s))",
                    phase=Phase.REVIEW,
                    iteration=self.iteration,
                    data=review,
                )
            )

        return updates

    def _observe_content(self, event: Any, author: str) -> list[Update]:
        """Surface tool activity and visible text, never private reasoning."""
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return []

        updates: list[Update] = []
        for part in parts:
            # `thought` parts carry the model's private reasoning. They are
            # dropped here as well as in the provider layer.
            if getattr(part, "thought", None):
                continue

            call = getattr(part, "function_call", None)
            if call is not None:
                self.tool_calls += 1
                updates.append(
                    Update(
                        kind="tool",
                        text=f"{call.name}({_brief_args(call.args)})",
                        phase=self.current_phase,
                        agent=author,
                        iteration=self.iteration,
                    )
                )
                continue

            text = getattr(part, "text", None)
            if text and text.strip():
                updates.append(
                    Update(
                        kind="message",
                        text=text.strip(),
                        phase=self.current_phase,
                        agent=author,
                        iteration=self.iteration,
                    )
                )

        return updates

    def finish(self) -> None:
        """Mark the run complete, so the checklist does not end mid-phase."""
        for phase in self.order:
            if self.phase_status[phase] is StageStatus.RUNNING:
                self.phase_status[phase] = StageStatus.COMPLETE


def _brief_args(args: Any, limit: int = 80) -> str:
    """Render tool arguments compactly for a one-line log."""
    if not args:
        return ""
    rendered = ", ".join(f"{k}={_brief_value(v)}" for k, v in args.items())
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _brief_value(value: Any, limit: int = 40) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."
