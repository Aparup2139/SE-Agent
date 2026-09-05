"""Tester: run the real test suite and report exactly what happened.

This agent has no LLM. That is the point.

A language model asked to "run the tests and report results" can produce a
plausible-looking pass without running anything, and the failure mode is silent.
So the tester is a deterministic ADK agent: it invokes the Docker sandbox,
parses the actual process output, and writes a :class:`TestReport` built purely
from the exit code and stdout. There is no path through this code by which a
model's opinion becomes a test result.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from ..schemas import TestReport
from ..tools.sandbox import DockerSandbox, get_sandbox

logger = logging.getLogger(__name__)

DEFAULT_TEST_COMMAND = "pytest -q"


def _resolve_command(state: dict[str, Any]) -> str:
    """Use the command the repository analysis found, not a guess.

    Projects differ: `pytest -q`, `npm test`, `go test ./...`. The repository
    agent reads this out of the manifest; falling back to a default is a last
    resort, not the normal path.
    """
    repository = state.get("repository") or {}
    if isinstance(repository, dict):
        command = (repository.get("test_command") or "").strip()
        if command:
            return command
    return DEFAULT_TEST_COMMAND


def _summarize(report: TestReport) -> str:
    """A short, honest, human-readable line for the event stream."""
    if report.execution_error:
        return f"Tests could not run: {report.execution_error}"

    parts = [f"{report.passed} passed"]
    if report.failed:
        parts.append(f"{report.failed} failed")
    if report.errors:
        parts.append(f"{report.errors} errors")
    if report.skipped:
        parts.append(f"{report.skipped} skipped")
    headline = ", ".join(parts) + f" in {report.duration_seconds}s"

    if report.is_green:
        return f"All tests passed: {headline}"

    lines = [f"Tests failed: {headline}"]
    if report.failure_summary:
        lines.append("Failures:")
        lines.extend(f"  - {failure}" for failure in report.failure_summary[:10])
        if len(report.failure_summary) > 10:
            lines.append(f"  ... and {len(report.failure_summary) - 10} more")
    return "\n".join(lines)


class TesterAgent(BaseAgent):
    """Executes the project's tests in Docker and records the real outcome."""

    sandbox: DockerSandbox | None = None

    def __init__(self, name: str = "tester_agent", **kwargs) -> None:
        super().__init__(
            name=name,
            description=(
                "Runs the project's test suite inside a Docker sandbox and "
                "reports the real pass/fail counts and failure details."
            ),
            **kwargs,
        )

    def _sandbox(self) -> DockerSandbox:
        return self.sandbox or get_sandbox()

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        command = _resolve_command(state)

        logger.info("running tests: %s", command)
        report = self._sandbox().run_tests(command)

        summary = _summarize(report)
        if report.execution_error:
            logger.error("test execution failed: %s", report.execution_error)

        # The report goes into state as structured data; the summary goes into
        # the conversation so the next agent reads the failures in context.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=summary)]),
            actions=EventActions(
                state_delta={
                    "tests": report.model_dump(mode="json"),
                    "tests_passed": report.is_green,
                }
            ),
        )


