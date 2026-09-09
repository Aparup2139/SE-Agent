"""Tester: run the real test suite and report exactly what happened.

This agent has no LLM. That is the point.

A language model asked to "run the tests and report results" can produce a
plausible-looking pass without running anything, and the failure mode is silent.
So the tester is a deterministic ADK agent: it invokes the sandbox,
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
from ..tools.sandbox import Sandbox, get_sandbox
from ..tools.workspace import Workspace, get_workspace

logger = logging.getLogger(__name__)

DEFAULT_TEST_COMMAND = "pytest -q"

# pytest only collects files matching these. A test file named anything else --
# `tester.py`, `checks.py` -- is invisible to it, and "0 tests collected" is
# indistinguishable from "this change is untested".
PYTEST_PATTERNS = ("test_*.py", "*_test.py")

# A declared config means the project has already chosen pytest; believe it
# rather than sniffing around for something else.
PYTEST_CONFIG_FILES = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")

MANIFEST_COMMANDS = (
    ("package.json", "npm test"),
    ("go.mod", "go test ./..."),
    ("Cargo.toml", "cargo test"),
)

# Enough to name the test modules of a small project without building a command
# line that no shell will accept.
MAX_UNITTEST_MODULES = 20


def _resolve_command(state: dict[str, Any]) -> str:
    """Use the command the repository analysis found, else work it out.

    Projects differ: `pytest -q`, `npm test`, `go test ./...`. The repository
    agent reads this out of the manifest, and on the direct path it never runs
    at all -- so the fallback is a real detection pass, not a guess.
    """
    repository = state.get("repository") or {}
    if isinstance(repository, dict):
        command = (repository.get("test_command") or "").strip()
        if command:
            return command
    return detect_test_command()


def detect_test_command(workspace: Workspace | None = None) -> str:
    """Work out how to run this project's tests, by looking at it.

    Deterministic on purpose. Asking a model "how do I test this?" costs a call
    and can be wrong in a way nothing catches; the filesystem already knows.
    """
    workspace = workspace or get_workspace()
    root = workspace.root

    if any((root / name).is_file() for name in PYTEST_CONFIG_FILES):
        return DEFAULT_TEST_COMMAND

    for pattern in PYTEST_PATTERNS:
        if next(iter(workspace.iter_files(pattern=f"**/{pattern}")), None):
            return DEFAULT_TEST_COMMAND

    modules = _unittest_modules(workspace)
    if modules:
        # Named explicitly rather than discovered: `unittest discover` with a
        # wide pattern imports every module in the tree, side effects included.
        return "python -m unittest -v " + " ".join(modules)

    for name, command in MANIFEST_COMMANDS:
        if (root / name).is_file():
            return command

    return DEFAULT_TEST_COMMAND


def _unittest_modules(workspace: Workspace) -> list[str]:
    """Importable module names of files holding ``unittest`` test cases.

    These are the files pytest would skip on name alone. pytest can run a
    unittest suite, but only if it collects the file in the first place.
    """
    found: list[str] = []
    for path in workspace.iter_files(pattern="**/*.py"):
        relative = workspace.relative(path)
        # A module name has to be importable: no dashes, no dotted directories.
        stem = relative[:-3]
        if not all(part.isidentifier() for part in stem.split("/")):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "unittest.TestCase" in text or "unittest.IsolatedAsyncioTestCase" in text:
            found.append(stem.replace("/", "."))
        if len(found) >= MAX_UNITTEST_MODULES:
            break
    return found


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
    """Executes the project's tests in the sandbox and records the real outcome."""

    sandbox: Sandbox | None = None

    def __init__(self, name: str = "tester_agent", **kwargs) -> None:
        super().__init__(
            name=name,
            description=(
                "Runs the project's test suite inside a sandbox and "
                "reports the real pass/fail counts and failure details."
            ),
            **kwargs,
        )

    def _sandbox(self) -> Sandbox:
        return self.sandbox or get_sandbox()

    async def _save_log(self, ctx: InvocationContext, log: str) -> str:
        """Store the full test output as an artifact, not in session state.

        A failing suite's output runs to thousands of lines. Session state is
        copied into every downstream prompt, so the log goes to the artifact
        service and only its name is kept in the report.
        """
        service = getattr(ctx, "artifact_service", None)
        if service is None or not log.strip():
            return ""

        iteration = ctx.session.state.get("iteration", 0)
        filename = f"test_output_iteration_{iteration}.txt"
        try:
            await service.save_artifact(
                app_name=ctx.session.app_name,
                user_id=ctx.session.user_id,
                session_id=ctx.session.id,
                filename=filename,
                artifact=types.Part(text=log),
            )
        except Exception as exc:  # noqa: BLE001 - losing a log must not fail the run
            logger.warning("could not save test output artifact: %s", exc)
            return ""
        return filename

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        command = _resolve_command(state)

        logger.info("running tests: %s", command)
        report, raw_output = self._sandbox().run_tests_with_output(command)
        report.output_artifact = await self._save_log(ctx, raw_output)

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


