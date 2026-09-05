"""SE-Agent entry point.

Wires the workflow graph to an ADK ``Runner`` with session, artifact and memory
services, and streams the run as human-readable progress.

Usage::

    python -m app.agent "add a /health endpoint that returns build info"
    python -m app.agent --repo ../my-project --max-iterations 3 "fix the flaky retry test"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .llm.adk_model import KimiK3Llm
from .llm.config import MissingAPIKeyError, get_config
from .schemas import FinalReport
from .telemetry import RunTracker, Update
from .tools.sandbox import sandbox_available
from .tools.workspace import get_workspace
from .workflows.graph import build_workflow, initial_state, max_iterations

logger = logging.getLogger(__name__)

APP_NAME = "se_agent"


@dataclass
class RunResult:
    """What a completed run produced."""

    session_id: str
    state: dict[str, Any]
    tracker: RunTracker

    @property
    def report(self) -> FinalReport | None:
        raw = self.state.get("report")
        if not raw:
            return None
        try:
            return FinalReport.model_validate(raw)
        except Exception:  # noqa: BLE001 - a malformed report is not a crash
            return None

    @property
    def tests_passed(self) -> bool:
        return bool(self.state.get("tests_passed"))

    @property
    def review_approved(self) -> bool:
        return bool(self.state.get("review_approved"))

    @property
    def succeeded(self) -> bool:
        """Success means both gates actually cleared, not just that it finished."""
        return self.tests_passed and self.review_approved


def preflight(repo: str | None = None) -> list[str]:
    """Check everything a run depends on, and report all problems at once.

    Returning a list rather than raising on the first failure means a user
    fixes one round of setup instead of three.
    """
    problems: list[str] = []

    try:
        get_config()
    except MissingAPIKeyError as exc:
        problems.append(str(exc))

    try:
        workspace = get_workspace(repo)
        logger.info("workspace: %s", workspace.root)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as text
        problems.append(f"workspace unusable: {exc}")

    # Whichever backend is selected must be able to run, or the tester cannot
    # produce a real result -- and a run that cannot test is a run that cannot
    # honestly report success.
    available, detail = sandbox_available()
    if not available:
        problems.append(
            f"the test sandbox is not usable: {detail}. Start Docker Desktop and "
            "build the image (docker build -t se-agent-sandbox:latest sandbox/), "
            "or set SE_AGENT_SANDBOX=script to use sandbox/local.sh instead."
        )

    return problems


def build_app() -> App:
    """Wrap the workflow in an ADK ``App`` with context compaction enabled.

    A full run is long: eight stages, up to five implementation iterations, and
    a tool call for every file read. Without compaction the event history grows
    until it crowds out the prompt.

    The summarizer is given an explicit :class:`KimiK3Llm` so compaction runs on
    the same model as everything else -- left to default, this would be the one
    place another model could quietly enter the system.
    """
    return App(
        name=APP_NAME,
        root_agent=build_workflow(),
        events_compaction_config=EventsCompactionConfig(
            summarizer=LlmEventSummarizer(llm=KimiK3Llm()),
            # Compact once a run has accumulated a few invocations' worth of
            # history, keeping an overlap so continuity is not lost at the seam.
            compaction_interval=3,
            overlap_size=1,
        ),
    )


def build_runner() -> Runner:
    """Assemble the runner with the services the workflow relies on.

    * Session service holds the current run's structured state.
    * Artifact service holds large outputs (diffs, logs) that must not be
      inlined into state.
    * Memory service holds knowledge worth carrying across runs.

    These are the in-memory implementations, which is right for a single
    process. Swapping in the database-backed session service (``google-adk[db]``)
    is a constructor change, not a code change.
    """
    return Runner(
        app=build_app(),
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService(),
    )


async def stream_run(
    feature_request: str,
    *,
    repo: str | None = None,
    limit: int | None = None,
    user_id: str = "local",
    runner: Runner | None = None,
) -> AsyncIterator[Update | RunResult]:
    """Run one feature request, yielding progress updates then the result."""
    if repo:
        get_workspace(repo)

    runner = runner or build_runner()
    tracker = RunTracker()
    session_id = f"run-{uuid.uuid4().hex[:12]}"

    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state(feature_request, limit),
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=f"Feature request:\n\n{feature_request}")],
    )

    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=message
    ):
        for update in tracker.observe(event):
            yield update

    tracker.finish()

    refreshed = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session.id
    )
    state = dict(refreshed.state) if refreshed else {}

    # Keep what this run learned available to later runs.
    try:
        if refreshed is not None:
            await runner.memory_service.add_session_to_memory(refreshed)
    except Exception as exc:  # noqa: BLE001 - memory is a nicety, not the run
        logger.warning("could not add this session to memory: %s", exc)

    yield RunResult(session_id=session.id, state=state, tracker=tracker)


async def run(
    feature_request: str,
    *,
    repo: str | None = None,
    limit: int | None = None,
) -> RunResult:
    """Run to completion and return only the result."""
    result: RunResult | None = None
    async for item in stream_run(feature_request, repo=repo, limit=limit):
        if isinstance(item, RunResult):
            result = item
    assert result is not None, "the workflow always yields a result"
    return result


# ----------------------------------------------------------------------- CLI


def _print(update: Update) -> None:
    """One line per update, prefixed so a long run stays scannable."""
    if update.kind == "phase":
        print(f"\n=== {update.text} ===")
    elif update.kind == "tool":
        print(f"  . {update.text}")
    elif update.kind == "tests":
        print(f"  TESTS  {update.text}")
    elif update.kind == "review":
        print(f"  REVIEW {update.text}")
    elif update.kind == "error":
        print(f"  ERROR  {update.text}", file=sys.stderr)
    else:
        text = update.text
        if len(text) > 500:
            text = text[:500] + " [...]"
        prefix = f"  [{update.agent}] " if update.agent else "  "
        print(prefix + text.replace("\n", "\n    "))


def _print_report(result: RunResult) -> None:
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    for line in result.tracker.checklist():
        print(f"  {line}")

    print(f"\n  iterations : {result.state.get('iteration', 0)}")
    print(f"  tool calls : {result.tracker.tool_calls}")
    print(f"  tests      : {'passed' if result.tests_passed else 'NOT passing'}")
    print(f"  review     : {'approved' if result.review_approved else 'not approved'}")

    report = result.report
    if report:
        print(f"\n  outcome    : {report.outcome}")
        print(f"\n{report.summary}")
        if report.files_changed:
            print("\n  files changed:")
            for path in report.files_changed:
                print(f"    - {path}")
        if report.requirements_unmet:
            print("\n  requirements NOT met:")
            for item in report.requirements_unmet:
                print(f"    - {item}")
        if report.known_limitations:
            print("\n  known limitations:")
            for item in report.known_limitations:
                print(f"    - {item}")
        if report.follow_up:
            print("\n  follow up:")
            for item in report.follow_up:
                print(f"    - {item}")
    else:
        print("\n  No final report was produced.")

    if result.tracker.errors:
        print("\n  errors:")
        for error in result.tracker.errors:
            print(f"    - {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="se-agent",
        description="Run a feature request through the SE-Agent workflow.",
    )
    parser.add_argument("request", help="the feature request, in plain language")
    parser.add_argument(
        "--repo", default=None, help="repository to work in (default: current directory)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=f"implementation attempts before stopping (default: {max_iterations()})",
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    problems = preflight(args.repo)
    if problems:
        print("Cannot start:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}\n", file=sys.stderr)
        return 1

    async def _go() -> RunResult:
        result: RunResult | None = None
        async for item in stream_run(
            args.request, repo=args.repo, limit=args.max_iterations
        ):
            if isinstance(item, RunResult):
                result = item
            else:
                _print(item)
        assert result is not None
        return result

    try:
        result = asyncio.run(_go())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    _print_report(result)
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
