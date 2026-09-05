"""Evaluation harness.

Runs SE-Agent against fixture projects and measures the outcome objectively:
each task's success is decided by executing a verification script in the
sandbox, never by reading the agent's own report.

Optionally compares against a single-agent baseline -- one Kimi K3 call with the
same tools and no workflow -- so the multi-agent structure has to earn its cost.

Usage::

    python -m evaluation.run_eval                       # all tasks
    python -m evaluation.run_eval --task add-sentence-count
    python -m evaluation.run_eval --baseline            # also run the baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.agent import RunResult, stream_run
from app.telemetry import Update
from app.tools.sandbox import DockerSandbox
from app.tools.workspace import Workspace

from .tasks import Task, get_tasks

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Everything measured for one task."""

    task_id: str
    system: str  # "se-agent" or "baseline"
    verified: bool = False
    verification_output: str = ""
    tests_passed: bool = False
    review_approved: bool = False
    iterations: int = 0
    tool_calls: int = 0
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    files_changed: list[str] = field(default_factory=list)
    outcome_reported: str = ""
    failure_mode: str = ""
    errors: list[str] = field(default_factory=list)


def prepare_workspace(task: Task, root: Path) -> Path:
    """Copy the fixture into a scratch directory with fresh git history.

    Each run gets a pristine copy: a task must not be able to benefit from, or
    be poisoned by, a previous attempt.
    """
    target = root / task.id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(task.fixture_path, target)

    for args in (
        ["init", "-q"],
        ["config", "user.email", "eval@se-agent.local"],
        ["config", "user.name", "SE-Agent Eval"],
        ["add", "-A"],
        ["commit", "-q", "-m", "fixture baseline"],
    ):
        subprocess.run(["git", *args], cwd=target, capture_output=True, check=False)

    return target


def verify(task: Task, workspace: Path) -> tuple[bool, str]:
    """Run the task's verification script in the sandbox.

    This is the only thing that decides success. The agent's own claims are
    recorded but never trusted.
    """
    script = workspace / "_verify_task.py"
    try:
        script.write_text(task.verification, encoding="utf-8")
        result = DockerSandbox(Workspace(workspace)).run(
            "python _verify_task.py", timeout=120
        )
    except Exception as exc:  # noqa: BLE001 - a broken check is a failed check
        return False, f"verification could not run: {exc}"
    finally:
        script.unlink(missing_ok=True)

    if result.sandbox_error:
        return False, f"sandbox error: {result.sandbox_error}"
    return result.exit_code == 0, result.combined_output.strip()[:1000]


def changed_files(workspace: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        line[3:].strip() for line in result.stdout.splitlines() if len(line) > 3
    ]


def classify_failure(result: TaskResult) -> str:
    """Name the failure mode, so results aggregate into something actionable."""
    if result.verified:
        return ""
    if result.errors:
        return "run_error"
    if not result.files_changed:
        return "no_changes_made"
    if not result.tests_passed:
        return "tests_never_passed"
    if not result.review_approved:
        return "review_never_approved"
    return "passed_own_tests_but_failed_verification"


async def run_se_agent(task: Task, workspace: Path, limit: int) -> TaskResult:
    """Run the full multi-agent workflow against one task."""
    result = TaskResult(task_id=task.id, system="se-agent")
    started = time.monotonic()

    try:
        async for item in stream_run(
            task.request, repo=str(workspace), limit=limit
        ):
            if isinstance(item, RunResult):
                result.tests_passed = item.tests_passed
                result.review_approved = item.review_approved
                result.iterations = int(item.state.get("iteration", 0))
                result.tool_calls = item.tracker.tool_calls
                result.errors = list(item.tracker.errors)
                report = item.report
                if report:
                    result.outcome_reported = report.outcome
            elif isinstance(item, Update) and item.kind == "error":
                logger.warning("[%s] %s", task.id, item.text)
    except Exception as exc:  # noqa: BLE001 - a crashed run is a failed task
        logger.exception("task %s crashed", task.id)
        result.errors.append(f"{type(exc).__name__}: {exc}")

    result.latency_seconds = round(time.monotonic() - started, 1)
    result.files_changed = changed_files(workspace)
    result.verified, result.verification_output = verify(task, workspace)
    result.failure_mode = classify_failure(result)
    return result


async def run_baseline(task: Task, workspace: Path) -> TaskResult:
    """A single Kimi K3 agent with the same tools and no workflow.

    The comparison the multi-agent design has to justify: same model, same
    authority, no requirements stage, no independent review, no forced test run.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from app.agents.common import build_agent
    from app.tools import CODING_TOOLS
    from app.tools.workspace import get_workspace

    get_workspace(str(workspace))

    agent = build_agent(
        name="baseline_agent",
        description="Single-agent baseline.",
        instruction=(
            "You are a software engineer. Implement the user's request in this "
            "repository. Read the relevant files first, make the change, write "
            "tests for it, and run the test suite with run_tests. Never weaken "
            "a test to make it pass. Report what you changed."
        ),
        tools=CODING_TOOLS,
    )

    result = TaskResult(task_id=task.id, system="baseline")
    started = time.monotonic()
    tool_calls = 0

    try:
        runner = InMemoryRunner(agent=agent, app_name="baseline")
        session = await runner.session_service.create_session(
            app_name="baseline", user_id="eval"
        )
        message = types.Content(
            role="user", parts=[types.Part(text=task.request)]
        )
        async for event in runner.run_async(
            user_id="eval", session_id=session.id, new_message=message
        ):
            for part in getattr(event.content, "parts", None) or []:
                if getattr(part, "function_call", None):
                    tool_calls += 1
                    if part.function_call.name == "run_tests":
                        result.tests_passed = True  # provisional; verified below
            if event.error_message:
                result.errors.append(f"{event.error_code}: {event.error_message}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("baseline %s crashed", task.id)
        result.errors.append(f"{type(exc).__name__}: {exc}")

    result.tool_calls = tool_calls
    result.iterations = 1
    result.latency_seconds = round(time.monotonic() - started, 1)
    result.files_changed = changed_files(workspace)
    result.verified, result.verification_output = verify(task, workspace)
    result.failure_mode = classify_failure(result)
    return result


def summarize(results: list[TaskResult]) -> str:
    """A short report a human can read at the end of a long run."""
    lines: list[str] = []
    by_system: dict[str, list[TaskResult]] = {}
    for result in results:
        by_system.setdefault(result.system, []).append(result)

    for system, group in by_system.items():
        solved = sum(1 for r in group if r.verified)
        lines.append(f"\n{system}: {solved}/{len(group)} tasks verified")
        lines.append(
            f"  mean latency   : "
            f"{sum(r.latency_seconds for r in group) / len(group):.0f}s"
        )
        lines.append(
            f"  mean tool calls: "
            f"{sum(r.tool_calls for r in group) / len(group):.1f}"
        )
        lines.append(
            f"  mean iterations: "
            f"{sum(r.iterations for r in group) / len(group):.1f}"
        )

        failures = [r for r in group if not r.verified]
        if failures:
            lines.append("  failures:")
            for failure in failures:
                lines.append(f"    - {failure.task_id}: {failure.failure_mode}")

    honesty = [
        r
        for r in results
        if r.outcome_reported == "completed" and not r.verified
    ]
    if honesty:
        lines.append(
            "\n  WARNING: reported 'completed' but failed verification: "
            + ", ".join(r.task_id for r in honesty)
        )
    else:
        lines.append("\n  No task was reported complete while failing verification.")

    return "\n".join(lines)


async def main_async(args) -> int:
    tasks = get_tasks(args.task)
    results: list[TaskResult] = []
    root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="se_eval_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"workspaces: {root}\n")

    for task in tasks:
        print(f"--- {task.id} ({task.difficulty}) : se-agent ---")
        workspace = prepare_workspace(task, root / "se-agent")
        result = await run_se_agent(task, workspace, args.max_iterations)
        results.append(result)
        print(
            f"    verified={result.verified} tests={result.tests_passed} "
            f"review={result.review_approved} iters={result.iterations} "
            f"tools={result.tool_calls} {result.latency_seconds}s"
        )
        if not result.verified:
            print(f"    {result.verification_output[:300]}")

        if args.baseline:
            print(f"--- {task.id} : baseline ---")
            workspace = prepare_workspace(task, root / "baseline")
            result = await run_baseline(task, workspace)
            results.append(result)
            print(
                f"    verified={result.verified} tools={result.tool_calls} "
                f"{result.latency_seconds}s"
            )

    print(summarize(results))

    if args.output:
        Path(args.output).write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")

    return 0 if all(r.verified for r in results if r.system == "se-agent") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SE-Agent.")
    parser.add_argument("--task", action="append", help="task id (repeatable)")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also run the single-agent baseline",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--workdir", default=None, help="where to create scratch repos")
    parser.add_argument("--output", default=None, help="write results as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
