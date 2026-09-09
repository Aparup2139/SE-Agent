"""Preflight: everything a run depends on, checked before the run starts."""

from __future__ import annotations

from app import agent as agent_module
from app.agent import RunResult
from app.schemas import RunMode
from app.telemetry import RunTracker
from app.tools import sandbox as sandbox_module
from app.tools.sandbox import ExecutionResult

# ------------------------------------------- the test runner must exist first


def test_preflight_reports_a_sandbox_without_pytest(tmp_path, monkeypatch):
    """Catch it in seconds, not eight minutes into a run.

    Observed: a console session reached the test phase after 10 minutes, found
    no pytest, and the coder then spent 100+ model calls trying to pip install
    into a sandbox with no network.
    """
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.setattr(agent_module, "sandbox_available", lambda: (True, "unshare"))

    class Failing:
        def run(self, command, timeout=None):
            return ExecutionResult(exit_code=127, stderr="sh: 1: pytest: not found")

    monkeypatch.setattr(sandbox_module, "get_sandbox", lambda: Failing())

    problems = agent_module._test_runner_problems(str(tmp_path))

    assert len(problems) == 1
    assert "cannot run pytest" in problems[0]
    assert "SE_AGENT_SANDBOX_PATH" in problems[0]  # names the fix
    assert "pytest: not found" in problems[0]  # and the evidence


def test_preflight_is_quiet_when_the_runner_works(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    class Working:
        def run(self, command, timeout=None):
            return ExecutionResult(exit_code=0, stdout="pytest 9.1.1")

    monkeypatch.setattr(sandbox_module, "get_sandbox", lambda: Working())
    assert agent_module._test_runner_problems(str(tmp_path)) == []


def test_a_non_python_project_is_not_probed(tmp_path, monkeypatch):
    """Only Python has a runner knowable before the repo is analysed."""

    def explode():  # pragma: no cover - must never be reached
        raise AssertionError("the sandbox was probed for a non-Python project")

    monkeypatch.setattr(sandbox_module, "get_sandbox", explode)
    assert agent_module._test_runner_problems(str(tmp_path)) == []


def test_a_broken_sandbox_probe_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    class Exploding:
        def run(self, command, timeout=None):
            raise OSError("wsl.exe is gone")

    monkeypatch.setattr(sandbox_module, "get_sandbox", lambda: Exploding())
    problems = agent_module._test_runner_problems(str(tmp_path))
    assert len(problems) == 1
    assert "wsl.exe is gone" in problems[0]


def test_a_verified_no_op_succeeds_in_both_modes():
    """A request already satisfied is answered, not failed."""
    for mode in (RunMode.NORMAL, RunMode.THINKING):
        result = RunResult(
            "s",
            {
                "mode": mode.value,
                "tests_passed": True,
                "files_changed": [],
                "no_change_needed": True,
            },
            RunTracker(mode),
        )
        assert result.succeeded, mode


def test_a_no_op_without_that_flag_still_fails():
    """Only the gate sets it, and only when the plan pre-committed."""
    result = RunResult(
        "s",
        {"mode": "normal", "tests_passed": True, "files_changed": []},
        RunTracker(RunMode.NORMAL),
    )
    assert not result.succeeded


def test_the_measured_file_list_replaces_the_reported_one():
    """A reporter has credited a run with a file it never touched."""
    result = RunResult(
        "s",
        {
            "files_changed": ["actually_changed.py"],
            "report": {
                "feature_request": "x",
                "outcome": "completed",
                "summary": "s",
                "files_changed": ["invented.py", "also_invented.py"],
            },
        },
        RunTracker(),
    )
    assert result.report.files_changed == ["actually_changed.py"]


def test_an_unmeasured_run_keeps_the_reported_list():
    """With no measurement there is nothing better to substitute."""
    result = RunResult(
        "s",
        {
            "report": {
                "feature_request": "x",
                "outcome": "completed",
                "summary": "s",
                "files_changed": ["claimed.py"],
            }
        },
        RunTracker(),
    )
    assert result.report.files_changed == ["claimed.py"]
