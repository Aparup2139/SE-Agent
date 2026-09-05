"""The shell-script sandbox backend.

Two things are load-bearing here and get real, unstubbed coverage: that no host
secret reaches the command, and that a sandbox which failed to start is never
reported as a test suite that failed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from app.tools import sandbox as sandbox_module
from app.tools.sandbox import (
    SANDBOX_FAILED_EXIT,
    ExecutionResult,
    Sandbox,
    SandboxConfig,
    _base_env,
)
from app.tools.workspace import Workspace

needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is not on PATH"
)


@pytest.fixture(autouse=True)
def _accept_missing_boundary(monkeypatch):
    """These tests exercise mechanics, not the policy gate."""
    monkeypatch.setenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", "1")


@pytest.fixture
def box(tmp_path):
    return Sandbox(Workspace(tmp_path), session_id="test-session")


@pytest.fixture
def ready_box(box, monkeypatch):
    monkeypatch.setattr(Sandbox, "preflight", lambda self: (True, "ok"))
    return box


def stub_popen(monkeypatch, *, returncode=0, stdout="", stderr=""):
    """Replace Popen with something that reports a fixed result."""

    class FakeProc:
        pid = 4242

        def __init__(self, *a, **kw):
            self.returncode = returncode
            self.args = a[0] if a else kw.get("args")
            seen["args"] = self.args
            seen["env"] = kw.get("env")

        def communicate(self, timeout=None):
            seen["timeout"] = timeout
            return stdout, stderr

    seen: dict = {}
    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    return seen


# ---------------------------------------------------------------- environment


def test_no_host_secret_is_forwarded_to_the_script(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-must-not-leak")
    monkeypatch.setenv("SE_AGENT_API_TOKEN", "tok-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    env = _base_env()

    assert "NVIDIA_API_KEY" not in env
    assert "SE_AGENT_API_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "nvapi-must-not-leak" not in "".join(env.values())


def test_the_workspace_is_passed_to_the_script(ready_box, monkeypatch, tmp_path):
    seen = stub_popen(monkeypatch)
    ready_box.run("pytest")
    assert seen["env"]["SE_AGENT_WORKSPACE"] == str(tmp_path.resolve())


def test_network_is_off_by_default_and_opt_in(ready_box, monkeypatch):
    seen = stub_popen(monkeypatch)
    ready_box.run("pytest")
    assert seen["env"]["SE_AGENT_SANDBOX_NETWORK"] == "0"

    ready_box.run("pip install x", network=True)
    assert seen["env"]["SE_AGENT_SANDBOX_NETWORK"] == "1"


# -------------------------------------------------------------------- command


def test_the_script_receives_session_timeout_and_command(ready_box, monkeypatch):
    seen = stub_popen(monkeypatch)
    ready_box.run("pytest -q", timeout=45)
    args = seen["args"]
    assert args[-3:] == ["test-session", "45", "pytest -q"]
    assert args[1].endswith("local.sh")


def test_an_empty_command_is_refused(box):
    assert "command is required" in box.run("   ").sandbox_error


def test_exit_codes_are_propagated(ready_box, monkeypatch):
    stub_popen(monkeypatch, returncode=3, stdout="nope")
    result = ready_box.run("pytest")
    assert result.exit_code == 3
    assert result.stdout == "nope"
    assert not result.sandbox_error


# ------------------------------------------------------- sandbox vs. command


def test_the_reserved_code_means_the_sandbox_failed_not_the_tests(
    ready_box, monkeypatch
):
    """Exit 97 must never look like a red suite."""
    stub_popen(
        monkeypatch,
        returncode=SANDBOX_FAILED_EXIT,
        stderr="sandbox: SE_AGENT_WORKSPACE is not set",
    )
    result = ready_box.run("pytest")
    assert result.sandbox_error == "sandbox: SE_AGENT_WORKSPACE is not set"
    assert not result.ok


def test_a_failed_sandbox_produces_no_test_failures(ready_box, monkeypatch):
    """The debugger must not be sent after a bug that does not exist."""
    stub_popen(monkeypatch, returncode=SANDBOX_FAILED_EXIT, stderr="bash missing")
    report = ready_box.run_tests("pytest -q")
    assert report.failed == 0
    assert report.passed == 0
    assert report.execution_error == "bash missing"
    assert not report.is_green


def test_a_missing_script_fails_preflight(tmp_path):
    box = Sandbox(Workspace(tmp_path), SandboxConfig(script="no/such.sh"))
    ok, detail = box.preflight()
    assert not ok
    assert "not found" in detail


# ----------------------------------------------------------- the unsafe gate


def test_a_run_refuses_to_start_with_no_boundary(box, monkeypatch):
    """Unisolated execution must be chosen, never stumbled into."""
    monkeypatch.delenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", raising=False)
    monkeypatch.setattr(Sandbox, "tier", lambda self: "none")
    monkeypatch.setattr(Sandbox, "_ask_script", lambda self, flag: "none: clean env")

    ok, detail = box.preflight()
    assert not ok
    assert "no isolation boundary" in detail
    assert "SE_AGENT_SANDBOX_ALLOW_UNSAFE" in detail


def test_the_gate_can_be_opened_deliberately(box, monkeypatch):
    monkeypatch.setenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", "1")
    monkeypatch.setattr(Sandbox, "tier", lambda self: "none")
    monkeypatch.setattr(Sandbox, "_ask_script", lambda self, flag: "none: clean env")

    ok, detail = box.preflight()
    assert ok
    # ...but the result still says so, so no UI can imply containment.
    assert "UNSAFE" in detail


def test_a_real_tier_needs_no_gate(box, monkeypatch):
    monkeypatch.delenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", raising=False)
    monkeypatch.setattr(Sandbox, "tier", lambda self: "unshare")
    monkeypatch.setattr(
        Sandbox, "_ask_script", lambda self, flag: "unshare: network namespace"
    )

    ok, detail = box.preflight()
    assert ok
    assert "UNSAFE" not in detail


def test_the_sandbox_is_process_wide_and_resettable(monkeypatch):
    monkeypatch.setenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", "1")
    sandbox_module.reset_sandbox()
    first = sandbox_module.get_sandbox()
    assert first is sandbox_module.get_sandbox()
    sandbox_module.reset_sandbox()
    assert sandbox_module.get_sandbox() is not first
    sandbox_module.reset_sandbox()


# ------------------------------------------------------------- the real script


@needs_bash
def test_the_script_reports_which_tier_is_in_force(box):
    from app.tools.sandbox import TIERS

    ok, detail = box.preflight()
    assert ok, detail
    # It must name a tier it actually probed, never a hopeful default.
    assert box.tier() in TIERS
    assert detail.startswith(box.tier())


@needs_bash
def test_the_real_script_runs_a_command(box):
    result = box.run("echo sandbox-ok")
    assert result.ok, result.sandbox_error or result.stderr
    assert "sandbox-ok" in result.stdout


@needs_bash
def test_the_real_script_hides_the_nvidia_key(box, monkeypatch):
    """The one guarantee that holds on every platform."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-must-not-leak")
    result = box.run("env")
    assert result.ok, result.sandbox_error
    assert "nvapi-must-not-leak" not in result.combined_output
    assert "NVIDIA_API_KEY" not in result.combined_output


@needs_bash
def test_the_real_script_propagates_a_failing_exit_code(box):
    assert box.run("exit 42").exit_code == 42


@needs_bash
def test_the_real_script_reports_a_missing_workspace_as_a_sandbox_failure(tmp_path):
    box = Sandbox(Workspace(tmp_path))
    box.workspace.root = tmp_path / "deleted"
    result = box.run("echo hi")
    assert result.sandbox_error
    assert result.exit_code == SANDBOX_FAILED_EXIT


@needs_bash
def test_the_command_runs_in_the_workspace(tmp_path):
    (tmp_path / "marker.txt").write_text("here", encoding="utf-8")
    box = Sandbox(Workspace(tmp_path))
    assert "marker.txt" in box.run("ls").stdout


def test_a_sandbox_error_is_never_ok():
    """`ok` is what the tester and the router read; a zero exit is not enough."""
    assert ExecutionResult(exit_code=0).ok
    assert not ExecutionResult(exit_code=0, sandbox_error="boom").ok


# ------------------------------------------------------- test-result honesty


def test_no_tests_collected_is_an_error_not_a_pass(ready_box, monkeypatch):
    """pytest exits 5 when it collects nothing; that must not read as success."""
    stub_popen(monkeypatch, returncode=5, stdout="no tests ran in 0.01s")
    report = ready_box.run_tests("pytest -q")
    assert report.execution_error == "no tests were collected"
    assert not report.is_green


def test_collection_crash_is_reported_as_an_execution_error(ready_box, monkeypatch):
    stub_popen(monkeypatch, returncode=2, stderr="ImportError while loading conftest")
    report = ready_box.run_tests("pytest -q")
    assert "failed before any test ran" in report.execution_error
    assert not report.is_green


def test_counts_and_exit_code_come_from_the_process(ready_box, monkeypatch):
    stub_popen(
        monkeypatch,
        returncode=1,
        stdout=(
            "FAILED tests/test_x.py::test_y - assert 1 == 2\n"
            "===== 1 failed, 4 passed in 2.0s ====="
        ),
    )
    report = ready_box.run_tests("pytest -q")
    assert (report.passed, report.failed, report.exit_code) == (4, 1, 1)
    assert report.command == "pytest -q"
    assert not report.is_green


def test_a_green_run_is_green(ready_box, monkeypatch):
    stub_popen(monkeypatch, returncode=0, stdout="===== 6 passed in 0.4s =====")
    assert ready_box.run_tests("pytest -q").is_green


def test_timeout_is_reported_as_a_sandbox_error(ready_box, monkeypatch):
    """A hung suite is an infrastructure outcome, not a verdict on the code."""

    class Hanging:
        pid = 9999
        returncode = None

        def __init__(self, *a, **kw):
            pass

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="bash", timeout=timeout)
            return "", ""

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", Hanging)
    monkeypatch.setattr(sandbox_module, "_terminate", lambda proc: None)

    result = ready_box.run("sleep 999", timeout=1)
    assert result.timed_out
    assert result.exit_code == 124
    assert "time limit" in result.sandbox_error

    report = ready_box.run_tests("pytest -q", timeout=1)
    assert not report.is_green
    assert not report.ran, "a timeout must never look like a test result"
