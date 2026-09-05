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
    DockerSandbox,
    ExecutionResult,
    ScriptConfig,
    ScriptSandbox,
    _base_env,
)
from app.tools.workspace import Workspace

needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is not on PATH"
)


@pytest.fixture
def box(tmp_path):
    return ScriptSandbox(Workspace(tmp_path), session_id="test-session")


@pytest.fixture
def ready_box(box, monkeypatch):
    monkeypatch.setattr(ScriptSandbox, "preflight", lambda self: (True, "ok"))
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
    box = ScriptSandbox(Workspace(tmp_path), ScriptConfig(script="no/such.sh"))
    ok, detail = box.preflight()
    assert not ok
    assert "not found" in detail


# ------------------------------------------------------------------ selection


def test_the_backend_is_chosen_by_environment(monkeypatch):
    monkeypatch.setenv("SE_AGENT_SANDBOX", "script")
    sandbox_module.reset_sandbox()
    assert isinstance(sandbox_module.get_sandbox(), ScriptSandbox)

    monkeypatch.setenv("SE_AGENT_SANDBOX", "docker")
    sandbox_module.reset_sandbox()
    assert isinstance(sandbox_module.get_sandbox(), DockerSandbox)
    sandbox_module.reset_sandbox()


def test_docker_remains_the_default(monkeypatch):
    monkeypatch.delenv("SE_AGENT_SANDBOX", raising=False)
    sandbox_module.reset_sandbox()
    assert isinstance(sandbox_module.get_sandbox(), DockerSandbox)
    sandbox_module.reset_sandbox()


# ------------------------------------------------------------- the real script


@needs_bash
def test_the_script_reports_what_it_isolates(box):
    ok, detail = box.preflight()
    assert ok, detail
    # Whatever the platform offers, the environment allowlist is unconditional.
    assert "clean-env" in detail


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
    box = ScriptSandbox(Workspace(tmp_path))
    box.workspace.root = tmp_path / "deleted"
    result = box.run("echo hi")
    assert result.sandbox_error
    assert result.exit_code == SANDBOX_FAILED_EXIT


@needs_bash
def test_the_command_runs_in_the_workspace(tmp_path):
    (tmp_path / "marker.txt").write_text("here", encoding="utf-8")
    box = ScriptSandbox(Workspace(tmp_path))
    assert "marker.txt" in box.run("ls").stdout


def test_execution_result_shape_is_shared_with_docker():
    """Both backends feed the same downstream code."""
    assert ExecutionResult(exit_code=0).ok
    assert not ExecutionResult(exit_code=0, sandbox_error="boom").ok
