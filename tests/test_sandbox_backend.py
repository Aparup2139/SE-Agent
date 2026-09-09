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
from app.tools.workspace import Workspace, get_workspace

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")


@pytest.fixture(autouse=True)
def _accept_missing_boundary(monkeypatch):
    """These tests exercise mechanics, not the policy gate.

    ``SE_AGENT_SANDBOX_WSL`` is cleared too: ``SandboxConfig`` reads it from
    the environment, so a developer with it set in ``.env`` would otherwise
    run these against WSL, where the argv and the workspace path are
    deliberately different. The WSL form is asserted on its own below, with
    the distro passed explicitly.
    """
    monkeypatch.setenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", "1")
    monkeypatch.delenv("SE_AGENT_SANDBOX_WSL", raising=False)


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

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def kill(self):
            pass

        def poll(self):
            return self.returncode

        def communicate(self, input=None, timeout=None):
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


def test_the_reserved_code_means_the_sandbox_failed_not_the_tests(ready_box, monkeypatch):
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
    monkeypatch.setattr(Sandbox, "_ask_script", lambda self, flag: "unshare: network namespace")

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
            "FAILED tests/test_x.py::test_y - assert 1 == 2\n===== 1 failed, 4 passed in 2.0s ====="
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


# ---------------------------------------------------------------- finding bash


def test_the_wsl_shim_is_not_mistaken_for_a_shell(monkeypatch):
    """`bash` on PATH in PowerShell is C:/Windows/System32/bash.exe -- the WSL
    launcher. Using it silently changes which OS the command runs on, and fails
    confusingly when handed a Windows path."""
    monkeypatch.setattr(
        sandbox_module.shutil, "which", lambda name: r"C:\Windows\System32\bash.exe"
    )
    monkeypatch.setattr(sandbox_module.Path, "is_file", lambda self: False)

    with pytest.raises(sandbox_module.SandboxError) as exc:
        sandbox_module.find_bash()
    assert "WSL launcher" in str(exc.value)


def test_a_real_bash_on_path_is_used(monkeypatch, tmp_path):
    shell = tmp_path / "bash.exe"
    shell.write_text("", encoding="utf-8")
    monkeypatch.setattr(sandbox_module.shutil, "which", lambda name: str(shell))
    assert sandbox_module.find_bash() == str(shell)


# ------------------------------------------- who may grant unguarded execution


def test_the_unsafe_flag_is_honored_from_the_environment(monkeypatch):
    monkeypatch.setenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", "1")
    monkeypatch.setattr(sandbox_module, "_came_from_dotenv", lambda name: False)
    assert sandbox_module.allow_unsafe() is True


def test_the_unsafe_flag_is_ignored_when_it_comes_from_dotenv(monkeypatch):
    """A run must not be able to disarm the guard for the next run.

    Observed: an agent wrote SE_AGENT_SANDBOX_ALLOW_UNSAFE=1 into .env while
    trying to make its own tests pass, and in doing so overwrote the API key
    in that file and the project virtualenv.
    """
    monkeypatch.setenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", "1")
    monkeypatch.setattr(sandbox_module, "_came_from_dotenv", lambda name: True)
    assert sandbox_module.allow_unsafe() is False


def test_dotenv_detection_reads_the_real_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SE_AGENT_SANDBOX_ALLOW_UNSAFE=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert sandbox_module._came_from_dotenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE") is True
    assert sandbox_module._came_from_dotenv("SOMETHING_ELSE") is False


# ------------------------------------------------------------ WSL path mapping


def test_a_drive_path_is_mapped_without_launching_wsl(tmp_path, monkeypatch):
    """Asking wslpath meant one WSL launch per path, and it failed under load.

    Observed: a non-zero exit with an empty stderr, surfacing as
    "could not translate ... for WSL:" and ending the run.
    """
    box = Sandbox(Workspace(tmp_path), SandboxConfig(wsl_distro="Ubuntu-24.04"))

    def explode(*a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError("wslpath was launched for a drive path")

    monkeypatch.setattr(subprocess, "run", explode)
    assert box._to_wsl_path(r"C:\Users\me\project") == "/mnt/c/Users/me/project"
    assert box._to_wsl_path(r"D:\Qoder IDE\bin") == "/mnt/d/Qoder IDE/bin"


def test_a_non_drive_path_still_asks_wslpath(tmp_path, monkeypatch):
    box = Sandbox(Workspace(tmp_path), SandboxConfig(wsl_distro="Ubuntu-24.04"))
    asked = {}

    class Result:
        returncode = 0
        stdout = "/mnt/wsl/share\n"
        stderr = ""

    def fake_run(args, **kw):
        asked["args"] = args
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert box._to_wsl_path("//server/share") == "/mnt/wsl/share"
    assert "wslpath" in asked["args"]


def test_the_interpreter_override_crosses_into_wsl(tmp_path, monkeypatch):
    """SE_AGENT_SANDBOX_PATH is the only way to name a usable interpreter.

    On Windows the workspace's own .venv is a Windows venv that Linux cannot
    run, so the script needs this to find pytest. Nothing carries a variable
    across `wsl.exe` implicitly, and omitting it here left the setting
    silently inert: the script fell back to whatever .venv/bin existed and
    reported "pytest: not found".
    """
    monkeypatch.setenv("SE_AGENT_SANDBOX_PATH", "/home/me/.venv/bin")
    monkeypatch.setenv("SE_AGENT_SANDBOX_REQUIRE", "unshare")
    box = Sandbox(Workspace(tmp_path), SandboxConfig(wsl_distro="Ubuntu-24.04"))
    monkeypatch.setattr(type(box), "_to_wsl_path", lambda self, p: "/mnt/c/ws")

    args = box._args("pytest -q", 60)

    assert "SE_AGENT_SANDBOX_PATH=/home/me/.venv/bin" in args
    assert "SE_AGENT_SANDBOX_REQUIRE=unshare" in args
    # ...and they must precede the shell, or `env` would not apply them.
    assert args.index("SE_AGENT_SANDBOX_PATH=/home/me/.venv/bin") < args.index("bash")


def test_an_unset_override_is_not_forwarded_as_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("SE_AGENT_SANDBOX_PATH", raising=False)
    monkeypatch.delenv("SE_AGENT_SANDBOX_REQUIRE", raising=False)
    box = Sandbox(Workspace(tmp_path), SandboxConfig(wsl_distro="Ubuntu-24.04"))
    monkeypatch.setattr(type(box), "_to_wsl_path", lambda self, p: "/mnt/c/ws")

    args = box._args("pytest -q", 60)

    assert not any(a.startswith("SE_AGENT_SANDBOX_PATH") for a in args)
    assert not any(a.startswith("SE_AGENT_SANDBOX_REQUIRE") for a in args)


def test_a_new_workspace_gets_a_new_sandbox(tmp_path, monkeypatch):
    """Otherwise the second run in a server process runs in the first run's
    directory: its tools read one repository while its shell is in another.

    That is not a subtle failure -- the test suite it runs is a different
    project's, so a correct change "fails" and the debugger is sent to explain
    why.
    """
    monkeypatch.setenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE", "1")
    first_repo = tmp_path / "first"
    second_repo = tmp_path / "second"
    first_repo.mkdir()
    second_repo.mkdir()

    sandbox_module.reset_sandbox()
    get_workspace(str(first_repo))
    first = sandbox_module.get_sandbox()
    assert first.workspace.root == first_repo.resolve()

    get_workspace(str(second_repo))
    second = sandbox_module.get_sandbox()
    assert second is not first
    assert second.workspace.root == second_repo.resolve()

    # Same workspace, same sandbox: this must not rebuild on every call.
    assert sandbox_module.get_sandbox() is second
    sandbox_module.reset_sandbox()
