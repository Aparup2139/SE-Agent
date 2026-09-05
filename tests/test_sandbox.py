"""Sandbox isolation flags, pytest output parsing, and honest failure reporting."""

from __future__ import annotations

import subprocess

import pytest

from app.tools import sandbox as sandbox_module
from app.tools.sandbox import (
    CONTAINER_WORKDIR,
    DockerSandbox,
    SandboxConfig,
    parse_pytest_output,
)
from app.tools.workspace import Workspace


@pytest.fixture
def box(tmp_path):
    return DockerSandbox(Workspace(tmp_path), SandboxConfig())


@pytest.fixture
def ready_box(box, monkeypatch):
    """A sandbox whose preflight always succeeds, so run() can be exercised."""
    monkeypatch.setattr(DockerSandbox, "preflight", lambda self: (True, "ok"))
    return box


# ------------------------------------------------------------------ isolation


def test_workspace_is_mounted_at_the_container_workdir(box, tmp_path):
    args = box._docker_args("pytest")
    assert "--workdir" in args
    assert args[args.index("--workdir") + 1] == CONTAINER_WORKDIR
    mount = args[args.index("--volume") + 1]
    assert mount == f"{tmp_path.resolve()}:{CONTAINER_WORKDIR}"


def test_network_is_disabled_by_default(box):
    args = box._docker_args("pytest")
    assert args[args.index("--network") + 1] == "none"


def test_network_can_be_enabled_explicitly(box):
    assert "--network" not in box._docker_args("pip install x", network=True)


def test_container_drops_capabilities_and_privileges(box):
    args = box._docker_args("pytest")
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    assert "--read-only" in args
    assert "--rm" in args


def test_resource_limits_are_applied(box):
    args = box._docker_args("pytest")
    assert args[args.index("--memory") + 1] == "2g"
    assert args[args.index("--cpus") + 1] == "2"
    assert args[args.index("--pids-limit") + 1] == "512"


def test_writable_tmpfs_is_provided(box):
    """A read-only root still needs scratch space for pytest and pip."""
    args = box._docker_args("pytest")
    assert any(a.startswith("/tmp:rw") for a in args)


# ------------------------------------------------------- credential isolation


def test_nvidia_key_never_reaches_the_container(box, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-must-not-leak")
    rendered = " ".join(box._docker_args("printenv"))
    assert "nvapi-must-not-leak" not in rendered
    assert "aws-must-not-leak" not in rendered
    assert "NVIDIA_API_KEY" not in rendered


def test_environment_is_an_allowlist_not_an_inheritance(box):
    args = box._docker_args("printenv")
    passed = {
        args[i + 1].split("=", 1)[0] for i, a in enumerate(args) if a == "--env"
    }
    assert passed == set(box.config.env_allowlist)


# -------------------------------------------------------------- preflight


def test_missing_docker_is_reported_not_raised(box, monkeypatch):
    monkeypatch.setattr(
        sandbox_module, "docker_available", lambda: (False, "docker daemon unreachable")
    )
    result = box.run("pytest")
    assert result.sandbox_error == "docker daemon unreachable"
    assert not result.ok


def test_sandbox_failure_is_distinguished_from_a_test_failure(box, monkeypatch):
    monkeypatch.setattr(
        sandbox_module, "docker_available", lambda: (False, "no docker")
    )
    report = box.run_tests("pytest -q")
    assert report.execution_error
    assert not report.is_green
    assert not report.ran, "a sandbox failure must never look like a test result"


def test_empty_command_is_rejected(box):
    assert "command is required" in box.run("   ").sandbox_error


def test_timeout_is_reported_as_a_sandbox_error(ready_box, monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = ready_box.run("sleep 999", timeout=1)
    assert result.timed_out
    assert result.exit_code == 124
    assert "time limit" in result.sandbox_error


# ------------------------------------------------------- pytest output parsing


def test_all_passing_summary_is_parsed():
    report = parse_pytest_output("===== 12 passed in 1.23s =====")
    report.exit_code = 0
    assert report.passed == 12
    assert report.test_count == 12
    assert report.is_green


def test_mixed_summary_is_parsed():
    output = "===== 3 failed, 12 passed, 1 skipped in 4.10s ====="
    report = parse_pytest_output(output)
    assert (report.passed, report.failed, report.skipped) == (12, 3, 1)
    assert report.test_count == 16
    assert not report.is_green


def test_errors_are_counted():
    report = parse_pytest_output("===== 2 errors, 1 passed in 0.5s =====")
    assert report.errors == 2
    assert not report.is_green


def test_failure_lines_are_summarized():
    output = (
        "FAILED tests/test_api.py::test_create - AssertionError: expected 201\n"
        "FAILED tests/test_api.py::test_delete - KeyError: 'id'\n"
        "===== 2 failed, 5 passed in 1.0s ====="
    )
    report = parse_pytest_output(output)
    assert len(report.failure_summary) == 2
    assert "test_create" in report.failure_summary[0]
    assert "expected 201" in report.failure_summary[0]


def test_error_lines_are_summarized():
    output = (
        "ERROR tests/test_db.py - ImportError: no module named psycopg\n"
        "===== 1 error in 0.2s ====="
    )
    report = parse_pytest_output(output)
    assert report.failure_summary[0].startswith("ERROR")


def test_only_the_final_summary_banner_is_trusted():
    """A failing test's own output can contain summary-looking text."""
    output = (
        "captured stdout: the script printed '99 passed in 0.1s' from a fixture\n"
        "===== 1 failed, 2 passed in 3.0s ====="
    )
    report = parse_pytest_output(output)
    assert report.passed == 2
    assert report.failed == 1


def test_empty_output_yields_an_empty_report():
    report = parse_pytest_output("")
    assert report.test_count == 0
    assert not report.is_green


def test_no_tests_collected_is_an_error_not_a_pass(ready_box, monkeypatch):
    """pytest exits 5 when it collects nothing; that must not read as success."""

    class Completed:
        returncode = 5
        stdout = "no tests ran in 0.01s"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Completed())
    report = ready_box.run_tests("pytest -q")
    assert report.execution_error == "no tests were collected"
    assert not report.is_green


def test_collection_crash_is_reported_as_an_execution_error(ready_box, monkeypatch):
    class Completed:
        returncode = 2
        stdout = ""
        stderr = "ImportError while loading conftest"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Completed())
    report = ready_box.run_tests("pytest -q")
    assert "failed before any test ran" in report.execution_error
    assert not report.is_green


def test_real_run_populates_counts_and_exit_code(ready_box, monkeypatch):
    class Completed:
        returncode = 1
        stdout = (
            "FAILED tests/test_x.py::test_y - assert 1 == 2\n"
            "===== 1 failed, 4 passed in 2.0s ====="
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Completed())
    report = ready_box.run_tests("pytest -q")
    assert report.passed == 4
    assert report.failed == 1
    assert report.exit_code == 1
    assert report.command == "pytest -q"
    assert not report.is_green


# ------------------------------------------------------------------ tool layer


def test_run_tests_tool_reports_failure_honestly(monkeypatch):
    from app.schemas import TestReport

    monkeypatch.setattr(
        sandbox_module,
        "get_sandbox",
        lambda: type(
            "S",
            (),
            {
                "run_tests": lambda self, c, timeout=None: TestReport(
                    failed=2, passed=1, test_count=3, exit_code=1, command=c
                )
            },
        )(),
    )
    result = sandbox_module.run_tests("pytest -q")
    assert result["ok"] is False
    assert result["failed"] == 2


def test_run_command_timeout_is_bounded(monkeypatch):
    seen = {}

    class Box:
        def run(self, command, timeout=None):
            seen["timeout"] = timeout
            from app.tools.sandbox import ExecutionResult

            return ExecutionResult(exit_code=0, command=command)

    monkeypatch.setattr(sandbox_module, "get_sandbox", lambda: Box())
    sandbox_module.run_command("echo hi", timeout=99999)
    assert seen["timeout"] == 900


# ------------------------------------------------------------- live docker


def _skip_without_sandbox_image():
    available, detail = sandbox_module.docker_available()
    if not available:
        pytest.skip(detail)
    if not sandbox_module.image_exists(sandbox_module.DEFAULT_IMAGE):
        pytest.skip(
            f"{sandbox_module.DEFAULT_IMAGE} not built "
            "(docker build -t se-agent-sandbox:latest sandbox/)"
        )


@pytest.mark.docker
def test_live_sandbox_runs_a_command(tmp_path):
    _skip_without_sandbox_image()

    box = DockerSandbox(Workspace(tmp_path), SandboxConfig())
    result = box.run("python -c \"print('sandbox-ok')\"")
    assert result.ok, result.sandbox_error or result.stderr
    assert "sandbox-ok" in result.stdout


@pytest.mark.docker
def test_live_sandbox_hides_the_nvidia_key(tmp_path, monkeypatch):
    available, detail = sandbox_module.docker_available()
    if not available:
        pytest.skip(detail)

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-must-not-leak")
    box = DockerSandbox(Workspace(tmp_path), SandboxConfig())
    result = box.run("printenv")
    assert "nvapi-must-not-leak" not in result.combined_output
