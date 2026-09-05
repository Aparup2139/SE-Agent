"""Isolated execution for repository code.

Nothing the agents write should be executed on the host unguarded. Two backends
provide that, chosen with ``SE_AGENT_SANDBOX``:

``docker`` (default)
    A short-lived container with the repository bind-mounted and a deliberately
    hostile environment: no host credentials, no network, a read-only root
    filesystem apart from the workspace, dropped capabilities, and hard limits
    on CPU, memory, processes and wall-clock time. A real security boundary.

``script``
    Delegates to ``sandbox/local.sh`` (override with ``SE_AGENT_SANDBOX_CMD``),
    which applies whatever the host OS actually offers -- a network namespace
    and process ceiling on Linux and WSL, only a clean environment and a
    working-directory change on Windows. Read that script before relying on it:
    on Windows it is **not** a security boundary.

What both guarantee is the environment allowlist. The child is built from an
explicit set of variables rather than inheriting yours, so ``NVIDIA_API_KEY``
and every other host secret stay out of repository code.

Both share one test-output parser and one :class:`TestReport` path, so no
backend can report a pass the other would not.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import TestReport
from .workspace import MAX_OUTPUT_CHARS, Workspace, clamp, get_workspace

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "se-agent-sandbox:latest"
FALLBACK_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT = 600
CONTAINER_WORKDIR = "/workspace"

# Resource ceilings. A runaway test suite should die, not take the host with it.
MEMORY_LIMIT = "2g"
CPU_LIMIT = "2"
PID_LIMIT = "512"


# The script backend's reserved code for "the sandbox failed", as distinct from
# "the command failed". Conflating the two would let a broken sandbox read as a
# red test suite and send the debugger after a bug that does not exist.
SANDBOX_FAILED_EXIT = 97

DEFAULT_SCRIPT = Path(__file__).resolve().parents[2] / "sandbox" / "local.sh"


class SandboxError(Exception):
    """The sandbox itself failed -- distinct from the command failing."""


@dataclass
class ExecutionResult:
    """Outcome of one sandboxed command."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    command: str = ""
    sandbox_error: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.sandbox_error

    @property
    def combined_output(self) -> str:
        parts = [p for p in (self.stdout, self.stderr) if p.strip()]
        return "\n".join(parts)


@dataclass
class SandboxConfig:
    image: str = DEFAULT_IMAGE
    timeout: int = DEFAULT_TIMEOUT
    network: bool = False
    memory: str = MEMORY_LIMIT
    cpus: str = CPU_LIMIT
    # Only these variables cross into the container. Host secrets never do.
    env_allowlist: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CI": "1",
            "HOME": "/tmp",
        }
    )


def docker_available() -> tuple[bool, str]:
    """Check that a Docker daemon is actually reachable, not just installed."""
    if shutil.which("docker") is None:
        return False, "docker is not installed or not on PATH"
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return False, "docker daemon did not respond within 20s"
    except OSError as exc:
        return False, f"could not invoke docker: {exc}"

    if result.returncode != 0:
        return False, f"docker daemon unreachable: {(result.stderr or '').strip()[:200]}"
    return True, f"docker server {result.stdout.strip()}"


def image_exists(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class _BaseSandbox:
    """What every backend must provide, and what they all share.

    Backends differ only in :meth:`run` and :meth:`preflight`. Everything that
    turns a process result into a :class:`TestReport` lives here, once, so a
    green suite means the same thing regardless of where it ran.
    """

    def preflight(self) -> tuple[bool, str]:
        raise NotImplementedError

    def run(
        self,
        command: str,
        timeout: int | None = None,
        network: bool | None = None,
    ) -> ExecutionResult:
        raise NotImplementedError

    def run_tests(self, command: str = "pytest -q", timeout: int | None = None) -> TestReport:
        """Run the project's test suite and parse the real result."""
        return self.run_tests_with_output(command, timeout=timeout)[0]

    def run_tests_with_output(
        self, command: str = "pytest -q", timeout: int | None = None
    ) -> tuple[TestReport, str]:
        """Run the tests, returning the parsed report *and* the raw output.

        The raw output is returned separately so the caller can store it as an
        artifact instead of inlining a multi-thousand-line log into session
        state.

        The returned :class:`TestReport` is derived entirely from the process
        exit code and output. Nothing here is ever authored by a model.
        """
        result = self.run(command, timeout=timeout)

        if result.sandbox_error:
            return (
                TestReport(
                    command=command,
                    exit_code=result.exit_code,
                    execution_error=result.sandbox_error,
                    duration_seconds=round(result.duration_seconds, 2),
                ),
                result.combined_output,
            )

        report = parse_pytest_output(result.combined_output)
        report.command = command
        report.exit_code = result.exit_code
        report.duration_seconds = round(result.duration_seconds, 2)

        # Exit code 5 is pytest's "no tests collected" -- a real signal, not a pass.
        if result.exit_code == 5 and report.test_count == 0:
            report.execution_error = "no tests were collected"
        elif report.test_count == 0 and result.exit_code != 0:
            report.execution_error = (
                "the test command failed before any test ran: "
                f"{result.combined_output.strip()[:400]}"
            )
        return report, result.combined_output


class DockerSandbox(_BaseSandbox):
    """Runs commands against the workspace inside a throwaway container."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        config: SandboxConfig | None = None,
    ) -> None:
        self.workspace = workspace or get_workspace()
        self.config = config or SandboxConfig()

    # ------------------------------------------------------------------ setup

    def preflight(self) -> tuple[bool, str]:
        """Verify the sandbox can run before an agent depends on it."""
        available, detail = docker_available()
        if not available:
            return False, detail
        if not image_exists(self.config.image):
            if self.config.image != FALLBACK_IMAGE and image_exists(FALLBACK_IMAGE):
                logger.warning(
                    "sandbox image %s missing; falling back to %s",
                    self.config.image,
                    FALLBACK_IMAGE,
                )
                self.config.image = FALLBACK_IMAGE
                return True, f"{detail}; using fallback image {FALLBACK_IMAGE}"
            return False, (
                f"sandbox image {self.config.image} is not present. "
                f"Build it with: docker build -t {DEFAULT_IMAGE} sandbox/"
            )
        return True, detail

    def _docker_args(self, command: str, network: bool | None = None) -> list[str]:
        """Assemble the `docker run` invocation for one command."""
        mount = f"{self.workspace.root}:{CONTAINER_WORKDIR}"
        args = [
            "docker", "run", "--rm",
            "--workdir", CONTAINER_WORKDIR,
            "--volume", mount,
            # Isolation: no new privileges, no capabilities, bounded resources.
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--memory", self.config.memory,
            "--cpus", self.config.cpus,
            "--pids-limit", PID_LIMIT,
            # Writable scratch space so pytest and pip can work with a
            # read-only root filesystem.
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=512m",
        ]
        use_network = self.config.network if network is None else network
        if not use_network:
            args.extend(["--network", "none"])

        for key, value in self.config.env_allowlist.items():
            args.extend(["--env", f"{key}={value}"])

        args.extend([self.config.image, "sh", "-lc", command])
        return args

    # ------------------------------------------------------------- execution

    def run(
        self,
        command: str,
        timeout: int | None = None,
        network: bool | None = None,
    ) -> ExecutionResult:
        """Execute a shell command inside the container."""
        if not command or not command.strip():
            return ExecutionResult(exit_code=1, sandbox_error="command is required")

        ok, detail = self.preflight()
        if not ok:
            return ExecutionResult(
                exit_code=1, command=command, sandbox_error=detail
            )

        limit = timeout or self.config.timeout
        args = self._docker_args(command, network=network)
        started = time.monotonic()

        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=limit,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=124,
                command=command,
                duration_seconds=time.monotonic() - started,
                timed_out=True,
                sandbox_error=f"command exceeded the {limit}s time limit",
            )
        except OSError as exc:
            return ExecutionResult(
                exit_code=1, command=command, sandbox_error=f"docker run failed: {exc}"
            )

        stdout, _ = clamp(completed.stdout or "", MAX_OUTPUT_CHARS)
        stderr, _ = clamp(completed.stderr or "", MAX_OUTPUT_CHARS // 2)

        return ExecutionResult(
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            command=command,
        )


# --------------------------------------------------------------- output parsing

# pytest's summary line, e.g. "3 failed, 12 passed, 1 skipped in 2.31s"
_SUMMARY_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")
_FAILURE_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$", re.MULTILINE)


def parse_pytest_output(output: str) -> TestReport:
    """Extract structured counts from pytest's terminal output.

    Only the last summary line is trusted: a failing test's own output can
    easily contain text that looks like a summary.
    """
    report = TestReport()
    if not output:
        return report

    summary_lines = [
        line for line in output.splitlines() if re.search(r"=+.*\bin\s[\d.]+s", line)
    ]
    counts: dict[str, int] = {}
    if summary_lines:
        for value, label in _SUMMARY_COUNT.findall(summary_lines[-1]):
            counts[label] = counts.get(label, 0) + int(value)
    else:
        # Fall back to scanning the whole output when there is no banner.
        for value, label in _SUMMARY_COUNT.findall(output):
            counts[label] = max(counts.get(label, 0), int(value))

    report.passed = counts.get("passed", 0) + counts.get("xpassed", 0)
    report.failed = counts.get("failed", 0)
    report.errors = counts.get("error", 0) + counts.get("errors", 0)
    report.skipped = counts.get("skipped", 0) + counts.get("xfailed", 0)
    report.test_count = report.passed + report.failed + report.errors + report.skipped

    failures: list[str] = []
    for kind, node_id, message in _FAILURE_LINE.findall(output):
        entry = f"{kind} {node_id}"
        if message:
            entry = f"{entry} - {message.strip()}"
        failures.append(entry[:400])
        if len(failures) >= 20:
            break
    report.failure_summary = failures

    return report


# ------------------------------------------------------------- script backend


def _base_env() -> dict[str, str]:
    """The minimum needed to start a shell -- and nothing else.

    The script does its own ``env -i``, so this is belt and braces; it also
    means an accidental edit to the script cannot leak a variable this layer
    never passed in.
    """
    env = {"PATH": os.environ.get("PATH", "")}
    for key in (
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        # Lets the caller point the sandbox at a specific interpreter.
        "SE_AGENT_SANDBOX_PATH",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _terminate(proc: subprocess.Popen) -> None:
    """Kill the command *and its children*.

    ``subprocess`` only signals the direct child, which for a shell script is
    the shell. Killing that leaves the test suite running, which is how a
    "timed out" run keeps burning CPU long after it was given up on.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    proc.kill()


@dataclass
class ScriptConfig:
    script: str = ""
    timeout: int = DEFAULT_TIMEOUT
    network: bool = False

    def __post_init__(self) -> None:
        if not self.script:
            self.script = os.environ.get("SE_AGENT_SANDBOX_CMD") or str(DEFAULT_SCRIPT)


class ScriptSandbox(_BaseSandbox):
    """Runs commands through a shell script that applies the isolation.

    The script owns every OS-specific decision, so swapping in a different
    mechanism later -- OpenShell, bubblewrap, a VM -- is a new script rather
    than new Python. The contract is deliberately tiny::

        script <session-id> <timeout-seconds> <command>   -> the command's exit
        exit 97                                           -> the sandbox failed
    """

    def __init__(
        self,
        workspace: Workspace | None = None,
        config: ScriptConfig | None = None,
        session_id: str = "default",
    ) -> None:
        self.workspace = workspace or get_workspace()
        self.config = config or ScriptConfig()
        self.session_id = session_id
        self._probe: str | None = None

    # ------------------------------------------------------------------ setup

    def preflight(self) -> tuple[bool, str]:
        """Confirm the script can run, and report what it actually isolates."""
        script = Path(self.config.script)
        if not script.is_file():
            return False, f"sandbox script not found: {script}"

        shell = shutil.which("bash")
        if shell is None:
            return False, (
                "bash is required for the script sandbox and is not on PATH "
                "(install Git Bash, or use SE_AGENT_SANDBOX=docker)"
            )

        if self._probe is None:
            try:
                probe = subprocess.run(
                    [shell, str(script), "--probe"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=_base_env(),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return False, f"sandbox script is not runnable: {exc}"
            if probe.returncode != 0:
                detail = (probe.stderr or "").strip()[:200]
                return False, f"sandbox script probe failed: {detail}"
            self._probe = probe.stdout.strip()

        return True, f"script sandbox ({self._probe})"

    def _args(self, command: str, timeout: int) -> list[str]:
        return [
            shutil.which("bash") or "bash",
            str(Path(self.config.script)),
            self.session_id,
            str(timeout),
            command,
        ]

    # ------------------------------------------------------------- execution

    def run(
        self,
        command: str,
        timeout: int | None = None,
        network: bool | None = None,
    ) -> ExecutionResult:
        if not command or not command.strip():
            return ExecutionResult(exit_code=1, sandbox_error="command is required")

        ok, detail = self.preflight()
        if not ok:
            return ExecutionResult(exit_code=1, command=command, sandbox_error=detail)

        limit = timeout or self.config.timeout
        use_network = self.config.network if network is None else network

        env = _base_env()
        env["SE_AGENT_WORKSPACE"] = str(self.workspace.root)
        env["SE_AGENT_SANDBOX_NETWORK"] = "1" if use_network else "0"

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                self._args(command, limit),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                # A process group of its own, so a timeout can take the whole
                # tree rather than just the shell.
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
        except OSError as exc:
            return ExecutionResult(
                exit_code=1, command=command, sandbox_error=f"could not start: {exc}"
            )

        try:
            # A little slack over the script's own timeout, so its polite
            # termination gets a chance before this forcible one.
            stdout, stderr = proc.communicate(timeout=limit + 15)
        except subprocess.TimeoutExpired:
            _terminate(proc)
            stdout, stderr = proc.communicate()
            return ExecutionResult(
                exit_code=124,
                command=command,
                duration_seconds=time.monotonic() - started,
                timed_out=True,
                sandbox_error=f"command exceeded the {limit}s time limit",
            )

        duration = time.monotonic() - started
        stdout_text, _ = clamp(stdout or "", MAX_OUTPUT_CHARS)
        stderr_text, _ = clamp(stderr or "", MAX_OUTPUT_CHARS // 2)

        # The reserved code means the sandbox never ran the command. Reporting
        # that as an ordinary failure would be indistinguishable from the tests
        # genuinely failing, and would send the debugger after a phantom bug.
        if proc.returncode == SANDBOX_FAILED_EXIT:
            return ExecutionResult(
                exit_code=proc.returncode,
                command=command,
                duration_seconds=duration,
                sandbox_error=(stderr_text.strip() or "the sandbox failed to run"),
            )

        return ExecutionResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            command=command,
        )


_sandbox: _BaseSandbox | None = None


def get_sandbox() -> _BaseSandbox:
    """Process-wide sandbox, chosen by ``SE_AGENT_SANDBOX``."""
    global _sandbox
    if _sandbox is None:
        backend = (os.environ.get("SE_AGENT_SANDBOX") or "docker").strip().lower()
        _sandbox = ScriptSandbox() if backend == "script" else DockerSandbox()
    return _sandbox


def reset_sandbox() -> None:
    """Drop the cached instance, so a changed backend takes effect."""
    global _sandbox
    _sandbox = None


def sandbox_available() -> tuple[bool, str]:
    """Preflight whichever backend is selected."""
    return get_sandbox().preflight()


# --------------------------------------------------------------- agent tools


def run_command(command: str, timeout: int = 300) -> dict:
    """Run a shell command against the repository inside a Docker sandbox.

    The command runs with no network access and no host credentials. Use it for
    builds, linters, and one-off checks; use ``run_tests`` for the test suite.

    Args:
        command: Shell command to execute, e.g. "python -m ruff check app".
        timeout: Seconds to allow before the command is killed (max 900).

    Returns:
        A dict with the exit code, stdout, stderr and duration.
    """
    bounded = max(10, min(int(timeout or 300), 900))
    result = get_sandbox().run(command, timeout=bounded)

    if result.sandbox_error:
        return {
            "ok": False,
            "error": result.sandbox_error,
            "command": command,
            "hint": "the sandbox could not run; this is not a test failure",
        }
    return {
        "ok": result.ok,
        "command": command,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_seconds": round(result.duration_seconds, 2),
    }


def run_tests(command: str = "pytest -q", timeout: int = 600) -> dict:
    """Run the project's tests inside a Docker sandbox and report real results.

    Results are parsed from the actual process output and exit code. A suite
    that collects no tests is reported as an error, never as a pass.

    Args:
        command: Test command to run, e.g. "pytest -q" or "npm test".
        timeout: Seconds to allow before the run is killed (max 900).

    Returns:
        A dict with passed/failed/error counts and a summary of each failure.
    """
    bounded = max(10, min(int(timeout or 600), 900))
    report = get_sandbox().run_tests(command, timeout=bounded)

    return {
        "ok": report.is_green,
        "ran": report.ran,
        "command": report.command,
        "exit_code": report.exit_code,
        "passed": report.passed,
        "failed": report.failed,
        "errors": report.errors,
        "skipped": report.skipped,
        "test_count": report.test_count,
        "duration_seconds": report.duration_seconds,
        "failure_summary": report.failure_summary,
        "execution_error": report.execution_error,
    }


EXECUTION_TOOLS = [run_command, run_tests]
