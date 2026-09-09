"""Isolated execution for repository code.

Everything an agent writes is executed through ``sandbox/local.sh``, which
applies the strongest isolation the host actually provides and reports which
that was. The script owns every OS-specific decision; this module owns the
process, the timeout, and the distinction between a failed command and a
failed sandbox.

Tiers, strongest first -- the script probes each by running it, so a mechanism
that is present but broken demotes rather than failing every run:

``bwrap``
    A real boundary. Fresh mount namespace, the workspace the only writable
    path, no network, its own PID and IPC space.
``unshare``
    Network and user namespace. Bounds what the command can reach, not what it
    can read.
``none``
    A clean environment only. Not a boundary -- and a run refuses to start on
    it unless ``SE_AGENT_SANDBOX_ALLOW_UNSAFE=1`` says that is intended.

On Windows only ``none`` is reachable, because the namespaces are Linux's.
``SE_AGENT_SANDBOX_WSL`` runs the script inside WSL to reach the others; note
that this also means the tests run with WSL's toolchain, not Windows'.

What holds at every tier is the environment allowlist: the command starts from
``env -i`` and receives only what the script lists, so ``NVIDIA_API_KEY`` and
every other host secret are absent rather than merely unmentioned.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..schemas import TestReport
from .workspace import MAX_OUTPUT_CHARS, Workspace, clamp, get_workspace

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600

# The script's reserved code for "the sandbox failed", as distinct from "the
# command failed". Conflating the two would let a broken sandbox read as a red
# test suite and send the debugger after a bug that does not exist.
SANDBOX_FAILED_EXIT = 97

DEFAULT_SCRIPT = Path(__file__).resolve().parents[2] / "sandbox" / "local.sh"

# Tiers in ascending order of isolation, as the script reports them.
TIERS = ("none", "unshare", "bwrap")


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
        # The isolation floor a run refuses to drop below.
        "SE_AGENT_SANDBOX_REQUIRE",
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


def allow_unsafe() -> bool:
    """Whether running without an isolation boundary has been accepted.

    Only a real environment variable counts. ``.env`` is a file inside the
    repository the agent is editing, and a run that could write this flag
    there would be granting the *next* run unguarded execution on the host --
    a decision that has to stay with the person, not the agent. Observed: a
    run wrote ``SE_AGENT_SANDBOX_ALLOW_UNSAFE=1`` into ``.env`` while trying
    to get its own tests to pass.
    """
    if _came_from_dotenv("SE_AGENT_SANDBOX_ALLOW_UNSAFE"):
        logger.warning(
            "ignoring SE_AGENT_SANDBOX_ALLOW_UNSAFE from .env: unguarded "
            "execution must be granted in the environment, not in a file the "
            "agent can write"
        )
        return False
    return (os.environ.get("SE_AGENT_SANDBOX_ALLOW_UNSAFE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _came_from_dotenv(name: str) -> bool:
    """Whether ``name`` is set in a ``.env`` file rather than the environment."""
    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard dependency
        return False
    path = find_dotenv(usecwd=True)
    return bool(path) and name in dotenv_values(path)


def find_bash() -> str:
    """Locate a *native* bash.

    On Windows the ``bash`` on PATH is often ``C:/Windows/System32/bash.exe``,
    which is not a shell at all -- it is the WSL launcher. Handing it a Windows
    path fails with a confusing ``execvpe(/bin/bash)`` error, and it silently
    changes which OS the command would have run on. PowerShell resolves to that
    shim while Git Bash resolves to its own, which is how the same code worked
    in one shell and not the other.
    """
    candidates: list[str] = []

    found = shutil.which("bash")
    if found and "system32" not in found.replace("\\", "/").lower():
        candidates.append(found)

    if os.name == "nt":
        git_bash = os.path.join("Git", "bin", "bash.exe")
        candidates += [
            os.path.join(r"C:\Program Files", git_bash),
            os.path.join(r"C:\Program Files (x86)", git_bash),
            os.path.join(os.path.expandvars(r"%LOCALAPPDATA%\Programs"), git_bash),
        ]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate

    detail = (
        "the `bash` on PATH is the WSL launcher, not a shell" if found else "no bash is on PATH"
    )
    raise SandboxError(
        f"no native bash was found to run the sandbox script ({detail}). "
        "Install Git Bash, or set SE_AGENT_SANDBOX_WSL=<distro> to run inside "
        "WSL deliberately."
    )


@dataclass
class SandboxConfig:
    script: str = ""
    timeout: int = DEFAULT_TIMEOUT
    network: bool = False
    # A WSL distribution to run the script in, for hosts whose own OS offers
    # no namespaces. Empty means run on this host directly.
    wsl_distro: str = ""

    def __post_init__(self) -> None:
        if not self.script:
            self.script = os.environ.get("SE_AGENT_SANDBOX_CMD") or str(DEFAULT_SCRIPT)
        if not self.wsl_distro:
            self.wsl_distro = (os.environ.get("SE_AGENT_SANDBOX_WSL") or "").strip()


class Sandbox:
    """Runs commands through the isolation script.

    The script's contract is deliberately tiny, so a different mechanism is a
    new script rather than new Python::

        script <session-id> <timeout-seconds> <command>   -> the command's exit
        exit 97                                           -> the sandbox failed
    """

    def __init__(
        self,
        workspace: Workspace | None = None,
        config: SandboxConfig | None = None,
        session_id: str = "default",
    ) -> None:
        self.workspace = workspace or get_workspace()
        self.config = config or SandboxConfig()
        self.session_id = session_id
        self._tier: str | None = None

    # ------------------------------------------------------------------ setup

    @property
    def uses_wsl(self) -> bool:
        return bool(self.config.wsl_distro) and os.name == "nt"

    def _script_path(self) -> str:
        """The script's path as the shell that will run it understands it."""
        native = str(Path(self.config.script).resolve())
        if not self.uses_wsl:
            return native
        return self._to_wsl_path(native)

    def _to_wsl_path(self, windows_path: str) -> str:
        """Translate a Windows path to its WSL form.

        A drive path is mapped here rather than by asking ``wslpath``, because
        that meant launching WSL once per path -- and observed failing under
        load with a non-zero exit and an empty stderr, which surfaced as
        "could not translate ... for WSL:" and killed the run. WSL also warns
        on stderr about untranslatable Windows PATH entries (any entry with a
        space, e.g. 'D:\\Qoder IDE\\bin'), which is noise in that same channel.

        ``wslpath`` still handles anything that is not a drive path -- a UNC
        share, say.
        """
        path = Path(windows_path)
        drive = path.drive
        # ponytail: assumes the default /mnt automount root; a custom
        # automount.root in /etc/wsl.conf would need reading that file.
        if len(drive) == 2 and drive.endswith(":"):
            tail = path.as_posix()[len(drive) :].lstrip("/")
            return f"/mnt/{drive[0].lower()}/{tail}"

        result = subprocess.run(
            ["wsl.exe", "-d", self.config.wsl_distro, "wslpath", "-a", windows_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise SandboxError(
                f"could not translate {windows_path} for WSL: {result.stderr.strip()}"
            )
        return result.stdout.strip().strip("\x00")

    def _launcher(self) -> list[str]:
        """The argv prefix that gets us a shell on the target OS."""
        if self.uses_wsl:
            return ["wsl.exe", "-d", self.config.wsl_distro, "--", "env"]
        return [find_bash()]

    def tier(self) -> str:
        """Which isolation tier is actually in force. Cached per instance."""
        if self._tier is None:
            self._tier = self._ask_script("--tier") or "none"
        return self._tier

    def _ask_script(self, flag: str) -> str:
        try:
            launcher = self._launcher()
            prefix = [*launcher, "bash"] if self.uses_wsl else launcher
            args = [*prefix, self._script_path(), flag]
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=60, env=_base_env()
            )
        except (OSError, subprocess.TimeoutExpired, SandboxError):
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.strip().splitlines()[-1].strip() if result.stdout else ""

    def preflight(self) -> tuple[bool, str]:
        """Confirm the sandbox can run, and report what it actually isolates."""
        script = Path(self.config.script)
        if not script.is_file():
            return False, f"sandbox script not found: {script}"

        try:
            self._launcher()
        except SandboxError as exc:
            return False, str(exc)

        probe = self._ask_script("--probe")
        if not probe:
            return False, (
                "the sandbox script did not run"
                + (f" in WSL distro {self.config.wsl_distro}" if self.uses_wsl else "")
            )

        where = f" via WSL:{self.config.wsl_distro}" if self.uses_wsl else ""

        # An unboundaried run has to be chosen, not stumbled into.
        if self.tier() == "none" and not allow_unsafe():
            return False, (
                f"no isolation boundary is available here ({probe}){where}. "
                "Repository code would run as you, with your files and network. "
                "Set SE_AGENT_SANDBOX_WSL=<distro> to run inside WSL, install "
                "bubblewrap for a full boundary, or set "
                "SE_AGENT_SANDBOX_ALLOW_UNSAFE=1 to accept this deliberately."
            )

        unsafe = " -- ACCEPTED AS UNSAFE" if self.tier() == "none" else ""
        return True, f"{probe}{where}{unsafe}"

    def _args(self, command: str, timeout: int, network: bool = False) -> list[str]:
        launcher = self._launcher()
        tail = [self._script_path(), self.session_id, str(timeout), command]
        if self.uses_wsl:
            # `wsl -- env VAR=... bash script` is how the environment crosses
            # the boundary; WSLENV is fiddly and this is explicit.
            crossing = [
                f"SE_AGENT_WORKSPACE={self._workspace_path()}",
                f"SE_AGENT_SANDBOX_NETWORK={'1' if network else '0'}",
            ]
            # The script reads these too, and nothing else carries them across:
            # a Windows host's workspace venv is unusable from Linux, so
            # SE_AGENT_SANDBOX_PATH is the only way to name an interpreter that
            # works -- and dropping it here left the setting silently inert on
            # the one platform that needs it, with the script falling back to
            # whatever `.venv/bin` happened to exist.
            for name in ("SE_AGENT_SANDBOX_PATH", "SE_AGENT_SANDBOX_REQUIRE"):
                value = (os.environ.get(name) or "").strip()
                if value:
                    crossing.append(f"{name}={value}")
            return [*launcher, *crossing, "bash", *tail]
        return [*launcher, *tail]

    def _workspace_path(self) -> str:
        native = str(self.workspace.root)
        return self._to_wsl_path(native) if self.uses_wsl else native

    # ------------------------------------------------------------- execution

    def run(
        self,
        command: str,
        timeout: int | None = None,
        network: bool | None = None,
    ) -> ExecutionResult:
        """Execute one command inside the sandbox."""
        if not command or not command.strip():
            return ExecutionResult(exit_code=1, sandbox_error="command is required")

        ok, detail = self.preflight()
        if not ok:
            return ExecutionResult(exit_code=1, command=command, sandbox_error=detail)

        limit = timeout or self.config.timeout
        use_network = self.config.network if network is None else network

        env = _base_env()
        try:
            args = self._args(command, limit, network=use_network)
            env["SE_AGENT_WORKSPACE"] = self._workspace_path()
        except SandboxError as exc:
            return ExecutionResult(exit_code=1, command=command, sandbox_error=str(exc))
        env["SE_AGENT_SANDBOX_NETWORK"] = "1" if use_network else "0"

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                # A process group of its own, so a timeout can take the whole
                # tree rather than just the shell.
                start_new_session=os.name != "nt",
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
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

    # ------------------------------------------------------------------ tests

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


# --------------------------------------------------------------- output parsing

# pytest's summary line, e.g. "3 failed, 12 passed, 1 skipped in 2.31s"
_SUMMARY_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")
_FAILURE_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$", re.MULTILINE)


# unittest reports nothing pytest's patterns match: no banner line, and its
# tallies read "FAILED (failures=2, errors=1)".
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in", re.MULTILINE)
_UNITTEST_TALLY = re.compile(r"(failures|errors|skipped|expected failures)=(\d+)")
_UNITTEST_FAILURE = re.compile(r"^(FAIL|ERROR): (\S+)(?:\s+\((\S+)\))?$", re.MULTILINE)
# unittest reports a suite it could not even import as a synthetic failing test.
_UNITTEST_IMPORT_FAILURE = re.compile(
    r"unittest\.loader\._FailedTest|Failed to import test module"
)
_UNITTEST_IMPORT_NAME = re.compile(r"Failed to import test module: (\S+)")


def parse_pytest_output(output: str) -> TestReport:
    """Extract structured counts from pytest's terminal output.

    Only the last summary line is trusted: a failing test's own output can
    easily contain text that looks like a summary.
    """
    report = TestReport()
    if not output:
        return report

    ran = _UNITTEST_RAN.search(output)
    if ran and not re.search(r"=+.*in\s[\d.]+s", output):
        return _parse_unittest_output(output, int(ran.group(1)))

    summary_lines = [line for line in output.splitlines() if re.search(r"=+.*\bin\s[\d.]+s", line)]
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


_sandbox: Sandbox | None = None


def _parse_unittest_output(output: str, total: int) -> TestReport:
    """Counts from ``python -m unittest``, which reports only what went wrong.

    "Ran 5 tests" plus "FAILED (failures=2)" means three passed -- the passing
    count is never stated, so it is what is left over.
    """
    report = TestReport(test_count=total)
    tallies = {label: int(value) for label, value in _UNITTEST_TALLY.findall(output)}
    report.failed = tallies.get("failures", 0)
    report.errors = tallies.get("errors", 0)
    report.skipped = tallies.get("skipped", 0) + tallies.get("expected failures", 0)
    report.passed = max(0, total - report.failed - report.errors - report.skipped)

    failures: list[str] = []
    for kind, name, where in _UNITTEST_FAILURE.findall(output):
        failures.append(f"{kind} {where or name}"[:400])
        if len(failures) >= 20:
            break
    report.failure_summary = failures

    # A suite that could not be imported is a broken environment, not a failing
    # implementation. Left as "1 error" it sends the debugger hunting for a bug
    # in code that may be perfectly correct; as an execution error it says what
    # actually happened.
    if _UNITTEST_IMPORT_FAILURE.search(output):
        match = _UNITTEST_IMPORT_NAME.search(output)
        module = f" ({match.group(1)})" if match else ""
        report.execution_error = (
            f"the test suite could not be imported{module}. The tests never ran, "
            "so this says nothing about whether the code is correct."
        )
    return report


def get_sandbox() -> Sandbox:
    """Process-wide sandbox instance, rebuilt if the workspace has moved.

    A ``Sandbox`` captures its workspace when it is constructed, so a cached one
    keeps executing in the directory of whichever run built it. In a server
    process that outlives a single run, the next run against a different
    repository would then read one directory through its tools and run its
    commands in another -- tests "failing" against a suite that is not even
    there, and a debugger sent to find out why.
    """
    global _sandbox
    root = get_workspace().root
    if _sandbox is None or _sandbox.workspace.root != root:
        _sandbox = Sandbox()
    return _sandbox


def reset_sandbox() -> None:
    """Drop the cached instance, so changed configuration takes effect."""
    global _sandbox
    _sandbox = None


def sandbox_available() -> tuple[bool, str]:
    """Preflight the sandbox."""
    return get_sandbox().preflight()


# --------------------------------------------------------------- agent tools


def run_command(command: str, timeout: int = 300) -> dict:
    """Run a shell command against the repository inside the sandbox.

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
    """Run the project's tests inside the sandbox and report real results.

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
