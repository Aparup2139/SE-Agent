"""Parsing pytest's output, and the tool layer agents actually call.

Both are backend-independent: the parser is a pure function, and the tools
speak to whatever :func:`get_sandbox` returns.
"""

from __future__ import annotations

from app.schemas import TestReport
from app.tools import sandbox as sandbox_module
from app.tools.sandbox import ExecutionResult, parse_pytest_output

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

# ------------------------------------------------------------------ tool layer


def test_run_tests_tool_reports_failure_honestly(monkeypatch):
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
            return ExecutionResult(exit_code=0, command=command)

    monkeypatch.setattr(sandbox_module, "get_sandbox", lambda: Box())
    sandbox_module.run_command("echo hi", timeout=99999)
    assert seen["timeout"] == 900


def test_a_sandbox_failure_is_flagged_as_not_a_test_failure(monkeypatch):
    """The tool's own wording has to keep the two apart for the agent."""

    class Box:
        def run(self, command, timeout=None):
            return ExecutionResult(
                exit_code=97, command=command, sandbox_error="bash missing"
            )

    monkeypatch.setattr(sandbox_module, "get_sandbox", lambda: Box())
    result = sandbox_module.run_command("pytest")
    assert result["ok"] is False
    assert result["error"] == "bash missing"
    assert "not a test failure" in result["hint"]


# ------------------------------------------------------- unittest's output


UNITTEST_PASS = """
test_01_ok (tester.TestKnapsack.test_01_ok) ... ok
test_02_ok (tester.TestKnapsack.test_02_ok) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
"""

UNITTEST_FAIL = """
test_01_bad (tester.TestKnapsack.test_01_bad) ... FAIL

======================================================================
FAIL: test_01_bad (tester.TestKnapsack.test_01_bad)
----------------------------------------------------------------------
AssertionError: ValueError not raised

----------------------------------------------------------------------
Ran 5 tests in 0.002s

FAILED (failures=1, errors=1, skipped=1)
"""

UNITTEST_IMPORT_ERROR = """
tester (unittest.loader._FailedTest.tester) ... ERROR

======================================================================
ERROR: tester (unittest.loader._FailedTest.tester)
----------------------------------------------------------------------
ImportError: Failed to import test module: tester
ModuleNotFoundError: No module named 'tester'

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (errors=1)
"""


def test_a_green_unittest_run_is_recognized_as_green():
    """pytest is not the only runner; its output format is not the only one."""
    report = parse_pytest_output(UNITTEST_PASS)
    assert report.test_count == 2
    assert report.passed == 2
    assert report.is_green


def test_unittest_counts_the_passes_it_never_prints():
    """unittest reports only what went wrong; the rest passed."""
    report = parse_pytest_output(UNITTEST_FAIL)
    assert report.test_count == 5
    assert (report.failed, report.errors, report.skipped) == (1, 1, 1)
    assert report.passed == 2
    assert not report.is_green
    assert report.failure_summary == ["FAIL tester.TestKnapsack.test_01_bad"]


def test_a_unittest_run_that_collected_nothing_is_not_green():
    """An empty suite reports Ran 0 tests ... OK, which is not a pass."""
    report = parse_pytest_output("Ran 0 tests in 0.000s\n\nOK\n")
    assert report.test_count == 0
    assert not report.is_green


def test_a_suite_that_would_not_import_is_an_execution_error():
    """A broken environment is not a failing implementation."""
    report = parse_pytest_output(UNITTEST_IMPORT_ERROR)
    assert report.execution_error
    assert "could not be imported (tester)" in report.execution_error
    assert not report.is_green


def test_an_import_error_does_not_masquerade_as_a_code_bug():
    """Left as 1 error it sends the debugger after a bug that may not exist."""
    report = parse_pytest_output(UNITTEST_IMPORT_ERROR)
    assert report.failed == 0
    assert "says nothing about whether the code is correct" in report.execution_error


def test_a_genuine_unittest_failure_is_not_called_an_execution_error():
    """The distinction has to cut both ways to be worth anything."""
    report = parse_pytest_output(UNITTEST_FAIL)
    assert not report.execution_error
    assert report.failed == 1
