"""Repository and execution tools available to SE-Agent's ADK agents.

Grouped by capability so each agent gets exactly the authority its role needs:
the reviewer is read-only by construction, not by instruction.
"""

from .repo_tools import (
    READ_ONLY_TOOLS,
    WRITE_TOOLS,
    edit_file,
    git_diff,
    git_log,
    git_status,
    list_files,
    read_file,
    search_code,
    write_file,
)
from .sandbox import (
    EXECUTION_TOOLS,
    ExecutionResult,
    Sandbox,
    SandboxConfig,
    SandboxError,
    get_sandbox,
    parse_pytest_output,
    reset_sandbox,
    run_command,
    run_tests,
    sandbox_available,
)
from .workspace import Workspace, WorkspaceError, get_workspace

# Read the repository, but change nothing and run nothing.
INSPECTION_TOOLS = list(READ_ONLY_TOOLS)

# Full implementation authority: read, edit, and run inside the sandbox.
CODING_TOOLS = [*READ_ONLY_TOOLS, *WRITE_TOOLS, *EXECUTION_TOOLS]

# The tester executes; it never edits, so it cannot make a failure disappear.
TESTING_TOOLS = [*READ_ONLY_TOOLS, *EXECUTION_TOOLS]

__all__ = [
    "CODING_TOOLS",
    "Sandbox",
    "EXECUTION_TOOLS",
    "ExecutionResult",
    "INSPECTION_TOOLS",
    "READ_ONLY_TOOLS",
    "SandboxConfig",
    "SandboxError",
    "TESTING_TOOLS",
    "WRITE_TOOLS",
    "Workspace",
    "WorkspaceError",
    "edit_file",
    "get_sandbox",
    "reset_sandbox",
    "sandbox_available",
    "get_workspace",
    "git_diff",
    "git_log",
    "git_status",
    "list_files",
    "parse_pytest_output",
    "read_file",
    "run_command",
    "run_tests",
    "search_code",
    "write_file",
]
