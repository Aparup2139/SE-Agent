"""Read-only repository tools exposed to ADK agents.

Each function is a plain Python function with type hints and a docstring, which
is all ADK needs to build a function declaration. They return dicts rather than
raw strings so an agent can branch on ``ok`` instead of pattern-matching prose.

Every path argument goes through :class:`~app.tools.workspace.Workspace`, so
traversal outside the repository, reads of credential files, and unbounded
output are all impossible from inside a tool call.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any

from .workspace import (
    MAX_LISTING_ENTRIES,
    MAX_SEARCH_MATCHES,
    Workspace,
    WorkspaceError,
    clamp,
    get_workspace,
)

logger = logging.getLogger(__name__)

GIT_TIMEOUT = 30


def _fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _git_prefix(workspace: Workspace) -> str:
    """Where the workspace sits inside its git repository, "" if it is the root.

    A workspace is often a subdirectory of a larger repository. Git does not
    care about the current directory -- it reports the whole repository -- so
    without this a workspace confined to ``config/`` would be handed the parent
    project's changes and, through ``git_diff``, the contents of files it is not
    allowed to read.
    """
    result = _run_git(workspace, ["rev-parse", "--show-prefix"])
    return result["output"].strip() if result["ok"] else ""


def _inside(path: str, prefix: str) -> str | None:
    """``path`` (repository-relative) as workspace-relative, or None if outside."""
    # Git C-quotes any path with a space or a quote in it.
    path = path.strip()
    if len(path) > 1 and path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if not prefix:
        return path
    if not path.startswith(prefix):
        return None
    return path[len(prefix) :]


def _run_git(workspace: Workspace, args: list[str]) -> dict[str, Any]:
    """Run a read-only git command inside the workspace."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace.root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return _fail("git is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        return _fail(f"git {' '.join(args)} timed out after {GIT_TIMEOUT}s")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if "not a git repository" in stderr.lower():
            # Spelled out, because the natural inference from a bare failure is
            # "so nothing changed" -- and an agent that draws it reports the
            # opposite of the truth about work it just did.
            return _fail(
                "this workspace is not a git repository, so there is no diff to "
                "read. This says nothing about whether files changed: read them, "
                "or use the files_changed list in state."
            )
        return _fail(f"git {' '.join(args)} failed: {stderr[:500]}")

    output, truncation = clamp(result.stdout)
    return {"ok": True, "output": output + truncation.note(), "truncated": truncation.truncated}


# ------------------------------------------------------------------ discovery


def list_files(directory: str = ".", pattern: str = "*") -> dict[str, Any]:
    """List files in the repository, skipping vendored and build directories.

    Args:
        directory: Repository-relative directory to list. Defaults to the root.
        pattern: Glob pattern applied recursively, e.g. "*.py" or "test_*.py".

    Returns:
        A dict with the matched file paths and a count.
    """
    workspace = get_workspace()
    try:
        paths = list(workspace.iter_files(directory, pattern))
    except WorkspaceError as exc:
        return _fail(str(exc))

    shown = paths[:MAX_LISTING_ENTRIES]
    result = {
        "ok": True,
        "directory": directory,
        "pattern": pattern,
        "count": len(paths),
        "files": [workspace.relative(p) for p in shown],
    }
    if len(paths) > MAX_LISTING_ENTRIES:
        result["note"] = (
            f"showing {MAX_LISTING_ENTRIES} of {len(paths)} matches; "
            "use a narrower pattern or directory"
        )
    return result


def read_file(path: str, start_line: int = 1, end_line: int = 0) -> dict[str, Any]:
    """Read a text file from the repository, optionally a line range.

    Args:
        path: Repository-relative path to the file.
        start_line: 1-indexed first line to return. Defaults to the start.
        end_line: 1-indexed last line to return. 0 means read to the end.

    Returns:
        A dict with the file content and the line range actually returned.
    """
    workspace = get_workspace()
    try:
        content, truncation = workspace.read_text(path)
    except WorkspaceError as exc:
        return _fail(str(exc))

    lines = content.splitlines()
    total = len(lines)

    start = max(1, start_line)
    end = total if end_line in (0, None) else min(end_line, total)
    if start > total:
        return _fail(f"{path} has {total} lines; start_line {start} is past the end")

    selected = lines[start - 1 : end]
    body, body_truncation = clamp("\n".join(selected))

    return {
        "ok": True,
        "path": path,
        "content": body + body_truncation.note() + truncation.note("bytes"),
        "start_line": start,
        "end_line": end,
        "total_lines": total,
        "truncated": truncation.truncated or body_truncation.truncated,
    }


def search_code(
    query: str, pattern: str = "*", directory: str = ".", regex: bool = False
) -> dict[str, Any]:
    """Search repository file contents for a string or regular expression.

    Args:
        query: Text to find, or a regex when ``regex`` is true.
        pattern: Glob limiting which files are searched, e.g. "*.py".
        directory: Repository-relative directory to search under.
        regex: Treat ``query`` as a regular expression instead of literal text.

    Returns:
        A dict of matches, each with a path, line number and the matching line.
    """
    if not query or not query.strip():
        return _fail("query is required")

    workspace = get_workspace()
    try:
        matcher = (
            re.compile(query) if regex else re.compile(re.escape(query), re.IGNORECASE)
        )
    except re.error as exc:
        return _fail(f"invalid regular expression: {exc}")

    matches: list[dict[str, Any]] = []
    files_searched = 0
    try:
        for path in workspace.iter_files(directory, pattern):
            if len(matches) >= MAX_SEARCH_MATCHES:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            files_searched += 1
            for number, line in enumerate(text.splitlines(), start=1):
                if matcher.search(line):
                    matches.append(
                        {
                            "path": workspace.relative(path),
                            "line": number,
                            "text": line.strip()[:300],
                        }
                    )
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        break
    except WorkspaceError as exc:
        return _fail(str(exc))

    result = {
        "ok": True,
        "query": query,
        "files_searched": files_searched,
        "match_count": len(matches),
        "matches": matches,
    }
    if len(matches) >= MAX_SEARCH_MATCHES:
        result["note"] = f"stopped at {MAX_SEARCH_MATCHES} matches; narrow the search"
    return result


# ------------------------------------------------------------------------ git


def git_status() -> dict[str, Any]:
    """Show which files in the repository are modified, added or untracked.

    Returns:
        A dict with the porcelain status output and a parsed file list.
    """
    workspace = get_workspace()
    # The trailing pathspec confines the answer to this workspace; a workspace
    # inside a larger repository must not be told about the rest of it.
    result = _run_git(workspace, ["status", "--porcelain=v1", "--", "."])
    if not result["ok"]:
        return result

    prefix = _git_prefix(workspace)
    changes = []
    for line in result["output"].splitlines():
        if len(line) > 3:
            path = _inside(line[3:], prefix)
            if path is None:
                continue
            changes.append({"status": line[:2].strip(), "path": path})
    return {
        "ok": True,
        "clean": not changes,
        "change_count": len(changes),
        "changes": changes[:MAX_LISTING_ENTRIES],
    }


def git_diff(path: str = "", staged: bool = False) -> dict[str, Any]:
    """Show uncommitted changes as a unified diff.

    Args:
        path: Limit the diff to one repository-relative path. Empty means all.
        staged: Diff the staging area instead of the working tree.

    Returns:
        A dict with the diff text and a per-file summary.
    """
    workspace = get_workspace()
    # --relative both rewrites paths relative to this directory and drops
    # changes outside it, so a subdirectory workspace cannot read out the
    # parent project through its diff.
    base = ["diff", "--relative"]
    if staged:
        base.append("--cached")

    scope: list[str] = []
    if path:
        try:
            workspace.resolve(path)
        except WorkspaceError as exc:
            return _fail(str(exc))
        scope = ["--", path]

    if not scope:
        scope = ["--", "."]

    result = _run_git(workspace, [*base, *scope])
    if not result["ok"]:
        return result

    # --stat must go before the "--" separator, or git reads it as a path.
    stat = _run_git(workspace, [*base, "--stat", *scope])
    return {
        "ok": True,
        "diff": result["output"] or "(no changes)",
        "summary": stat.get("output", "").strip() if stat["ok"] else "",
        "truncated": result["truncated"],
    }


def git_log(limit: int = 10, path: str = "") -> dict[str, Any]:
    """Show recent commits, most recent first.

    Args:
        limit: How many commits to return (1-50).
        path: Limit history to one repository-relative path. Empty means all.

    Returns:
        A dict with one entry per commit: hash, author, date and subject.
    """
    workspace = get_workspace()
    bounded = max(1, min(int(limit or 10), 50))
    args = ["log", f"-{bounded}", "--pretty=format:%h%x1f%an%x1f%ar%x1f%s"]
    if path:
        try:
            workspace.resolve(path)
        except WorkspaceError as exc:
            return _fail(str(exc))
        args.extend(["--", path])

    result = _run_git(workspace, args)
    if not result["ok"]:
        return result

    commits = []
    for line in result["output"].splitlines():
        fields = line.split("\x1f")
        if len(fields) == 4:
            commits.append(
                {
                    "hash": fields[0],
                    "author": fields[1],
                    "date": fields[2],
                    "subject": fields[3],
                }
            )
    return {"ok": True, "count": len(commits), "commits": commits}


# ----------------------------------------------------------------- authoring


def write_file(path: str, content: str) -> dict[str, Any]:
    """Create or overwrite a file in the repository.

    Prefer :func:`edit_file` for changing part of an existing file -- a full
    overwrite risks discarding code you did not read.

    Args:
        path: Repository-relative path to write.
        content: Complete new contents of the file.

    Returns:
        A dict confirming the path and the number of bytes written.
    """
    workspace = get_workspace()
    try:
        target = workspace.write_text(path, content)
    except WorkspaceError as exc:
        return _fail(str(exc))
    except OSError as exc:
        return _fail(f"could not write {path}: {exc}")

    return {
        "ok": True,
        "path": workspace.relative(target),
        "bytes_written": len(content.encode("utf-8")),
        "lines": len(content.splitlines()),
    }


def edit_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    """Replace an exact snippet in a file, leaving the rest untouched.

    The snippet must appear exactly once, so an ambiguous edit fails loudly
    instead of changing the wrong occurrence.

    Args:
        path: Repository-relative path to edit.
        old_text: Exact text to replace, including indentation.
        new_text: Replacement text.

    Returns:
        A dict confirming the edit, or an error explaining why it did not apply.
    """
    workspace = get_workspace()
    try:
        content, _ = workspace.read_text(path)
    except WorkspaceError as exc:
        return _fail(str(exc))

    if not old_text:
        return _fail("old_text is required; use write_file to create a new file")

    occurrences = content.count(old_text)
    if occurrences == 0:
        return _fail(
            f"old_text not found in {path}. Read the file first and copy the "
            "snippet exactly, including indentation."
        )
    if occurrences > 1:
        return _fail(
            f"old_text appears {occurrences} times in {path}; include more "
            "surrounding context so the edit is unambiguous"
        )

    try:
        workspace.write_text(path, content.replace(old_text, new_text, 1))
    except (WorkspaceError, OSError) as exc:
        return _fail(f"could not write {path}: {exc}")

    return {
        "ok": True,
        "path": path,
        "lines_removed": len(old_text.splitlines()),
        "lines_added": len(new_text.splitlines()),
    }


READ_ONLY_TOOLS = [
    list_files,
    read_file,
    search_code,
    git_status,
    git_diff,
    git_log,
]

WRITE_TOOLS = [write_file, edit_file]


# A fingerprint of a large tree is taken twice per iteration, so it is bounded.
# Past this many files the comparison is still sound -- iter_files walks in
# sorted order, so both snapshots cover the same prefix.
FINGERPRINT_LIMIT = 5000


def _filesystem_state(workspace: Workspace) -> dict[str, str] | None:
    """Fingerprint by size and modification time, for a directory without git.

    Coarser than a diff: a rewrite that preserves both size and mtime reads as
    unchanged. Every ordinary edit changes the mtime, so in practice this
    answers the only question asked of it -- did anything here change?

    ``iter_files`` skips ``__pycache__`` and ``.pytest_cache``, so running the
    tests does not by itself look like a change to the source.
    """
    state: dict[str, str] = {}
    try:
        for path in workspace.iter_files():
            try:
                info = path.stat()
            except OSError:
                continue
            state[workspace.relative(path)] = f"{info.st_size}/{info.st_mtime_ns}"
            if len(state) >= FINGERPRINT_LIMIT:
                break
    except (WorkspaceError, OSError) as exc:
        logger.warning("could not fingerprint the workspace: %s", exc)
        return None
    return state


# Not an agent tool: the workflow uses this to tell its own edits apart from a
# working tree that was already dirty when the run started.
def working_tree_state() -> dict[str, str] | None:
    """Fingerprint every uncommitted change in the workspace.

    Maps workspace-relative path -> an opaque marker of that path's current
    content. Two of these compared tell you which files a run actually touched;
    a plain path list cannot, because on an already-dirty tree every path looks
    changed before the run begins.

    Confined to the workspace like every other tool. A workspace inside a larger
    repository must not measure the parent project's changes -- doing so credits
    a run with edits someone else made in a directory it cannot even read.

    A directory that is not a git repository still gets a fingerprint, taken
    from the filesystem instead. Plenty of real work happens in directories
    nobody has run ``git init`` in, and the alternative -- no evidence, so every
    run reports having changed nothing -- makes the whole system unusable there.

    Returns ``None`` only when neither git nor the filesystem can answer. That
    is deliberately distinct from an empty dict (a genuinely clean tree) -- a
    tool failure must not be read as evidence either way.
    """
    workspace = get_workspace()
    # "diff HEAD" covers staged and unstaged edits both; plain "diff" would miss
    # anything the agent staged. On a repository with no commits yet there is no
    # HEAD to diff against, so fall back.
    scope = ["--relative", "--numstat", "--", "."]
    numstat = _run_git(workspace, ["diff", "HEAD", *scope])
    if not numstat["ok"]:
        numstat = _run_git(workspace, ["diff", *scope])
    status = _run_git(workspace, ["status", "--porcelain=v1", "--", "."])
    if not numstat["ok"] or not status["ok"]:
        return _filesystem_state(workspace)

    prefix = _git_prefix(workspace)
    state: dict[str, str] = {}
    for line in numstat["output"].splitlines():
        parts = line.split("\t")
        # --relative has already made these workspace-relative.
        if len(parts) == 3:
            state[parts[2].strip().strip('"')] = f"{parts[0]}/{parts[1]}"

    for line in status["output"].splitlines():
        if line.startswith("??") and len(line) > 3:
            path = _inside(line[3:], prefix)
            if path is None:
                continue
            # ponytail: size, not a hash -- an equal-length rewrite of an
            # untracked file reads as unchanged. Hash it if that ever matters.
            try:
                marker = f"untracked/{(workspace.root / path).stat().st_size}"
            except OSError:
                marker = "untracked"
            state.setdefault(path, marker)

    return state
