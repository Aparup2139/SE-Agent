"""Repository tools: path confinement, secret refusal, output caps, git, edits."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.tools import repo_tools
from app.tools.workspace import (
    MAX_SEARCH_MATCHES,
    Workspace,
    WorkspaceError,
    clamp,
    is_secret_path,
)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A small throwaway repository, wired in as the active workspace."""
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)

    (tmp_path / "app" / "main.py").write_text(
        "def greet(name):\n    return f'hello {name}'\n", encoding="utf-8"
    )
    (tmp_path / "app" / "util.py").write_text("VALUE = 42\n", encoding="utf-8")
    (tmp_path / "tests" / "test_main.py").write_text(
        "def test_greet():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET_TOKEN=hunter2\n", encoding="utf-8")
    (tmp_path / "id_rsa.pem").write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    workspace = Workspace(tmp_path)
    monkeypatch.setattr(repo_tools, "get_workspace", lambda: workspace)
    return tmp_path


@pytest.fixture
def git_repo(repo):
    """The same repository, with git history."""

    def run(*args):
        subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)

    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    run("add", "-A")
    run("commit", "-q", "-m", "initial commit")
    return repo


# ------------------------------------------------------------- path security


def test_parent_traversal_is_blocked(repo):
    workspace = Workspace(repo)
    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        workspace.resolve("../../../etc/passwd")


def test_deep_traversal_inside_a_subdir_is_blocked(repo):
    workspace = Workspace(repo)
    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        workspace.resolve("app/../../outside.txt")


def test_absolute_path_outside_root_is_blocked(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    outside.write_text("data", encoding="utf-8")
    workspace = Workspace(repo)
    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        workspace.resolve(str(outside))


def test_absolute_path_inside_root_is_allowed(repo):
    workspace = Workspace(repo)
    assert workspace.resolve(str(repo / "app" / "main.py")).name == "main.py"


def test_symlink_out_of_the_workspace_is_blocked(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "target.txt"
    outside.write_text("secret", encoding="utf-8")
    link = repo / "escape_link"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted in this environment")

    workspace = Workspace(repo)
    with pytest.raises(WorkspaceError, match="escapes the workspace"):
        workspace.resolve("escape_link")


def test_read_file_rejects_traversal(repo):
    result = repo_tools.read_file("../../../etc/passwd")
    assert result["ok"] is False
    assert "escapes the workspace" in result["error"]


# ---------------------------------------------------------------- secrets


def test_env_file_is_recognized_as_secret(repo):
    assert is_secret_path(repo / ".env")
    assert is_secret_path(repo / ".env.production")
    assert is_secret_path(repo / "id_rsa.pem")
    assert not is_secret_path(repo / "app" / "main.py")


def test_reading_dotenv_is_refused(repo):
    result = repo_tools.read_file(".env")
    assert result["ok"] is False
    assert "credentials" in result["error"]
    assert "hunter2" not in str(result)


def test_reading_a_private_key_is_refused(repo):
    result = repo_tools.read_file("id_rsa.pem")
    assert result["ok"] is False
    assert "BEGIN PRIVATE KEY" not in str(result)


def test_writing_to_a_secret_file_is_refused(repo):
    result = repo_tools.write_file(".env", "NVIDIA_API_KEY=leaked")
    assert result["ok"] is False
    assert (repo / ".env").read_text(encoding="utf-8") == "SECRET_TOKEN=hunter2\n"


def test_binary_files_are_refused(repo):
    result = repo_tools.read_file("logo.png")
    assert result["ok"] is False
    assert "binary" in result["error"]


# ------------------------------------------------------------------ listing


def test_list_files_finds_python_sources(repo):
    result = repo_tools.list_files(".", "*.py")
    assert result["ok"]
    assert "app/main.py" in result["files"]
    assert "tests/test_main.py" in result["files"]


def test_listing_skips_vendored_directories(repo):
    files = repo_tools.list_files(".", "*")["files"]
    assert not any("node_modules" in f for f in files)


def test_listing_never_stats_an_ignored_tree(repo, monkeypatch):
    """A venv built under WSL leaves links Windows cannot stat -- don't try.

    ``is_ignored`` is a string check, so an ignored tree must be skipped
    before anything touches the filesystem.
    """
    (repo / "venv" / "bin").mkdir(parents=True)
    (repo / "venv" / "bin" / "python").write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "is_file", _is_file_but_not_in("venv"))

    files = repo_tools.list_files(".", "*")["files"]
    assert "app/main.py" in files
    assert not any("venv" in name for name in files)


def test_listing_skips_a_path_it_cannot_stat(repo, monkeypatch):
    """One unreadable entry must not take the whole listing down with it."""
    (repo / "broken.py").write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "is_file", _is_file_but_not_in("broken.py"))

    files = repo_tools.list_files(".", "*.py")["files"]
    assert "app/main.py" in files
    assert "broken.py" not in files


def _is_file_but_not_in(unstattable: str):
    """``Path.is_file`` that raises WinError 1920 for one path component."""
    real = Path.is_file

    def is_file(self, *args, **kwargs):
        if unstattable in self.parts:
            raise OSError(1920, "The file cannot be accessed by the system")
        return real(self, *args, **kwargs)

    return is_file


def test_listing_a_missing_directory_reports_clearly(repo):
    result = repo_tools.list_files("does_not_exist")
    assert result["ok"] is False


def test_listing_is_capped(repo, monkeypatch):
    for i in range(60):
        (repo / f"gen_{i}.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(repo_tools, "MAX_LISTING_ENTRIES", 10)
    result = repo_tools.list_files(".", "*.py")
    assert len(result["files"]) == 10
    assert "note" in result


# ------------------------------------------------------------------ reading


def test_read_file_returns_content_and_line_count(repo):
    result = repo_tools.read_file("app/main.py")
    assert result["ok"]
    assert "def greet" in result["content"]
    assert result["total_lines"] == 2


def test_read_file_honours_a_line_range(repo):
    (repo / "long.py").write_text("\n".join(f"line{i}" for i in range(1, 21)), encoding="utf-8")
    result = repo_tools.read_file("long.py", start_line=5, end_line=7)
    assert result["content"].splitlines() == ["line5", "line6", "line7"]
    assert result["total_lines"] == 20


def test_read_past_end_of_file_is_an_error(repo):
    result = repo_tools.read_file("app/main.py", start_line=500)
    assert result["ok"] is False
    assert "past the end" in result["error"]


def test_reading_a_directory_suggests_list_files(repo):
    result = repo_tools.read_file("app")
    assert result["ok"] is False
    assert "list_files" in result["error"]


def test_missing_file_is_reported(repo):
    result = repo_tools.read_file("nope.py")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_clamp_announces_truncation():
    text, truncation = clamp("x" * 100, limit=10)
    assert len(text) == 10
    assert truncation.truncated
    assert "truncated" in truncation.note()


def test_clamp_leaves_short_text_alone():
    text, truncation = clamp("short", limit=100)
    assert text == "short"
    assert truncation.note() == ""


# ---------------------------------------------------------------- searching


def test_search_finds_a_literal_string(repo):
    result = repo_tools.search_code("greet", "*.py")
    assert result["ok"]
    assert result["match_count"] >= 1
    assert any(m["path"] == "app/main.py" for m in result["matches"])


def test_search_is_case_insensitive_by_default(repo):
    assert repo_tools.search_code("GREET", "*.py")["match_count"] >= 1


def test_search_supports_regex(repo):
    result = repo_tools.search_code(r"def \w+\(", "*.py", regex=True)
    assert result["match_count"] >= 2


def test_invalid_regex_is_reported_not_raised(repo):
    result = repo_tools.search_code("[unclosed", regex=True)
    assert result["ok"] is False
    assert "invalid regular expression" in result["error"]


def test_empty_query_is_rejected(repo):
    assert repo_tools.search_code("  ")["ok"] is False


def test_search_skips_vendored_directories(repo):
    result = repo_tools.search_code("x", "*.js")
    assert all("node_modules" not in m["path"] for m in result["matches"])


def test_search_results_are_capped(repo, monkeypatch):
    (repo / "many.py").write_text("needle\n" * 500, encoding="utf-8")
    result = repo_tools.search_code("needle", "*.py")
    assert result["match_count"] <= MAX_SEARCH_MATCHES


# ---------------------------------------------------------------------- git


def test_git_status_is_clean_after_commit(git_repo):
    result = repo_tools.git_status()
    assert result["ok"]
    assert result["clean"]


def test_git_status_reports_a_modification(git_repo):
    (git_repo / "app" / "main.py").write_text("changed\n", encoding="utf-8")
    result = repo_tools.git_status()
    assert not result["clean"]
    assert any(c["path"] == "app/main.py" for c in result["changes"])


def test_git_diff_shows_the_change(git_repo):
    (git_repo / "app" / "util.py").write_text("VALUE = 99\n", encoding="utf-8")
    result = repo_tools.git_diff()
    assert result["ok"]
    assert "VALUE = 99" in result["diff"]
    assert "util.py" in result["summary"]


def test_git_diff_can_be_scoped_to_one_path(git_repo):
    (git_repo / "app" / "util.py").write_text("VALUE = 99\n", encoding="utf-8")
    (git_repo / "README.md").write_text("# Changed\n", encoding="utf-8")
    result = repo_tools.git_diff(path="app/util.py")
    assert "util.py" in result["diff"]
    assert "README" not in result["diff"]


def test_git_diff_rejects_a_path_outside_the_repo(git_repo):
    result = repo_tools.git_diff(path="../../etc/passwd")
    assert result["ok"] is False


def test_git_log_returns_structured_commits(git_repo):
    result = repo_tools.git_log(limit=5)
    assert result["ok"]
    assert result["commits"][0]["subject"] == "initial commit"
    assert result["commits"][0]["hash"]


def test_git_log_limit_is_bounded(git_repo):
    assert repo_tools.git_log(limit=9999)["ok"]


def test_git_tools_explain_a_non_repository(repo):
    result = repo_tools.git_status()
    assert result["ok"] is False
    assert "not a git repository" in result["error"]


# ------------------------------------------------------------------ writing


def test_write_file_creates_nested_directories(repo):
    result = repo_tools.write_file("app/sub/new.py", "x = 1\n")
    assert result["ok"]
    assert (repo / "app" / "sub" / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_write_file_rejects_traversal(repo):
    assert repo_tools.write_file("../escaped.py", "x")["ok"] is False


def test_edit_file_replaces_an_exact_snippet(repo):
    result = repo_tools.edit_file("app/util.py", "VALUE = 42", "VALUE = 43")
    assert result["ok"]
    assert "VALUE = 43" in (repo / "app" / "util.py").read_text(encoding="utf-8")


def test_edit_file_refuses_an_ambiguous_snippet(repo):
    (repo / "dup.py").write_text("a = 1\na = 1\n", encoding="utf-8")
    result = repo_tools.edit_file("dup.py", "a = 1", "a = 2")
    assert result["ok"] is False
    assert "appears 2 times" in result["error"]
    assert (repo / "dup.py").read_text(encoding="utf-8") == "a = 1\na = 1\n"


def test_edit_file_reports_a_missing_snippet(repo):
    result = repo_tools.edit_file("app/util.py", "NOT PRESENT", "x")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_edit_file_requires_old_text(repo):
    assert repo_tools.edit_file("app/util.py", "", "x")["ok"] is False


# ------------------------------------------- confinement, and life without git


def _narrow_to(monkeypatch, subdir):
    """Point the tools at a subdirectory of the same git repository."""
    workspace = Workspace(subdir)
    monkeypatch.setattr(repo_tools, "get_workspace", lambda: workspace)
    return workspace


def test_working_tree_state_distinguishes_two_dirty_trees(git_repo):
    """The whole point: an edit on top of an already-dirty tree must show up."""
    (git_repo / "app" / "main.py").write_text("ambient", encoding="utf-8")
    baseline = repo_tools.working_tree_state()
    assert baseline == repo_tools.working_tree_state(), "must be stable when idle"

    (git_repo / "app" / "util.py").write_text("VALUE = 99", encoding="utf-8")
    (git_repo / "new.py").write_text("print(1)", encoding="utf-8")
    after = repo_tools.working_tree_state()

    touched = {p for p, mark in after.items() if baseline.get(p) != mark}
    assert touched == {"app/util.py", "new.py"}


def test_working_tree_state_sees_a_staged_edit(git_repo):
    """Staging a change must not make it invisible."""
    (git_repo / "app" / "main.py").write_text("staged", encoding="utf-8")
    subprocess.run(["git", "add", "app/main.py"], cwd=git_repo, check=True)
    assert "app/main.py" in repo_tools.working_tree_state()


def test_working_tree_state_falls_back_to_the_filesystem_without_git(repo):
    """Real work happens in directories nobody ran git init in."""
    state = repo_tools.working_tree_state()
    assert state is not None
    assert "app/main.py" in state

    before = dict(state)
    (repo / "app" / "main.py").write_text("edited", encoding="utf-8")
    after = repo_tools.working_tree_state()
    assert after["app/main.py"] != before["app/main.py"]


def test_the_filesystem_fingerprint_ignores_test_artifacts(repo):
    """Running the tests must not itself look like a change to the source."""
    before = repo_tools.working_tree_state()
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "main.cpython-312.pyc").write_text("c", encoding="utf-8")
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "CACHEDIR.TAG").write_text("x", encoding="utf-8")
    assert repo_tools.working_tree_state() == before


def test_working_tree_state_is_none_when_nothing_can_answer(repo, monkeypatch):
    """None, not {} -- a tool failure is not evidence of a clean tree."""

    def unreadable(*a, **k):
        raise OSError("the disk is on fire")

    monkeypatch.setattr(Workspace, "iter_files", unreadable)
    assert repo_tools.working_tree_state() is None


def test_git_status_is_confined_to_the_workspace(git_repo, monkeypatch):
    """A workspace inside a bigger repository must not see the rest of it."""
    (git_repo / "outside.py").write_text("x = 1", encoding="utf-8")
    (git_repo / "app" / "inside.py").write_text("y = 2", encoding="utf-8")

    _narrow_to(monkeypatch, git_repo / "app")
    paths = {c["path"] for c in repo_tools.git_status()["changes"]}
    assert paths == {"inside.py"}, paths


def test_a_confined_git_diff_cannot_read_out_the_parent_project(git_repo, monkeypatch):
    """git_diff would otherwise print files Workspace.resolve forbids reading."""
    (git_repo / "app" / "util.py").write_text("SECRET_ABOVE = 1", encoding="utf-8")
    (git_repo / "tests" / "test_main.py").write_text(
        "SECRET_ASIDE = 2", encoding="utf-8"
    )

    _narrow_to(monkeypatch, git_repo / "tests")
    diff = repo_tools.git_diff()["diff"]
    assert "SECRET_ASIDE" in diff
    assert "SECRET_ABOVE" not in diff


def test_the_fingerprint_is_confined_to_the_workspace(git_repo, monkeypatch):
    """Otherwise a run is credited with edits made where it cannot even read."""
    (git_repo / "app" / "main.py").write_text("changed", encoding="utf-8")
    (git_repo / "tests" / "test_main.py").write_text("changed too", encoding="utf-8")

    _narrow_to(monkeypatch, git_repo / "tests")
    assert set(repo_tools.working_tree_state()) == {"test_main.py"}
