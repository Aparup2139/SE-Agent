"""Safety primitives shared by every repository tool.

Two rules are enforced here rather than in each tool:

* **Every path is confined to the workspace root.** Paths are resolved (which
  collapses ``..`` and follows symlinks) and then checked against the resolved
  root, so neither ``../../etc/passwd`` nor a symlink pointing outside can
  escape.
* **Every output is capped.** Tool results are fed straight back into a prompt,
  so an unbounded ``cat`` of a vendored bundle would blow the context budget.
  Truncation is always announced, never silent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Caps chosen to stay well inside a single prompt while still being useful.
MAX_FILE_BYTES = 200_000
MAX_OUTPUT_CHARS = 30_000
MAX_LISTING_ENTRIES = 500
MAX_SEARCH_MATCHES = 200

# Directories that are never worth reading and would dominate any listing.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".next",
        "target",
        ".gradle",
        "vendor",
        "site-packages",
    }
)

# Files whose contents are secrets, and files that are not text at all.
SECRET_FILENAMES = frozenset(
    {".env", ".env.local", ".env.production", ".npmrc", ".netrc", ".pypirc"}
)
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")
BINARY_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp", ".pdf",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".class",
    ".pyc", ".pyd", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".wav",
    ".sqlite", ".db",
)


class WorkspaceError(Exception):
    """A tool was asked to do something outside its sandbox, or impossible."""


@dataclass(frozen=True)
class Truncation:
    """Records that output was cut, so the tool can say so."""

    truncated: bool
    original_size: int
    shown_size: int

    def note(self, unit: str = "characters") -> str:
        if not self.truncated:
            return ""
        return (
            f"\n\n[truncated: showing {self.shown_size} of {self.original_size} "
            f"{unit}. Narrow the request to see more.]"
        )


def clamp(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, Truncation]:
    """Cut ``text`` to ``limit`` characters, reporting whether it was cut."""
    if len(text) <= limit:
        return text, Truncation(False, len(text), len(text))
    return text[:limit], Truncation(True, len(text), limit)


def is_probably_binary(path: Path) -> bool:
    """Cheap check: known binary suffix, or a NUL byte in the first block."""
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(4096)
    except OSError:
        return False


def is_secret_path(path: Path) -> bool:
    """Files a repository tool must refuse to read back into a prompt."""
    name = path.name.lower()
    if name in SECRET_FILENAMES or name.startswith(".env."):
        return True
    return path.suffix.lower() in SECRET_SUFFIXES


class Workspace:
    """A root directory plus the rules for staying inside it."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        raw = root or os.environ.get("SE_AGENT_TARGET_REPO") or "."
        resolved = Path(raw).expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {resolved}")
        self.root = resolved

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Workspace({str(self.root)!r})"

    # ------------------------------------------------------------ path safety

    def resolve(self, relative_path: str) -> Path:
        """Resolve a caller-supplied path, refusing anything outside the root.

        Resolution happens *before* the containment check so that ``..``
        segments and symlinks are both defeated.
        """
        if relative_path is None:
            raise WorkspaceError("path is required")

        candidate = Path(str(relative_path).strip().strip('"').strip("'"))
        if candidate.is_absolute() or (candidate.drive or ""):
            # Accept an absolute path only if it already points inside the root.
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()

        if not self.contains(resolved):
            raise WorkspaceError(
                f"path escapes the workspace: {relative_path!r} resolves outside "
                f"{self.root}"
            )
        return resolved

    def contains(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        return True

    def relative(self, path: Path) -> str:
        """Workspace-relative, forward-slashed path for display."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:  # pragma: no cover - defensive
            return path.as_posix()

    def is_ignored(self, path: Path) -> bool:
        """True when any component of the path is an ignored directory."""
        return any(part in IGNORED_DIRS for part in self.relative(path).split("/"))

    # -------------------------------------------------------------- file I/O

    def read_text(self, relative_path: str) -> tuple[str, Truncation]:
        """Read a text file, refusing secrets and binaries, capping the size."""
        path = self.resolve(relative_path)

        if not path.exists():
            raise WorkspaceError(f"file not found: {self.relative(path)}")
        if path.is_dir():
            raise WorkspaceError(
                f"{self.relative(path)} is a directory -- use list_files instead"
            )
        if is_secret_path(path):
            raise WorkspaceError(
                f"refusing to read {self.relative(path)}: this file holds "
                "credentials and must never enter a prompt"
            )
        if is_probably_binary(path):
            raise WorkspaceError(
                f"refusing to read {self.relative(path)}: file is binary"
            )

        size = path.stat().st_size
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                content = handle.read(MAX_FILE_BYTES)
        except OSError as exc:
            raise WorkspaceError(f"could not read {self.relative(path)}: {exc}") from exc

        if size > MAX_FILE_BYTES:
            return content, Truncation(True, size, len(content))
        return content, Truncation(False, size, len(content))

    def write_text(self, relative_path: str, content: str) -> Path:
        """Write a text file inside the workspace, creating parent directories."""
        path = self.resolve(relative_path)
        if is_secret_path(path):
            raise WorkspaceError(f"refusing to write to secret file {relative_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def iter_files(self, subdir: str = ".", pattern: str = "*"):
        """Yield non-ignored files under ``subdir``, sorted for stable output."""
        base = self.resolve(subdir)
        if not base.is_dir():
            raise WorkspaceError(f"not a directory: {self.relative(base)}")

        for path in sorted(base.rglob(pattern)):
            if path.is_file() and not self.is_ignored(path):
                yield path


_default: Workspace | None = None


def get_workspace(root: str | None = None) -> Workspace:
    """Process-wide workspace, overridable for tests and alternate repos."""
    global _default
    if root is not None:
        _default = Workspace(root)
    elif _default is None:
        _default = Workspace()
    return _default
