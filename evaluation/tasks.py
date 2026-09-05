"""Evaluation tasks: realistic feature requests with objective success checks.

Each task carries its own verification. Whether SE-Agent succeeded is decided by
running the resulting code, not by reading its report -- the whole point is to
measure the system rather than take its word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class Task:
    """One evaluation task."""

    id: str
    fixture: str
    request: str
    # Python executed inside the sandbox after the run. It must exit non-zero on
    # failure, so success is a real signal rather than a self-report.
    verification: str
    difficulty: str = "medium"
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def fixture_path(self) -> Path:
        return FIXTURES / self.fixture


TASKS: list[Task] = [
    Task(
        id="add-sentence-count",
        fixture="textstats",
        difficulty="easy",
        tags=["feature", "pure-function"],
        request=(
            "Add a sentence_count(text) function to textstats that counts "
            "sentences terminated by '.', '!' or '?'. Consecutive terminators "
            "like '!!!' count as one sentence. Export it from the package and "
            "cover it with tests."
        ),
        verification="""
from textstats import sentence_count

assert sentence_count("One. Two. Three.") == 3, "basic sentences"
assert sentence_count("Wow!!! Really?") == 2, "consecutive terminators are one"
assert sentence_count("") == 0, "empty text"
assert sentence_count("No terminator") == 1, "trailing text counts"
print("OK")
""",
        notes="Straightforward addition. Tests the happy path end to end.",
    ),
    Task(
        id="fix-char-count-tabs",
        fixture="textstats",
        difficulty="medium",
        tags=["bugfix", "regression"],
        request=(
            "char_count(text, include_spaces=False) only strips spaces, so tabs "
            "and newlines are still counted as characters. It should exclude all "
            "whitespace. Fix it without breaking the existing behaviour, and add "
            "a regression test."
        ),
        verification="""
from textstats import char_count

assert char_count("a\\tb\\nc", include_spaces=False) == 3, "all whitespace excluded"
assert char_count("ab cd", include_spaces=False) == 4, "spaces still excluded"
assert char_count("ab cd") == 5, "default behaviour unchanged"
print("OK")
""",
        notes=(
            "A real bug with an existing passing test that must keep passing. "
            "Catches an agent that fixes by weakening the old test."
        ),
    ),
    Task(
        id="add-word-frequencies",
        fixture="textstats",
        difficulty="medium",
        tags=["feature", "api-design"],
        request=(
            "Add word_frequencies(text) returning a dict mapping each lowercased "
            "word to how many times it appears. Strip surrounding punctuation "
            "from words. Export it and add tests."
        ),
        verification="""
from textstats import word_frequencies

result = word_frequencies("The cat, the CAT!")
assert result.get("the") == 2, f"case-insensitive count, got {result}"
assert result.get("cat") == 2, f"punctuation stripped, got {result}"
assert word_frequencies("") == {}, "empty text"
print("OK")
""",
        notes="Requires a judgement call on punctuation handling.",
    ),
]

TASKS_BY_ID = {task.id: task for task in TASKS}


def get_tasks(ids: list[str] | None = None) -> list[Task]:
    """Select tasks by id, or all of them."""
    if not ids:
        return list(TASKS)
    missing = set(ids) - set(TASKS_BY_ID)
    if missing:
        raise KeyError(f"unknown task id(s): {', '.join(sorted(missing))}")
    return [TASKS_BY_ID[i] for i in ids]
