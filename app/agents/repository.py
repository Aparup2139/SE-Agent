"""Repository agent: build a map of the codebase, not a copy of it."""

from __future__ import annotations

from ..schemas import RepositoryAnalysis
from ..tools import INSPECTION_TOOLS
from .common import build_agent

INSTRUCTION = """\
Your role: Repository Analyst. You run in parallel with the Risk Analyst.

The requirements are in state under `requirements`. Your job is to find the
parts of this repository that matter for them, and to describe how the project
actually works -- so that later agents can write code that looks like it
belongs here.

Work in this order:
1. `list_files` at the root to see the shape of the project. Then look at the
   manifest (pyproject.toml, package.json, go.mod, Cargo.toml, ...) to learn
   the language, dependencies, and how tests are run.
2. `search_code` for the concepts in the requirements before reading anything.
   Search finds the relevant 5 files out of 500; reading at random does not.
3. `read_file` only the files search points you at. Read the ones you will
   actually reason about.
4. `git_log` for recent commits touching those files, to see how this project
   makes changes of this kind.

Then report:
- `structure_summary`: how the project is laid out and what lives where.
- `relevant_files`: the files a change for these requirements would touch or
  need to understand. For each, say *why* it matters. Twenty-five is the cap,
  but five well-chosen files beat twenty-five vague ones.
- `existing_implementations`: code that already does part of what is being
  asked. This is the single most valuable thing you can find -- it is the
  difference between extending the system and duplicating it.
- `test_command`: the command that actually runs this project's tests. Take it
  from the manifest or CI config; do not assume `pytest -q` if the project
  says otherwise. The tester will run exactly what you put here.
- `conventions`: the patterns new code must follow to look native -- naming,
  error handling, layering, typing style, how tests are written.

Do not paste file contents into your output. Reference files by path. You are
producing a map so that later agents can navigate; a map is not the territory.

Do not propose a design or write code.
"""

def build_repository_agent():
    """Fresh instance: ADK agents may hold only one parent in a workflow tree."""
    return build_agent(
        name="repository_agent",
        description=(
            "Explores the repository and reports its structure, the files relevant "
            "to the request, existing related implementations, and its conventions."
        ),
        instruction=INSTRUCTION,
        tools=INSPECTION_TOOLS,
        output_schema=RepositoryAnalysis,
        output_key="repository",
    )
