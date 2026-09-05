"""Implementation coordinator and final report agent.

The coordinator is the collaborative node of the workflow: it holds the coder's
authority and can consult specialists as tools when a problem genuinely needs
one. Consulting a specialist costs a full model call, so the instruction pushes
hard against reflexive delegation.
"""

from __future__ import annotations

from google.adk.tools.agent_tool import AgentTool

from ..schemas import FinalReport
from ..tools import CODING_TOOLS, INSPECTION_TOOLS
from .common import build_agent
from .debugger import build_debugger_agent, build_security_agent, build_testing_agent

COORDINATOR_INSTRUCTION = """\
Your role: Implementation Coordinator. You implement the plan, and you may
consult specialists when a problem is genuinely outside routine implementation.

You have the full coding toolset and the same obligations as the engineer:
read before you write, match the project's conventions, run the tests, inspect
the diff, never weaken a test to make it pass, fix root causes.

You also have three specialists available as tools:
- `debugger_agent`: consult when tests fail and the cause is not obvious after
  you have read the failure and the code yourself.
- `security_agent`: consult when the change touches authentication,
  authorization, untrusted input, secrets, or a network boundary.
- `testing_agent`: consult when you are unsure what to test, or when tests are
  flaky or passing for the wrong reason.

When to consult, and when not to:
- Consult when you have actually tried and are stuck, or when the question is
  squarely in a specialist's domain and getting it wrong is expensive.
- Do **not** consult to check work you are already confident in. Do not consult
  all three because the change seems important. Each consultation costs a model
  call and adds to everyone's context; the default is to do the work yourself.
- When you do consult, ask a specific question and include what you already
  tried. "Tests fail" wastes the call. "test_rate_limit asserts a 60s
  retry_after but gets None; I checked the header is present in the response --
  where is it being dropped?" gets an answer.

Act on what a specialist tells you, or explain why you did not. A consultation
you ignore was a wasted call.

Report concisely at the end: tasks completed, files changed, test results, and
anything blocked with the reason.
"""

def build_coordinator_agent():
    """Fresh instance: ADK agents may hold only one parent in a workflow tree.

    The specialists are wrapped as tools rather than sub-agents so consulting
    one returns control here, instead of handing the rest of the run over.
    """
    return build_agent(
        name="coordinator_agent",
        description=(
            "Implements the plan and delegates to debugging, security and testing "
            "specialists when a problem requires one."
        ),
        instruction=COORDINATOR_INSTRUCTION,
        tools=[
            *CODING_TOOLS,
            AgentTool(agent=build_debugger_agent()),
            AgentTool(agent=build_security_agent()),
            AgentTool(agent=build_testing_agent()),
        ],
    )


REPORT_INSTRUCTION = """\
Your role: Reporter. You write the final account of this run for a human.

State holds everything: `requirements`, `architecture`, `plan`, `tests`,
`review`, `files_changed`, and the iteration count. Use `git_diff` and
`git_status` to confirm what actually changed on disk -- report the repository's
state, not the plan's intentions.

Be accurate above all. This report is how a human decides whether to trust and
merge the work, so an over-favourable summary is worse than useless.

Set `outcome` honestly:
- `completed`: every acceptance criterion is met, tests genuinely passed, and
  the review approved.
- `partial`: real progress, but something is unmet, untested, or unapproved.
  This is the correct outcome when the iteration limit was reached.
- `failed`: the work does not function or nothing meaningful was achieved.

Rules:
- `requirements_met` lists only criteria you can point to evidence for. If the
  review flagged it missed, it is not met.
- If tests did not pass, or could not run, say so in the summary. Never describe
  a run with failing tests as successful.
- `known_limitations` is where you are most useful: what a reader would
  otherwise discover the hard way. Assumptions that were made, edge cases not
  handled, cases not covered by tests, shortcuts taken under the iteration
  limit.
- `follow_up` is concrete next actions, not aspirations.

Write the `summary` for someone who did not watch the run: what was asked, what
was built, whether it works, and what to check before merging. A few sentences.
No marketing.
"""

def build_report_agent():
    """Fresh instance: ADK agents may hold only one parent in a workflow tree."""
    return build_agent(
        name="report_agent",
        description=(
            "Writes the final report: what was built, whether it works, what is "
            "unmet, and what to follow up."
        ),
        instruction=REPORT_INSTRUCTION,
        tools=INSPECTION_TOOLS,
        output_schema=FinalReport,
        output_key="report",
    )
