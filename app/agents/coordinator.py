"""Implementation coordinator and final report agent.

The coordinator is the collaborative node of the workflow: it holds the coder's
authority and can consult specialists as tools when a problem genuinely needs
one. Consulting a specialist costs a full model call, so the instruction pushes
hard against reflexive delegation.
"""

from __future__ import annotations

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.agent_tool import AgentTool

from ..schemas import FinalReport, ImplementationPlan
from ..tools import CODING_TOOLS, INSPECTION_TOOLS
from .common import build_agent
from .debugger import build_debugger_agent, build_security_agent, build_testing_agent

COORDINATOR_INSTRUCTION = """\
Your role: Implementation Coordinator. You execute the plan printed below, and
you may consult specialists when a problem is genuinely outside routine
implementation.

The plan is your instruction. It is not context, background, or a suggestion --
it is the complete statement of what this run is for. Work through its tasks in
order and do those things. Nothing else is your job.

Specifically, and these are hard rules:
- Change only files a task names. Reading another file to understand a
  convention is fine; editing one no task names is not.
- Do not survey the repository, audit its state, read its git history, or
  investigate anything a task does not require. If you find yourself running
  `git log`, `git status` or `git diff` to work out what is going on, stop: that
  is not this run's question. Read the files your tasks name and change them.
- Do not look for other work. A repository always has more that could be
  improved; none of it is in the plan.
- If a task cannot be done, name the task, say what blocked it, and stop. Do not
  substitute a task you can do instead, and never conclude that no instructions
  were given -- the plan below is the instruction.
- When every task is done, report and stop.

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
anything blocked with the reason. Report against the plan's task ids, so the
next stage can tell which tasks are actually done.
"""

NO_PLAN = """## Your plan

No plan reached this agent, which should not happen -- both run modes plan
first. Do not improvise a substitute and do not survey the repository. Say
plainly that no plan was provided, and stop.
"""


def _render_plan(state) -> str:
    """The plan, in the prompt, as text a model cannot skim past.

    Session state is not automatically visible to an agent: it arrives only
    through conversation history, where a structured tool result competes with
    every other event and can be summarized away by context compaction. A run
    once spent twenty tool calls on repository archaeology and concluded "no
    additional implementation instructions were provided" while a perfectly good
    plan sat in state. Putting it in the instruction removes the question.
    """
    raw = state.get("plan")
    if not raw:
        return NO_PLAN
    try:
        plan = (
            raw
            if isinstance(raw, ImplementationPlan)
            else ImplementationPlan.model_validate(raw)
        )
    except Exception:  # noqa: BLE001 - an unreadable plan is not a crash
        return NO_PLAN
    if not plan.tasks:
        return NO_PLAN

    lines = ["## Your plan", ""]
    if plan.ordering_rationale:
        lines += [f"Order: {plan.ordering_rationale}", ""]
    for task in plan.tasks:
        lines.append(f"### {task.task_id}: {task.objective}")
        lines.append(f"- Files: {', '.join(task.files) or '(none named)'}")
        if task.dependencies:
            lines.append(f"- After: {', '.join(task.dependencies)}")
        if task.implementation_notes:
            lines.append(f"- Notes: {task.implementation_notes}")
        lines.append(f"- Done when: {task.validation_criteria}")
        lines.append("")
    return "\n".join(lines)


def coordinator_instruction(ctx: ReadonlyContext) -> str:
    """The role, with this run's plan appended."""
    return COORDINATOR_INSTRUCTION + "\n" + _render_plan(ctx.state)


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
        instruction=coordinator_instruction,
        tools=[
            *CODING_TOOLS,
            AgentTool(agent=build_debugger_agent()),
            AgentTool(agent=build_security_agent()),
            AgentTool(agent=build_testing_agent()),
        ],
    )


REPORT_INSTRUCTION = """\
Your role: Reporter. You write the final account of this run for a human.

State holds what the run produced: `requirements`, `architecture`, `plan`,
`tests`, `review`, `files_changed`, and the iteration count. Use `git_diff` and
`git_status` to confirm what actually changed on disk -- report the repository's
state, not the plan's intentions.

`files_changed` in state is authoritative: it was measured, not claimed. If the
git tools return an error -- a directory with no git in it is ordinary -- that
is a missing instrument, not a finding. Never turn it into "no changes were
made": read the files named in the plan and report what they contain now. A
false negative about work that is sitting on disk is as damaging as a false
positive about work that is not.

On a direct run (`mode` is `normal`) there is no requirements, architecture or
review, because those stages were skipped on purpose. Their absence is not a
failure and not a limitation worth listing; judge the run on what it changed
and what the tests said. Do report, under `known_limitations`, that the change
was not independently reviewed.

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
- If no tests were collected at all, say that plainly: the change is unverified.
  "Tests passed" is false when no test ran.
- `known_limitations` is where you are most useful: what a reader would
  otherwise discover the hard way. Assumptions that were made, edge cases not
  handled, cases not covered by tests, shortcuts taken under the iteration
  limit.
- `follow_up` is concrete next actions, not aspirations.

Write the `summary` for someone who did not watch the run: what was asked, what
was built, whether it works, and what to check before merging. A few sentences.
No marketing.

Report on **this** run only. The request and the plan are printed below; they
are the run. A repository can contain a previous task's files, a previous task's
tests, and a previous task's fixes -- none of that is this run's work, and
describing it as such is the worst kind of wrong report, because it reads as a
success. If the plan's tasks and what you see on disk disagree, say so rather
than picking whichever makes a better story.
"""

NO_REQUEST = """## What was asked

The request did not reach this agent. Do not reconstruct one from the
repository's contents -- whatever you find there may predate this run entirely.
Say that the request was unavailable, and report only what you can verify.
"""


def report_instruction(ctx: ReadonlyContext) -> str:
    """The role, with this run's request and plan appended.

    A reporter that is not told what was asked will infer it from whatever the
    repository happens to contain. One did exactly that: handed a directory that
    still held a previous task's files, it reported having fixed bugs in
    `knapsack.py` on a run whose only act was creating `binarytree.py`. The
    request is a fact about the run, so it belongs in the prompt, not in a
    guess.
    """
    request = (ctx.state.get("feature_request") or "").strip()
    asked = f"## What was asked\n\n{request}\n" if request else NO_REQUEST
    return "\n".join([REPORT_INSTRUCTION, asked, _render_plan(ctx.state)])

def build_report_agent():
    """Fresh instance: ADK agents may hold only one parent in a workflow tree."""
    return build_agent(
        name="report_agent",
        description=(
            "Writes the final report: what was built, whether it works, what is "
            "unmet, and what to follow up."
        ),
        instruction=report_instruction,
        tools=INSPECTION_TOOLS,
        output_schema=FinalReport,
        output_key="report",
    )
