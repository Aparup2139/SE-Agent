"""Shared construction rules for SE-Agent's agents.

Every agent in the system runs on Kimi K3. What differs between them is role,
authority (which tools they hold), and the schema they must fill. Centralizing
that here keeps the individual agent modules to their prompt and their contract.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from ..llm.config import DEFAULT_MODEL

# The one model, named once. Agent modules import this rather than a literal.
MODEL = DEFAULT_MODEL

# Prepended to every agent instruction. States the shared rules that hold
# regardless of role -- most importantly, that claims must be grounded in tool
# output rather than assumption.
SHARED_PREAMBLE = """\
You are one agent in SE-Agent, an automated software engineering team.

Rules that apply to you regardless of your role:
- Ground every claim in something you actually observed. If you did not read a
  file, do not describe its contents. If you did not run a command, do not
  report its output.
- Prefer reading the repository over guessing. Tools are cheap; wrong
  assumptions are expensive.
- Be concise. Your output is fed into the next agent's prompt, so every
  unnecessary sentence costs the whole workflow context budget.
- Never invent file paths, function names, test results, or dependencies.
- If something is genuinely unknown, say so explicitly rather than filling the
  gap with a plausible guess.
- Do not reveal your private reasoning. Report conclusions and evidence.
"""

# Appended to agents that must fill a Pydantic schema.
SCHEMA_DISCIPLINE = """\

Fill every field of the required output schema. Respect the length and count
limits: they exist because your output becomes another agent's context. If a
list would exceed its limit, keep the most important entries rather than
truncating arbitrarily.
"""


def build_agent(
    *,
    name: str,
    description: str,
    instruction: str,
    tools: list | None = None,
    output_schema: type | None = None,
    output_key: str | None = None,
    sub_agents: list | None = None,
    **kwargs,
) -> LlmAgent:
    """Create an ADK ``LlmAgent`` backed by Kimi K3.

    Args:
        name: Agent id, used in events and for delegation.
        description: What this agent is for. Other agents read this when
            deciding whether to delegate, so it must be accurate.
        instruction: The role-specific prompt, appended to the shared preamble.
        tools: The agent's authority. Omitted entirely for agents that should
            not touch the repository.
        output_schema: Pydantic model the agent must fill. ADK reconciles this
            with tools automatically via its structured-output tool.
        output_key: Session-state key the validated output is written to.
        sub_agents: Agents this one may delegate to.
    """
    full_instruction = SHARED_PREAMBLE + "\n" + instruction
    if output_schema is not None:
        full_instruction += SCHEMA_DISCIPLINE

    return LlmAgent(
        name=name,
        model=MODEL,
        description=description,
        instruction=full_instruction,
        tools=tools or [],
        output_schema=output_schema,
        output_key=output_key,
        sub_agents=sub_agents or [],
        **kwargs,
    )
