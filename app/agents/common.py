"""Shared construction rules for SE-Agent's agents.

Every agent in the system runs on the same configured model. What differs
between them is role, authority (which tools they hold), and the schema they
must fill. Centralizing that here keeps the individual agent modules to their
prompt and their contract.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.base_llm import BaseLlm

from ..llm.config import get_config
from ..llm.gemini_model import build_gemini

# None means "ask get_config() for the live model", so a provider switch
# (SE_AGENT_LLM_PROVIDER) takes effect with no code change. Tests set this to
# a stub model name to bypass the real provider entirely.
MODEL: str | None = None

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


def resolve_model(agent: str) -> str | BaseLlm:
    """The model one agent runs on: a name ADK resolves, or an instance.

    A name is enough for the OpenAI-compatible providers -- ADK's registry
    matches it to ``KimiK3Llm``, and one model serves every agent. Gemini is
    handed over as a configured instance instead: retries, schema fixes and
    the per-agent model choice all have to be attached to something, and a
    bare name would get ADK's untouched defaults.
    """
    if MODEL is not None:
        return MODEL
    config = get_config()
    if config.provider != "gemini":
        return config.model
    return build_gemini(agent)


def build_agent(
    *,
    name: str,
    description: str,
    instruction: str | Callable[[ReadonlyContext], str],
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
            May be a callable taking ADK's ``ReadonlyContext``, for an agent
            whose prompt must *carry* state -- the plan it has to execute, say --
            rather than hope to find it in the conversation history.
        tools: The agent's authority. Omitted entirely for agents that should
            not touch the repository.
        output_schema: Pydantic model the agent must fill. ADK reconciles this
            with tools automatically via its structured-output tool.
        output_key: Session-state key the validated output is written to.
        sub_agents: Agents this one may delegate to.
    """
    full_instruction: Any
    if callable(instruction):
        # ADK skips its own state templating for a provider, so the provider
        # itself is responsible for everything the prompt needs to contain.
        async def full_instruction(ctx: ReadonlyContext) -> str:
            role = instruction(ctx)
            if inspect.isawaitable(role):
                role = await role
            text = SHARED_PREAMBLE + "\n" + role
            return text + SCHEMA_DISCIPLINE if output_schema is not None else text

    else:
        full_instruction = SHARED_PREAMBLE + "\n" + instruction
        if output_schema is not None:
            full_instruction += SCHEMA_DISCIPLINE

    return LlmAgent(
        name=name,
        model=resolve_model(name),
        description=description,
        instruction=full_instruction,
        tools=tools or [],
        output_schema=output_schema,
        output_key=output_key,
        sub_agents=sub_agents or [],
        **kwargs,
    )
