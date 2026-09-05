"""SE-Agent's agents. Every one of them runs on Kimi K3 via NVIDIA NIM.

The only exception is :class:`~app.agents.tester.TesterAgent`, which has no
model at all -- test results must come from a process, never from a prediction.

Agents are exposed as factories rather than singletons. ADK allows an agent
instance only one parent in a workflow tree, and several of these appear in more
than one position (the architect runs in the main pipeline and again when a
review sends the design back), so each site needs its own instance.
"""

from .architecture import build_architecture_agent
from .coder import build_coder_agent
from .common import MODEL, build_agent
from .coordinator import build_coordinator_agent, build_report_agent
from .debugger import (
    build_debugger_agent,
    build_security_agent,
    build_specialists,
    build_testing_agent,
)
from .planner import build_planner_agent
from .repository import build_repository_agent
from .requirements import build_requirements_agent
from .reviewer import build_reviewer_agent
from .risk import build_risk_agent
from .tester import TesterAgent

__all__ = [
    "MODEL",
    "TesterAgent",
    "build_agent",
    "build_architecture_agent",
    "build_coder_agent",
    "build_coordinator_agent",
    "build_debugger_agent",
    "build_planner_agent",
    "build_report_agent",
    "build_repository_agent",
    "build_requirements_agent",
    "build_reviewer_agent",
    "build_risk_agent",
    "build_security_agent",
    "build_specialists",
    "build_testing_agent",
]
