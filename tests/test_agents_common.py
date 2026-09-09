"""build_agent must route to whichever provider/model is currently configured."""

from __future__ import annotations

import pytest

from app.agents.common import build_agent
from app.llm.config import get_config


@pytest.fixture(autouse=True)
def clear_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_build_agent_uses_nvidia_model_by_default(monkeypatch):
    monkeypatch.delenv("SE_AGENT_LLM_PROVIDER", raising=False)
    agent = build_agent(name="x", description="d", instruction="i")
    assert agent.model == "moonshotai/kimi-k3"


def test_build_agent_uses_ollama_model_when_configured(monkeypatch):
    monkeypatch.setenv("SE_AGENT_LLM_PROVIDER", "ollama")
    agent = build_agent(name="x", description="d", instruction="i")
    assert agent.model == "qwen3:4b"
