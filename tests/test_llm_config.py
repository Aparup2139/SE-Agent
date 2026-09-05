"""Configuration loading, key hygiene, and startup validation."""

from __future__ import annotations

import pytest

from app.llm.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MissingAPIKeyError,
    NVIDIAConfig,
    get_config,
)

ENV_NAMES = (
    "NVIDIA_API_KEY",
    "NVIDIA_API_KEY_",
    "Nvidia_API_key",
    "NVIDIA_APIKEY",
    "NVIDIA_MODEL",
    "NVIDIA_BASE_URL",
    "NVIDIA_TIMEOUT",
    "NVIDIA_MAX_RETRIES",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    get_config.cache_clear()
    yield monkeypatch
    get_config.cache_clear()


def test_missing_key_raises_actionable_error(clean_env):
    with pytest.raises(MissingAPIKeyError, match="NVIDIA_API_KEY is not set"):
        get_config()


def test_blank_key_is_treated_as_missing(clean_env):
    clean_env.setenv("NVIDIA_API_KEY", "   ")
    with pytest.raises(MissingAPIKeyError):
        get_config()


def test_bearer_prefix_and_quotes_are_stripped():
    config = NVIDIAConfig(api_key='"Bearer nvapi-abc123"')
    assert config.api_key == "nvapi-abc123"


def test_plain_key_is_unchanged():
    assert NVIDIAConfig(api_key="nvapi-abc123").api_key == "nvapi-abc123"


def test_api_key_never_appears_in_repr_or_str():
    config = NVIDIAConfig(api_key="nvapi-supersecretvalue123")
    assert "supersecret" not in repr(config)
    assert "supersecret" not in str(config)
    assert "nvapi-" in config.masked_key()


def test_masked_key_hides_the_middle():
    masked = NVIDIAConfig(api_key="nvapi-0123456789abcdef").masked_key()
    assert masked.startswith("nvapi-")
    assert masked.endswith("cdef")
    assert "0123456789" not in masked


def test_short_key_is_fully_masked():
    assert NVIDIAConfig(api_key="abc").masked_key() == "****"


def test_defaults_point_at_nim_and_kimi(clean_env):
    clean_env.setenv("NVIDIA_API_KEY", "nvapi-test")
    config = get_config()
    assert config.model == DEFAULT_MODEL == "moonshotai/kimi-k3"
    assert config.base_url == DEFAULT_BASE_URL
    assert config.chat_completions_url.endswith("/v1/chat/completions")


def test_trailing_slash_is_normalized():
    config = NVIDIAConfig(api_key="k", base_url="https://example.com/v1/")
    assert config.chat_completions_url == "https://example.com/v1/chat/completions"


def test_malformed_numeric_env_falls_back_to_default(clean_env):
    clean_env.setenv("NVIDIA_API_KEY", "nvapi-test")
    clean_env.setenv("NVIDIA_TIMEOUT", "not-a-number")
    assert get_config().timeout == 300.0


def test_retries_are_bounded():
    assert NVIDIAConfig(api_key="k", max_retries=999).max_retries == 10
    with pytest.raises(ValueError):
        NVIDIAConfig(api_key="k", max_retries=-1)
