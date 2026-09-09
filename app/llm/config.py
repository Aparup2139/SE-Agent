"""Centralized configuration for the NVIDIA NIM / Kimi K3 backend.

Kimi K3 on NVIDIA NIM is the *only* model backend in SE-Agent. Every agent
resolves its credentials and endpoint through this module so there is exactly
one place where the API key is read and validated.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

DEFAULT_MODEL = "moonshotai/kimi-k3"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Gemini is reached through ADK's own native client, not the OpenAI-compatible
# endpoint, so it needs no base URL here. 2.5 Flash is deliberately not the
# default: Google refuses it to keys created after its retirement ("no longer
# available to new users"), and names 3.6 Flash as the replacement.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

# Providers whose API is reached over the OpenAI-compatible client share one
# config shape; only the defaults and the API-key requirement differ. Gemini
# rides the same config but is served by ADK's native model class.
_PROVIDER_DEFAULTS = {
    "nvidia": {"model": DEFAULT_MODEL, "base_url": DEFAULT_BASE_URL},
    "ollama": {"model": DEFAULT_OLLAMA_MODEL, "base_url": DEFAULT_OLLAMA_BASE_URL},
    "gemini": {"model": DEFAULT_GEMINI_MODEL, "base_url": ""},
}

# Providers that are useless without a key, and so must fail at startup.
_KEYED_PROVIDERS = ("nvidia", "gemini")


class MissingAPIKeyError(RuntimeError):
    """Raised at startup when NVIDIA_API_KEY is absent or blank."""


class NVIDIAConfig(BaseModel):
    """Validated NVIDIA NIM settings.

    The API key is intentionally excluded from ``repr``/``str`` so it can never
    reach a log line or a traceback frame rendering.
    """

    api_key: str = Field(default="", repr=False)
    provider: str = "nvidia"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 300.0
    max_retries: int = 5
    max_tokens: int = 16384
    temperature: float = 1.0
    seed: int = 0
    reasoning_effort: str = "max"

    @field_validator("api_key")
    @classmethod
    def _clean_api_key(cls, value: str) -> str:
        # Tolerate keys pasted with the HTTP scheme still attached
        # (e.g. NVIDIA_API_KEY="Bearer nvapi-...").
        cleaned = value.strip().strip('"').strip("'").strip()
        if cleaned.lower().startswith("bearer "):
            cleaned = cleaned[len("bearer ") :].strip()
        return cleaned

    @model_validator(mode="after")
    def _require_key_for_hosted_providers(self) -> NVIDIAConfig:
        # Ollama and other local/keyless providers don't need one.
        if self.provider in _KEYED_PROVIDERS and not self.api_key:
            raise ValueError(f"an API key is required for provider {self.provider!r}")
        return self

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("max_retries")
    @classmethod
    def _bounded_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("NVIDIA_MAX_RETRIES must be >= 0")
        return min(value, 10)

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def masked_key(self) -> str:
        """A safe-to-log fingerprint of the key."""
        if len(self.api_key) <= 10:
            return "****"
        return f"{self.api_key[:6]}...{self.api_key[-4:]}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"NVIDIAConfig(model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key={self.masked_key()!r})"
        )


def _env_key() -> str | None:
    """Read the API key, accepting a few historical spellings."""
    for name in ("NVIDIA_API_KEY", "NVIDIA_API_KEY_", "Nvidia_API_key", "NVIDIA_APIKEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value
    return None


@lru_cache(maxsize=1)
def get_config() -> NVIDIAConfig:
    """Load and cache configuration, failing loudly on a missing key.

    ``SE_AGENT_LLM_PROVIDER`` selects the backend (``nvidia``, the default,
    ``ollama``, or ``gemini``); an unrecognized value falls back to ``nvidia``
    rather than silently picking one at random.
    """
    provider = (os.environ.get("SE_AGENT_LLM_PROVIDER") or "nvidia").strip().lower()
    if provider not in _PROVIDER_DEFAULTS:
        provider = "nvidia"
    defaults = _PROVIDER_DEFAULTS[provider]

    if provider == "nvidia":
        api_key = _env_key()
        if not api_key:
            raise MissingAPIKeyError(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and add your "
                "NVIDIA NIM key (get one at https://build.nvidia.com/). SE-Agent "
                "cannot run without it."
            )
        model_env, base_url_env = "NVIDIA_MODEL", "NVIDIA_BASE_URL"
    elif provider == "gemini":
        api_key = (
            os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        ).strip()
        if not api_key:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is not set. Create a Gemini API key at "
                "https://aistudio.google.com/apikey and add it to .env. "
                "SE-Agent cannot run without it."
            )
        # ADK builds its own google-genai client and reads the key from the
        # environment, so a key supplied only as GEMINI_API_KEY (or only in
        # .env) has to be published under the name that client looks for.
        os.environ["GOOGLE_API_KEY"] = api_key
        model_env, base_url_env = "GEMINI_MODEL", "GEMINI_BASE_URL"
    else:
        api_key = os.environ.get("OLLAMA_API_KEY") or ""
        model_env, base_url_env = "OLLAMA_MODEL", "OLLAMA_BASE_URL"

    def _float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        try:
            return float(raw) if raw not in (None, "") else default
        except ValueError:
            return default

    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        try:
            return int(raw) if raw not in (None, "") else default
        except ValueError:
            return default

    return NVIDIAConfig(
        api_key=api_key,
        provider=provider,
        model=os.environ.get(model_env) or defaults["model"],
        base_url=os.environ.get(base_url_env) or defaults["base_url"],
        timeout=_float("NVIDIA_TIMEOUT", 300.0),
        max_retries=_int("NVIDIA_MAX_RETRIES", 5),
        max_tokens=_int("NVIDIA_MAX_TOKENS", 16384),
        temperature=_float("NVIDIA_TEMPERATURE", 1.0),
        seed=_int("NVIDIA_SEED", 0),
        reasoning_effort=os.environ.get("NVIDIA_REASONING_EFFORT") or "max",
    )
