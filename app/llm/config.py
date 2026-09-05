"""Centralized configuration for the NVIDIA NIM / Kimi K3 backend.

Kimi K3 on NVIDIA NIM is the *only* model backend in SE-Agent. Every agent
resolves its credentials and endpoint through this module so there is exactly
one place where the API key is read and validated.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

DEFAULT_MODEL = "moonshotai/kimi-k3"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class MissingAPIKeyError(RuntimeError):
    """Raised at startup when NVIDIA_API_KEY is absent or blank."""


class NVIDIAConfig(BaseModel):
    """Validated NVIDIA NIM settings.

    The API key is intentionally excluded from ``repr``/``str`` so it can never
    reach a log line or a traceback frame rendering.
    """

    api_key: str = Field(repr=False)
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
        if not cleaned:
            raise ValueError("NVIDIA_API_KEY is empty")
        return cleaned

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
    """Load and cache configuration, failing loudly on a missing key."""
    api_key = _env_key()
    if not api_key:
        raise MissingAPIKeyError(
            "NVIDIA_API_KEY is not set. Copy .env.example to .env and add your "
            "NVIDIA NIM key (get one at https://build.nvidia.com/). SE-Agent "
            "cannot run without it."
        )

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
        model=os.environ.get("NVIDIA_MODEL") or DEFAULT_MODEL,
        base_url=os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL,
        timeout=_float("NVIDIA_TIMEOUT", 300.0),
        max_retries=_int("NVIDIA_MAX_RETRIES", 5),
        max_tokens=_int("NVIDIA_MAX_TOKENS", 16384),
        temperature=_float("NVIDIA_TEMPERATURE", 1.0),
        seed=_int("NVIDIA_SEED", 0),
        reasoning_effort=os.environ.get("NVIDIA_REASONING_EFFORT") or "max",
    )
