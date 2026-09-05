"""NVIDIA NIM / Kimi K3 provider layer -- the only LLM backend in SE-Agent."""

from .adk_model import KimiK3Llm
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MissingAPIKeyError,
    NVIDIAConfig,
    get_config,
)
from .nvidia_kimi import (
    Completion,
    NVIDIAAuthError,
    NVIDIAError,
    NVIDIAKimiClient,
    NVIDIARateLimitError,
    NVIDIAResponseError,
    NVIDIAServerError,
    NVIDIATimeoutError,
    ToolCall,
    get_client,
)

__all__ = [
    "Completion",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MissingAPIKeyError",
    "NVIDIAAuthError",
    "NVIDIAConfig",
    "NVIDIAError",
    "NVIDIAKimiClient",
    "NVIDIARateLimitError",
    "NVIDIAResponseError",
    "NVIDIAServerError",
    "NVIDIATimeoutError",
    "KimiK3Llm",
    "ToolCall",
    "get_client",
    "get_config",
]
