"""Shared test setup.

Unit tests must never depend on a real credential, so a placeholder key is
installed when the environment has none. Tests marked ``integration`` do need a
real key and are skipped without one.
"""

from __future__ import annotations

import os

import pytest

_PLACEHOLDER = "nvapi-test-placeholder-key"


def pytest_configure(config):
    if not os.environ.get("NVIDIA_API_KEY", "").strip():
        os.environ["NVIDIA_API_KEY"] = _PLACEHOLDER


def has_real_api_key() -> bool:
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    return bool(key) and key != _PLACEHOLDER


requires_api_key = pytest.mark.skipif(
    not has_real_api_key(), reason="NVIDIA_API_KEY not configured"
)
