# Execution sandbox for repository code.
#
# Agent-written code runs here, never on the host. The image carries only what a
# test run needs; it holds no credentials, and the container that uses it is
# started with no network, a read-only root filesystem, and dropped
# capabilities (see app/tools/sandbox.py).
#
# Build:  docker build -t se-agent-sandbox:latest sandbox/

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# git is needed for diff-based checks; build-essential covers packages with
# native extensions that would otherwise fail to install during a test run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Test tooling is baked in so a sandboxed run needs no network to start.
RUN pip install --no-cache-dir \
        pytest==8.3.4 \
        pytest-asyncio==0.25.0 \
        pytest-cov==6.0.0 \
        pydantic==2.10.4 \
        requests==2.32.3 \
        httpx==0.28.1

# Run as an unprivileged user: agent code must not be able to write outside the
# bind-mounted workspace even if the container is misconfigured.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox

WORKDIR /workspace

CMD ["pytest", "-q"]
