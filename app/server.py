"""FastAPI server exposing SE-Agent runs as a live event stream.

A run takes minutes, so the only sensible interface is streaming: the client
posts a feature request and receives server-sent events as each phase, tool
call, test result and review lands. It is the same
:func:`~app.agent.stream_run` the CLI consumes, rendered as JSON.

Run it with::

    uvicorn app.server:app --reload

Security
--------
This endpoint starts an agent that **writes files and executes code**. Two
controls bound that authority, both configured by environment variable:

``SE_AGENT_REPO_ROOT``
    The only directory tree a request may target. Without it, the ``repo``
    field is refused outright and every run uses the server's own working
    directory -- a caller cannot point the agent at an arbitrary host path.

``SE_AGENT_API_TOKEN``
    A bearer token required on every run request. Without it the server binds
    to localhost only by convention and refuses to accept a token-free request
    from a non-loopback client.

Neither default is "open": an unconfigured server is usable locally and
inert to anyone else.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import RunResult, preflight, stream_run
from .telemetry import PHASE_LABELS, PHASE_ORDER, Update
from .workflows.graph import DEFAULT_MAX_ITERATIONS, ITERATION_CEILING

logger = logging.getLogger(__name__)

LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}

app = FastAPI(
    title="SE-Agent",
    description="Multi-agent software engineering on ADK 2.8.0 and Kimi K3.",
    version="0.1.0",
)


class RunRequest(BaseModel):
    request: str = Field(min_length=1, max_length=8000)
    repo: str | None = None
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS, ge=1, le=ITERATION_CEILING
    )


def authorize(request: Request, authorization: str | None) -> str:
    """Authenticate the caller and return the identity to attribute the run to.

    A configured token is required from everyone. With no token configured, only
    loopback callers are served -- otherwise starting the server on 0.0.0.0
    would silently hand code execution to the network.
    """
    expected = (os.environ.get("SE_AGENT_API_TOKEN") or "").strip()

    if expected:
        supplied = ""
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        # Constant-time comparison: a token check must not leak its prefix.
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid or missing token")
        return f"token:{expected[:6]}"

    host = (request.client.host if request.client else "") or ""
    if host not in LOOPBACK:
        raise HTTPException(
            status_code=403,
            detail=(
                "this server executes code and is unauthenticated. Set "
                "SE_AGENT_API_TOKEN to accept requests from outside localhost."
            ),
        )
    return f"local:{host or 'unknown'}"


def resolve_repo(requested: str | None) -> str | None:
    """Confine the target repository to the configured root.

    Without ``SE_AGENT_REPO_ROOT`` a caller may not choose a directory at all:
    an HTTP client picking an arbitrary host path for an agent that writes files
    and runs commands is exactly the capability this must not hand out.
    """
    if not requested:
        return None

    root = (os.environ.get("SE_AGENT_REPO_ROOT") or "").strip()
    if not root:
        raise HTTPException(
            status_code=403,
            detail=(
                "choosing a repository over HTTP requires SE_AGENT_REPO_ROOT to "
                "be set; without it the server only works in its own directory"
            ),
        )

    try:
        allowed = Path(root).expanduser().resolve(strict=True)
        target = Path(requested).expanduser()
        target = (allowed / target if not target.is_absolute() else target).resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"unusable path: {exc}") from exc

    if target != allowed and allowed not in target.parents:
        raise HTTPException(
            status_code=403, detail="repository path is outside SE_AGENT_REPO_ROOT"
        )
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="repository path is not a directory")
    return str(target)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Report whether a run could actually start right now."""
    problems = preflight()
    return {"ready": not problems, "problems": problems}


@app.get("/api/phases")
async def phases() -> dict[str, Any]:
    """The phase list, so a client can render the checklist before starting."""
    return {
        "phases": [
            {"id": phase.value, "label": PHASE_LABELS[phase]} for phase in PHASE_ORDER
        ]
    }


@app.post("/api/run")
async def run(
    body: RunRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """Start a run and stream its progress as server-sent events.

    Event types: ``phase``, ``agent``, ``tool``, ``tests``, ``review``,
    ``message``, ``error``, and finally ``result``.

    Authentication and path confinement are resolved *before* the stream opens,
    so a rejection is an ordinary 401/403 rather than an error buried in a
    200 response body.
    """
    identity = authorize(request, authorization)
    repo = resolve_repo(body.repo)

    async def events():
        problems = preflight(repo)
        if problems:
            yield _sse("error", {"text": "; ".join(problems), "fatal": True})
            return

        try:
            async for item in stream_run(
                body.request, repo=repo, limit=body.max_iterations, user_id=identity
            ):
                if isinstance(item, RunResult):
                    report = item.report
                    yield _sse(
                        "result",
                        {
                            "session_id": item.session_id,
                            "succeeded": item.succeeded,
                            "tests_passed": item.tests_passed,
                            "review_approved": item.review_approved,
                            "iterations": item.state.get("iteration", 0),
                            "tool_calls": item.tracker.tool_calls,
                            "checklist": item.tracker.checklist(),
                            "errors": item.tracker.errors,
                            "report": report.model_dump() if report else None,
                        },
                    )
                elif isinstance(item, Update):
                    payload = asdict(item)
                    payload["phase"] = item.phase.value if item.phase else None
                    yield _sse(item.kind, payload)

                # Let the event loop flush to the client between updates.
                await asyncio.sleep(0)
        except Exception as exc:  # noqa: BLE001 - the client needs to hear about it
            logger.exception("run failed")
            yield _sse("error", {"text": f"{type(exc).__name__}: {exc}", "fatal": True})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


CONSOLE = Path(__file__).parent / "static" / "console.html"


@app.get("/")
async def index() -> FileResponse:
    """A single-file console, so the server is usable without a build step."""
    return FileResponse(CONSOLE, media_type="text/html")
