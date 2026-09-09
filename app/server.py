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
    Confines the directory tree a request may target. Without it, a caller on
    loopback -- the person who started the server, with the reach the CLI
    already gives them -- may pick any directory on the machine, and a remote
    caller may pick none at all.

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
from .schemas import RunMode
from .telemetry import PHASE_LABELS, Update, phase_order
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
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, le=ITERATION_CEILING)
    mode: RunMode = RunMode.THINKING


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


def is_local(request: Request) -> bool:
    """Whether the caller is the machine's own user rather than the network."""
    return ((request.client.host if request.client else "") or "") in LOOPBACK


def resolve_repo(requested: str | None, *, local: bool = False) -> str | None:
    """Resolve the target repository, confined to the configured root if any.

    A loopback caller is the person who started the server, and so may pick any
    directory on the machine -- the same reach the CLI already has. A remote
    caller may not: handing an arbitrary host path to an agent that writes files
    and runs commands is exactly the capability this must not give the network,
    so ``SE_AGENT_REPO_ROOT`` must be set and the path must live inside it.
    """
    if not requested:
        return None

    root = (os.environ.get("SE_AGENT_REPO_ROOT") or "").strip()
    if not root:
        if not local:
            raise HTTPException(
                status_code=403,
                detail=(
                    "choosing a repository from off this machine requires "
                    "SE_AGENT_REPO_ROOT to be set"
                ),
            )
        try:
            target = Path(requested).expanduser().resolve()
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"unusable path: {exc}") from exc
        if not target.is_dir():
            raise HTTPException(status_code=400, detail="repository path is not a directory")
        return str(target)

    try:
        allowed = Path(root).expanduser().resolve(strict=True)
        target = Path(requested).expanduser()
        target = (allowed / target if not target.is_absolute() else target).resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"unusable path: {exc}") from exc

    if target != allowed and allowed not in target.parents:
        raise HTTPException(status_code=403, detail="repository path is outside SE_AGENT_REPO_ROOT")
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
async def phases(mode: RunMode = RunMode.THINKING) -> dict[str, Any]:
    """The phase list for one mode, so a client can render the checklist early.

    The two modes have different checklists, so the client asks again when the
    user switches rather than showing rows a normal run will never reach.
    """
    return {
        "mode": mode.value,
        "phases": [
            {"id": phase.value, "label": PHASE_LABELS[phase]}
            for phase in phase_order(mode)
        ],
    }


@app.get("/api/dirs")
async def dirs(
    request: Request,
    path: str | None = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """List the subdirectories of one directory, so a client can pick a repo.

    Reach is ``resolve_repo``'s: the whole machine for a loopback caller,
    nothing outside ``SE_AGENT_REPO_ROOT`` for anyone else. An empty ``path``
    means the top -- the machine's drives, or the configured root.

    ``entries`` are ready to navigate to as-is, and ``up`` is the parent or
    null at the top, so a client never has to join paths itself.
    """
    authorize(request, authorization)
    local = is_local(request)
    confined = (os.environ.get("SE_AGENT_REPO_ROOT") or "").strip()

    if not confined and not local:
        raise HTTPException(
            status_code=403,
            detail="browsing this machine from off it requires SE_AGENT_REPO_ROOT",
        )

    if not path and not confined:
        drives = getattr(os, "listdrives", lambda: ["/"])()
        return {"path": "", "up": None, "entries": _entries(drives)}

    target = Path(resolve_repo(path or ".", local=local))
    if confined:
        top = Path(confined).expanduser().resolve()
        # "" would mean the drive list, which the confined caller may not see.
        up = None if target == top else str(target.parent)
    else:
        up = "" if target == target.parent else str(target.parent)
    return {
        "path": str(target),
        "up": up,
        "entries": _entries(_subdirectories(target)),
    }


def _subdirectories(target: Path) -> list[str]:
    """Immediate subdirectories, skipping dotted and unreadable ones."""
    found = []
    try:
        children = list(target.iterdir())
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"unreadable: {exc}") from exc
    for child in children:
        try:
            if child.is_dir() and not child.name.startswith("."):
                found.append(str(child))
        except OSError:  # a path we may not stat is not a path we can offer
            continue
    return sorted(found, key=str.casefold)


def _entries(paths: list[str]) -> list[dict[str, str]]:
    return [{"value": p, "label": Path(p).name or p} for p in paths]


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
    repo = resolve_repo(body.repo, local=is_local(request))

    async def events():
        problems = preflight(repo)
        if problems:
            yield _sse("error", {"text": "; ".join(problems), "fatal": True})
            return

        try:
            async for item in stream_run(
                body.request,
                repo=repo,
                limit=body.max_iterations,
                mode=body.mode,
                user_id=identity,
            ):
                if isinstance(item, RunResult):
                    report = item.report
                    yield _sse(
                        "result",
                        {
                            "session_id": item.session_id,
                            "mode": item.mode.value,
                            "succeeded": item.succeeded,
                            "tests_passed": item.tests_passed,
                            "tests_inconclusive": bool(
                                item.state.get("tests_inconclusive")
                            ),
                            "no_change_needed": bool(
                                item.state.get("no_change_needed")
                            ),
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
