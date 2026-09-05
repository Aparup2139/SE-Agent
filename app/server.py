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
from fastapi.responses import HTMLResponse, StreamingResponse
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


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """A minimal console, so the server is usable without a build step."""
    return _CONSOLE_HTML


_CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SE-Agent</title>
<style>
  :root { color-scheme: light dark; --bg:#fbfbfa; --fg:#1a1a18; --muted:#6b6b63;
          --line:#e2e2dc; --accent:#2f6f4f; --warn:#a33; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --fg:#e8e8e3; --muted:#9a9a92; --line:#2c2c33;
            --accent:#7fc8a0; --warn:#e08c8c; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing:-0.01em; }
  .sub { color: var(--muted); margin: 0 0 28px; }
  form { display: grid; gap: 12px; margin-bottom: 28px; }
  textarea, input {
    width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px;
    background:transparent; color:inherit; font:inherit;
  }
  textarea { min-height: 92px; resize: vertical; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
  .row label { color:var(--muted); font-size:13px; }
  button { padding:9px 18px; border:0; border-radius:8px; background:var(--accent);
           color:#fff; font:inherit; font-weight:600; cursor:pointer; }
  button[disabled] { opacity:.5; cursor:not-allowed; }
  .grid { display:grid; grid-template-columns: 260px 1fr; gap:24px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  .panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.07em;
              color:var(--muted); margin:0 0 10px; }
  ol { list-style:none; margin:0; padding:0; }
  ol li { padding:5px 0; color:var(--muted); }
  ol li.done { color:var(--fg); }
  ol li.active { color:var(--accent); font-weight:600; }
  #log { height:60vh; overflow-y:auto; font-family:ui-monospace, monospace;
         font-size:12.5px; }
  .e { padding:3px 0; border-bottom:1px solid var(--line); white-space:pre-wrap;
       word-break:break-word; }
  .e:last-child { border-bottom:0; }
  .k { color:var(--muted); margin-right:8px; }
  .tool .k { color:#7a7ac0; }
  .tests .k, .review .k { color:var(--accent); }
  .error { color:var(--warn); }
</style>
</head>
<body>
<div class="wrap">
  <h1>SE-Agent</h1>
  <p class="sub">Requirements to reviewed change, with tests actually run in Docker.</p>

  <form id="f">
    <textarea id="req" placeholder="Describe the change you want..."></textarea>
    <div class="row">
      <input id="repo" placeholder="repository path (optional)" style="flex:1;min-width:220px">
      <label>iterations
        <input id="iters" type="number" value="3" min="1" max="10" style="width:64px">
      </label>
      <button id="go" type="submit">Run</button>
    </div>
  </form>

  <div class="grid">
    <div class="panel">
      <h2>Progress</h2>
      <ol id="phases"></ol>
    </div>
    <div class="panel">
      <h2>Activity</h2>
      <div id="log"></div>
    </div>
  </div>
</div>
<script>
const log = document.getElementById('log');
const phasesEl = document.getElementById('phases');
let phases = [], current = null, done = new Set();

fetch('/api/phases').then(r => r.json()).then(d => { phases = d.phases; renderPhases(); });

function renderPhases() {
  phasesEl.innerHTML = phases.map(p => {
    const cls = p.id === current ? 'active' : (done.has(p.id) ? 'done' : '');
    const mark = p.id === current ? '>' : (done.has(p.id) ? '\\u2713' : '\\u00b7');
    return `<li class="${cls}">${mark} ${p.label}</li>`;
  }).join('');
}

function add(kind, text) {
  const el = document.createElement('div');
  el.className = 'e ' + kind;
  el.innerHTML = `<span class="k">${kind}</span>${escapeHtml(text)}`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

document.getElementById('f').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const btn = document.getElementById('go');
  btn.disabled = true;
  log.innerHTML = ''; done = new Set(); current = null; renderPhases();

  const res = await fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      request: document.getElementById('req').value,
      repo: document.getElementById('repo').value || null,
      max_iterations: Number(document.getElementById('iters').value)
    })
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done: fin} = await reader.read();
    if (fin) break;
    buf += decoder.decode(value, {stream: true});
    const chunks = buf.split('\\n\\n');
    buf = chunks.pop();
    for (const chunk of chunks) {
      const ev = /event: (.*)/.exec(chunk);
      const dt = /data: ([\\s\\S]*)/.exec(chunk);
      if (!ev || !dt) continue;
      let d; try { d = JSON.parse(dt[1]); } catch { continue; }
      handle(ev[1], d);
    }
  }
  btn.disabled = false;
});

function handle(kind, d) {
  if (kind === 'phase') {
    if (current) done.add(current);
    current = d.phase; renderPhases();
    add('phase', d.text);
  } else if (kind === 'result') {
    if (current) done.add(current);
    current = null; renderPhases();
    add('result', `${d.succeeded ? 'SUCCEEDED' : 'DID NOT SUCCEED'} - tests ` +
      `${d.tests_passed ? 'passed' : 'not passing'}, review ` +
      `${d.review_approved ? 'approved' : 'not approved'}, ` +
      `${d.iterations} iteration(s), ${d.tool_calls} tool call(s)`);
    if (d.report) add('report', d.report.summary || '');
  } else {
    add(kind, d.text || '');
  }
}
</script>
</body>
</html>
"""
