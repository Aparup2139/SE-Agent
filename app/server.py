"""FastAPI server exposing SE-Agent runs as a live event stream.

A run takes minutes, so the only sensible interface is streaming: the client
posts a feature request and receives server-sent events as each phase, tool
call, test result and review lands. It is the same
:func:`~app.agent.stream_run` the CLI consumes, rendered as JSON.

Run it with::

    uvicorn app.server:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import RunResult, preflight, stream_run
from .telemetry import PHASE_LABELS, PHASE_ORDER, Update
from .workflows.graph import DEFAULT_MAX_ITERATIONS, ITERATION_CEILING

logger = logging.getLogger(__name__)

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
async def run(body: RunRequest) -> StreamingResponse:
    """Start a run and stream its progress as server-sent events.

    Event types: ``phase``, ``agent``, ``tool``, ``tests``, ``review``,
    ``message``, ``error``, and finally ``result``.
    """

    async def events():
        problems = preflight(body.repo)
        if problems:
            yield _sse("error", {"text": "; ".join(problems), "fatal": True})
            return

        try:
            async for item in stream_run(
                body.request, repo=body.repo, limit=body.max_iterations
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
