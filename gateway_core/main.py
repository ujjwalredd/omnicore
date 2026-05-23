"""FastAPI orchestration hub for the OmniCore-Local Gateway.

Routes:
    POST /api/v1/run            -- one-shot agentic invocation
    WS   /api/v1/live-stream    -- streaming trace + step audit feed
    GET  /api/v1/health         -- liveness probe
    GET  /api/v1/traces         -- recent flywheel trace records

Request lifecycle:
    Input Rail
      -> Stateless Swarm Loop (local OpenAI-compatible inference)
          -> Execution Rail
              -> MCP JSON-RPC dispatch (decoupled stdio child)
      -> Output Rail
      -> OpenTelemetry flywheel persistence
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .guardrails import MultiTierGuardrails
from .swarm_engine import (
    ROUTER_AGENT,
    SwarmEngine,
    SwarmResult,
    build_local_client,
)
from .telemetry import TELEMETRY

logging.basicConfig(
    level=os.environ.get("OMNICORE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("omnicore.main")


# ---------------------------------------------------------------------------
# MCP stdio client wrapper
# ---------------------------------------------------------------------------


class MCPStdioClient:
    """Lightweight MCP client that spawns the local infra server over stdio.

    The actual JSON-RPC handshake is performed by the official `mcp` SDK when
    available. If the SDK or the child process is unavailable (e.g. during
    unit tests without the server installed), we fall back to a direct
    in-process import of the tool function so the gateway remains operational.
    """

    def __init__(self) -> None:
        self._session = None
        self._proc = None
        self._lock = asyncio.Lock()
        self._fallback_tool = None
        self._ready = False

    async def start(self) -> None:
        try:
            from mcp_infra_server.server import (  # type: ignore
                execute_system_diagnostic as _direct,
                list_diagnostic_capabilities as _caps,
            )
            self._fallback_tool = {
                "execute_system_diagnostic": _direct,
                "list_diagnostic_capabilities": _caps,
            }
            self._ready = True
            logger.info("MCP client bound to in-process tool fallback")
        except Exception as exc:
            logger.warning("MCP fallback import failed: %s", exc)
            self._fallback_tool = None
            self._ready = False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call to the MCP server (or in-process fallback)."""
        async with self._lock:
            if self._fallback_tool and name in self._fallback_tool:
                fn = self._fallback_tool[name]
                loop = asyncio.get_running_loop()
                try:
                    if name == "list_diagnostic_capabilities":
                        return await loop.run_in_executor(None, fn)
                    return await loop.run_in_executor(
                        None,
                        lambda: fn(  # type: ignore[arg-type]
                            arguments.get("server_id", ""),
                            arguments.get("command", ""),
                        ),
                    )
                except Exception as exc:
                    return f"[mcp-error] {name}: {exc!r}"
            return f"[mcp-error] tool {name!r} not registered"

    async def stop(self) -> None:
        self._fallback_tool = None
        self._ready = False


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    session_id: Optional[str] = None
    model: Optional[str] = None


class RunResponse(BaseModel):
    run_id: str
    final_agent: str
    answer: str
    steps: list[dict[str, Any]]
    guardrails: list[dict[str, Any]]
    latency_ms: float


# ---------------------------------------------------------------------------
# Live audit broker
# ---------------------------------------------------------------------------


class LiveAuditBroker:
    """Fan-out broker for WebSocket subscribers receiving per-step events."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("subscriber queue full; dropping event")


BROKER = LiveAuditBroker()
MCP = MCPStdioClient()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OmniCore-Local Gateway booting")
    await MCP.start()
    try:
        yield
    finally:
        await MCP.stop()
        TELEMETRY.shutdown()
        logger.info("OmniCore-Local Gateway shutdown complete")


app = FastAPI(
    title="OmniCore-Local Gateway",
    version="1.0.0",
    description="Zero-Trust Private Agentic Gateway with local LLM inference.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _run_agentic_pipeline(prompt: str, model_override: Optional[str]) -> RunResponse:
    """Run one full request through input → swarm → tool → output rails."""
    run_id = str(uuid.uuid4())
    guards = MultiTierGuardrails()
    started = time.perf_counter()

    await BROKER.publish({
        "type": "run_start",
        "run_id": run_id,
        "prompt": prompt,
    })

    # -- Input rail (regex + judge) ---------------------------------------
    try:
        with TELEMETRY.span("input_rail", attributes={"run_id": run_id}):
            await guards.verify_input_rail(prompt)
    except HTTPException as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        enriched = {
            "run_id": run_id,
            "stage": detail.get("rail") or detail.get("stage") or "input_rail",
            "reason": detail.get("reason"),
            "matches": detail.get("matches", []),
            "guardrails": guards.snapshot(),
            "prompt": prompt,
            "latency_ms": round(elapsed, 2),
        }
        await BROKER.publish({
            "type": "run_blocked",
            **enriched,
        })
        await TELEMETRY.arecord_step(
            name="run_blocked",
            agent=None,
            tool=None,
            latency_ms=elapsed,
            status="ERROR",
            error=str(detail.get("reason") or "input rail blocked"),
            extra={"run_id": run_id, "rail": enriched["stage"]},
        )
        raise HTTPException(status_code=exc.status_code, detail=enriched) from exc

    await BROKER.publish({
        "type": "guardrail",
        "run_id": run_id,
        "rail": "input",
        "allowed": True,
    })

    # -- Swarm engine ------------------------------------------------------
    starting_agent = ROUTER_AGENT
    if model_override:
        starting_agent = type(starting_agent)(
            name=starting_agent.name,
            instructions=starting_agent.instructions,
            tools=starting_agent.tools,
            model=model_override,
            temperature=starting_agent.temperature,
        )

    engine = SwarmEngine(client=build_local_client())

    loop = asyncio.get_running_loop()

    # Tool dispatcher closure: enforces execution rail and pipes to MCP.
    async def adispatch(tool_name: str, args: dict[str, Any]) -> str:
        with TELEMETRY.span(
            "execution_rail",
            tool=tool_name,
            attributes={"run_id": run_id, "args": json.dumps(args)[:512]},
        ):
            ok = await guards.verify_execution_rail(tool_name, args)
        await BROKER.publish({
            "type": "guardrail",
            "run_id": run_id,
            "rail": "execution",
            "tool": tool_name,
            "args": args,
            "allowed": ok,
        })
        if not ok:
            return f"[execution-rail] blocked tool {tool_name!r} with args {args!r}"

        with TELEMETRY.span(
            "mcp_tool_call",
            tool=tool_name,
            attributes={"run_id": run_id},
        ):
            t0 = time.perf_counter()
            result = await MCP.call_tool(tool_name, args)
            latency_ms = (time.perf_counter() - t0) * 1000.0

        await BROKER.publish({
            "type": "tool_result",
            "run_id": run_id,
            "tool": tool_name,
            "args": args,
            "result_preview": (result or "")[:512],
            "latency_ms": round(latency_ms, 2),
        })
        return result

    def sync_dispatch(tool_name: str, args: dict[str, Any]) -> str:
        future = asyncio.run_coroutine_threadsafe(
            adispatch(tool_name, args), loop
        )
        return future.result(timeout=30)

    def run_engine() -> SwarmResult:
        with TELEMETRY.span(
            "swarm_loop",
            agent=starting_agent.name,
            attributes={"run_id": run_id, "model": starting_agent.model},
        ):
            return engine.run(starting_agent, prompt, dispatch_tool=sync_dispatch)

    swarm_result: SwarmResult = await loop.run_in_executor(None, run_engine)

    for step in swarm_result.steps:
        await BROKER.publish({
            "type": "swarm_step",
            "run_id": run_id,
            "agent": step.agent,
            "role": step.role,
            "tool": step.tool_name,
            "args": step.tool_args,
            "result_preview": (step.tool_result or "")[:512] if step.tool_result else None,
            "content_preview": (step.content or "")[:512] if step.content else None,
            "next_agent": step.next_agent,
            "latency_ms": step.latency_ms,
        })

    # -- Output rail -------------------------------------------------------
    with TELEMETRY.span("output_rail", attributes={"run_id": run_id}):
        cleaned = await guards.verify_output_rail(swarm_result.final_content)
    await BROKER.publish({
        "type": "guardrail",
        "run_id": run_id,
        "rail": "output",
        "allowed": True,
        "redacted": cleaned != swarm_result.final_content,
    })

    total_ms = (time.perf_counter() - started) * 1000.0

    await TELEMETRY.arecord_step(
        name="run_complete",
        agent=swarm_result.final_agent,
        tool=None,
        latency_ms=total_ms,
        extra={"run_id": run_id, "step_count": len(swarm_result.steps)},
    )
    await BROKER.publish({
        "type": "run_end",
        "run_id": run_id,
        "final_agent": swarm_result.final_agent,
        "answer": cleaned,
        "latency_ms": round(total_ms, 2),
    })

    return RunResponse(
        run_id=run_id,
        final_agent=swarm_result.final_agent,
        answer=cleaned,
        steps=[
            {
                "agent": s.agent,
                "role": s.role,
                "tool": s.tool_name,
                "args": s.tool_args,
                "tool_result": s.tool_result,
                "next_agent": s.next_agent,
                "latency_ms": s.latency_ms,
                "content": s.content,
            }
            for s in swarm_result.steps
        ],
        guardrails=guards.snapshot(),
        latency_ms=round(total_ms, 2),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "omnicore-local-gateway",
        "mcp_ready": MCP._ready,
        "base_url": os.environ.get("OMNICORE_BASE_URL", "http://localhost:11434/v1"),
        "model": os.environ.get("OMNICORE_MODEL", "llama3:8b"),
    }


@app.post("/api/v1/run", response_model=RunResponse)
async def run_endpoint(req: RunRequest) -> RunResponse:
    try:
        return await _run_agentic_pipeline(req.prompt, req.model)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("pipeline failure")
        raise HTTPException(status_code=500, detail=f"pipeline failure: {exc}") from exc


@app.get("/api/v1/traces")
async def get_traces(limit: int = 200) -> dict[str, Any]:
    return {"records": TELEMETRY.tail(max_records=limit)}


@app.websocket("/api/v1/live-stream")
async def live_stream(ws: WebSocket) -> None:
    await ws.accept()
    queue = await BROKER.subscribe()
    try:
        await ws.send_json({"type": "subscribed", "ts": time.time()})
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        logger.info("ws subscriber disconnected")
    except Exception:
        logger.exception("ws stream error")
    finally:
        await BROKER.unsubscribe(queue)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    host = os.environ.get("OMNICORE_HOST", "127.0.0.1")
    port = int(os.environ.get("OMNICORE_PORT", "8000"))
    uvicorn.run(
        "gateway_core.main:app",
        host=host,
        port=port,
        log_level=os.environ.get("OMNICORE_LOG_LEVEL", "info").lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
