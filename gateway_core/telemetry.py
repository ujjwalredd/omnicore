"""OpenTelemetry span capture + local flywheel log aggregator.

Every Swarm step and MCP tool invocation is wrapped in an explicit OTel span
and additionally serialized as a structured JSON line to `traces_flywheel.json`
in the project root. The Streamlit dashboard tails this file to render the
live audit matrix.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

logger = logging.getLogger("omnicore.telemetry")

TRACE_LOG_PATH = Path(
    os.environ.get(
        "OMNICORE_TRACE_LOG",
        str(Path(__file__).resolve().parent.parent / "traces_flywheel.json"),
    )
)


# ---------------------------------------------------------------------------
# Trace record schema
# ---------------------------------------------------------------------------


@dataclass
class TraceRecord:
    """Structured span record persisted to the flywheel log."""

    timestamp: str
    trace_id: str
    span_id: str
    name: str
    agent: Optional[str] = None
    tool: Optional[str] = None
    status: str = "OK"
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# JSONL file exporter (local flywheel sink)
# ---------------------------------------------------------------------------


class _FlywheelJsonExporter(SpanExporter):
    """Serialize spans as one JSON object per line at TRACE_LOG_PATH."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def export(self, spans) -> SpanExportResult:  # type: ignore[override]
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as fh:
                for span in spans:
                    record = self._span_to_record(span)
                    fh.write(json.dumps(record, default=str) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            logger.exception("flywheel span export failed")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None

    @staticmethod
    def _span_to_record(span) -> dict[str, Any]:
        ctx = span.get_span_context()
        attrs = dict(span.attributes or {})
        status = span.status.status_code.name if span.status else "OK"
        duration_ns = (span.end_time or 0) - (span.start_time or 0)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": f"{ctx.trace_id:032x}",
            "span_id": f"{ctx.span_id:016x}",
            "name": span.name,
            "status": status,
            "latency_ms": round(duration_ns / 1_000_000.0, 3),
            "attributes": attrs,
        }


# ---------------------------------------------------------------------------
# TelemetryManager
# ---------------------------------------------------------------------------


class TelemetryManager:
    """Configures OTel and exposes ergonomic span helpers for the gateway."""

    _initialized: bool = False
    _provider: Optional[TracerProvider] = None
    _tracer = None

    def __init__(
        self,
        service_name: str = "omnicore-local-gateway",
        log_path: Path = TRACE_LOG_PATH,
        console: bool = False,
    ) -> None:
        self.service_name = service_name
        self.log_path = log_path
        self._configure(console=console)
        self._tail_lock = threading.Lock()

    def _configure(self, console: bool) -> None:
        if TelemetryManager._initialized:
            self._tracer = trace.get_tracer(self.service_name)
            return

        resource = Resource.create({"service.name": self.service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(_FlywheelJsonExporter(self.log_path))
        )
        if console:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        TelemetryManager._initialized = True
        TelemetryManager._provider = provider
        self._tracer = trace.get_tracer(self.service_name)

    # ---- span helpers ----------------------------------------------------

    @contextmanager
    def span(
        self,
        name: str,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Iterator[Any]:
        """Context manager wrapping a unit of agentic work in an OTel span."""
        attrs: dict[str, Any] = dict(attributes or {})
        if agent:
            attrs["omnicore.agent"] = agent
        if tool:
            attrs["omnicore.tool"] = tool
        attrs["omnicore.service"] = self.service_name

        start = time.perf_counter()
        tracer = self._tracer or trace.get_tracer(self.service_name)
        with tracer.start_as_current_span(name, attributes=attrs) as span:
            try:
                yield span
                span.set_attribute(
                    "omnicore.latency_ms",
                    round((time.perf_counter() - start) * 1000.0, 3),
                )
            except Exception as exc:
                span.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def record_step(
        self,
        name: str,
        agent: Optional[str],
        tool: Optional[str],
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        status: str = "OK",
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit a non-context-managed trace record (for post-hoc logging)."""
        record = TraceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trace_id="post-hoc",
            span_id="post-hoc",
            name=name,
            agent=agent,
            tool=tool,
            status=status,
            latency_ms=round(latency_ms, 3),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            attributes=extra or {},
            error=error,
        )
        line = json.dumps(asdict(record), default=str)
        try:
            with self._tail_lock, self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            logger.exception("failed to append trace record")

    async def arecord_step(self, *args, **kwargs) -> None:
        """Async wrapper around record_step (offloaded to default executor)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.record_step(*args, **kwargs))

    # ---- log readers (used by dashboard) --------------------------------

    def tail(self, max_records: int = 200) -> list[dict[str, Any]]:
        """Return the most recent N trace records from the flywheel log."""
        if not self.log_path.exists():
            return []
        try:
            with self.log_path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()[-max_records:]
        except Exception:
            logger.exception("failed to tail flywheel log")
            return []
        out: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def shutdown(self) -> None:
        if TelemetryManager._provider is not None:
            try:
                TelemetryManager._provider.shutdown()
            except Exception:
                logger.exception("telemetry shutdown error")


# Module-level default instance (lazy users may import this directly).
TELEMETRY = TelemetryManager()
