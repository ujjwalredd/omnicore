"""OmniCore-Local Gateway Core package.

Implements the Zero-Trust Private Agentic Gateway runtime:
- Stateless Swarm orchestration adapted for local OpenAI-compatible endpoints
- Multi-tier programmatic guardrails (NVIDIA NeMo pattern)
- OpenTelemetry distributed tracing flywheel (Google SRE pattern)
- FastAPI ingress with WebSocket live audit stream
"""

__version__ = "1.0.0"
__all__ = ["swarm_engine", "guardrails", "telemetry", "main"]
