"""OmniCore-Local Streamlit dashboard.

A dual-panel live audit UI:

    Left  : chat panel — message history + prompt input
    Right : engineering audit matrix — auto-refreshing OTel trace records,
            Swarm transitions, guardrail decisions, raw JSON-RPC payloads.

Launch:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st

API_BASE = os.environ.get("OMNICORE_API_BASE", "http://127.0.0.1:8000")
TRACE_PATH = Path(
    os.environ.get(
        "OMNICORE_TRACE_LOG",
        str(Path(__file__).resolve().parent.parent / "traces_flywheel.json"),
    )
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OmniCore-Local Gateway",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for failure / blocked rows.
st.markdown(
    """
    <style>
    .blocked   { color: #ff5252; font-weight: 600; }
    .ok        { color: #2ecc71; font-weight: 600; }
    .agent-tag { background: #1f2937; color: #93c5fd; padding: 2px 8px;
                 border-radius: 4px; font-family: monospace; }
    .tool-tag  { background: #111827; color: #fbbf24; padding: 2px 8px;
                 border-radius: 4px; font-family: monospace; }
    .latency   { color: #94a3b8; font-family: monospace; }
    div[data-testid="stHorizontalBlock"] { gap: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "chat" not in st.session_state:
    st.session_state.chat: list[dict[str, Any]] = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("OmniCore-Local")
    st.caption("Zero-Trust Private Agentic Gateway")

    st.subheader("Inference target")
    st.code(
        os.environ.get("OMNICORE_BASE_URL", "http://localhost:11434/v1"),
        language="bash",
    )
    st.code(os.environ.get("OMNICORE_MODEL", "llama3:8b"), language="bash")

    st.subheader("Gateway API")
    st.code(API_BASE, language="bash")

    try:
        health = requests.get(f"{API_BASE}/api/v1/health", timeout=2).json()
        st.success(f"Gateway: {health.get('status', 'unknown')}")
        st.json(health)
    except Exception as exc:
        st.error(f"Gateway unreachable: {exc}")

    refresh_secs = st.slider("Audit refresh (s)", 1, 10, 2)
    max_records = st.slider("Audit window (records)", 20, 500, 150, step=10)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.session_state.last_run = None
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛰️ OmniCore-Local Gateway")
st.caption(
    "Anthropic MCP · OpenAI Swarm · NVIDIA Guardrails · Google OTel — all local."
)


# ---------------------------------------------------------------------------
# Layout: split viewport
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1])


# ---------------------------------------------------------------------------
# Left: chat panel
# ---------------------------------------------------------------------------

with left:
    st.subheader("Operator Console")
    chat_box = st.container(height=520, border=True)

    with chat_box:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("meta"):
                    st.caption(msg["meta"])

    user_prompt = st.chat_input("Ask OmniCore about your infrastructure...")

    if user_prompt:
        st.session_state.chat.append({"role": "user", "content": user_prompt})
        with st.spinner("Routing through gateway..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/run",
                    json={"prompt": user_prompt},
                    timeout=120,
                )
                if resp.status_code != 200:
                    # Parse FastAPI error body to populate audit tabs even
                    # when the gateway rejected the prompt at a rail.
                    try:
                        body = resp.json()
                    except Exception:
                        body = {"detail": resp.text}
                    detail = body.get("detail", body)
                    if isinstance(detail, str):
                        rail = "unknown"
                        reason = detail
                        guardrails = []
                        matches = []
                        latency_ms = 0
                    else:
                        rail = detail.get("stage") or detail.get("rail") or "unknown"
                        reason = detail.get("reason", "blocked")
                        guardrails = detail.get("guardrails", []) or []
                        matches = detail.get("matches", []) or []
                        latency_ms = detail.get("latency_ms", 0)

                    # Synthesize a "blocked run" so Last Run / Guardrails /
                    # Raw JSON-RPC tabs render meaningful data on rejection.
                    st.session_state.last_run = {
                        "run_id": detail.get("run_id", "blocked"),
                        "final_agent": f"BLOCKED@{rail}",
                        "answer": f"Blocked by rail `{rail}`: {reason}",
                        "latency_ms": latency_ms,
                        "blocked": True,
                        "blocked_rail": rail,
                        "blocked_reason": reason,
                        "matches": matches,
                        "steps": [{
                            "agent": "gateway",
                            "role": "blocked",
                            "tool": None,
                            "args": None,
                            "tool_result": None,
                            "next_agent": None,
                            "latency_ms": latency_ms,
                            "content": (
                                f"Rail `{rail}` rejected this prompt.\n"
                                f"Reason: {reason}\n"
                                f"Matches: {matches}"
                            ),
                        }],
                        "guardrails": guardrails,
                        "http_status": resp.status_code,
                        "raw_detail": detail,
                    }
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": (
                            f"❌ Blocked at rail **{rail}** — {reason}"
                            + (f"\n\nMatches: `{matches}`" if matches else "")
                        ),
                        "meta": f"HTTP {resp.status_code} · rail={rail}",
                    })
                else:
                    data = resp.json()
                    data["blocked"] = False
                    st.session_state.last_run = data
                    answer = data.get("answer", "(empty)")
                    meta = (
                        f"agent={data.get('final_agent')} · "
                        f"latency={data.get('latency_ms')} ms · "
                        f"steps={len(data.get('steps', []))}"
                    )
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": meta,
                    })
            except Exception as exc:
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": f"❌ Local request failed: `{exc}`",
                })
        st.rerun()


# ---------------------------------------------------------------------------
# Right: engineering audit matrix
# ---------------------------------------------------------------------------


def _read_trace_log(limit: int) -> list[dict[str, Any]]:
    if not TRACE_PATH.exists():
        return []
    try:
        with TRACE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
    except Exception:
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


def _render_step(step: dict[str, Any]) -> None:
    agent = step.get("agent") or "?"
    role = step.get("role") or "?"
    tool = step.get("tool") or ""
    latency = step.get("latency_ms", 0)

    header_bits = [
        f"<span class='agent-tag'>{agent}</span>",
        f"<b>{role}</b>",
    ]
    if tool:
        header_bits.append(f"<span class='tool-tag'>{tool}</span>")
    header_bits.append(f"<span class='latency'>{latency:.1f} ms</span>")
    st.markdown(" ".join(header_bits), unsafe_allow_html=True)

    if step.get("args"):
        st.code(json.dumps(step["args"], indent=2), language="json")
    if step.get("tool_result"):
        with st.expander("tool result", expanded=False):
            st.code(step["tool_result"], language="bash")
    if step.get("content"):
        with st.expander("assistant content", expanded=False):
            st.text(step["content"])
    if step.get("next_agent"):
        st.info(f"↪ handoff to **{step['next_agent']}**")


def _render_guardrails(decisions: list[dict[str, Any]]) -> None:
    for d in decisions:
        css = "ok" if d.get("allowed") else "blocked"
        label = "ALLOW" if d.get("allowed") else "BLOCK"
        st.markdown(
            f"<span class='{css}'>[{label}]</span> "
            f"<b>{d.get('rail')}</b> — {d.get('reason')}",
            unsafe_allow_html=True,
        )
        if d.get("matches"):
            with st.expander("matched patterns", expanded=False):
                for m in d["matches"]:
                    st.code(m)


with right:
    st.subheader("Engineering Audit Matrix")

    tabs = st.tabs([
        "Last Run",
        "Trace Flywheel",
        "Guardrail Decisions",
        "Raw JSON-RPC",
    ])

    # ----- Last run -------------------------------------------------------
    with tabs[0]:
        if st.session_state.last_run is None:
            st.info("No run executed yet. Send a prompt to populate the matrix.")
        else:
            run = st.session_state.last_run
            blocked = run.get("blocked")

            if blocked:
                st.error(
                    f"🛑 **BLOCKED** at rail `{run.get('blocked_rail')}` — "
                    f"{run.get('blocked_reason')}"
                )

            cols = st.columns(3)
            cols[0].metric("Run ID", str(run.get("run_id", ""))[:8])
            cols[1].metric("Final agent", run.get("final_agent", "?"))
            latency = run.get("latency_ms", 0) or 0
            cols[2].metric("Total latency (ms)", f"{float(latency):.0f}")

            if blocked and run.get("matches"):
                st.markdown("**Matched patterns**")
                for m in run["matches"]:
                    st.code(m)

            st.markdown("**Swarm steps**" if not blocked else "**Rejection trace**")
            for idx, step in enumerate(run.get("steps", []), start=1):
                with st.container(border=True):
                    st.caption(f"Step {idx}")
                    _render_step(step)

            if blocked and run.get("raw_detail"):
                with st.expander("Raw 4xx response body", expanded=False):
                    st.json(run["raw_detail"])

    # ----- Live trace log ------------------------------------------------
    with tabs[1]:
        st.caption(f"Tailing `{TRACE_PATH}` (auto-refresh {refresh_secs}s)")
        records = _read_trace_log(max_records)
        if not records:
            st.info("No trace records yet.")
        else:
            for rec in reversed(records[-50:]):
                status = rec.get("status", "OK")
                css = "blocked" if status not in {"OK", "UNSET"} else "ok"
                attrs = rec.get("attributes", {}) or {}
                with st.container(border=True):
                    st.markdown(
                        f"<span class='{css}'>[{status}]</span> "
                        f"<b>{rec.get('name')}</b> "
                        f"<span class='latency'>{rec.get('latency_ms', 0):.1f} ms</span> "
                        f"<span class='latency'>{rec.get('timestamp', '')}</span>",
                        unsafe_allow_html=True,
                    )
                    if attrs:
                        with st.expander("attributes", expanded=False):
                            st.json(attrs)

    # ----- Guardrails ----------------------------------------------------
    with tabs[2]:
        run = st.session_state.last_run
        if not run or not run.get("guardrails"):
            st.info("No guardrail decisions for the current session yet.")
        else:
            _render_guardrails(run["guardrails"])

    # ----- Raw JSON-RPC --------------------------------------------------
    with tabs[3]:
        run = st.session_state.last_run
        if not run:
            st.info("Trigger a tool-using prompt to see MCP payloads.")
        else:
            tool_steps = [
                s for s in run.get("steps", [])
                if s.get("role") == "tool" and s.get("tool")
            ]
            if not tool_steps:
                st.info("No MCP tool calls in the last run.")
            for s in tool_steps:
                with st.container(border=True):
                    st.markdown(
                        f"<span class='tool-tag'>{s.get('tool')}</span> "
                        f"<span class='latency'>{s.get('latency_ms', 0):.1f} ms</span>",
                        unsafe_allow_html=True,
                    )
                    payload = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": s.get("tool"),
                            "arguments": s.get("args", {}),
                        },
                    }
                    st.markdown("**Outbound JSON-RPC**")
                    st.code(json.dumps(payload, indent=2), language="json")
                    st.markdown("**Inbound result**")
                    st.code(s.get("tool_result", ""), language="bash")


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

time.sleep(refresh_secs)
st.rerun()
