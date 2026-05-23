"""Continuous-optimization data flywheel.

Scans completed agent transcripts, scores each run, and exports low-quality
runs to a fine-tuning JSONL dataset for future local-model improvement.

Selection priority formula (lower-is-better => higher priority):
    Selection Priority = 1.0 - (0.4 * Accuracy + 0.6 * OperationalSuccess)

A run is harvested for the tuning dataset when its priority exceeds the
configured threshold (default 0.5).

Run as a script:
    python -m tests.test_flywheel

Or via pytest:
    pytest tests/test_flywheel.py -v
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_LOG = Path(
    os.environ.get(
        "OMNICORE_TRACE_LOG", str(REPO_ROOT / "traces_flywheel.json")
    )
)
TRANSCRIPT_DIR = Path(
    os.environ.get(
        "OMNICORE_TRANSCRIPTS", str(REPO_ROOT / "flywheel_data" / "transcripts")
    )
)
TUNING_DATASET = Path(
    os.environ.get(
        "OMNICORE_TUNING_PATH",
        str(REPO_ROOT / "flywheel_data" / "tuning_dataset.json"),
    )
)
SELECTION_THRESHOLD = float(os.environ.get("OMNICORE_FLYWHEEL_THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# Transcript record
# ---------------------------------------------------------------------------


@dataclass
class Transcript:
    run_id: str
    user_prompt: str
    final_answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    guardrails: list[dict[str, Any]] = field(default_factory=list)
    final_agent: str = ""
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


_ERROR_HINTS = (
    "[execution-rail]", "[mcp-error]", "[engine-error]", "[tool-error]",
    "Traceback", "NoneType",
)


def accuracy_score(t: Transcript) -> float:
    """Heuristic accuracy estimate on [0, 1].

    Penalizes empty answers, redacted output, and final answers that simply
    echo error markers from the runtime. A real deployment would replace this
    with a learned judge or human-labeled ground truth.
    """
    answer = (t.final_answer or "").strip()
    if not answer:
        return 0.0
    if "[REDACTED]" in answer:
        return 0.2
    if any(h in answer for h in _ERROR_HINTS):
        return 0.1
    if len(answer) < 20:
        return 0.4
    if re.search(r"\{\{.*?\}\}", answer):
        return 0.2
    return 0.9


def operational_success_score(t: Transcript) -> float:
    """How cleanly the runtime executed on [0, 1]."""
    if not t.steps:
        return 0.3
    blocked = sum(
        1 for g in t.guardrails if not g.get("allowed", True)
    )
    tool_errors = sum(
        1 for s in t.steps
        if (s.get("tool_result") or "").startswith(("[mcp-error]", "[execution-rail]", "[tool-error]"))
    )
    handoffs = sum(1 for s in t.steps if s.get("role") == "handoff")

    score = 1.0
    score -= 0.25 * blocked
    score -= 0.35 * tool_errors
    if handoffs > 3:
        score -= 0.1 * (handoffs - 3)
    if t.latency_ms > 30_000:
        score -= 0.1
    return max(0.0, min(1.0, score))


def selection_priority(t: Transcript) -> float:
    """`1.0 - (0.4 * accuracy + 0.6 * operational_success)` — higher = pick."""
    acc = accuracy_score(t)
    ops = operational_success_score(t)
    return round(1.0 - (0.4 * acc + 0.6 * ops), 4)


# ---------------------------------------------------------------------------
# Transcript loaders
# ---------------------------------------------------------------------------


def load_transcripts_from_dir(directory: Path) -> list[Transcript]:
    """Load transcripts saved as JSON files by the gateway runtime."""
    if not directory.exists():
        return []
    out: list[Transcript] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(Transcript(
            run_id=data.get("run_id", path.stem),
            user_prompt=data.get("prompt", ""),
            final_answer=data.get("answer", ""),
            steps=data.get("steps", []) or [],
            guardrails=data.get("guardrails", []) or [],
            final_agent=data.get("final_agent", ""),
            latency_ms=float(data.get("latency_ms", 0.0)),
        ))
    return out


def load_transcripts_from_flywheel(path: Path) -> list[Transcript]:
    """Reconstruct transcripts by grouping flywheel trace lines by run_id."""
    if not path.exists():
        return []

    by_run: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                attrs = rec.get("attributes", {}) or {}
                run_id = attrs.get("run_id") or attrs.get("omnicore.run_id")
                if not run_id:
                    continue
                bucket = by_run.setdefault(run_id, {
                    "run_id": run_id,
                    "prompt": "",
                    "answer": "",
                    "steps": [],
                    "guardrails": [],
                    "final_agent": attrs.get("omnicore.agent", ""),
                    "latency_ms": 0.0,
                })
                bucket["latency_ms"] += float(rec.get("latency_ms", 0.0))
                bucket["steps"].append({
                    "name": rec.get("name"),
                    "status": rec.get("status"),
                    "latency_ms": rec.get("latency_ms"),
                    "attrs": attrs,
                })
    except Exception:
        return []

    return [
        Transcript(
            run_id=b["run_id"],
            user_prompt=b["prompt"],
            final_answer=b["answer"],
            steps=b["steps"],
            guardrails=b["guardrails"],
            final_agent=b["final_agent"],
            latency_ms=b["latency_ms"],
        )
        for b in by_run.values()
    ]


# ---------------------------------------------------------------------------
# Cleaning & export
# ---------------------------------------------------------------------------


def clean_prompt(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:4000]


def clean_response(text: str) -> str:
    text = (text or "").strip()
    for marker in _ERROR_HINTS:
        text = text.replace(marker, "")
    text = re.sub(r"\[REDACTED\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:4000]


def to_tuning_record(t: Transcript) -> dict[str, Any]:
    """Map a low-quality transcript into a supervised fine-tuning record."""
    return {
        "run_id": t.run_id,
        "priority": selection_priority(t),
        "accuracy": accuracy_score(t),
        "operational_success": operational_success_score(t),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are OmniCore-Local, a private on-prem infrastructure "
                    "assistant. Answer concisely and call tools only when needed."
                ),
            },
            {"role": "user", "content": clean_prompt(t.user_prompt)},
            {"role": "assistant", "content": clean_response(t.final_answer)},
        ],
        "instruction": clean_prompt(t.user_prompt),
        "response": clean_response(t.final_answer),
        "metadata": {
            "final_agent": t.final_agent,
            "step_count": len(t.steps),
            "latency_ms": t.latency_ms,
        },
    }


def harvest(
    transcripts: Iterable[Transcript],
    threshold: float = SELECTION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Select transcripts whose selection priority meets the threshold."""
    harvested: list[dict[str, Any]] = []
    for t in transcripts:
        prio = selection_priority(t)
        if prio >= threshold:
            harvested.append(to_tuning_record(t))
    return harvested


def append_to_dataset(records: list[dict[str, Any]], path: Path = TUNING_DATASET) -> int:
    """Append harvested records to the tuning dataset JSONL file."""
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def run_flywheel_pipeline() -> dict[str, Any]:
    """End-to-end harvest: read transcripts, score, export, return summary."""
    transcripts = load_transcripts_from_dir(TRANSCRIPT_DIR)
    if not transcripts:
        transcripts = load_transcripts_from_flywheel(TRACE_LOG)

    harvested = harvest(transcripts)
    written = append_to_dataset(harvested)

    return {
        "scanned": len(transcripts),
        "harvested": written,
        "threshold": SELECTION_THRESHOLD,
        "tuning_path": str(TUNING_DATASET),
    }


# ---------------------------------------------------------------------------
# pytest cases
# ---------------------------------------------------------------------------


def _failing_transcript() -> Transcript:
    return Transcript(
        run_id="fail-001",
        user_prompt="check uptime on edge-node-7",
        final_answer="[execution-rail] blocked tool 'execute_system_diagnostic' ...",
        steps=[
            {"role": "tool", "tool": "execute_system_diagnostic",
             "tool_result": "[execution-rail] blocked ..."}
        ],
        guardrails=[
            {"rail": "execution", "allowed": False,
             "reason": "forbidden token", "matches": ["sudo"]}
        ],
        final_agent="system_admin_agent",
        latency_ms=1234.0,
    )


def _passing_transcript() -> Transcript:
    return Transcript(
        run_id="ok-001",
        user_prompt="what is uptime on edge-node-7?",
        final_answer=(
            "Edge-node-7 has been up for 14 days. Load average is 0.42. "
            "No anomalies detected."
        ),
        steps=[
            {"role": "tool", "tool": "execute_system_diagnostic",
             "tool_result": "up 14 days, load 0.42"},
            {"role": "assistant", "content": "..."}
        ],
        guardrails=[
            {"rail": "input", "allowed": True, "reason": "clean"},
            {"rail": "execution", "allowed": True, "reason": "validated"},
            {"rail": "output", "allowed": True, "reason": "clean"},
        ],
        final_agent="system_admin_agent",
        latency_ms=910.0,
    )


def test_accuracy_score_bounds() -> None:
    assert 0.0 <= accuracy_score(_failing_transcript()) <= 0.4
    assert accuracy_score(_passing_transcript()) >= 0.8


def test_operational_success_bounds() -> None:
    assert operational_success_score(_failing_transcript()) < 0.6
    assert operational_success_score(_passing_transcript()) > 0.8


def test_selection_priority_picks_failures() -> None:
    fail_prio = selection_priority(_failing_transcript())
    ok_prio = selection_priority(_passing_transcript())
    assert fail_prio > ok_prio
    assert fail_prio >= SELECTION_THRESHOLD
    assert ok_prio < SELECTION_THRESHOLD


def test_harvest_writes_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "tuning_dataset.json"
    monkeypatch.setattr(
        "tests.test_flywheel.TUNING_DATASET", target, raising=False
    )
    records = harvest([_failing_transcript(), _passing_transcript()])
    assert len(records) == 1
    written = append_to_dataset(records, path=target)
    assert written == 1
    assert target.exists()
    line = target.read_text(encoding="utf-8").strip().splitlines()[0]
    parsed = json.loads(line)
    assert parsed["run_id"] == "fail-001"
    assert "messages" in parsed
    assert parsed["priority"] >= SELECTION_THRESHOLD


def test_clean_helpers_strip_markers() -> None:
    dirty = "  [execution-rail] something happened  [REDACTED] "
    assert "[execution-rail]" not in clean_response(dirty)
    assert "[REDACTED]" not in clean_response(dirty)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    summary = run_flywheel_pipeline()
    print(json.dumps(summary, indent=2))
