from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List, Any

from jinx.micro.memory.schema import MemoryItem
from jinx.micro.memory.storage import read_compact, read_evergreen
from jinx.micro.embeddings.retrieval import retrieve_top_k as _emb_retrieve
from jinx.micro.embeddings.text_clean import is_noise_text as _is_noise


def _lines(s: str) -> List[str]:
    return [ln.strip() for ln in (s or "").splitlines() if (ln or "").strip()]


async def ephemeral_for(query: str, *, k: int = 8, max_time_ms: int = 200) -> List[MemoryItem]:
    """Return top-k ephemeral items (dialogue/state/sandbox) for a query.

    Uses the shared embeddings runtime store under a strict time budget.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        overfetch = max(k * 3, int(os.getenv("JINX_MEM_LEVELS_OVERFETCH", str(k * 4))))
    except Exception:
        overfetch = k * 4
    try:
        hits = await _emb_retrieve(q, k=overfetch, max_time_ms=max_time_ms)
    except Exception:
        hits = []
    out: List[MemoryItem] = []
    seen: set[str] = set()
    now_ms = int(time.time() * 1000)
    for _score, src, obj in hits:
        meta: Dict[str, Any] = obj.get("meta", {}) if isinstance(obj, dict) else {}
        pv = (meta.get("text_preview") or "").strip()
        if not pv or pv in seen or _is_noise(pv):
            continue
        try:
            ts_ms = int(float(meta.get("ts") or 0.0) * 1000.0)
        except Exception:
            ts_ms = now_ms
        out.append(MemoryItem(text=pv[:160], source=str(meta.get("source") or src or "recent"), ts_ms=ts_ms, meta=meta))
        seen.add(pv)
        if len(out) >= k:
            break
    return out


async def session_tail(max_lines: int = 200) -> List[MemoryItem]:
    """Return recent session memory lines (from compact.md)."""
    try:
        raw = await read_compact()
    except Exception:
        raw = ""
    lines = _lines(raw)
    if max_lines > 0:
        lines = lines[-max_lines:]
    out: List[MemoryItem] = []
    now_ms = int(time.time() * 1000)
    for idx, ln in enumerate(lines):
        out.append(MemoryItem(text=ln[:320], source="compact", ts_ms=now_ms, meta={"idx": idx}))
    return out


def _channel(line: str) -> str:
    low = (line or "").lower()
    if low.startswith("path: "):
        return "paths"
    if low.startswith("symbol: "):
        return "symbols"
    if low.startswith("pref: "):
        return "prefs"
    if low.startswith("decision: "):
        return "decisions"
    if low.startswith("setting: "):
        return "settings"
    return "evergreen"


async def evergreen_all(max_lines: int = 400) -> List[MemoryItem]:
    """Return evergreen lines as MemoryItems with channel metadata."""
    try:
        raw = await read_evergreen()
    except Exception:
        raw = ""
    lines = _lines(raw)
    if max_lines > 0:
        lines = lines[-max_lines:]
    out: List[MemoryItem] = []
    now_ms = int(time.time() * 1000)
    for idx, ln in enumerate(lines):
        ch = _channel(ln)
        out.append(MemoryItem(text=ln[:320], source="evergreen", ts_ms=now_ms, meta={"channel": ch, "idx": idx}))
    return out


__all__ = [
    "MemoryItem",
    "ephemeral_for",
    "session_tail",
    "evergreen_all",
]
