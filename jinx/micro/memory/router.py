from __future__ import annotations

import os
import time
from typing import List, Tuple

from jinx.micro.memory.storage import read_compact as _read_compact, read_evergreen as _read_evergreen
from jinx.micro.memory.pin_store import load_pins as _pins_load
from jinx.micro.memory.graph_reasoner import activate as _graph_activate
from jinx.micro.memory.search import rank_memory as _rank_memory


def _lines_of(txt: str) -> List[str]:
    return [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]


def _trim(s: str, lim: int) -> str:
    s2 = " ".join((s or "").split())
    return s2[:lim]


async def assemble_memroute(query: str, k: int = 12, preview_chars: int = 160) -> List[str]:
    """Assemble the best memory slate (pins + graph-aligned + ranked) under RT budget.

    Priorities:
      1) Pinned lines (head)
      2) Lines matching graph activation winners
      3) Ranker-selected lines from compact/evergreen (mix)
    Controls:
      JINX_MEMROUTE_MAX_MS (default 45)
    """
    q = (query or "").strip()
    try:
        max_ms = float(os.getenv("JINX_MEMROUTE_MAX_MS", "45"))
    except Exception:
        max_ms = 45.0
    t0 = time.perf_counter()

    # Load base texts
    try:
        compact = await _read_compact()
    except Exception:
        compact = ""
    try:
        evergreen = await _read_evergreen()
    except Exception:
        evergreen = ""
    c_lines = _lines_of(compact)
    e_lines = _lines_of(evergreen)

    # Pinned
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    out: List[str] = []
    for p in pins:
        if p and p not in out:
            out.append(_trim(p, preview_chars))
            if len(out) >= k:
                return out[:k]

    # Graph winners -> harvest matching lines
    winners: List[Tuple[str, float]] = []
    try:
        winners = await _graph_activate(q, k=max(1, k), steps=2)
    except Exception:
        winners = []
    keys = [key for key, _ in winners]
    if keys:
        pool = c_lines[-(k * 10):] + e_lines[: (k * 5)] + e_lines[-(k * 5):]
        for ln in pool:
            low = ln.lower()
            if any((key.lower() in low) for key in keys):
                if ln not in out:
                    out.append(_trim(ln, preview_chars))
                    if len(out) >= k:
                        return out[:k]
    if (time.perf_counter() - t0) * 1000.0 > max_ms:
        return out[:k]

    # Ranker
    ranked: List[str] = []
    try:
        ranked = await _rank_memory(q, scope="any", k=k, preview_chars=preview_chars)
    except Exception:
        ranked = []
    for ln in ranked:
        if ln and ln not in out:
            out.append(_trim(ln, preview_chars))
            if len(out) >= k:
                break

    return out[:k]
