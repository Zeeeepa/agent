from __future__ import annotations

import os
import time
import asyncio
from typing import List, Tuple

from jinx.micro.memory.storage import read_compact as _read_compact, read_evergreen as _read_evergreen
from jinx.micro.memory.pin_store import load_pins as _pins_load
from jinx.micro.memory.graph_reasoner import activate as _graph_activate
from jinx.micro.memory.search import rank_memory as _rank_memory
from jinx.micro.memory.vector_index import search as _vec_search
from jinx.micro.memory.telemetry import log_memroute_event as _log_mr
from jinx.micro.memory.usage_store import bump_usage as _bump_usage
from jinx.micro.memory.kb_store import search_lines as _kb_search


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
    t_prev = t0
    try:
        tm_on = str(os.getenv("JINX_MEMROUTE_TELEMETRY", "0")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        tm_on = False

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
    before_pins = 0
    for p in pins:
        if p and p not in out:
            out.append(_trim(p, preview_chars))
            if len(out) >= k:
                if tm_on:
                    try:
                        now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0
                        asyncio.create_task(_log_mr("pins", len(out) - before_pins, stage_ms, k, max_ms))
                    except Exception:
                        pass
                try:
                    asyncio.create_task(_bump_usage(list(out[:k])))
                except Exception:
                    pass
                return out[:k]
    # pins stage done
    if tm_on:
        try:
            now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0; t_prev = now_t
            asyncio.create_task(_log_mr("pins", len(out) - before_pins, stage_ms, k, max_ms))
        except Exception:
            t_prev = time.perf_counter()

    # Graph winners -> harvest matching lines
    winners: List[Tuple[str, float]] = []
    try:
        winners = await _graph_activate(q, k=max(1, k), steps=2)
    except Exception:
        winners = []
    keys = [key for key, _ in winners]
    if keys:
        before_graph = len(out)
        pool = c_lines[-(k * 10):] + e_lines[: (k * 5)] + e_lines[-(k * 5):]
        for ln in pool:
            low = ln.lower()
            if any((key.lower() in low) for key in keys):
                if ln not in out:
                    out.append(_trim(ln, preview_chars))
                    if len(out) >= k:
                        if tm_on:
                            try:
                                now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0
                                asyncio.create_task(_log_mr("graph", len(out) - before_graph, stage_ms, k, max_ms))
                            except Exception:
                                pass
                        try:
                            asyncio.create_task(_bump_usage(list(out[:k])))
                        except Exception:
                            pass
                        return out[:k]
        if tm_on:
            try:
                now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0; t_prev = now_t
                asyncio.create_task(_log_mr("graph", len(out) - before_graph, stage_ms, k, max_ms))
            except Exception:
                t_prev = time.perf_counter()
    # Optional: vector-based retrieval over embedded memory/state under remaining budget
    try:
        vec_on = str(os.getenv("JINX_MEMROUTE_VEC_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        vec_on = True
    if vec_on and len(out) < k:
        before_vec = len(out)
        try:
            try:
                k_vec = int(os.getenv("JINX_MEMROUTE_K_VEC", str(max(1, k // 2))))
            except Exception:
                k_vec = max(1, k // 2)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rem_ms = int(max(1.0, max_ms - elapsed_ms)) if max_ms > 0 else 30
            if rem_ms > 1:
                vhits = await _vec_search(q, k=min(k_vec, max(1, k - len(out))), max_time_ms=rem_ms, preview_chars=preview_chars)
            else:
                vhits = []
        except Exception:
            vhits = []
        for ln in vhits:
            if ln and ln not in out:
                out.append(_trim(ln, preview_chars))
                if len(out) >= k:
                    if tm_on:
                        try:
                            now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0
                            asyncio.create_task(_log_mr("vector", len(out) - before_vec, stage_ms, k, max_ms))
                        except Exception:
                            pass
                    try:
                        asyncio.create_task(_bump_usage(list(out[:k])))
                    except Exception:
                        pass
                    return out[:k]
        if tm_on:
            try:
                now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0; t_prev = now_t
                asyncio.create_task(_log_mr("vector", len(out) - before_vec, stage_ms, k, max_ms))
            except Exception:
                t_prev = time.perf_counter()
    # Optional: KB retrieval under remaining budget
    try:
        kb_on = str(os.getenv("JINX_MEMROUTE_KB_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        kb_on = True
    if kb_on and len(out) < k:
        before_kb = len(out)
        try:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rem_ms = int(max(1.0, max_ms - elapsed_ms)) if max_ms > 0 else 20
            if rem_ms > 1:
                khits = await _kb_search(q, k=max(1, min(3, k - len(out))), max_time_ms=rem_ms, preview_chars=preview_chars)
            else:
                khits = []
        except Exception:
            khits = []
        for ln in khits:
            if ln and ln not in out:
                out.append(_trim(ln, preview_chars))
                if len(out) >= k:
                    if tm_on:
                        try:
                            now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0
                            asyncio.create_task(_log_mr("kb", len(out) - before_kb, stage_ms, k, max_ms))
                        except Exception:
                            pass
                    try:
                        asyncio.create_task(_bump_usage(list(out[:k])))
                    except Exception:
                        pass
                    return out[:k]
        if tm_on:
            try:
                now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0; t_prev = now_t
                asyncio.create_task(_log_mr("kb", len(out) - before_kb, stage_ms, k, max_ms))
            except Exception:
                t_prev = time.perf_counter()

    if (time.perf_counter() - t0) * 1000.0 > max_ms:
        return out[:k]

    # Ranker
    before_rank = len(out)
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
    if tm_on:
        try:
            now_t = time.perf_counter(); stage_ms = (now_t - t_prev) * 1000.0
            asyncio.create_task(_log_mr("rank", len(out) - before_rank, stage_ms, k, max_ms))
        except Exception:
            pass
    return out[:k]
