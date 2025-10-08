from __future__ import annotations

import os
from typing import List

from jinx.micro.memory.router import assemble_memroute as _memroute
from jinx.micro.embeddings.project_retrieval import (
    build_project_context_for as _build_proj_ctx_single,
    build_project_context_multi_for as _build_proj_ctx_multi,
)
from jinx.micro.memory.storage import read_channel as _read_channel


async def build_project_context_enriched(query: str, user_text: str = "", synth: str = "") -> str:
    """Compose a project code context aligned with the current task and memory.

    Strategy:
    - Build enriched query from task + routed memory hints (memroute).
    - Generate several sub-queries (task+hints, task+top paths/symbols) under a strict cap.
    - Use multi-query aggregation within a bounded RT budget; fallback to single-query.
    """
    _q = (query or "").strip()
    if not _q:
        return ""
    # Routed memory hints
    try:
        mem_hints: List[str] = await _memroute(_q, k=8, preview_chars=120)
    except Exception:
        mem_hints = []
    # Caps and dynamics
    try:
        max_q_chars = int(os.getenv("JINX_PROJ_QUERY_MAX_CHARS", "800"))
    except Exception:
        max_q_chars = 800
    _q_proj = (" ".join([_q] + mem_hints)).strip()
    if len(_q_proj) > max_q_chars:
        _q_proj = _q_proj[:max_q_chars]
    qlow = _q_proj.lower()
    codey = any(sym in _q_proj for sym in "=[](){}.:,") or any(
        kw in qlow for kw in ["def ", "class ", "import ", "from ", "return ", "async ", "await ", " for ", " in ", " = "]
    )
    first_budget = 1200 if codey else None
    try:
        proj_k = int(os.getenv("JINX_PROJ_CTX_K", ("10" if codey else "6")))
    except Exception:
        proj_k = 10 if codey else 6
    try:
        subq_cap = int(os.getenv("JINX_PROJ_SUBQ_MAX", "3"))
    except Exception:
        subq_cap = 3
    subqs: List[str] = [_q_proj]
    # 1) task + top memory hints
    for h in (mem_hints or [])[: max(0, subq_cap)]:
        qh = f"{_q} {h}".strip()
        if len(qh) > max_q_chars:
            qh = qh[:max_q_chars]
        if qh not in subqs:
            subqs.append(qh)
    # 2) task + top channels (paths/symbols)
    try:
        ch_paths = (await _read_channel("paths") or "").splitlines()
    except Exception:
        ch_paths = []
    try:
        ch_syms = (await _read_channel("symbols") or "").splitlines()
    except Exception:
        ch_syms = []

    def _after(prefix: str, ln: str) -> str:
        low = (ln or "").lower()
        return ln[len(prefix) :].strip() if low.startswith(prefix) else ""

    add_more = max(0, subq_cap - max(0, len(subqs) - 1))
    for ln in ch_paths[:2]:
        if add_more <= 0:
            break
        p = _after("path: ", ln)
        if not p:
            continue
        qh = f"{_q} {p}".strip()
        if len(qh) > max_q_chars:
            qh = qh[:max_q_chars]
        if qh not in subqs:
            subqs.append(qh)
            add_more -= 1
    for ln in ch_syms[:2]:
        if add_more <= 0:
            break
        s = _after("symbol: ", ln)
        if not s:
            continue
        qh = f"{_q} {s}".strip()
        if len(qh) > max_q_chars:
            qh = qh[:max_q_chars]
        if qh not in subqs:
            subqs.append(qh)
            add_more -= 1

    # Build context using multi-query if we have more than one sub-query
    try:
        if len(subqs) > 1:
            ctx = await _build_proj_ctx_multi(subqs, k=proj_k, max_time_ms=first_budget)
        else:
            ctx = await _build_proj_ctx_single(_q_proj, k=proj_k, max_time_ms=first_budget)
    except Exception:
        ctx = ""
    if ctx:
        return ctx
    # Fallback with larger budget
    try:
        if len(subqs) > 1:
            ctx = await _build_proj_ctx_multi(subqs, k=proj_k, max_time_ms=2000)
        else:
            ctx = await _build_proj_ctx_single(_q_proj, k=proj_k, max_time_ms=2000)
    except Exception:
        ctx = ""
    return ctx or ""
