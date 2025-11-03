from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

# Fast locator for simple "where/usage" queries. Uses project retrieval core directly with tight budgets.

async def find_usages(query: str, *, k: int | None = None, max_time_ms: int | None = None) -> List[str]:
    try:
        from jinx.micro.embeddings.retrieval_core import retrieve_project_top_k as _retr
    except Exception:
        return []
    q = (query or "").strip()
    if not q:
        return []
    try:
        kk = int(os.getenv("JINX_LOCATOR_K", "10")) if k is None else int(k)
    except Exception:
        kk = 10
    try:
        budget = int(os.getenv("JINX_LOCATOR_MS", "220")) if max_time_ms is None else int(max_time_ms)
    except Exception:
        budget = 220
    try:
        hits: List[Tuple[float, str, Dict[str, Any]]] = await _retr(q, k=kk, max_time_ms=budget)
    except Exception:
        return []
    lines: List[str] = []
    for sc, rel, obj in (hits or [])[: kk]:
        try:
            m = obj.get("meta", {}) if isinstance(obj, dict) else {}
        except Exception:
            m = {}
        try:
            file_rel = str(m.get("file_rel") or rel or "")
            ls = int(m.get("line_start") or 0)
            pv = (m.get("text_preview") or "").strip().splitlines()[0] if m else ""
        except Exception:
            file_rel = str(rel or ""); ls = 0; pv = ""
        row = f"{file_rel}:{ls} {pv}".strip()
        if row:
            lines.append(row[:240])
    return lines


__all__ = ["find_usages"]
