from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks
from .flex_pattern import make_flex_code_pattern_from_query


_WS = re.compile(r"\s+", re.MULTILINE)


def stage_lineexact_hits(query: str, k: int, *, max_time_ms: int | None = 160) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Whitespace-insensitive literal matching across project files (Python only).

    Builds a flexible regex from the query to match the literal code fragment,
    ignoring whitespace differences. Returns a small window around the match.
    """
    q = (query or "").strip()
    if not q:
        return []
    t0 = time.perf_counter()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    pat = make_flex_code_pattern_from_query(q)
    if pat is None:
        return []

    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def process(abs_p: str, rel_p: str) -> bool:
        if time_up():
            return True
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception:
            return False
        if not txt:
            return False
        m = pat.search(txt)
        if not m:
            return False
        pos0, pos1 = m.start(), m.end()
        pre = txt[:pos0]
        ls = pre.count("\n") + 1
        le = ls + max(1, txt[pos0:pos1].count("\n"))
        lines = txt.splitlines()
        a = max(1, ls - 12)
        b = min(len(lines), le + 12)
        snip = "\n".join(lines[a-1:b]).strip()
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": snip or txt[max(0, pos0-160):min(len(txt), pos0+160)].strip(),
                "line_start": a,
                "line_end": b,
            },
        }
        hits.append((0.998, rel_p, obj))
        return len(hits) >= k

    # Prefer embeddings-known files first
    try:
        seen: set[str] = set()
        rel_files: List[str] = []
        for fr, obj in iter_project_chunks():
            rel = fr or str((obj.get("meta") or {}).get("file_rel") or "")
            if rel and rel not in seen:
                seen.add(rel)
                rel_files.append(rel)
        for rel in rel_files:
            ap = os.path.join(ROOT, rel)
            if process(ap, rel):
                return hits[:k]
    except Exception:
        pass

    # Fallback: scan .py files
    for ap, rel in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if process(ap, rel):
            return hits[:k]

    return hits[:k]


__all__ = ["stage_lineexact_hits"]
