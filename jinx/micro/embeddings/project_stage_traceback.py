from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT


_TB_FILE_PATTERNS = [
    # CPython traceback: File "path", line N, in func
    re.compile(r"File\s+\"(?P<path>[^\"]+)\"\s*,\s*line\s+(?P<line>\d+)", re.IGNORECASE),
    # filename.py:123 style
    re.compile(r"(?P<path>[^\s:<>\"\']+\.py)[:\(](?P<line>\d+)\)?", re.IGNORECASE),
]


def _norm_path(p: str) -> str:
    p = (p or "").strip().strip("\u200b\ufeff")
    if not p:
        return ""
    # Normalize separators
    p = p.replace("\\", os.sep).replace("/", os.sep)
    # If absolute under ROOT, make it relative
    try:
        ap = os.path.abspath(p)
        ar = os.path.abspath(ROOT)
        if ap.startswith(ar + os.sep):
            return os.path.relpath(ap, ar)
    except Exception:
        pass
    # If already relative from ROOT
    return p


def _extract_frames(q: str) -> List[Tuple[str, int]]:
    frames: List[Tuple[str, int]] = []
    s = q or ""
    for pat in _TB_FILE_PATTERNS:
        for m in pat.finditer(s):
            path = _norm_path(m.group("path") or "")
            try:
                line = int(m.group("line"))
            except Exception:
                line = 0
            if path and line > 0:
                frames.append((path, line))
    # Deduplicate preserving order
    out: List[Tuple[str, int]] = []
    seen: set[Tuple[str, int]] = set()
    for it in frames:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out[:4]


def stage_traceback_hits(query: str, k: int, *, max_time_ms: int | None = 100) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: parse Python traceback-like text and return precise file windows.

    Extremely precise when users paste error logs with file/line info.
    """
    q = (query or "").strip()
    if not q:
        return []
    frames = _extract_frames(q)
    if not frames:
        return []

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    for rel_p, ln in frames:
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
        abs_p = os.path.join(ROOT, rel_p)
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if not text:
            continue
        lines = text.splitlines()
        a = max(1, ln - 12)
        b = min(len(lines), ln + 12)
        preview = "\n".join(lines[a - 1 : b]).strip()
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": preview,
                "line_start": a,
                "line_end": b,
            },
        }
        # Highest precision among heuristics
        hits.append((0.996, rel_p, obj))
        if len(hits) >= k:
            break
    return hits


__all__ = ["stage_traceback_hits"]
