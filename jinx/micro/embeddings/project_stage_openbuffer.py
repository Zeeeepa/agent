from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT
from .flex_pattern import make_flex_code_pattern_from_query

# Location under the project root
_OPEN_BUFFERS_PATH = os.path.join(ROOT, ".jinx", "memory", "open_buffers.jsonl")

_WS = re.compile(r"\s+", re.MULTILINE)


def _snippet_from_pos(txt: str, pos0: int, length: int, around: int = 12) -> tuple[int, int, str]:
    if pos0 < 0:
        return (0, 0, "")
    pos1 = min(len(txt), pos0 + max(1, length))
    pre = txt[:pos0]
    ls = pre.count("\n") + 1
    le = ls + max(1, txt[pos0:pos1].count("\n"))
    lines = txt.splitlines()
    a = max(1, ls - around)
    b = min(len(lines), le + around)
    snip = "\n".join(lines[a - 1 : b]).strip()
    return (a, b, snip)


def _iter_open_buffers(path: str) -> List[Tuple[str, str]]:
    """Yield (name, text) for open buffers from a jsonl file if present."""
    out: List[Tuple[str, str]] = []
    try:
        if not os.path.exists(path):
            return out
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                name = (obj.get("name") or obj.get("path") or "buffer").strip() or "buffer"
                text = (obj.get("text") or "")
                if text:
                    out.append((name, text))
    except Exception:
        return []
    return out


def stage_openbuffer_hits(query: str, k: int, *, max_time_ms: int | None = 140) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Search unsaved open buffers stored at .jinx/memory/open_buffers.jsonl.

    Returns best-effort literal/flex matches with small windows. This stage
    is cheap and improves recall for code not saved to disk yet.
    """
    q = (query or "").strip()
    if not q:
        return []
    # Build code-flex pattern (code-core aware) with ignore_case for editor variability
    pat = make_flex_code_pattern_from_query(q, prefer_core=True, ignore_case=True)
    if pat is None:
        return []
    t0 = time.perf_counter()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    hits: List[Tuple[float, str, Dict[str, Any]]] = []
    for name, text in _iter_open_buffers(_OPEN_BUFFERS_PATH):
        if time_up():
            break
        try:
            m = pat.search(text)
        except Exception:
            m = None
        if not m:
            continue
        a, b, snip = _snippet_from_pos(text, m.start(), m.end() - m.start())
        meta = {
            "file_rel": f"open_buffer:{name}",
            "text_preview": snip or text[:300].strip(),
            "line_start": a,
            "line_end": b,
        }
        hits.append((0.9965, f"open_buffer:{name}", {"embedding": [], "meta": meta}))
        if len(hits) >= k:
            break
    return hits[:k]


__all__ = ["stage_openbuffer_hits"]
