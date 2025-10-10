from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks
from .project_retrieval_config import PROJ_COOCCUR_MAX_DIST

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")
_STOP = {
    "and", "or", "not", "if", "else", "elif", "for", "in", "while", "return", "true", "false",
    "none", "class", "def", "with", "as", "try", "except", "finally", "from", "import", "pass",
}


def _extract_tokens(q: str, limit: int = 4) -> List[str]:
    s = (q or "").strip()
    if not s:
        return []
    # Normalize C-like bool ops to improve detection
    s = s.replace("&&", " and ").replace("||", " or ")
    toks: List[str] = []
    seen: set[str] = set()
    for m in _TOKEN_RE.finditer(s):
        t = (m.group(0) or "").strip()
        tl = t.lower()
        if len(t) < 3:
            continue
        if tl in _STOP:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        toks.append(t)
        if len(toks) >= limit:
            break
    return toks


def _find_lines(text: str, token: str) -> List[int]:
    # Case-sensitive then insensitive fallback
    lines = text.splitlines()
    hits: List[int] = []
    t_cs = token
    t_ci = token.lower()
    for idx, ln in enumerate(lines, start=1):
        if t_cs in ln or t_ci in ln.lower():
            hits.append(idx)
    return hits


def _window(lines: List[str], s: int, e: int, around: int = 12) -> tuple[int, int, str]:
    a = max(1, min(s, e) - around)
    b = min(len(lines), max(s, e) + around)
    return a, b, "\n".join(lines[a-1:b]).strip()


def stage_cooccur_hits(query: str, k: int, *, max_time_ms: int | None = 220) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Find snippets where multiple query tokens co-occur within a small line distance.

    - Extract up to 4 tokens from the query (ignore stopwords).
    - Scan Python files and detect windows where at least 2 tokens appear within PROJ_COOCCUR_MAX_DIST lines.
    - Score favors more tokens and tighter proximity.
    """
    q = (query or "").strip()
    if not q:
        return []
    tokens = _extract_tokens(q, limit=4)
    # Require at least 2 informative tokens
    if len(tokens) < 2:
        return []
    t0 = time.perf_counter()
    out: List[Tuple[float, str, Dict[str, Any]]] = []

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

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
        lines = txt.splitlines()
        # Map token -> line numbers
        locs: List[List[int]] = [_find_lines(txt, t) for t in tokens]
        present = [i for i, arr in enumerate(locs) if arr]
        if len(present) < 2:
            return False
        # Evaluate co-occurrence windows using pairwise distances
        best_score = 0.0
        best_a = best_b = 0
        maxd = max(1, int(PROJ_COOCCUR_MAX_DIST))
        for i in range(len(tokens)):
            li = locs[i]
            if not li:
                continue
            for j in range(i + 1, len(tokens)):
                lj = locs[j]
                if not lj:
                    continue
                # Two-pointer over sorted line lists
                p = 0
                q2 = 0
                while p < len(li) and q2 < len(lj):
                    l1 = li[p]
                    l2 = lj[q2]
                    d = abs(l1 - l2)
                    if d <= maxd:
                        a = min(l1, l2)
                        b = max(l1, l2)
                        # Score: base for 2 tokens + proximity bonus
                        score = 0.992 + 0.006 * (1.0 - (d / (maxd + 1)))
                        if score > best_score:
                            best_score = score
                            best_a, best_b = a, b
                        # advance closer pointer
                        if l1 <= l2:
                            p += 1
                        else:
                            q2 += 1
                    else:
                        if l1 < l2:
                            p += 1
                        else:
                            q2 += 1
                    if time_up():
                        break
                if time_up():
                    break
            if time_up():
                break
        if best_score > 0 and best_a > 0:
            a, b, snip = _window(lines, best_a, best_b)
            obj = {
                "embedding": [],
                "meta": {
                    "file_rel": rel_p,
                    "text_preview": snip or "\n".join(lines[a-1:b]).strip(),
                    "line_start": a,
                    "line_end": b,
                },
            }
            out.append((best_score, rel_p, obj))
            return len(out) >= k
        return False

    # Prefer embeddings-known files first
    try:
        seen: set[str] = set()
        rel_files: List[str] = []
        for fr, obj in iter_project_chunks():
            rel = fr or str((obj.get("meta") or {}).get("file_rel") or "")
            if rel and rel.endswith(".py") and rel not in seen:
                seen.add(rel)
                rel_files.append(rel)
        for rel in rel_files:
            ap = os.path.join(ROOT, rel)
            if process(ap, rel):
                return out[:k]
    except Exception:
        pass

    # Fallback: walk .py files
    for ap, rel in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if process(ap, rel):
            return out[:k]

    return out[:k]


__all__ = ["stage_cooccur_hits"]
