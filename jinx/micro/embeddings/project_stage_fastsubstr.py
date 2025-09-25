from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks
from .project_query_tokens import expand_strong_tokens, codeish_tokens


def _anchors(q: str, limit: int = 4) -> List[str]:
    toks: List[str] = []
    # prefer strong tokens then simple ones
    toks.extend(expand_strong_tokens(q, max_items=32))
    for t in codeish_tokens(q):
        if t not in toks:
            toks.append(t)
    # filter trivial tokens
    bad = {"for", "in", "def", "class", "return", "async", "await"}
    toks2 = [t for t in toks if len(t) >= 3 and t.lower() not in bad]
    # sort by length desc and uniqueness
    uniq: List[str] = []
    seen: set[str] = set()
    for t in sorted(toks2, key=len, reverse=True):
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        uniq.append(t)
    return uniq[:limit]


def _code_core(q: str) -> str | None:
    # Try to extract a likely code fragment; mirror of textscan logic, simplified
    try:
        import re as _re, ast as _ast
        cands = list(_re.finditer(r"[A-Za-z0-9_\./:\-+*<>=!\"'\[\]\(\)\{\),\s]+", q or ""))
        def _can(s: str) -> bool:
            s = (s or "").strip()
            if len(s) < 3:
                return False
            try:
                _ast.parse(s, mode="exec"); return True
            except Exception:
                pass
            try:
                _ast.parse(s, mode="eval"); return True
            except Exception:
                pass
            try:
                _ast.parse(f"({s})", mode="eval"); return True
            except Exception:
                return False
        best = ""; best_len = 0
        for m in cands:
            frag = (m.group(0) or "").strip()
            if len(frag) < 6:
                continue
            if _can(frag) and len(frag) > best_len:
                best = frag; best_len = len(frag)
        if best:
            return best
        if cands:
            longest = max((m.group(0) or '').strip() for m in cands if (m.group(0) or '').strip())
            return longest or None
        return None
    except Exception:
        return None


def stage_fastsubstr_hits(query: str, k: int, *, max_time_ms: int | None = 150) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: fast substring-based hit using code-core or anchor tokens.

    Extremely cheap and robust line-window retrieval when indexing may be cold.
    """
    q = (query or "").strip()
    if not q:
        return []
    t0 = time.perf_counter()

    core = _code_core(q)
    anchors = _anchors(q, limit=3)

    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    def process(abs_p: str, rel_p: str) -> bool:
        if time_up():
            return True
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return False
        if not text:
            return False
        low = text.lower()
        # Try fuzzy core first using 'regex' for small typos/whitespace differences
        if core:
            try:
                import regex as _rx  # type: ignore
                err = max(1, min(6, len(core) // 10))
                pat = _rx.compile(f"({_rx.escape(core)}){{e<={err}}}", _rx.BESTMATCH | _rx.IGNORECASE | _rx.DOTALL)
                mm = pat.search(text)
                if mm:
                    pos0, pos1 = mm.start(), mm.end()
                    pre = text[:pos0]
                    ls = pre.count("\n") + 1
                    le = ls + max(1, text[pos0:pos1].count("\n"))
                    lines = text.splitlines()
                    a = max(1, ls - 12)
                    b = min(len(lines), le + 12)
                    snip = "\n".join(lines[a-1:b]).strip()
                    obj = {
                        "embedding": [],
                        "meta": {
                            "file_rel": rel_p,
                            "text_preview": snip or text[:300].strip(),
                            "line_start": a,
                            "line_end": b,
                        },
                    }
                    hits.append((0.994, rel_p, obj))
                    return len(hits) >= k
            except Exception:
                pass
        # Try exact core next
        ls = le = 0
        snip = ""
        if core and core.lower() in low:
            pos0 = low.find(core.lower())
            pre = low[:pos0]
            ls = pre.count("\n") + 1
            le = ls + max(1, low[pos0:pos0+len(core)].count("\n"))
        else:
            # Require multiple anchors when available to avoid word-only false positives
            present = [a for a in anchors if a.lower() in low]
            need = 2 if len(anchors) >= 2 else 1
            if len(present) < need:
                return False
            # window around first present anchor
            a0 = present[0].lower()
            p0 = low.find(a0)
            pre = low[:p0]
            ls = pre.count("\n") + 1
            le = ls + 1
        lines = text.splitlines()
        a = max(1, ls - 12)
        b = min(len(lines), le + 12)
        snip = "\n".join(lines[a-1:b]).strip()
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": snip or text[:300].strip(),
                "line_start": a,
                "line_end": b,
            },
        }
        hits.append((0.994, rel_p, obj))
        return len(hits) >= k

    # Scan embeddings-known files first
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

    # Fallback: full walk
    for ap, rel in iter_candidate_files(ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if process(ap, rel):
            return hits[:k]

    return hits[:k]


__all__ = ["stage_fastsubstr_hits"]
