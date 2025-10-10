from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_scan_store import iter_project_chunks
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from .flex_pattern import make_flex_code_pattern_from_query


def _snippet_from_pos(txt: str, pos0: int, length: int, around: int = 12) -> tuple[int, int, str]:
    if pos0 < 0:
        return (0, 0, "")
    pos1 = min(len(txt), pos0 + max(1, length))
    pre = txt[:pos0]
    ls = pre.count("\n") + 1
    # include lines spanned by the match
    le = ls + max(1, txt[pos0:pos1].count("\n"))
    lines = txt.splitlines()
    a = max(1, ls - around)
    b = min(len(lines), le + around)
    snip = "\n".join(lines[a-1:b]).strip()
    return (a, b, snip)


def stage_literal_hits(query: str, k: int, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Literal substring search across all included project files.

    Pass order:
    1) Case-sensitive exact substring
    2) Case-insensitive substring
    3) Whitespace-insensitive regex (flex)
    """
    q = (query or "").strip()
    if len(q) < 3:
        return []
    t0 = time.perf_counter()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    # Precompute flexible regex once (code-core aware)
    flex_pat = make_flex_code_pattern_from_query(q)
    codey = _is_code_like(q)
    include_exts = ["py"] if codey else INCLUDE_EXTS

    # Pass 1: scan embeddings-known files first (fast, limited set)
    try:
        seen_f: set[str] = set()
        rel_files: List[str] = []
        for fr, obj in iter_project_chunks():
            rel = fr or str((obj.get("meta") or {}).get("file_rel") or "")
            if rel and rel not in seen_f:
                seen_f.add(rel)
                if not codey or rel.endswith(".py"):
                    rel_files.append(rel)
        for rel_p in rel_files:
            if time_up():
                break
            abs_p = os.path.join(ROOT, rel_p)
            try:
                with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue
            if not txt:
                continue
            pos = txt.find(q)
            found_kind = None
            ls = le = 0
            snip = ""
            if pos != -1:
                ls, le, snip = _snippet_from_pos(txt, pos, len(q))
                found_kind = "cs"
            else:
                posi = txt.lower().find(q.lower())
                if posi != -1:
                    ls, le, snip = _snippet_from_pos(txt, posi, len(q))
                    found_kind = "ci"
                elif flex_pat is not None:
                    m = flex_pat.search(txt)
                    if m:
                        ls, le, snip = _snippet_from_pos(txt, m.start(), m.end() - m.start())
                        found_kind = "flex"
            if found_kind:
                obj = {
                    "embedding": [],
                    "meta": {
                        "file_rel": rel_p,
                        "text_preview": snip or txt[:300].strip(),
                        "line_start": ls,
                        "line_end": le,
                    },
                }
                score = 0.997 if found_kind == "cs" else (0.996 if found_kind == "ci" else 0.995)
                hits.append((score, rel_p, obj))
                if len(hits) >= k:
                    return hits[:k]
    except Exception:
        pass

    # Pass 2: general project walk (may be slower)
    for abs_p, rel_p in iter_candidate_files(
        ROOT,
        include_exts=include_exts,
        exclude_dirs=EXCLUDE_DIRS,
        max_file_bytes=MAX_FILE_BYTES,
    ):
        if time_up():
            break
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception:
            continue
        if not txt:
            continue
        pos = txt.find(q)
        found_kind = None
        ls = le = 0
        snip = ""
        if pos != -1:
            ls, le, snip = _snippet_from_pos(txt, pos, len(q))
            found_kind = "cs"
        else:
            posi = txt.lower().find(q.lower())
            if posi != -1:
                ls, le, snip = _snippet_from_pos(txt, posi, len(q))
                found_kind = "ci"
            elif flex_pat is not None:
                m = flex_pat.search(txt)
                if m:
                    ls, le, snip = _snippet_from_pos(txt, m.start(), m.end() - m.start())
                    found_kind = "flex"
        if found_kind:
            obj = {
                "embedding": [],
                "meta": {
                    "file_rel": rel_p,
                    "text_preview": snip or txt[:300].strip(),
                    "line_start": ls,
                    "line_end": le,
                },
            }
            # Score by strength of match type
            score = 0.997 if found_kind == "cs" else (0.996 if found_kind == "ci" else 0.995)
            hits.append((score, rel_p, obj))
            if len(hits) >= k:
                break
    return hits[:k]

__all__ = ["stage_literal_hits"]
