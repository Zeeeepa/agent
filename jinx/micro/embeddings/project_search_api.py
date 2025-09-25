from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .project_retrieval import retrieve_project_top_k
from .project_snippet import build_snippet
from .project_retrieval_config import (
    PROJ_DEFAULT_TOP_K,
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_ALWAYS_FULL_PY_SCOPE,
    PROJ_FULL_SCOPE_TOP_N,
)
from .project_config import ROOT
from .project_py_scope import get_python_symbol_at_line
from .project_lang import lang_for_file


async def search_project(query: str, *, k: int | None = None, max_time_ms: int | None = 300) -> List[Dict[str, Any]]:
    """Structured project code search.

    Returns a list of dicts with fields:
    - file: relative file path
    - line_start, line_end: 1-based inclusive range of the snippet
    - lang: language id for the file
    - code: raw code text of the snippet (no fences)
    - header: compact header like "[file:ls-le def name]" when applicable
    - score: retrieval score (stage-dependent)
    - is_full_scope: whether the snippet spans full def/class scope (Python)
    - symbol_name, symbol_kind: Python symbol metadata when available
    - meta: lightweight meta (preview, terms)
    """
    q = (query or "").strip()
    if not q:
        return []

    k_eff = k or PROJ_DEFAULT_TOP_K
    hits = await retrieve_project_top_k(q, k=k_eff, max_time_ms=max_time_ms)
    if not hits:
        return []

    # Preserve chronological order when possible
    hits_sorted = sorted(
        hits,
        key=lambda h: float((h[2].get("meta", {}).get("ts") or 0.0)),
    )

    out: List[Dict[str, Any]] = []
    full_scope_used = 0
    for score, file_rel, obj in hits_sorted:
        meta = obj.get("meta", {})
        prefer_full = PROJ_ALWAYS_FULL_PY_SCOPE and (
            PROJ_FULL_SCOPE_TOP_N <= 0 or (full_scope_used < PROJ_FULL_SCOPE_TOP_N)
        )
        header, code_block, ls, le, is_full_scope = build_snippet(
            file_rel,
            meta,
            q,
            max_chars=PROJ_SNIPPET_PER_HIT_CHARS,
            prefer_full_scope=prefer_full,
        )
        if is_full_scope:
            full_scope_used += 1

        # Strip code fences to return raw code
        code_text = code_block
        if code_text.startswith("```"):
            try:
                first_nl = code_text.find("\n")
                if first_nl != -1:
                    body = code_text[first_nl + 1 :]
                    if body.endswith("```"):
                        body = body[: -3]
                    code_text = body.strip("\n")
            except Exception:
                # Best effort; keep as-is on parsing issues
                pass

        # Symbol metadata (Python only)
        sym_name: str | None = None
        sym_kind: str | None = None
        try:
            if str(file_rel).endswith(".py"):
                abs_path = os.path.join(ROOT, file_rel)
                file_text = ""
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_text = f.read()
                except Exception:
                    file_text = ""
                if file_text:
                    cand_line = int((ls + le) // 2) if (ls and le) else int(ls or le or 0)
                    n, kind = get_python_symbol_at_line(file_text, cand_line)
                    sym_name, sym_kind = n, kind
        except Exception:
            pass

        out.append(
            {
                "file": str(file_rel or meta.get("file_rel") or ""),
                "line_start": int(ls or 0),
                "line_end": int(le or 0),
                "lang": lang_for_file(file_rel),
                "code": code_text,
                "header": header,
                "score": float(score),
                "is_full_scope": bool(is_full_scope),
                "symbol_name": sym_name,
                "symbol_kind": sym_kind,
                "meta": {
                    "text_preview": (meta.get("text_preview") or ""),
                    "terms": [str(x) for x in (meta.get("terms") or [])],
                },
            }
        )
    return out


__all__ = ["search_project"]
