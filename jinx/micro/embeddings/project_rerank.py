from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from .project_query_core import extract_code_core

_TOK_RE = re.compile(r"(?u)[\w\.]{3,}")


def _query_tokens(q: str) -> List[str]:
    toks: List[str] = []
    for m in _TOK_RE.finditer((q or "")):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 3:
            toks.append(t)
    # dedupe, keep order
    seen: set[str] = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def rerank_hits(hits: List[Tuple[float, str, Dict[str, Any]]], query: str) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Lightweight reranker: boosts filename/path token matches and preview matches.

    - Path/file match: +0.3 per token
    - Preview match: +0.1 per token
    """
    if not hits:
        return []
    # Prefer tokens from the code-core when present to better represent code fragments
    core_q = extract_code_core(query or "") or (query or "")
    qtok = _query_tokens(core_q)
    if not qtok:
        return sorted(hits, key=lambda h: float(h[0] or 0.0), reverse=True)
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for sc, rel, obj in hits:
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").lower()
        rel_l = (str(meta.get("file_rel") or rel) or "").lower()
        boost = 0.0
        pos_list: List[int] = []
        for t in qtok:
            if t in rel_l:
                boost += 0.3
            elif t in pv:
                boost += 0.15
                try:
                    pos = pv.find(t)
                    if pos >= 0:
                        pos_list.append(pos)
                except Exception:
                    pass
        # Additional proximity boost if multiple tokens occur close together in preview
        if len(pos_list) >= 2:
            pos_list.sort()
            try:
                span = max(pos_list) - min(pos_list)
                if span <= 24:
                    boost += 0.2
            except Exception:
                pass
        scored.append((float(sc or 0.0) + boost, rel, obj))
    return sorted(scored, key=lambda h: float(h[0] or 0.0), reverse=True)
