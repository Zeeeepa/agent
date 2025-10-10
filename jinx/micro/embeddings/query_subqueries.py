from __future__ import annotations

from typing import List

from .project_query_core import extract_code_core
from .project_query_tokens import expand_strong_tokens


def build_codecentric_subqueries(query: str, *, max_tokens: int = 8, max_compact: int = 5) -> List[str]:
    """Generate code-centric subqueries from a raw query.

    - code-core fragment (if present)
    - compacted strong-token phrase (up to max_compact unique tokens)
    - targeted 'isinstance ast.*' pair when applicable
    """
    q = (query or "").strip()
    if not q:
        return []
    subs: list[str] = []
    # 1) code-core
    core = extract_code_core(q)
    if core:
        subs.append(core)
    # 2) strong tokens compact
    try:
        toks = expand_strong_tokens(q, max_items=max_tokens)
    except Exception:
        toks = []
    if toks:
        uniq = list(dict.fromkeys(toks))
        compact = " ".join(uniq[: max(1, int(max_compact))]).strip()
        if compact:
            subs.append(compact)
        # 3) targeted pair
        try:
            has_inst = any(t.lower() == "isinstance" for t in uniq)
            ast_tok = next((t for t in uniq if t.startswith("ast.")), "")
            if has_inst and ast_tok:
                pair = f"isinstance {ast_tok}"
                subs.append(pair)
        except Exception:
            pass
    # de-duplicate preserving order
    out: list[str] = []
    seen: set[str] = set()
    for s in subs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


__all__ = ["build_codecentric_subqueries"]
