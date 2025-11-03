from __future__ import annotations

import os
import re
from typing import List, Tuple

from jinx.micro.embeddings.context_blocks import build_brain_block
from jinx.micro.embeddings.project_util import tokenize_terms


_TERM_RE = re.compile(r"(?u)[\w\.]{3,}")
_PATH_RE = re.compile(r"(?:(?:[A-Za-z]\:)?[A-Za-z0-9_\-./\\]+\.[A-Za-z0-9_]{1,8})")


def _extract_blocks_multi(text: str, tags: List[str], max_len: int) -> str:
    s = text or ""
    out: List[str] = []
    for tag in (tags or []):
        lt, rt = f"<{tag}>", f"</{tag}>"
        pos = 0
        while True:
            i = s.find(lt, pos)
            if i == -1:
                break
            j = s.find(rt, i)
            if j == -1:
                break
            core = s[i + len(lt): j]
            if core:
                out.append(core)
            pos = j + len(rt)
            if sum(len(x) for x in out) > max_len:
                break
    return ("\n\n".join(out))[:max_len]


def _top_terms_weighted(sources: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for text, w in sources:
        if not text:
            continue
        for t in tokenize_terms(text, top_k=256):
            scores[t] = scores.get(t, 0.0) + float(w)
    pairs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[: top_k]
    return [(k, v) for k, v in pairs]


def _paths_from_text(text: str, max_n: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _PATH_RE.finditer(text or ""):
        p = (m.group(0) or "").strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
            if len(out) >= max_n:
                break
    return out


def build_unified_brain_block(
    ctx_body: str,
    memory_text: str,
    task_text: str,
    evergreen_text: str = "",
    *,
    top_k: int | None = None,
) -> str:
    """Fuse <embeddings_*> + <memory> + <task> into a compact <embeddings_brain> block.

    Heuristics (RT-safe):
    - Extract inner bodies of <embeddings_code> and <embeddings_context> (clamped).
    - Weight tokens: task=3.0, memory=2.0, code/context=1.0, evergreen=1.0.
    - Include a few path anchors from context.
    - Return build_brain_block() lines like: '- term: foo (score)'.
    """
    try:
        _TOP = max(6, int(os.getenv("EMBED_BRAIN_BLOCK_TOP", "12")))
    except Exception:
        _TOP = 12
    try:
        _CTX_CLAMP = max(400, int(os.getenv("EMBED_UNIFY_CTX_CLAMP", "2400")))
    except Exception:
        _CTX_CLAMP = 2400
    try:
        _MEM_CLAMP = max(200, int(os.getenv("EMBED_UNIFY_MEM_CLAMP", "1200")))
    except Exception:
        _MEM_CLAMP = 1200

    code_body = _extract_blocks_multi(ctx_body or "", ["embeddings_code"], _CTX_CLAMP)
    ctx_extra = _extract_blocks_multi(ctx_body or "", ["embeddings_context"], _CTX_CLAMP)
    refs_body = _extract_blocks_multi(ctx_body or "", ["embeddings_refs"], max(400, _CTX_CLAMP // 2))
    graph_body = _extract_blocks_multi(ctx_body or "", ["embeddings_graph"], max(400, _CTX_CLAMP // 2))
    memsel_body = _extract_blocks_multi(ctx_body or "", ["memory_selected", "pins", "memory_context", "memory_graph"], _MEM_CLAMP)

    # Build weighted term list
    sources: list[tuple[str, float]] = []
    sources.append(((task_text or "").strip()[:800], 3.0))
    sources.append(((memory_text or "").strip()[:_MEM_CLAMP], 2.0))
    sources.append((memsel_body, 2.4))
    sources.append(((evergreen_text or "").strip()[:800], 0.9))
    sources.append((code_body, 1.2))
    sources.append((ctx_extra, 1.0))
    sources.append((refs_body, 0.9))
    sources.append((graph_body, 1.0))

    topn = int(top_k or _TOP)
    term_pairs = _top_terms_weighted(sources, topn * 2)

    # Convert into brain keys with prefix 'term:' and normalize score 0..1
    if term_pairs:
        maxv = max((v for _, v in term_pairs), default=1.0)
    else:
        maxv = 1.0
    brain_pairs: list[tuple[str, float]] = [(f"term: {k}", (v / maxv) if maxv > 0 else 0.0) for k, v in term_pairs[:topn]]

    # Add a couple of path anchors extracted from context
    paths = _paths_from_text("\n".join([ctx_body or "", memsel_body or "", memory_text or ""])[:4096], max_n=8)
    for p in paths[:max(2, min(6, topn // 3))]:
        brain_pairs.append((f"path: {p}", 0.85))

    # Build final brain block
    return build_brain_block(brain_pairs, topn=topn)


__all__ = ["build_unified_brain_block"]
