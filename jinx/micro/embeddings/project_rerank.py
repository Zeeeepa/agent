from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from .project_query_core import extract_code_core
from jinx.micro.memory.graph import read_graph_edges, read_graph_nodes
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.embeddings.project_identifiers import extract_identifiers as _extract_ids
from jinx.micro.brain.attention import get_attention_weights as _atten_get

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


def _edge_key(a: str, b: str) -> str:
    return a + "||" + b if a <= b else b + "||" + a


def _hit_tokens(meta: Dict[str, Any], rel: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (terms, paths, symbols) lowercased for a hit.

    - terms: derived from preview tokens
    - paths: from file_rel
    - symbols: identifiers extracted from preview
    """
    pv = (meta.get("text_preview") or "").lower()
    terms = []
    for m in _TOK_RE.finditer(pv):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 3:
            terms.append(t)
    file_rel = (meta.get("file_rel") or rel or "").strip()
    paths = [file_rel.lower()] if file_rel and not str(file_rel).startswith("memory://") else []
    syms: List[str] = []
    try:
        for tok in _extract_ids(pv, max_items=64):
            st = (tok or "").strip().lower()
            if st:
                syms.append(st)
    except Exception:
        pass
    return terms, paths, syms


def rerank_hits_unified(hits: List[Tuple[float, str, Dict[str, Any]]], query: str) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Unified reranker for project+memory hits with source-aware and KG-aware boosts.

    - Source-aware: prefer code hits for code-like queries; prefer memory hits for non-code queries.
    - Path/preview token boosts as in rerank_hits.
    - KG-aware: use knowledge graph edges between query terms and hit terms/paths.
    """
    if not hits:
        return []
    q = (query or "")
    core_q = extract_code_core(q) or q
    qtok = _query_tokens(core_q)
    code_like = _is_code_like(q or "")

    # Load KG once (best-effort)
    edges = read_graph_edges()  # {'a||b': w}
    nodes = read_graph_nodes()  # {key: w}
    # Snapshot of attention weights (short-term working memory)
    try:
        atten = _atten_get()
    except Exception:
        atten = {}
    def _node_w(k: str) -> float:
        try:
            return float(nodes.get(k, 0.0))
        except Exception:
            return 0.0

    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for sc, rel, obj in hits:
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").lower()
        rel_l = (str(meta.get("file_rel") or rel) or "").lower()
        is_mem = str(rel).startswith("memory://") or bool(meta.get("memory_id"))

        # Base path/preview boosts
        boost = 0.0
        pos_list: List[int] = []
        for t in qtok:
            if rel_l and t in rel_l:
                boost += 0.3
            elif t in pv:
                boost += 0.15
                try:
                    pos = pv.find(t)
                    if pos >= 0:
                        pos_list.append(pos)
                except Exception:
                    pass
        if len(pos_list) >= 2:
            pos_list.sort()
            try:
                span = max(pos_list) - min(pos_list)
                if span <= 24:
                    boost += 0.2
            except Exception:
                pass

        # Source-aware boost
        if code_like:
            boost += 0.12 if not is_mem else 0.05
        else:
            boost += 0.12 if is_mem else 0.05

        # KG-aware boost
        if edges:
            terms, paths, syms = _hit_tokens(meta, rel_l)
            hit_keys: List[str] = [f"term: {t}" for t in terms]
            if paths:
                for p in paths:
                    hit_keys.append(f"path: {p}")
            if syms:
                for s in syms:
                    hit_keys.append(f"symbol: {s}")
            # query keys: terms + symbols
            q_keys: List[str] = [f"term: {t}" for t in qtok]
            try:
                for st in _extract_ids(core_q, max_items=64):
                    s = (st or "").strip().lower()
                    if s:
                        q_keys.append(f"symbol: {s}")
            except Exception:
                pass
            kg_sum = 0.0
            for a in hit_keys:
                for b in q_keys:
                    ek = _edge_key(a, b)
                    w = float(edges.get(ek, 0.0))
                    # weigh by neighbor node importance
                    if w != 0.0:
                        kg_sum += w * max(_node_w(a), 0.5)
            try:
                beta = float((__import__('os').getenv("EMBED_RERANK_KG_BETA", "0.25") or "0.25"))
            except Exception:
                beta = 0.25
            boost += beta * kg_sum

        # Attention-aware boost: sum attention on hit keys
        if atten:
            try:
                gamma = float((__import__('os').getenv("EMBED_RERANK_ATTEN_GAMMA", "0.35") or "0.35"))
            except Exception:
                gamma = 0.35
            att_sum = 0.0
            try:
                for k in hit_keys:
                    try:
                        att_sum += float(atten.get(k, 0.0))
                    except Exception:
                        continue
            except Exception:
                att_sum = att_sum
            if att_sum != 0.0:
                boost += gamma * att_sum

        scored.append((float(sc or 0.0) + boost, rel, obj))

    return sorted(scored, key=lambda h: float(h[0] or 0.0), reverse=True)


__all__ = [
    "rerank_hits",
    "rerank_hits_unified",
]
