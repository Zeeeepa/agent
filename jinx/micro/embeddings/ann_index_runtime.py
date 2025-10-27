from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple, Optional

try:
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hnswlib = None  # type: ignore

from .similarity import score_cosine_batch

# ANN overlay for runtime embeddings (log/embeddings items).
# Items are (source, obj) where obj contains {"embedding": [...], "meta": {...}}.
# Falls back to brute-force cosine if ANN is disabled/unavailable.

class _AnnState:
    def __init__(self) -> None:
        self.key: Tuple[int, int] | None = None  # (dims, n_items)
        self.expires_at: float = 0.0
        self.labels: List[int] = []
        self.vectors: List[List[float]] = []
        self.meta: List[Tuple[str, Dict[str, Any]]] = []  # (src, obj)
        self.index: Any | None = None  # hnswlib.Index if available

_ANN = _AnnState()


def _ttl_sec() -> float:
    try:
        return max(0.5, float(os.getenv("EMBED_RUNTIME_ANN_TTL_SEC", "10")))
    except Exception:
        return 10.0


def _ann_enabled() -> bool:
    try:
        return os.getenv("EMBED_RUNTIME_ANN", "1").strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def _should_rebuild(dims: int, n: int) -> bool:
    if _ANN.key != (dims, n):
        return True
    return (time.time() > _ANN.expires_at)


def _reset_state() -> None:
    _ANN.key = None
    _ANN.expires_at = 0.0
    _ANN.labels = []
    _ANN.vectors = []
    _ANN.meta = []
    _ANN.index = None


def _normalize(vec: List[float]) -> List[float]:
    try:
        import math
        s = math.sqrt(sum((x or 0.0) * (x or 0.0) for x in (vec or [])))
        if s <= 0.0:
            return list(vec or [])
        return [(x or 0.0) / s for x in vec]
    except Exception:
        return list(vec or [])


def _build_hnsw(dims: int, vecs: List[List[float]]) -> Any | None:
    if hnswlib is None:
        return None
    try:
        idx = hnswlib.Index(space='cosine', dim=dims)
        M = max(8, int(os.getenv("EMBED_RUNTIME_HNSW_M", "24")))
        efc = max(50, int(os.getenv("EMBED_RUNTIME_HNSW_EF_CONSTRUCT", "100")))
        idx.init_index(max_elements=len(vecs), ef_construction=efc, M=M)
        idx.add_items(vecs, list(range(len(vecs))))
        ef = max(50, int(os.getenv("EMBED_RUNTIME_HNSW_EF", "200")))
        idx.set_ef(ef)
        return idx
    except Exception:
        return None


def ensure_index(items: List[Tuple[str, Dict[str, Any]]]) -> Tuple[int, List[List[float]], List[Tuple[str, Dict[str, Any]]], Any | None]:
    vecs: List[List[float]] = []
    meta: List[Tuple[str, Dict[str, Any]]] = []
    dims = 0
    for src, obj in (items or []):
        v = list((obj or {}).get("embedding") or [])
        if not v:
            continue
        if dims == 0:
            dims = len(v)
        if len(v) != dims:
            continue
        vecs.append(_normalize(v))
        meta.append((src, obj))
    n = len(vecs)
    if dims <= 0 or n == 0:
        _reset_state()
        return (0, [], [], None)
    if _should_rebuild(dims, n):
        _ANN.key = (dims, n)
        _ANN.expires_at = time.time() + _ttl_sec()
        _ANN.vectors = vecs
        _ANN.meta = meta
        _ANN.labels = list(range(n))
        _ANN.index = _build_hnsw(dims, vecs)
    return (dims, _ANN.vectors, _ANN.meta, _ANN.index)


def search_ann_items(qv: List[float], items: List[Tuple[str, Dict[str, Any]]], top_n: int) -> List[Tuple[int, float]]:
    if not _ann_enabled():
        vecs = [list((obj or {}).get("embedding") or []) for _src, obj in items]
        sims = score_cosine_batch(qv, vecs)
        order = list(range(len(sims)))
        order.sort(key=lambda i: float(sims[i] or 0.0), reverse=True)
        return [(i, sims[i]) for i in order[:top_n]]
    dims, vecs, meta, index = ensure_index(items)
    if dims <= 0 or not vecs:
        return []
    if index is None:
        sims = score_cosine_batch(qv, vecs)
        order = list(range(len(sims)))
        order.sort(key=lambda i: float(sims[i] or 0.0), reverse=True)
        return [(i, sims[i]) for i in order[:top_n]]
    try:
        qn = _normalize(qv)
        k = max(1, int(top_n) * 2)
        labels, distances = index.knn_query([qn], k=min(k, len(vecs)))  # type: ignore[attr-defined]
        labs = list(labels[0]) if labels is not None else []
        cand_vecs = [vecs[i] for i in labs]
        sims = score_cosine_batch(qn, cand_vecs)
        pairs = list(zip(labs, sims))
        pairs.sort(key=lambda p: float(p[1] or 0.0), reverse=True)
        return pairs[:top_n]
    except Exception:
        sims = score_cosine_batch(qv, vecs)
        order = list(range(len(sims)))
        order.sort(key=lambda i: float(sims[i] or 0.0), reverse=True)
        return [(i, sims[i]) for i in order[:top_n]]
