from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple, Optional

try:
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hnswlib = None  # type: ignore

from .similarity import score_cosine_batch
from .project_paths import PROJECT_INDEX_DIR

# Lightweight, best-effort ANN cache for project chunks.
# Falls back to brute-force cosine if ANN backend not available.

class _AnnState:
    def __init__(self) -> None:
        self.key: Tuple[int, int] | None = None  # (dims, n_items)
        self.expires_at: float = 0.0
        self.labels: List[int] = []
        self.vectors: List[List[float]] = []
        self.meta: List[Tuple[str, Dict[str, Any]]] = []  # (file_rel, obj)
        self.index: Any | None = None  # hnswlib.Index if available

_ANN = _AnnState()


def _ttl_sec() -> float:
    try:
        return max(0.5, float(os.getenv("EMBED_PROJECT_ANN_TTL_SEC", "10")))
    except Exception:
        return 10.0


def _ann_enabled() -> bool:
    try:
        return os.getenv("EMBED_PROJECT_ANN", "1").strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def _persist_enabled() -> bool:
    try:
        return os.getenv("EMBED_PROJECT_ANN_PERSIST", "1").strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def _persist_path() -> str:
    try:
        os.makedirs(PROJECT_INDEX_DIR, exist_ok=True)
    except Exception:
        pass
    return os.path.join(PROJECT_INDEX_DIR, "ann_state.pkl")


def _persist_save_async(dims: int, vecs: List[List[float]], meta: List[Tuple[str, Dict[str, Any]]]) -> None:
    if not _persist_enabled():
        return
    path = _persist_path()
    payload = {
        "dims": int(dims),
        "n": int(len(vecs)),
        "vectors": vecs,
        "meta": meta,
        "ts": time.time(),
    }
    import asyncio
    import pickle
    def _write() -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    # Prefer scheduling on the running loop; otherwise fall back to a background thread
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread — use a daemon thread
        try:
            import threading
            threading.Thread(target=_write, daemon=True).start()
        except Exception:
            pass
        return
    try:
        # Fire-and-forget in default executor; no coroutine is created if scheduling fails
        loop.run_in_executor(None, _write)
    except Exception:
        # Best-effort fallback
        try:
            import threading
            threading.Thread(target=_write, daemon=True).start()
        except Exception:
            pass


def _persist_load(dims: int, n: int) -> Optional[Tuple[List[List[float]], List[Tuple[str, Dict[str, Any]]]]]:
    if not _persist_enabled():
        return None
    path = _persist_path()
    if not os.path.isfile(path):
        return None
    try:
        import pickle
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if int(payload.get("dims") or 0) != int(dims):
            return None
        if int(payload.get("n") or 0) != int(n):
            return None
        vecs = list(payload.get("vectors") or [])
        meta = list(payload.get("meta") or [])
        if not vecs or not meta or len(vecs) != n or len(meta) != n:
            return None
        return (vecs, meta)
    except Exception:
        return None


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


def _build_hnsw(dims: int, vecs: List[List[float]]) -> Any | None:
    if hnswlib is None:
        return None
    try:
        idx = hnswlib.Index(space='cosine', dim=dims)
        M = max(8, int(os.getenv("EMBED_PROJECT_HNSW_M", "24")))
        efc = max(50, int(os.getenv("EMBED_PROJECT_HNSW_EF_CONSTRUCT", "100")))
        idx.init_index(max_elements=len(vecs), ef_construction=efc, M=M)
        idx.add_items(vecs, list(range(len(vecs))))
        # Set default ef for search; can be overridden in search call
        try:
            ef = max(50, int(os.getenv("EMBED_PROJECT_HNSW_EF", "200")))
        except Exception:
            ef = 200
        idx.set_ef(ef)
        return idx
    except Exception:
        return None


def _normalize(vec: List[float]) -> List[float]:
    # Safety: normalize to unit length to ensure cosine correctness if backend uses dot
    try:
        import math
        s = math.sqrt(sum((x or 0.0) * (x or 0.0) for x in (vec or [])))
        if s <= 0.0:
            return list(vec or [])
        return [(x or 0.0) / s for x in vec]
    except Exception:
        return list(vec or [])


def ensure_index(items: List[Tuple[str, Dict[str, Any]]]) -> Tuple[int, List[List[float]], List[Tuple[str, Dict[str, Any]]], Any | None]:
    """Ensure ANN state for given items; returns (dims, vectors, meta, index)."""
    # Extract vectors and filter invalids
    vecs: List[List[float]] = []
    meta: List[Tuple[str, Dict[str, Any]]] = []
    dims = 0
    for file_rel, obj in (items or []):
        v = list(obj.get("embedding") or [])
        if not v:
            continue
        if dims == 0:
            dims = len(v)
        if len(v) != dims:
            # skip mismatched dims defensively
            continue
        vecs.append(_normalize(v))
        meta.append((file_rel, obj))
    n = len(vecs)
    if dims <= 0 or n == 0:
        _reset_state()
        return (0, [], [], None)
    if _should_rebuild(dims, n):
        _ANN.key = (dims, n)
        _ANN.expires_at = time.time() + _ttl_sec()
        # Try persisted state first; fall back to freshly built vectors/meta
        loaded = _persist_load(dims, n)
        if loaded is not None:
            vecs_loaded, meta_loaded = loaded
            _ANN.vectors = vecs_loaded
            _ANN.meta = meta_loaded
        else:
            _ANN.vectors = vecs
            _ANN.meta = meta
        _ANN.labels = list(range(len(_ANN.vectors)))
        _ANN.index = _build_hnsw(dims, _ANN.vectors)
        # Persist new state asynchronously when we use fresh vectors
        if loaded is None:
            try:
                _persist_save_async(dims, _ANN.vectors, _ANN.meta)
            except Exception:
                pass
    return (dims, _ANN.vectors, _ANN.meta, _ANN.index)


def search_ann_items(qv: List[float], items: List[Tuple[str, Dict[str, Any]]], top_n: int, *, max_time_ms: Optional[int] = None) -> List[Tuple[int, float]]:
    """Return list of (idx, approx_distance_or_score) for candidate items.

    If ANN disabled/unavailable, returns indices ranked by brute-force cosine.
    """
    if not _ann_enabled():
        # brute-force path: compute cosine and return top indices by score
        vecs = [list(obj.get("embedding") or []) for _, obj in items]
        sims = score_cosine_batch(qv, vecs)
        order = list(range(len(sims)))
        order.sort(key=lambda i: float(sims[i] or 0.0), reverse=True)
        return [(i, sims[i]) for i in order[:top_n]]

    dims, vecs, meta, index = ensure_index(items)
    if dims <= 0 or not vecs:
        return []
    # If no ANN backend, return brute-force
    if index is None:
        sims = score_cosine_batch(qv, vecs)
        order = list(range(len(sims)))
        order.sort(key=lambda i: float(sims[i] or 0.0), reverse=True)
        return [(i, sims[i]) for i in order[:top_n]]
    # Use ANN to get labels, then re-score with exact cosine for top labels
    try:
        import math
        qn = _normalize(qv)
        k = max(1, int(top_n) * 2)  # overfetch to mitigate ANN error; will trim later
        labels, distances = index.knn_query([qn], k=min(k, len(vecs)))  # type: ignore[attr-defined]
        labs = list(labels[0]) if labels is not None else []
        # Retrieve candidate vecs and compute exact cosine
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
