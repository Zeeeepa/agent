from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from jinx.micro.memory.storage import memory_dir

_FAST_PATH = os.path.join(memory_dir(), "graph_fast.json")
_FAST_TMP = _FAST_PATH + ".tmp"

# In-process aggregator (best-effort)
_AGG: Dict[str, float] = {}
_AGG_COUNT: int = 0
_AGG_LAST_FLUSH_MS: int = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load() -> Dict[str, Any]:
    try:
        with open(_FAST_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj  # type: ignore[return-value]
    except Exception:
        pass
    return {"edges": {}, "updated_ts": 0}


def _atomic_save(d: Dict[str, Any]) -> None:
    try:
        os.makedirs(memory_dir(), exist_ok=True)
        with open(_FAST_TMP, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(_FAST_TMP, _FAST_PATH)
    except Exception:
        pass


def _decay(v: float, now_ms: int, last_ms: int, half_ms: float) -> float:
    if last_ms <= 0 or half_ms <= 0:
        return v
    import math
    return float(v) * (0.5 ** (max(0.0, (now_ms - last_ms)) / half_ms))


def read_fast_edges() -> Dict[str, float]:
    """Return decayed fast edges overlay."""
    d = _load()
    now = _now_ms()
    try:
        half_ms = float(os.getenv("JINX_MEM_GRAPH_FAST_HALF_LIFE_MS", str(10 * 60 * 1000)))
    except Exception:
        half_ms = float(10 * 60 * 1000)
    last = int(d.get("updated_ts") or 0)
    out: Dict[str, float] = {}
    edges = d.get("edges") or {}
    for ek, w in edges.items():
        try:
            dw = _decay(float(w), now, last, half_ms)
            if dw > 1e-6:
                out[str(ek)] = dw
        except Exception:
            continue
    return out


def _flush(now: int) -> None:
    global _AGG, _AGG_COUNT, _AGG_LAST_FLUSH_MS
    if not _AGG:
        return
    d = _load()
    edges: Dict[str, float] = dict(d.get("edges") or {})
    # apply aggregated deltas
    for ek, delta in _AGG.items():
        try:
            edges[ek] = float(edges.get(ek, 0.0)) + float(delta)
        except Exception:
            edges[ek] = float(delta)
    _AGG = {}
    _AGG_COUNT = 0
    d["edges"] = edges
    d["updated_ts"] = now
    # Prune & cap
    try:
        max_edges = int(os.getenv("JINX_MEM_GRAPH_FAST_MAX_EDGES", "0"))
    except Exception:
        max_edges = 0
    try:
        thr = float(os.getenv("JINX_MEM_GRAPH_FAST_MIN_ABS", "0"))
    except Exception:
        thr = 0.0
    # soft cap per edge
    try:
        edge_cap = float(os.getenv("JINX_MEM_GRAPH_FAST_EDGE_CAP", "0"))
    except Exception:
        edge_cap = 0.0
    # sparsify by threshold and cap
    pruned: Dict[str, float] = {}
    for k, v in edges.items():
        try:
            vv0 = float(v)
        except Exception:
            vv0 = 0.0
        if edge_cap > 0.0:
            vv0 = max(-edge_cap, min(edge_cap, vv0))
        if thr > 0.0:
            if abs(vv0) >= thr:
                pruned[k] = vv0
        else:
            pruned[k] = vv0
    if max_edges > 0 and len(pruned) > max_edges:
        items = sorted(pruned.items(), key=lambda kv: -abs(float(kv[1])))
        pruned = {k: v for k, v in items[:max_edges]}
    d["edges"] = pruned
    _atomic_save(d)
    _AGG_LAST_FLUSH_MS = now


def update_fast(seeds: List[str], winners: List[str], amount: float = 1.0) -> None:
    """Lightweight Hebbian overlay: connect seed->winner pairs, bump weight.

    Writes with current timestamp, no heavy read-modify cycles.
    """
    global _AGG, _AGG_COUNT
    now = _now_ms()
    try:
        flush_min_ms = int(os.getenv("JINX_MEM_GRAPH_FAST_FLUSH_MIN_MS", "200"))
    except Exception:
        flush_min_ms = 200
    try:
        flush_max_deltas = int(os.getenv("JINX_MEM_GRAPH_FAST_FLUSH_MAX_DELTAS", "1000"))
    except Exception:
        flush_max_deltas = 1000

    for a in seeds:
        for b in winners:
            if not a or not b or a == b:
                continue
            ek = a + "||" + b if a <= b else b + "||" + a
            _AGG[ek] = _AGG.get(ek, 0.0) + float(amount)
            _AGG_COUNT += 1
    # Debounced flush
    if (_AGG_COUNT >= flush_max_deltas) or (now - _AGG_LAST_FLUSH_MS >= flush_min_ms):
        _flush(now)
