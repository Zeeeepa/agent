from __future__ import annotations

import hashlib
import os
import time
from typing import Optional, Tuple
from collections import OrderedDict

# Size-limited TTL cache for prefetched contexts with simple LRU eviction.

_DEF_TTL = 12.0  # seconds

try:
    _MAX_PROJ = max(32, int(os.getenv("JINX_PREFETCH_CACHE_MAX_PROJECT", "256")))
except Exception:
    _MAX_PROJ = 256
try:
    _MAX_BASE = max(32, int(os.getenv("JINX_PREFETCH_CACHE_MAX_BASE", "256")))
except Exception:
    _MAX_BASE = 256

_proj: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
_base: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()

_hits_proj = 0
_miss_proj = 0
_hits_base = 0
_miss_base = 0


def _norm(q: str) -> str:
    s = (q or "").strip().lower()
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) > 512:
        s = s[:512]
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _evict(d: "OrderedDict[str, Tuple[float, str]]", cap: int) -> None:
    while len(d) > max(16, cap):
        try:
            d.popitem(last=False)  # oldest first
        except Exception:
            break


def put_project(q: str, ctx: str, *, ttl: float = _DEF_TTL) -> None:
    if not ctx:
        return
    k = _norm(q)
    _proj.pop(k, None)
    _proj[k] = (time.perf_counter() + max(0.5, float(ttl)), ctx)
    _evict(_proj, _MAX_PROJ)


def get_project(q: str) -> Optional[str]:
    global _hits_proj, _miss_proj
    k = _norm(q)
    v = _proj.get(k)
    if not v:
        _miss_proj += 1
        return None
    exp, ctx = v
    if time.perf_counter() > exp:
        _proj.pop(k, None)
        _miss_proj += 1
        return None
    # LRU bump
    _proj.move_to_end(k, last=True)
    _hits_proj += 1
    return ctx


def put_base(q: str, ctx: str, *, ttl: float = _DEF_TTL) -> None:
    if not ctx:
        return
    k = _norm(q)
    _base.pop(k, None)
    _base[k] = (time.perf_counter() + max(0.5, float(ttl)), ctx)
    _evict(_base, _MAX_BASE)


def get_base(q: str) -> Optional[str]:
    global _hits_base, _miss_base
    k = _norm(q)
    v = _base.get(k)
    if not v:
        _miss_base += 1
        return None
    exp, ctx = v
    if time.perf_counter() > exp:
        _base.pop(k, None)
        _miss_base += 1
        return None
    _base.move_to_end(k, last=True)
    _hits_base += 1
    return ctx


def get_metrics() -> dict:
    """Return simple cache metrics for observability and tuning."""
    try:
        return {
            "proj": {"hits": int(_hits_proj), "miss": int(_miss_proj), "size": len(_proj)},
            "base": {"hits": int(_hits_base), "miss": int(_miss_base), "size": len(_base)},
        }
    except Exception:
        return {"proj": {"hits": 0, "miss": 0, "size": 0}, "base": {"hits": 0, "miss": 0, "size": 0}}

__all__ = [
    "put_project",
    "get_project",
    "put_base",
    "get_base",
    "get_metrics",
]
