from __future__ import annotations

import os
import time
import threading
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .project_config import ROOT
from .project_retrieval_config import (
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_MULTI_SEGMENT_ENABLE,
    PROJ_SEGMENT_HEAD_LINES,
    PROJ_SEGMENT_TAIL_LINES,
    PROJ_SEGMENT_MID_WINDOWS,
    PROJ_SEGMENT_MID_AROUND,
    PROJ_SEGMENT_STRIP_COMMENTS,
    PROJ_SCOPE_MAX_CHARS,
)

def _env_int(name: str, default: int, *, minv: int | None = None, maxv: int | None = None) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except Exception:
        v = default
    if minv is not None:
        v = max(minv, v)
    if maxv is not None:
        v = min(maxv, v)
    return v

# TTL for snippet cache entries (milliseconds). Default-on, clamped 0..600_000
_SNIPPET_TTL_MS = _env_int("EMBED_PROJECT_SNIPPET_TTL_MS", 1200, minv=0, maxv=600_000)

# In-memory cache: key -> (ts_ms, (header, code_block, ls, le, is_full_scope))
_Cache: Dict[str, Tuple[int, Tuple[str, str, int, int, bool]]] = {}
_Lock = threading.Lock()
_MAX_ENTRIES = _env_int("EMBED_PROJECT_SNIPPET_CACHE_MAX", 1024, minv=16, maxv=1_000_000)

# Coalescing of concurrent builds
_Inflight: Dict[str, threading.Event] = {}
_COALESCE_WAIT_MS_DEFAULT = _env_int("EMBED_PROJECT_SNIPPET_COALESCE_WAIT_MS", 400, minv=0, maxv=2000)


def _now_ms() -> int:
    # Monotonic clock for TTL to avoid wall-clock jumps
    try:
        return int(time.monotonic_ns() // 1_000_000)
    except Exception:
        return int(time.time() * 1000)


def _hash_text(s: str) -> str:
    try:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    except Exception:
        return str(abs(hash(s)))


def _file_sig(file_rel: str) -> Tuple[int, int]:
    """Return (mtime_ns, size) signature; (0,0) on error.

    Uses high-resolution mtime when available to avoid collisions on rapid edits.
    """
    try:
        p = os.path.join(ROOT, file_rel)
        st = os.stat(p)
        mt_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
        return (mt_ns, int(st.st_size))
    except Exception:
        return (0, 0)


def file_signature(file_rel: str) -> Tuple[int, int]:
    """Public helper for other modules to snapshot current file signature."""
    return _file_sig(file_rel)


def make_snippet_cache_key(
    file_rel: str,
    meta: Dict[str, Any],
    query: str,
    *,
    prefer_full_scope: bool,
    expand_callees: bool,
    extra_centers_abs: Optional[List[int]],
    file_sig: Optional[Tuple[int, int]] = None,
) -> str:
    """Build a stable cache key for a snippet build request.

    Includes file signature, meta range, query hash, key shaping knobs and options.
    """
    ls = int((meta or {}).get("line_start") or 0)
    le = int((meta or {}).get("line_end") or 0)
    if file_sig is None:
        mt, sz = _file_sig(file_rel)
    else:
        mt, sz = file_sig
    qh = _hash_text(query or "")
    centers_key = ",".join(str(int(x)) for x in sorted(set(extra_centers_abs or []))[:16])
    knobs = (
        PROJ_SNIPPET_PER_HIT_CHARS,
        1 if PROJ_MULTI_SEGMENT_ENABLE else 0,
        PROJ_SEGMENT_HEAD_LINES,
        PROJ_SEGMENT_TAIL_LINES,
        PROJ_SEGMENT_MID_WINDOWS,
        PROJ_SEGMENT_MID_AROUND,
        1 if PROJ_SEGMENT_STRIP_COMMENTS else 0,
        PROJ_SCOPE_MAX_CHARS,
    )
    # Key version to avoid collisions across format changes
    key = f"v1|{file_rel}|{mt}|{sz}|{ls}|{le}|{qh}|pf{int(prefer_full_scope)}|xc{int(expand_callees)}|cent[{centers_key}]|knobs{knobs}"
    return key


def get_cached_snippet(key: str) -> Optional[Tuple[str, str, int, int, bool]]:
    if _SNIPPET_TTL_MS <= 0:
        return None
    now = _now_ms()
    with _Lock:
        ent = _Cache.get(key)
        if not ent:
            return None
        ts, val = ent
        if now - ts > _SNIPPET_TTL_MS:
            try:
                del _Cache[key]
            except Exception:
                pass
            return None
        return val


def invalidate_file(file_rel: str) -> int:
    """Drop all cache entries for a given relative file. Returns number removed."""
    if not file_rel:
        return 0
    pref = f"{file_rel}|"
    removed = 0
    with _Lock:
        try:
            to_del = [k for k in _Cache.keys() if k.startswith(pref)]
            for k in to_del:
                _Cache.pop(k, None)
                removed += 1
        except Exception:
            return removed
    return removed


def invalidate_all() -> int:
    """Clear the entire snippet cache. Returns previous size."""
    with _Lock:
        n = len(_Cache)
        _Cache.clear()
        _Inflight.clear()
        return n


def coalesce_enter(key: str, wait_ms: Optional[int] = None) -> Tuple[str, Optional[threading.Event]]:
    """Return (mode, event). mode is 'leader' or 'wait'.

    If 'wait', caller should wait on the event up to wait_ms and then retry cache.
    """
    with _Lock:
        ev = _Inflight.get(key)
        if ev is None:
            ev = threading.Event()
            _Inflight[key] = ev
            return ("leader", None)
        else:
            return ("wait", ev)


def coalesce_exit(key: str) -> None:
    ev: Optional[threading.Event]
    with _Lock:
        ev = _Inflight.pop(key, None)
    if ev is not None:
        try:
            ev.set()
        except Exception:
            pass


def coalesce_wait_ms() -> int:
    return int(_COALESCE_WAIT_MS_DEFAULT)


def stats() -> Dict[str, int]:
    """Return snapshot stats for debugging: size and inflight count."""
    with _Lock:
        return {"size": len(_Cache), "inflight": len(_Inflight)}


def put_cached_snippet(key: str, value: Tuple[str, str, int, int, bool]) -> None:
    if _SNIPPET_TTL_MS <= 0:
        return
    now = _now_ms()
    with _Lock:
        _Cache[key] = (now, value)
        # Simple LRU-ish cap: drop oldest entries when exceeding max size
        if _MAX_ENTRIES > 0 and len(_Cache) > _MAX_ENTRIES:
            try:
                # Evict ~1/16th oldest to amortize cost
                n_evict = max(1, len(_Cache) // 16)
                # Sort keys by timestamp ascending (oldest first)
                oldest = sorted(_Cache.items(), key=lambda kv: kv[1][0])[:n_evict]
                for k, _ in oldest:
                    if k == key:
                        continue
                    _Cache.pop(k, None)
            except Exception:
                pass
