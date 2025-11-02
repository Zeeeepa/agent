from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple
import time
import os
import asyncio

from jinx.micro.memory.storage import read_compact as _read_compact, read_evergreen as _read_evergreen, get_memory_mtimes as _get_mtimes
from jinx.micro.memory.graph_reasoner import activate as _graph_activate
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.memory.usage_store import weight_for as _usage_weight

# Module-level caches keyed by memory file mtimes
_C_MTIME: int = -1
_E_MTIME: int = -1
_C_CACHE: List[str] = []
_E_CACHE: List[str] = []

# Ranker TTL cache and coalescing
_RANK_CACHE: Dict[str, Tuple[int, List[str]]] = {}
_RANK_INFLIGHT: Dict[str, asyncio.Future] = {}
try:
    _RANK_TTL_MS = int(os.getenv("JINX_MEM_RANK_CACHE_TTL_MS", "800"))
except Exception:
    _RANK_TTL_MS = 800
try:
    _RANK_CONC_LIMIT = int(os.getenv("JINX_MEM_RANK_CONC_LIMIT", "8"))
except Exception:
    _RANK_CONC_LIMIT = 8
_RANK_SEM: asyncio.Semaphore = asyncio.Semaphore(max(1, _RANK_CONC_LIMIT))


def _tokens(s: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"(?u)[\w\.]+", s or ""):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 3:
            out.append(t)
    return out


def _lines_of(txt: str) -> List[str]:
    return [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]


def _build_idf(lines: List[str]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    N = max(1, len(lines))
    for ln in lines:
        seen: set[str] = set()
        for t in _tokens(ln):
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1.0) / (d + 1.0)) + 1.0
    return idf


def _is_error(ln: str) -> bool:
    low = ln.lower()
    return low.startswith("error:") or ("traceback" in low) or ("exception" in low)


def _is_codey(ln: str) -> bool:
    return _is_code_like(ln or "")


async def rank_memory(query: str, *, scope: str = "compact", k: int = 6, preview_chars: int = 160) -> List[str]:
    """Rank memory lines against a query using simple IDF and recency boosts.

    scope: "compact" | "evergreen" | "any"
    returns: list of top-k lines (trimmed to preview_chars)
    """
    q = (query or "").strip()
    if not q:
        return []
    # Cache by file mtimes (module-level cache)
    global _C_MTIME, _E_MTIME, _C_CACHE, _E_CACHE
    # TTL cache / coalescing key
    cache_key = f"{scope}|{k}|{preview_chars}|{q}"
    now_ms = int(time.time() * 1000)
    # fast hit
    ent = _RANK_CACHE.get(cache_key)
    if ent and (_RANK_TTL_MS <= 0 or (now_ms - ent[0]) <= _RANK_TTL_MS):
        return ent[1][:k]
    # coalesce in-flight
    fut = _RANK_INFLIGHT.get(cache_key)
    if fut is not None and not fut.done():
        try:
            res = await fut
            return list(res)[:k]
        except Exception:
            pass
    # prepare new future
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _RANK_INFLIGHT[cache_key] = fut

    try:
        m_c, m_e = _get_mtimes()
    except Exception:
        m_c, m_e = (0, 0)
    comp: List[str]
    ever: List[str]
    if _C_MTIME != m_c:
        try:
            comp = _lines_of(await _read_compact())
        except Exception:
            comp = []
        _C_CACHE = comp
        _C_MTIME = m_c
    else:
        comp = _C_CACHE or []
    if _E_MTIME != m_e:
        try:
            ever = _lines_of(await _read_evergreen())
        except Exception:
            ever = []
        _E_CACHE = ever
        _E_MTIME = m_e
    else:
        ever = _E_CACHE or []

    if scope == "evergreen":
        pool = ever
    elif scope == "any":
        pool = ever + comp
    else:
        pool = comp
    if not pool:
        # resolve coalesced future with empty result and cleanup
        try:
            _RANK_CACHE[cache_key] = (now_ms, [])
        except Exception:
            pass
        try:
            if not fut.done():
                fut.set_result([])
        except Exception:
            pass
        try:
            if _RANK_INFLIGHT.get(cache_key) is fut:
                _RANK_INFLIGHT.pop(cache_key, None)
        except Exception:
            pass
        return []
    # Clamp pool size and line length for RT safety
    try:
        max_lines = int(os.getenv("JINX_MEM_RANK_MAX_LINES", "1200"))
    except Exception:
        max_lines = 1200
    try:
        max_len = int(os.getenv("JINX_MEM_RANK_MAX_LINE_CHARS", "2000"))
    except Exception:
        max_len = 2000
    if max_lines > 0 and len(pool) > max_lines:
        # prefer tail for compact, balanced for any
        pool = pool[-max_lines:]
    if max_len > 0:
        pool = [ln[:max_len] for ln in pool]

    # Concurrency limit for heavy work
    async with _RANK_SEM:
        idf = _build_idf(pool)
    q_toks = _tokens(q)

    scored: List[Tuple[float, int, str]] = []
    L = len(pool)
    # time budget
    try:
        max_ms = float(os.getenv("JINX_MEM_RANK_MAX_MS", "40"))
    except Exception:
        max_ms = 40.0
    t0 = time.perf_counter()
    # Graph-based activation (optional, default ON)
    try:
        use_g = str(os.getenv("JINX_MEM_RANK_GRAPH", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        use_g = True
    g_hits: List[Tuple[str, float]] = []
    if use_g:
        try:
            gk = int(os.getenv("JINX_MEM_RANK_GRAPH_MAXNODES", "12"))
        except Exception:
            gk = 12
        try:
            g_hits = await _graph_activate(q, k=max(1, gk), steps=2)
        except Exception:
            g_hits = []
    try:
        gboost = float(os.getenv("JINX_MEM_RANK_GRAPH_BOOST", "0.15"))
    except Exception:
        gboost = 0.15

    # Optional usage-based weighting (default ON)
    try:
        use_usage = str(os.getenv("JINX_MEM_RANK_USAGE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        use_usage = True

    for i, ln in enumerate(pool):
        low = ln.lower()
        score = 0.0
        for t in q_toks:
            if t in low:
                score += idf.get(t, 1.0)
        # graph boost by matching activated node keys
        if g_hits and gboost > 0.0:
            for key, sc in g_hits:
                try:
                    if key and key.lower() in low:
                        score += gboost * float(sc)
                except Exception:
                    continue
        # boosts
        if _is_error(ln):
            score *= 1.15
        if _is_codey(ln):
            score *= 1.07
        # recency boost for compact (closer to tail)
        if scope != "evergreen" and L > 1:
            rec = 1.0 + 0.25 * (i / (L - 1))
            score *= rec
        # usage boost
        if use_usage and score > 0.0:
            try:
                w = float(_usage_weight(ln))
            except Exception:
                w = 1.0
            if w < 0.7:
                w = 0.7
            elif w > 2.0:
                w = 2.0
            score *= w
        scored.append((score, i, ln))
        if max_ms > 0 and (time.perf_counter() - t0) * 1000.0 > max_ms:
            break

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out: List[str] = []
    for sc, _idx, ln in scored:
        if sc <= 0:
            continue
        out.append(ln[:preview_chars])
        if len(out) >= max(1, k):
            break
    # store cache & resolve
    try:
        _RANK_CACHE[cache_key] = (now_ms, list(out))
    except Exception:
        pass
    try:
        if not fut.done():
            fut.set_result(list(out))
    except Exception:
        pass
    finally:
        # cleanup inflight
        try:
            if _RANK_INFLIGHT.get(cache_key) is fut:
                _RANK_INFLIGHT.pop(cache_key, None)
        except Exception:
            pass
    return out
