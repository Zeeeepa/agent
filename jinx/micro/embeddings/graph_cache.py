from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Dict, List, Tuple

from .project_callgraph import build_symbol_graph as _build_symbol_graph
from .project_refs import find_usages_in_project as _find_usages
from .snippet_cache import file_signature as _file_signature

# TTLs (ms)
try:
    _GRAPH_TTL_MS = int(os.getenv("EMBED_PROJECT_GRAPH_TTL_MS", "1800"))
except Exception:
    _GRAPH_TTL_MS = 1800
try:
    _USAGES_TTL_MS = int(os.getenv("EMBED_PROJECT_USAGES_TTL_MS", "1800"))
except Exception:
    _USAGES_TTL_MS = 1800

_graph_cache: Dict[str, Tuple[int, List[Tuple[str, str]]]] = {}
_usages_cache: Dict[str, Tuple[int, List[Tuple[str, str]]]] = {}


def _now_ms() -> int:
    try:
        return int(time.monotonic_ns() // 1_000_000)
    except Exception:
        return int(time.time() * 1000)


def _graph_key(file_rel: str, ls: int, le: int, callers_limit: int, callees_limit: int, around: int, scan_cap_files: int, time_budget_ms: int | None) -> str:
    sig = _file_signature(file_rel)
    return f"v1|{file_rel}|{sig[0]}|{sig[1]}|{ls}|{le}|c{callers_limit}|e{callees_limit}|a{around}|s{scan_cap_files}|t{int(time_budget_ms or 0)}"


def _usages_key(sym_name: str, file_rel: str, limit: int, around: int) -> str:
    sig = _file_signature(file_rel)
    return f"v1|{sym_name}|{file_rel}|{sig[0]}|{sig[1]}|l{limit}|a{around}"


async def get_symbol_graph_cached(
    file_rel: str,
    ls: int,
    le: int,
    *,
    callers_limit: int,
    callees_limit: int,
    around: int,
    scan_cap_files: int,
    time_budget_ms: int,
) -> List[Tuple[str, str]]:
    if _GRAPH_TTL_MS <= 0:
        def _work() -> List[Tuple[str, str]]:
            try:
                return _build_symbol_graph(
                    file_rel,
                    ls,
                    le,
                    callers_limit=callers_limit,
                    callees_limit=callees_limit,
                    around=around,
                    scan_cap_files=scan_cap_files,
                    time_budget_ms=time_budget_ms,
                )
            except Exception:
                return []
        return await asyncio.to_thread(_work)
    key = _graph_key(file_rel, ls, le, callers_limit, callees_limit, around, scan_cap_files, time_budget_ms)
    now = _now_ms()
    ent = _graph_cache.get(key)
    if ent and now - ent[0] <= _GRAPH_TTL_MS:
        return ent[1]
    def _work() -> List[Tuple[str, str]]:
        try:
            return _build_symbol_graph(
                file_rel,
                ls,
                le,
                callers_limit=callers_limit,
                callees_limit=callees_limit,
                around=around,
                scan_cap_files=scan_cap_files,
                time_budget_ms=time_budget_ms,
            )
        except Exception:
            return []
    pairs = await asyncio.to_thread(_work)
    _graph_cache[key] = (now, pairs or [])
    return pairs or []


async def find_usages_cached(sym_name: str, file_rel: str, *, limit: int, around: int) -> List[Tuple[str, int, int, str, str | None]]:
    if _USAGES_TTL_MS <= 0:
        def _work():
            try:
                return _find_usages_in_project(sym_name, file_rel, limit=limit, around=around)
            except Exception:
                return []
        return await asyncio.to_thread(_work)
    key = _usages_key(sym_name, file_rel, limit, around)
    now = _now_ms()
    ent = _usages_cache.get(key)
    if ent and now - ent[0] <= _USAGES_TTL_MS:
        return ent[1]  # type: ignore[return-value]
    def _work():
        try:
            return _find_usages(sym_name, file_rel, limit=limit, around=around)
        except Exception:
            return []
    out = await asyncio.to_thread(_work)
    _usages_cache[key] = (now, out or [])
    return out or []
