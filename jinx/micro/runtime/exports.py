from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from .api import list_programs, get_program


async def _maybe_await(x: Any) -> Any:
    if asyncio.iscoroutine(x):
        return await x
    return x


async def get_program_export(pid: str, key: str) -> str:
    """Best-effort fetch of a single program's export value for key.

    Supports either a `.exports` dict attribute or a `.get_export(key)` method
    (which may be async or sync). Returns an empty string if missing or on error.
    """
    try:
        prog = await get_program(pid)
        if not prog:
            return ""
        if hasattr(prog, "exports"):
            try:
                exports = getattr(prog, "exports")
                if isinstance(exports, dict):
                    v = exports.get(key)
                    return "" if v is None else str(v)
            except Exception:
                return ""
        if hasattr(prog, "get_export"):
            try:
                fn = getattr(prog, "get_export")
                v = await _maybe_await(fn(key))
                return "" if v is None else str(v)
            except Exception:
                return ""
    except Exception:
        return ""
    return ""


async def collect_export(key: str, limit: Optional[int] = None) -> List[str]:
    """Collect export values for key across active programs.

    Returns up to `limit` non-empty string values (order by registry listing).
    """
    out: List[str] = []
    try:
        pids = await list_programs()
    except Exception:
        pids = []
    for pid in pids:
        try:
            v = await get_program_export(pid, key)
            if v:
                out.append(v)
                if limit is not None and len(out) >= max(0, limit):
                    break
        except Exception:
            continue
    return out


async def collect_export_for_group(key: str, group: str, limit: Optional[int] = None) -> List[str]:
    """Collect export values preferring group-specific keys.

    Tries key formatted as "group:{group}:{key}" for each program, falling back to the global key.
    Returns up to `limit` non-empty values.
    """
    out: List[str] = []
    try:
        pids = await list_programs()
    except Exception:
        pids = []
    gk = f"group:{(group or 'main').strip().lower()}:{key}"
    for pid in pids:
        try:
            v = await get_program_export(pid, gk)
            if not v:
                v = await get_program_export(pid, key)
            if v:
                out.append(v)
                if limit is not None and len(out) >= max(0, limit):
                    break
        except Exception:
            continue
    return out
