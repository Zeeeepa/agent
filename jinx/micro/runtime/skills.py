from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
from typing import Callable, Dict, Optional, Tuple, List

from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT


_HandleFn = Callable[[str], "asyncio.Future[str]"] | Callable[[str], "str"]
_cache: Dict[str, Tuple[float, _HandleFn]] = {}
_cache_ts: float = 0.0


def _skills_dir() -> str:
    root = PROJECT_ROOT or os.getcwd()
    return os.path.join(root, "jinx", "skills")


def _list_skill_files() -> List[str]:
    d = _skills_dir()
    if not os.path.isdir(d):
        return []
    out: List[str] = []
    for fn in os.listdir(d):
        if not fn.lower().endswith(".py"):
            continue
        out.append(os.path.join(d, fn))
    return out


def _load_handle(path: str) -> Optional[_HandleFn]:
    try:
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(f"jinx.skills.{name}", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # type: ignore[index]
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        fn = getattr(mod, "handle", None)
        if fn is None:
            return None
        return fn  # type: ignore[return-value]
    except Exception:
        return None


def _refresh_cache() -> None:
    global _cache, _cache_ts
    now = time.time()
    # Refresh at most every 3 seconds
    if (now - _cache_ts) < 3.0:
        return
    files = _list_skill_files()
    new_cache: Dict[str, Tuple[float, _HandleFn]] = {}
    for p in files:
        try:
            mt = os.path.getmtime(p)
        except Exception:
            continue
        ent = _cache.get(p)
        if ent and ent[0] == mt:
            new_cache[p] = ent
            continue
        fn = _load_handle(p)
        if fn is not None:
            new_cache[p] = (mt, fn)
    _cache = new_cache
    _cache_ts = now


async def try_execute(query: str, *, budget_ms: int = 600) -> Optional[str]:
    """Try to execute a skill's handle(query) and return its string result.
    Returns None if no skill can handle it quickly.
    """
    q = (query or "").strip()
    if not q:
        return None
    _refresh_cache()
    if not _cache:
        return None
    # Try each handle with a small timeout
    for p, (mt, fn) in list(_cache.items()):
        try:
            res = fn(q)
            if asyncio.iscoroutine(res):
                out = await asyncio.wait_for(res, timeout=max(0.1, budget_ms) / 1000.0)
            else:
                # Offload sync handles to thread
                def _call_sync():
                    try:
                        return str(res) if isinstance(res, str) else str(fn(q))
                    except Exception:
                        return ""
                out = await asyncio.to_thread(_call_sync)
            s = (out or "").strip()
            if s:
                return s
        except asyncio.TimeoutError:
            continue
        except Exception:
            continue
    return None
