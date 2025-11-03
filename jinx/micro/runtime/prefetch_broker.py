from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Dict, Optional

# Central broker to dedupe and rate-limit prefetch requests across plugins.
# Keyed by (kind, normalized_query); single global semaphore for concurrency.

_INF: Dict[str, asyncio.Task[None]] = {}
_SEM: Optional[asyncio.Semaphore] = None


def _sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        try:
            conc = max(1, int(os.getenv("JINX_PREFETCH_BROKER_CONC", "3")))
        except Exception:
            conc = 3
        _SEM = asyncio.Semaphore(conc)
    return _SEM


def _norm(q: str) -> str:
    s = (q or "").strip().lower()
    if len(s) > 512:
        s = s[:512]
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _key(kind: str, q: str) -> str:
    return f"{kind}::{_norm(q)}"


async def prefetch_project_ctx(query: str, max_time_ms: int = 260) -> None:
    # Avoid duplication if already cached
    try:
        from jinx.micro.embeddings.prefetch_cache import get_project as _get
    except Exception:
        _get = None  # type: ignore
    if _get is not None:
        try:
            if _get(query):
                return
        except Exception:
            pass
    k = _key("proj", query)
    if k in _INF and not _INF[k].done():
        return

    async def _run() -> None:
        async with _sem():
            try:
                from jinx.micro.embeddings.project_retrieval import build_project_context_for as _build
                from jinx.micro.embeddings.prefetch_cache import put_project as _put
            except Exception:
                return
            try:
                ctx = await _build(query, max_time_ms=max_time_ms)
                if ctx:
                    _put(query, ctx)
            except Exception:
                return

    _INF[k] = asyncio.create_task(_run())
    try:
        await _INF[k]
    except Exception:
        pass


async def prefetch_base_ctx(query: str, max_time_ms: int = 120) -> None:
    try:
        from jinx.micro.embeddings.prefetch_cache import get_base as _get
    except Exception:
        _get = None  # type: ignore
    if _get is not None:
        try:
            if _get(query):
                return
        except Exception:
            pass
    k = _key("base", query)
    if k in _INF and not _INF[k].done():
        return

    async def _run() -> None:
        async with _sem():
            try:
                from jinx.micro.embeddings.retrieval import build_context_for as _build
                from jinx.micro.embeddings.prefetch_cache import put_base as _put
            except Exception:
                return
            try:
                ctx = await _build(query, max_time_ms=max_time_ms)
                if ctx:
                    _put(query, ctx)
            except Exception:
                return

    _INF[k] = asyncio.create_task(_run())
    try:
        await _INF[k]
    except Exception:
        pass


__all__ = [
    "prefetch_project_ctx",
    "prefetch_base_ctx",
]
