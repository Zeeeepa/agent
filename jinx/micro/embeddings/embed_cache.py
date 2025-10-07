from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Tuple

from jinx.net import get_openai_client

# Simple in-memory TTL cache with request coalescing and concurrency limiting
# Keys are (model, text) for single; for batch we fill per-text from the same cache.

_TTL_SEC = float(os.getenv("JINX_EMBED_TTL_SEC", "900"))  # 15 minutes default
try:
    _TIMEOUT_MS = int(os.getenv("JINX_EMBED_TIMEOUT_MS", "2500"))
except Exception:
    _TIMEOUT_MS = 2500
try:
    _MAX_CONC = int(os.getenv("JINX_EMBED_MAX_CONCURRENCY", "4"))
except Exception:
    _MAX_CONC = 4

_DUMP = str(os.getenv("JINX_EMBED_DUMP", "0")).lower() in {"1", "true", "on", "yes"}

_mem: Dict[Tuple[str, str], Tuple[float, List[float]]] = {}
_inflight: Dict[Tuple[str, str], asyncio.Future] = {}
_sem = asyncio.Semaphore(max(1, _MAX_CONC))


async def _dump_line(line: str) -> None:
    if not _DUMP:
        return
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        await _append(BLUE_WHISPERS, f"[embed_cache] {line}")
    except Exception:
        pass


def _now() -> float:
    return time.time()


def _cache_get(model: str, text: str) -> List[float] | None:
    k = (model, text)
    v = _mem.get(k)
    if not v:
        return None
    exp, vec = v
    if exp < _now():
        _mem.pop(k, None)
        return None
    return vec


def _cache_put(model: str, text: str, vec: List[float]) -> None:
    k = (model, text)
    _mem[k] = (_now() + max(1.0, _TTL_SEC), list(vec or []))


async def _call_single(model: str, text: str) -> List[float]:
    async with _sem:
        await _dump_line(f"call single model={model} len={len(text)}")
        def _worker() -> Any:
            client = get_openai_client()
            return client.embeddings.create(model=model, input=text)
        try:
            resp = await asyncio.wait_for(asyncio.to_thread(_worker), timeout=max(0.05, _TIMEOUT_MS / 1000))
            vec = resp.data[0].embedding if getattr(resp, "data", None) else []
        except Exception as e:
            await _dump_line(f"single error: {type(e).__name__}")
            vec = []
        return vec


async def _call_batch(model: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    async with _sem:
        await _dump_line(f"call batch model={model} n={len(texts)}")
        def _worker() -> Any:
            client = get_openai_client()
            return client.embeddings.create(model=model, input=texts)
        try:
            resp = await asyncio.wait_for(asyncio.to_thread(_worker), timeout=max(0.05, _TIMEOUT_MS / 1000))
            data = getattr(resp, "data", None) or []
            out: List[List[float]] = []
            for i in range(len(texts)):
                try:
                    vec = data[i].embedding  # type: ignore[index]
                except Exception:
                    vec = []
                out.append(vec)
            return out
        except Exception as e:
            await _dump_line(f"batch error: {type(e).__name__}")
            return [[] for _ in texts]


async def embed_text_cached(text: str, *, model: str) -> List[float]:
    t = (text or "").strip()
    if not t:
        return []
    # cache
    c = _cache_get(model, t)
    if c is not None:
        return c
    key = (model, t)
    # coalescing
    fut = _inflight.get(key)
    if fut is not None:
        try:
            res = await fut
            return list(res or [])
        except Exception:
            pass
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _inflight[key] = fut
    try:
        vec = await _call_single(model, t)
        _cache_put(model, t, vec)
        fut.set_result(vec)
        return vec
    except Exception as e:
        fut.set_result([])
        return []
    finally:
        _inflight.pop(key, None)


async def embed_texts_cached(texts: List[str], *, model: str) -> List[List[float]]:
    if not texts:
        return []
    # Normalize inputs
    items = [(i, (texts[i] or "").strip()) for i in range(len(texts))]
    out: List[List[float]] = [[] for _ in texts]

    # First, fulfill from cache or inflight
    missing_idx: List[int] = []
    missing_vals: List[str] = []
    inflight_waits: List[Tuple[int, asyncio.Future]] = []

    for i, t in items:
        if not t:
            out[i] = []
            continue
        c = _cache_get(model, t)
        if c is not None:
            out[i] = c
            continue
        key = (model, t)
        fut = _inflight.get(key)
        if fut is not None:
            inflight_waits.append((i, fut))
            continue
        # mark as missing
        missing_idx.append(i)
        missing_vals.append(t)

    # Await inflight results
    for i, fut in inflight_waits:
        try:
            res = await fut
            out[i] = list(res or [])
        except Exception:
            out[i] = []

    # Batch call for remaining missing
    if missing_vals:
        # Deduplicate in this batch while keeping mapping
        dedup_map: Dict[str, List[int]] = {}
        order: List[str] = []
        for pos, val in zip(missing_idx, missing_vals):
            if val not in dedup_map:
                dedup_map[val] = []
                order.append(val)
            dedup_map[val].append(pos)
        # Set inflight futures for deduped keys
        loop = asyncio.get_running_loop()
        futs_local: Dict[str, asyncio.Future] = {}
        for val in order:
            key = (model, val)
            if key not in _inflight:
                _inflight[key] = loop.create_future()
                futs_local[val] = _inflight[key]
        # Perform one batch API call
        vecs = await _call_batch(model, order)
        for idx, val in enumerate(order):
            vec = vecs[idx] if idx < len(vecs) else []
            _cache_put(model, val, vec)
            # resolve coalesced future if we created it here
            f = futs_local.get(val)
            if f is not None and not f.done():
                try:
                    f.set_result(vec)
                except Exception:
                    pass
            # fill all positions
            for pos in dedup_map.get(val, []):
                out[pos] = vec
            # clear inflight entry
            _inflight.pop((model, val), None)

    return out
