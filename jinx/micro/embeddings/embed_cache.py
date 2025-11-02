from __future__ import annotations

import asyncio
import os
import time
import hashlib
import math
import re
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


# ------------------------
# Offline fallback support
# ------------------------

_TOK_RE = re.compile(r"(?u)[\w]{2,}")


def _dims_for_model(model: str) -> int:
    m = (model or "").lower()
    # Known OpenAI embedding dims
    if "text-embedding-3-large" in m:
        return 3072
    if "text-embedding-3-small" in m:
        return 1536
    if "text-embedding-ada-002" in m:
        return 1536
    try:
        return max(64, int(os.getenv("JINX_EMBED_OFFLINE_DIMS", "1536")))
    except Exception:
        return 1536


def _online_enabled() -> bool:
    val = str(os.getenv("JINX_EMBED_ONLINE", "auto")).strip().lower()
    if val in ("0", "false", "off", "no"):
        return False
    if val in ("1", "true", "on", "yes"):
        return True
    # auto: enabled only if API key present
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))


def _normalize(vec: List[float]) -> List[float]:
    try:
        s = math.sqrt(sum((x or 0.0) * (x or 0.0) for x in (vec or [])))
        if s <= 0.0:
            return list(vec or [])
        return [(x or 0.0) / s for x in vec]
    except Exception:
        return list(vec or [])


def _offline_embed(model: str, text: str) -> List[float]:
    dims = _dims_for_model(model)
    vec = [0.0] * dims
    if not text:
        return vec
    salt = "jinx-offline-emb-v1"
    try:
        max_tokens = max(128, int(os.getenv("JINX_EMBED_OFFLINE_MAX_TOKENS", "2048")))
    except Exception:
        max_tokens = 2048
    # Tokenize and accumulate hashed contributions
    toks: List[str] = []
    for m in _TOK_RE.finditer(text.lower()):
        t = (m.group(0) or "").strip()
        if t:
            toks.append(t)
        if len(toks) >= max_tokens:
            break
    if not toks:
        return vec
    for t in toks:
        h = hashlib.blake2b((salt + t).encode("utf-8", errors="ignore"), digest_size=16).digest()
        idx = int.from_bytes(h[:8], "little", signed=False) % dims
        mag_raw = int.from_bytes(h[8:], "little", signed=False)
        mag = (mag_raw % 2001) / 1000.0 - 1.0  # in [-1.0, 1.0]
        vec[idx] += mag
    return _normalize(vec)


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
    # Offline short-circuit
    if not _online_enabled():
        return _offline_embed(model, text)
    async with _sem:
        await _dump_line(f"call single model={model} len={len(text)}")
        def _worker() -> Any:
            client = get_openai_client()
            return client.embeddings.create(model=model, input=text)
        try:
            resp = await asyncio.wait_for(asyncio.to_thread(_worker), timeout=max(0.05, _TIMEOUT_MS / 1000))
            vec = resp.data[0].embedding if getattr(resp, "data", None) else []
            if not vec:
                return _offline_embed(model, text)
            return vec
        except Exception as e:
            await _dump_line(f"single error: {type(e).__name__}")
            return _offline_embed(model, text)


async def _call_batch(model: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    # Offline short-circuit
    if not _online_enabled():
        return [_offline_embed(model, t) for t in texts]
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
                    if not vec:
                        vec = _offline_embed(model, texts[i])
                except Exception:
                    vec = _offline_embed(model, texts[i])
                out.append(vec)
            return out
        except Exception as e:
            await _dump_line(f"batch error: {type(e).__name__}")
            return [_offline_embed(model, t) for t in texts]


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
