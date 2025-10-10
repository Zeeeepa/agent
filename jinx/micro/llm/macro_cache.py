from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Dict, Tuple

_mem: Dict[str, Tuple[int, str]] = {}
_inflight: Dict[str, asyncio.Future[str]] = {}


def _now_ms() -> int:
    try:
        return int(time.monotonic_ns() // 1_000_000)
    except Exception:
        return int(time.time() * 1000)


async def memoized_call(key: str, ttl_ms: int, call: Callable[[], Awaitable[str]]) -> str:
    if ttl_ms <= 0:
        return await call()
    now = _now_ms()
    ent = _mem.get(key)
    if ent and now <= ent[0]:
        return ent[1]
    fut = _inflight.get(key)
    if fut is not None and not fut.done():
        try:
            return await fut
        except Exception:
            pass
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _inflight[key] = fut
    try:
        res = await call()
    except Exception as e:
        if not fut.done():
            fut.set_exception(e)
        _inflight.pop(key, None)
        raise
    else:
        _mem[key] = (now + max(1, ttl_ms), res or "")
        if not fut.done():
            fut.set_result(res or "")
        _inflight.pop(key, None)
        return res or ""
