from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from jinx.net import get_openai_client

# TTL cache + request coalescing + concurrency limiting + timeouts for LLM Responses API
# Keyed by a stable fingerprint of (instructions, model, input_text, extra_kwargs)

try:
    _TTL_SEC = float(os.getenv("JINX_LLM_TTL_SEC", "300"))  # 5 minutes default
except Exception:
    _TTL_SEC = 300.0
try:
    _TIMEOUT_MS = int(os.getenv("JINX_LLM_TIMEOUT_MS", "20000"))  # 20s default
except Exception:
    _TIMEOUT_MS = 20000
try:
    _MAX_CONC = int(os.getenv("JINX_LLM_MAX_CONCURRENCY", "4"))
except Exception:
    _MAX_CONC = 4

_DUMP = str(os.getenv("JINX_LLM_DUMP", "0")).lower() in {"1", "true", "on", "yes"}

_mem: Dict[str, Tuple[float, str]] = {}
_inflight: Dict[str, asyncio.Future] = {}
_sem = asyncio.Semaphore(max(1, _MAX_CONC))


def _now() -> float:
    return time.time()


def _safe_jsonable(obj: Any, depth: int = 0) -> Any:
    """Best-effort transform to jsonable structure without exploding on exotic types.

    Limits depth to avoid huge payloads; falls back to repr for unknowns.
    """
    if depth > 4:
        return f"<{type(obj).__name__}:depth>"
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v, depth + 1) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v, depth + 1) for v in obj[:100]]  # cap length for stability
    try:
        return json.loads(json.dumps(obj))  # type: ignore[arg-type]
    except Exception:
        try:
            r = repr(obj)
            # trim very long reprs to keep key stable and small
            if len(r) > 256:
                r = r[:256] + "..."
            return r
        except Exception:
            return f"<{type(obj).__name__}>"


def _fingerprint(instructions: str, model: str, input_text: str, extra_kwargs: Dict[str, Any]) -> str:
    payload = {
        "i": (instructions or ""),
        "m": (model or ""),
        "t": (input_text or ""),
        "k": _safe_jsonable(extra_kwargs or {}),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


async def _dump_line(line: str) -> None:
    if not _DUMP:
        return
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        await _append(BLUE_WHISPERS, f"[llm_cache] {line}")
    except Exception:
        pass


async def call_openai_cached(instructions: str, model: str, input_text: str, *, extra_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Cached/coalesced wrapper for OpenAI Responses API.

    Returns output_text (string). On API error, raises the exception (caller logs/handles).
    """
    ek = extra_kwargs or {}
    key = _fingerprint(instructions, model, input_text, ek)
    # TTL cache lookup
    item = _mem.get(key)
    if item is not None:
        exp, val = item
        if exp >= _now():
            return val
        else:
            _mem.pop(key, None)

    # Coalescing
    fut = _inflight.get(key)
    if fut is not None:
        try:
            res = await fut
            return str(res or "")
        except Exception:
            # If the inflight failed, proceed to execute
            pass

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _inflight[key] = fut
    soft_timeout = False
    async with _sem:
        await _dump_line(f"call key={key[:8]} model={model} ilen={len(instructions)} tlen={len(input_text)}")
        def _worker():
            client = get_openai_client()
            return client.responses.create(
                instructions=instructions,
                model=model,
                input=input_text,
                **ek,
            )
        # Launch background task so we can safely wait on shared fut even if a soft timeout occurs
        task: asyncio.Task = asyncio.create_task(asyncio.to_thread(_worker))

        def _on_done(t: asyncio.Task) -> None:
            try:
                if t.cancelled():
                    # Propagate cancellation to awaiters without leaking to event loop logs
                    if not fut.done():
                        fut.set_exception(asyncio.CancelledError())
                    return
                r = t.result()
                out = str(getattr(r, "output_text", ""))
                _mem[key] = (_now() + max(1.0, _TTL_SEC), out)
                if not fut.done():
                    fut.set_result(out)
            except BaseException as ex:
                try:
                    if not fut.done():
                        fut.set_exception(ex)
                except BaseException:
                    pass
            finally:
                _inflight.pop(key, None)

        task.add_done_callback(_on_done)
        try:
            # Try to get result within timeout
            r = await asyncio.wait_for(task, timeout=max(0.1, _TIMEOUT_MS / 1000))
            out = str(getattr(r, "output_text", ""))
            # Callback will also set cache/fut and pop inflight; just return out here
            return out
        except asyncio.TimeoutError:
            soft_timeout = True
            await _dump_line("soft_timeout")

    # If we timed out, release the semaphore first, then await the shared inflight future.
    if soft_timeout:
        await _dump_line("awaiting inflight outside semaphore")
        try:
            res = await fut
            return str(res or "")
        except Exception as ex:
            # Propagate the underlying error if the background task failed
            raise ex
