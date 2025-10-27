from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Tuple, List
from threading import Lock as _TLock

from jinx.net import get_openai_client
from jinx.micro.parser.api import parse_tagged_blocks as _parse_blocks

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
_family_inflight: Dict[str, asyncio.Future] = {}
_inflight_tlock: _TLock = _TLock()
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


def _fingerprint_family(instructions: str, model: str, input_text: str) -> str:
    """Family fingerprint ignoring extra kwargs.

    Used to coalesce outward calls for the same logical request shape regardless of
    small variations (like temperature) to guarantee single outbound call.
    """
    return _fingerprint(instructions, model, input_text, {})


async def _dump_line(line: str) -> None:
    if not _DUMP:
        return
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        await _append(BLUE_WHISPERS, f"[llm_cache] {line}")
    except Exception:
        pass


def _is_forbidden_region(ex: BaseException) -> bool:
    """Detect provider 'unsupported country/region/territory' 403 errors robustly.

    We rely on best-effort string and attribute checks because SDK types may vary.
    """
    try:
        # Common SDKs expose http_status / status_code
        sc = getattr(ex, "status_code", None) or getattr(ex, "http_status", None) or getattr(ex, "status", None)
        if sc == 403:
            s = str(ex).lower()
            if "unsupported_country_region_territory" in s:
                return True
            if "request_forbidden" in s and "country" in s and "supported" in s:
                return True
    except Exception:
        pass
    # Fallback: message-only detection
    s = str(ex).lower()
    if "unsupported_country_region_territory" in s:
        return True
    return False


async def call_openai_cached(instructions: str, model: str, input_text: str, *, extra_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Cached/coalesced wrapper for OpenAI Responses API.

    Returns output_text (string). On API error, raises the exception (caller logs/handles).
    """
    ek = extra_kwargs or {}
    # Strip internal control keys from fingerprinting and outbound SDK kwargs
    ek_fpr = {str(k): v for k, v in ek.items() if not str(k).startswith("__")}
    key = _fingerprint(instructions, model, input_text, ek_fpr)
    fam_key = _fingerprint_family(instructions, model, input_text)
    no_family = bool(ek.get("__no_family__", False))
    # TTL cache lookup
    item = _mem.get(key)
    if item is not None:
        exp, val = item
        if exp >= _now():
            return val
        else:
            _mem.pop(key, None)

    # Coalescing (with race-free creation)
    loop = asyncio.get_running_loop()
    to_wait: asyncio.Future | None = None
    # Cross-thread safe critical section for inflight maps
    with _inflight_tlock:
        # Exact-key inflight first
        existing_exact = _inflight.get(key)
        if existing_exact is not None:
            to_wait = existing_exact
        else:
            # Family-level inflight (unless disabled)
            if not no_family:
                existing_fam = _family_inflight.get(fam_key)
                if existing_fam is not None:
                    to_wait = existing_fam
            if to_wait is None:
                fut = loop.create_future()
                _inflight[key] = fut
                if not no_family:
                    _family_inflight[fam_key] = fut
    if to_wait is not None:
        try:
            res = await to_wait
            return str(res or "")
        except Exception:
            # If the inflight failed, continue to execute fresh
            pass
    soft_timeout = False
    async with _sem:
        await _dump_line(f"call key={key[:8]} model={model} ilen={len(instructions)} tlen={len(input_text)}")
        def _worker():
            client = get_openai_client()
            ek_api = {str(k): v for k, v in ek.items() if not str(k).startswith("__")}
            return client.responses.create(
                instructions=instructions,
                model=model,
                input=input_text,
                **ek_api,
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
                try:
                    _inflight.pop(key, None)
                except Exception:
                    pass
                # Clear family mapping if set
                if not no_family:
                    try:
                        if _family_inflight.get(fam_key) is fut:
                            _family_inflight.pop(fam_key, None)
                    except Exception:
                        pass

        task.add_done_callback(_on_done)
        # Implement soft timeout without cancelling the underlying task
        timeout_sec = max(0.1, _TIMEOUT_MS / 1000)
        timeout_task = asyncio.create_task(asyncio.sleep(timeout_sec))
        done, _ = await asyncio.wait({task, timeout_task}, return_when=asyncio.FIRST_COMPLETED)
        # Clean up timeout task if still pending
        if not timeout_task.done():
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass
        if task in done:
            try:
                r = await task
            except asyncio.CancelledError:
                # Should not happen since we didn't cancel; treat as transient
                soft_timeout = True
                await _dump_line("soft_timeout_cancelled")
            except BaseException as ex:
                # Convert immediate failure into soft path so we await the shared future.
                # This ensures we 'consume' the future's exception (set by the callback)
                # to avoid 'Future exception was never retrieved' warnings.
                soft_timeout = True
                if _is_forbidden_region(ex):
                    await _dump_line("provider_forbidden_region_immediate")
                else:
                    await _dump_line(f"task_failed:{type(ex).__name__}")
            else:
                out = str(getattr(r, "output_text", ""))
                # Callback will also set cache/fut and pop inflight; just return out here
                return out
        else:
            soft_timeout = True
            await _dump_line("soft_timeout")

    # If we timed out, release the semaphore first, then await the shared inflight future.
    if soft_timeout:
        await _dump_line("awaiting inflight outside semaphore")
        try:
            res = await fut
            return str(res or "")
        except BaseException as ex:
            # Propagate the underlying error if the background task failed
            if _is_forbidden_region(ex):
                # Negative cache for a short period to avoid rapid retries.
                try:
                    _mem[key] = (_now() + min(60.0, max(1.0, _TTL_SEC)), "")
                except Exception:
                    pass
                await _dump_line("provider_forbidden_region_soft")
                return ""
            raise ex
    

async def call_openai_multi_validated(
    instructions: str,
    model: str,
    input_text: str,
    *,
    code_id: str,
    base_extra_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Run multiple cached LLM calls in parallel and return the first valid output.

    - Variations are done via temperature tweaks (kept small to preserve determinism).
    - Validation: output must contain exactly one <python_{code_id}> block.
    - Does not cancel in-flight calls so they can populate the TTL cache for future turns.
    """
    try:
        n = max(1, int(os.getenv("JINX_LLM_MULTI_SAMPLES", "1")))
    except Exception:
        n = 1
    # Conservative small variations
    temps_all: List[float] = [0.2, 0.5, 0.8, 0.3, 0.7]
    temps = temps_all[:max(1, n)]
    extra = dict(base_extra_kwargs or {})
    try:
        hedge_ms = int(os.getenv("JINX_LLM_MULTI_HEDGE_MS", "0"))
    except Exception:
        hedge_ms = 0
    try:
        cancel_losers = (os.getenv("JINX_LLM_MULTI_CANCEL", "1").strip().lower() not in ("", "0", "false", "off", "no"))
    except Exception:
        cancel_losers = True

    async def _one(t: float, register_family: bool) -> str:
        kw = dict(extra)
        # Temperature is widely supported in Responses API kwargs
        kw["temperature"] = t
        # Only the first sample registers family inflight; others opt-out to avoid collapsing race
        if not register_family:
            kw["__no_family__"] = True
        return await call_openai_cached(instructions, model, input_text, extra_kwargs=kw)

    # Start first immediately
    tasks: List[asyncio.Task] = []
    if not temps:
        temps = [0.2]
    t0 = asyncio.create_task(_one(temps[0], True))
    tasks.append(t0)
    # Optional: start one additional hedged request after a short delay if first hasn't finished
    if len(temps) > 1 and hedge_ms > 0:
        try:
            await asyncio.wait_for(asyncio.sleep(max(0.0, hedge_ms) / 1000.0), timeout=max(0.05, hedge_ms / 1000.0))
        except Exception:
            pass
        if not t0.done():
            t1 = asyncio.create_task(_one(temps[1], False))
            tasks.append(t1)

    first: str | None = None
    for fut in asyncio.as_completed(tasks):
        try:
            out = await fut
        except Exception:
            continue
        if first is None:
            first = out  # remember earliest even if invalid, as fallback
        try:
            pairs = _parse_blocks(out, code_id)
        except Exception:
            pairs = []
        # Strict: exactly one matching code block and non-empty content
        good = 0
        for tag, core in pairs:
            if tag.strip() == f"python_{code_id}" and (core or "").strip():
                good += 1
        if good == 1:
            # Best-effort cancel losers to reduce outbound traffic
            if cancel_losers:
                for t in tasks:
                    if t is not fut and not t.done():
                        t.cancel()
                        try:
                            await t
                        except Exception:
                            pass
            return out
    # If none validated, return earliest completed output
    return first or ""
