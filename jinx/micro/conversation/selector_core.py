from __future__ import annotations

import os
import time as _time
from typing import Any, Callable, Optional, Dict

from jinx.micro.llm.prompt_filters import sanitize_prompt_for_external_api as _sanitize
from jinx.micro.llm.json_multi import call_json_multi_validated as _json_multi
from .debug import log_debug


_CB_MAP: Dict[str, Dict[str, float | int]] = {}


async def call_selector_json(
    *,
    instructions: str,
    text: str,
    parse: Callable[[str], Optional[Dict[str, Any]]],
    validate: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    model_env: str,
    default_model_env: str = "OPENAI_MODEL",
    default_model_value: str = "gpt-5",
    timeout_env: str,
    timeout_default_ms: int,
    cb_name: str,
    base_extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Shared hardened path for compact JSON selectors with strict RT.

    - instructions: fully built instruction string (with schema and examples)
    - text: user input (will be sanitized)
    - parse/validate: functions to parse and validate JSON output
    - model_env: env var to read model name, falling back to default_model_env/default value
    - timeout_env: env var for ms timeout
    - cb_name: circuit-breaker key (separate per feature)
    - base_extra_kwargs: extra kwargs (e.g., temperature) sent to OpenAI Responses API
    """
    # Resolve model
    model = os.getenv(model_env, os.getenv(default_model_env, default_model_value))
    stxt = _sanitize(text or "")

    # Circuit breaker
    cb = _CB_MAP.setdefault(cb_name, {"fails": 0, "until": 0.0})
    now = _time.time()
    if now < float(cb.get("until") or 0.0):
        await log_debug(cb_name, f"cb_skip until={cb.get('until'):.3f}")
        return None

    # Timeout
    try:
        lms = int(os.getenv(timeout_env, str(timeout_default_ms)))
    except Exception:
        lms = timeout_default_ms

    # Multi-sample strict JSON with hard RT
    try:
        import asyncio as _asyncio
        obj = await _asyncio.wait_for(
            _json_multi(
                instructions,
                model,
                stxt,
                parse=parse,
                validate=validate,
                base_extra_kwargs=dict(base_extra_kwargs or {}),
            ),
            timeout=max(0.1, lms / 1000.0),
        )
    except Exception as ex:
        await log_debug(cb_name, f"selector_timeout_or_error: {ex!r}")
        obj = None
        try:
            cb["fails"] = int(cb.get("fails", 0)) + 1
            if cb["fails"] >= 3:
                backoff = float(os.getenv(f"{cb_name.upper()}_CB_BACKOFF_SEC", "60"))
                cb["until"] = now + max(5.0, backoff)
                cb["fails"] = 0
        except Exception:
            pass
    if obj:
        await log_debug(cb_name, "selector_validated")
        return obj
    await log_debug(cb_name, "selector_fallback_none")
    return None
