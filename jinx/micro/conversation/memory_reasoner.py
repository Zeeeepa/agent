from __future__ import annotations

import os
import json
import re
from typing import Optional, Dict, Any
import time as _time

from jinx.micro.conversation.turns_router import detect_turn_query as _fast_turn
from jinx.micro.conversation.selector_core import call_selector_json as _call_selector
from jinx.micro.conversation.debug import log_debug
from jinx.prompts.selector_memory import build_memory_instructions

_ALLOWED = {"turn", "memroute", "pins", "none"}
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


# Instructions for memory selector are built via build_memory_instructions() from selector_specs


def _json_candidates(s: str):
    if not s:
        return
    # Prefer fenced json blocks if present
    for m in re.finditer(r"```json\s*\n([\s\S]*?)```", s, re.IGNORECASE):
        frag = (m.group(1) or "").strip()
        if frag:
            yield frag
    # Balanced brace scan for first-level JSON object
    depth = 0
    start: int | None = None
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield s[start:i+1]
                    start = None
    # Fallback: non-greedy regex matches
    for m in re.finditer(r"\{[\s\S]*?\}", s):
        yield (m.group(0) or "").strip()


def _extract_json(s: str) -> Optional[Dict[str, Any]]:
    for frag in _json_candidates(s or ""):
        try:
            obj = json.loads(frag)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _validate(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    action = str(obj.get("action", "")).strip().lower()
    if action not in _ALLOWED:
        return None
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        return None
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    if action == "turn":
        kind = str(params.get("kind", "")).strip().lower()
        idx = int(params.get("index", 0)) if str(params.get("index", "")).strip() else 0
        if kind not in ("user", "jinx", "pair") or idx <= 0:
            return None
    elif action == "memroute":
        q = str(params.get("query", ""))
        try:
            k = int(params.get("k", 6))
        except Exception:
            k = 6
        k = max(1, min(16, k))
        params = {"query": q, "k": k}
    elif action == "pins":
        params = {"op": "list"}
    elif action == "none":
        params = {}
    return {"action": action, "params": params, "confidence": max(0.0, min(1.0, conf))}




async def infer_memory_action(text: str, allowed: Optional[list[str]] = None, max_k: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Hybrid memory intent inference.

    1) Try fast 'turn' detection (no API).
    2) Else ask a compact LLM selector with TTL cache & coalescing.
    """
    t = text or ""
    ft = _fast_turn(t)
    if ft and int(ft.get("index", 0)) > 0:
        kind = (ft.get("kind") or "pair").strip().lower()
        if kind not in ("user", "jinx", "pair"):
            kind = "pair"
        await log_debug("JINX_MEMSEL", f"fast_turn kind={kind} idx={int(ft['index'])}")
        return {"action": "turn", "params": {"kind": kind, "index": int(ft["index"])}, "confidence": 0.66}

    # LLM selector path
    try:
        use = str(os.getenv("JINX_MEMORY_REASONER", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        use = True
    if not use:
        return None
    # Programmatic prompt: respect caller/env-driven allowed actions and k bound
    model = os.getenv("OPENAI_MODEL_MEMORY", os.getenv("OPENAI_MODEL", "gpt-5"))
    if allowed is not None:
        allowed_list = [a.strip().lower() for a in allowed if a.strip()]
    else:
        try:
            allowed_env = os.getenv("JINX_MEMSEL_ALLOWED", "turn,memroute,pins,none")
            allowed_list = [p.strip().lower() for p in allowed_env.split(',') if p.strip()]
        except Exception:
            allowed_list = ["turn","memroute","pins","none"]
    if max_k is not None:
        k_max_eff = int(max_k)
    else:
        try:
            k_max_eff = int(os.getenv("JINX_MEMSEL_MAX_K", "8"))
        except Exception:
            k_max_eff = 8
    instr = build_memory_instructions(allowed=allowed_list, max_k=k_max_eff, include_examples=True)
    # Shared selector core (circuit breaker, timeout, multi-sample, sanitization, debug)
    obj = await _call_selector_json(
        instructions=instr,
        text=t,
        parse=_extract_json,
        validate=_validate,
        model_env="OPENAI_MODEL_MEMORY",
        default_model_env="OPENAI_MODEL",
        default_model_value="gpt-5",
        timeout_env="JINX_MEMSEL_LLM_MS",
        timeout_default_ms=1100,
        cb_name="JINX_MEMSEL",
        base_extra_kwargs={"temperature": 0.2},
    )
    return obj


__all__ = ["infer_memory_action"]
