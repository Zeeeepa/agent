from __future__ import annotations

import os
import json
import re
from typing import Optional, Dict, Any
import time as _time

from jinx.micro.conversation.turns_router import detect_turn_query as _fast_detect
from jinx.micro.conversation.prefilter import likely_turn_query as _likely_turn
from jinx.micro.conversation.selector_core import call_selector_json as _call_selector
from jinx.micro.memory.turns import parse_active_turns as _parse_turns
from jinx.prompts.selector_turns import build_turns_instructions

_ALLOWED_KINDS = {"user", "jinx", "pair"}

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _mk_instructions() -> str:
    return build_turns_instructions(include_examples=True)


def _extract_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    m = _JSON_OBJ_RE.search(s)
    if not m:
        return None
    frag = m.group(0)
    try:
        obj = json.loads(frag)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _validate(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = str(obj.get("kind", "")).strip().lower()
    try:
        index = int(obj.get("index", 0))
    except Exception:
        index = 0
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    if kind not in _ALLOWED_KINDS or index <= 0:
        return None
    return {"kind": kind, "index": index, "confidence": max(0.0, min(1.0, conf))}


_REL_LAST_RE = re.compile(r"(?i)\b(last|latest|recent|последн\w*)\b")
_REL_PREV_RE = re.compile(r"(?i)\b(prev|previous|предыдущ\w*|предпослед\w*)\b")
_ROLE_USER_RE = re.compile(r"(?i)\b(i|my|me|я|мое|моя|мой)\b")
_ROLE_JINX_RE = re.compile(r"(?i)\b(jinx|assistant|bot|agent|ты|твой|твоя|тебя)\b")


async def _fast_relative_turn(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    role = "pair"
    tl = t.lower()
    try:
        if _ROLE_USER_RE.search(tl):
            role = "user"
        elif _ROLE_JINX_RE.search(tl):
            role = "jinx"
    except Exception:
        role = "pair"
    rel = None
    try:
        if _REL_LAST_RE.search(tl):
            rel = "last"
        elif _REL_PREV_RE.search(tl):
            rel = "prev"
    except Exception:
        rel = None
    if not rel:
        return None
    try:
        turns = await _parse_turns()
    except Exception:
        turns = []
    n = len(turns)
    if n <= 0:
        return None
    idx = n if rel == "last" else (n - 1)
    if idx <= 0:
        return None
    return {"kind": role, "index": idx, "confidence": 0.7}


async def infer_turn_query(text: str) -> Optional[Dict[str, Any]]:
    """Hybrid fast+LLM inference for (kind,index) turn queries.

    1) Try fast local detection from turns_router (no API).
    2) If not found or ambiguous, optionally call a compact LLM with TTL cache/coalescing.
    """
    # Step 1: fast path
    fast = _fast_detect(text or "")
    if fast and int(fast.get("index", 0)) > 0:
        kind = str(fast.get("kind") or "pair").strip().lower()
        if kind not in _ALLOWED_KINDS:
            kind = "pair"
        return {"kind": kind, "index": int(fast["index"]), "confidence": 0.66}
    # Fast relative references like 'last/previous message'
    rel = await _fast_relative_turn(text or "")
    if rel:
        return rel

    # Step 2: optional LLM fallback
    try:
        use_llm = str(os.getenv("JINX_TURNS_LLM", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        use_llm = True
    if not use_llm:
        return fast  # may be None
    # Prefilter: skip LLM when not likely a turns query
    try:
        if not _likely_turn(text or ""):
            return fast
    except Exception:
        pass

    # Circuit breaker: skip LLM if recently failing
    CB = globals().setdefault("_CB_TURNS", {"fails": 0, "until": 0.0})
    now = _time.time()
    if now < float(CB.get("until") or 0.0):
        return fast

    instr = _mk_instructions()
    try:
        obj = await _call_selector_json(
            instructions=instr,
            text=text or "",
            parse=_extract_json,
            validate=_validate,
            model_env="OPENAI_MODEL_TURNS",
            default_model_env="OPENAI_MODEL",
            default_model_value="gpt-5",
            timeout_env="JINX_TURNS_LLM_MS",
            timeout_default_ms=900,
            cb_name="JINX_TURNS",
            base_extra_kwargs={"temperature": 0.2},
        )
    except Exception:
        obj = None
    if obj:
        return obj
    # Final graceful fallback
    return fast


__all__ = ["infer_turn_query"]
