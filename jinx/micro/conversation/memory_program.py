from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional

from jinx.micro.conversation.selector_core import call_selector_json as _call_selector
from jinx.prompts.selector_memory import build_memory_program_instructions
from jinx.micro.conversation.memory_snapshot import build_memory_snapshot
from jinx.micro.conversation.memory_ops import exec_ops as _exec_ops

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")

_ALLOWED_ACTIONS = {"memroute", "pins", "append_channel", "write_topic"}
_ALLOWED_CHANNELS = {"paths", "symbols", "prefs", "decisions"}
_ALLOWED_PIN_OPS = {"list", "add", "remove"}


def _json_candidates(s: str):
    if not s:
        return
    # Prefer fenced json blocks
    for m in re.finditer(r"```json\s*\n([\s\S]*?)```", s, re.IGNORECASE):
        frag = (m.group(1) or "").strip()
        if frag:
            yield frag
    # Balanced braces
    depth = 0
    start: Optional[int] = None
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
    # Fallback
    for m in re.finditer(r"\{[\s\S]*?\}", s):
        yield (m.group(0) or "").strip()


def _parse_json(s: str) -> Optional[Dict[str, Any]]:
    for frag in _json_candidates(s or ""):
        try:
            obj = json.loads(frag)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _clip_line(x: str, lim: int) -> str:
    x = (x or "").strip()
    if lim <= 0 or len(x) <= lim:
        return x
    return x[:lim]


def _validate_ops(obj: Dict[str, Any], *, max_ops: int, kmax: int) -> Optional[Dict[str, Any]]:
    try:
        ops_in = obj.get("ops")
    except Exception:
        return None
    if not isinstance(ops_in, list) or not ops_in:
        return None
    out: List[Dict[str, Any]] = []
    for raw in ops_in[: max(1, max_ops)]:
        try:
            action = str(raw.get("action") or "").strip().lower()
            if action not in _ALLOWED_ACTIONS:
                continue
            params = raw.get("params") or {}
            if not isinstance(params, dict):
                continue
            if action == "memroute":
                q = str(params.get("query") or "").strip()
                try:
                    k = int(params.get("k") or kmax)
                except Exception:
                    k = kmax
                k = max(1, min(kmax, k))
                if not q:
                    continue
                out.append({"action": action, "params": {"query": _clip_line(q, 240), "k": k}})
            elif action == "pins":
                opx = str(params.get("op") or "list").strip().lower()
                if opx not in _ALLOWED_PIN_OPS:
                    opx = "list"
                if opx in ("add", "remove"):
                    line = _clip_line(str(params.get("line") or ""), 400)
                    if not line:
                        continue
                    out.append({"action": action, "params": {"op": opx, "line": line}})
                else:
                    out.append({"action": action, "params": {"op": "list"}})
            elif action == "append_channel":
                kind = str(params.get("kind") or "").strip().lower()
                if kind not in _ALLOWED_CHANNELS:
                    continue
                lines = [ _clip_line(str(x), 400) for x in (params.get("lines") or []) if str(x).strip()]
                if not lines:
                    continue
                out.append({"action": action, "params": {"kind": kind, "lines": lines[:8]}})
            elif action == "write_topic":
                name = _clip_line(str(params.get("name") or "topic"), 80)
                mode = str(params.get("mode") or "append").strip().lower()
                if mode not in ("append", "replace"):
                    mode = "append"
                lines = [ _clip_line(str(x), 400) for x in (params.get("lines") or []) if str(x).strip()]
                if not lines:
                    continue
                out.append({"action": action, "params": {"name": name, "lines": lines[:20], "mode": mode}})
        except Exception:
            continue
    if not out:
        return None
    return {"ops": out}


async def infer_memory_program(text: str) -> Dict[str, str]:
    """Plan and execute memory ops; return injectable blocks.

    - Gated by JINX_MEM_PROGRAM=1 (default on).
    - Uses a compact snapshot of durable memory to ground planning.
    - Strict RT via selector_core; circuit breaker under key JINX_MEMPROG.
    """
    try:
        on = str(os.getenv("JINX_MEM_PROGRAM", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        on = True
    if not on:
        return {}

    # Controls
    try:
        max_ops = int(os.getenv("JINX_MEM_PROGRAM_MAX_OPS", "3"))
    except Exception:
        max_ops = 3
    try:
        kmax = int(os.getenv("JINX_MEM_PROGRAM_MAX_K", "8"))
    except Exception:
        kmax = 8
    try:
        allowed_env = os.getenv("JINX_MEM_PROGRAM_ALLOWED", "memroute,pins,append_channel,write_topic")
        allowed = [p.strip().lower() for p in allowed_env.split(',') if p.strip()]
    except Exception:
        allowed = ["memroute","pins","append_channel","write_topic"]

    # Build snapshot
    try:
        snap_chars = int(os.getenv("JINX_MEM_SNAPSHOT_CHARS", "4000"))
    except Exception:
        snap_chars = 4000
    snapshot = await build_memory_snapshot(max_chars=snap_chars)

    # Build instruction and input JSON
    instr = build_memory_program_instructions(allowed=allowed, max_ops=max_ops, max_k=kmax, include_examples=True)
    input_json = json.dumps({"text": str(text or ""), "snapshot": snapshot}, ensure_ascii=False)

    # Selector core: parse+validate
    async def _parse(s: str) -> Optional[Dict[str, Any]]:
        return _parse_json(s)

    def _validate(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return _validate_ops(obj, max_ops=max_ops, kmax=kmax)

    obj = await _call_selector_json(
        instructions=instr,
        text=input_json,
        parse=_parse,
        validate=_validate,
        model_env="OPENAI_MODEL_MEMORY",
        default_model_env="OPENAI_MODEL",
        default_model_value="gpt-5",
        timeout_env="JINX_MEM_PROGRAM_LLM_MS",
        timeout_default_ms=1300,
        cb_name="JINX_MEMPROG",
        base_extra_kwargs={"temperature": 0.2},
    )
    if not obj:
        return {}
    ops = obj.get("ops") or []
    if not isinstance(ops, list) or not ops:
        return {}
    # Execute ops via micro-ops engine
    return await _exec_ops(ops)  # returns dict of blocks


__all__ = ["infer_memory_program"]
