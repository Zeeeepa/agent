from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

_STATE_PATH = os.path.join(".jinx", "state", "prompt_brain.json")


async def _ensure_dir() -> None:
    try:
        d = os.path.dirname(_STATE_PATH)
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass


async def _read_state() -> Dict[str, Any]:
    try:
        await _ensure_dir()
        if not os.path.isfile(_STATE_PATH):
            return {"counters": {}, "variants": {}, "last": {}}
        def _load() -> Dict[str, Any]:
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
        return await asyncio.to_thread(_load)
    except Exception:
        return {"counters": {}, "variants": {}, "last": {}}


async def _write_state(obj: Dict[str, Any]) -> None:
    try:
        await _ensure_dir()
        def _dump() -> None:
            tmp = _STATE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=True)
            os.replace(tmp, _STATE_PATH)
        await asyncio.to_thread(_dump)
    except Exception:
        pass


async def compose_policy_tail(tag: str, stxt: str, *, have_unified_ctx: bool) -> str:
    """Return a small ASCII-only instruction tail to bias prompts based on recent outcomes.

    Inputs:
    - tag: code tag ID for current prompt
    - stxt: sanitized user text
    - have_unified_ctx: whether unified embeddings context was injected

    Policy heuristics (minimal initial set):
    - If json_mismatch rate elevated -> re-emphasize RAW JSON ONLY and ASCII-only.
    - If fences_violations elevated -> forbid code fences.
    - If timeouts elevated -> ask for shorter output and strict budgets.
    - If no unified context -> strengthen retrieval hints.
    """
    st = await _read_state()
    c = dict(st.get("counters") or {})
    jm = int(c.get("json_mismatch", 0))
    fv = int(c.get("fences_violations", 0))
    to = int(c.get("timeouts", 0))
    lines: list[str] = []
    if jm >= 3:
        lines.append("- RAW JSON ONLY; ASCII-only; no commentary; strict schema.")
    if fv >= 3:
        lines.append("- Do NOT use code fences or backticks under any circumstance.")
    if to >= 3:
        lines.append("- Be concise; keep outputs minimal and deterministic; honor small budgets strictly.")
    if not have_unified_ctx:
        lines.append("- Provide compact, self-contained reasoning; avoid relying on external context.")
    # Always enforce basics in tail
    lines.append("- ASCII-only; avoid angle brackets in values unless required by the contract.")
    return ("\n" + "\n".join(lines) + "\n") if lines else ""


async def record_prompt_outcome(tag: str, out_text: str) -> None:
    """Record a few cheap signals about the output to bias future prompts.

    Signals:
    - json_mismatch: increments when output looks like JSON contract but fails basic JSON braces/brackets balance.
    - fences_violations: increments when backticks ``` appear.
    - timeouts: caller may pass empty string on timeout; we count it here when out_text == "".
    """
    st = await _read_state()
    counters: Dict[str, int] = dict(st.get("counters") or {})
    txt = out_text or ""
    # Backticks violation
    if "```" in txt:
        counters["fences_violations"] = int(counters.get("fences_violations", 0)) + 1
    # JSON mismatch (very naive, cheap): looks like JSON but cannot load
    if txt.strip().startswith("{") and txt.strip().endswith("}"):
        try:
            json.loads(txt)
        except Exception:
            counters["json_mismatch"] = int(counters.get("json_mismatch", 0)) + 1
    # Timeout/empty
    if not txt.strip():
        counters["timeouts"] = int(counters.get("timeouts", 0)) + 1
    st["counters"] = counters
    await _write_state(st)
