from __future__ import annotations

import os
from typing import Optional

# Lightweight digest generator for chunks. Prefers heuristics; can use LLM via env gate.

_MAX_LEN_DEFAULT = 200


def _heuristic_digest(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Prefer first non-empty line as a summary seed
    for ln in t.splitlines():
        s = (ln or "").strip()
        if len(s) >= 20:
            if len(s) <= max_len:
                return s
            return s[: max_len - 1] + "…"
    # Fallback: trim
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


async def make_digest(text: str, *, max_len: Optional[int] = None) -> str:
    ml = int(max_len or int(os.getenv("EMBED_PROJECT_DIGEST_MAX_LEN", str(_MAX_LEN_DEFAULT))))
    use_llm = (os.getenv("EMBED_PROJECT_DIGEST_LLM", "0").strip().lower() in {"1", "true", "on", "yes"})
    if not use_llm:
        return _heuristic_digest(text, ml)
    # Best-effort LLM summarization (guarded by env gate). Falls back to heuristic on any error.
    try:
        from jinx.micro.llm.openai_caller import call_openai
    except Exception:
        return _heuristic_digest(text, ml)
    try:
        instr = (
            "Summarize the following snippet into a compact single sentence (<= %d characters). "
            "No markdown, no code fences. Be concrete."
        ) % ml
        out = await call_openai(instr, os.getenv("JINX_SUMMARY_MODEL", "gpt-4o-mini"), text)
        s = (out or "").strip()
        # Strip any accidental fences
        s = s.replace("```", "").strip()
        if len(s) > ml:
            s = s[: ml - 1] + "…"
        return s or _heuristic_digest(text, ml)
    except Exception:
        return _heuristic_digest(text, ml)
