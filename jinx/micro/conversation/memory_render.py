from __future__ import annotations

import re

_MAX_PREVIEW = 240


def _clip(s: str, n: int = _MAX_PREVIEW) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s if n <= 0 or len(s) <= n else s[:n]


def summarize_agent_output_for_memory(agent_out: str) -> str:
    """Derive a compact, single-line preview for the agent reply to store in <memory> when
    the raw agent output is mostly tool/code tags.

    Preference order:
    1) <python_question_{key}>: extract the question text from print("...") inside the block.
    2) <machine_{key}>: use the first short line (analysis) as a one-line summary.
    3) <python_{key}>: if comments exist at top, use first non-empty comment; else say 'executed code: N lines'.
    4) Fallback: strip tags and collapse whitespace.
    """
    t = agent_out or ""
    if not t.strip():
        return ""

    # 1) python_question
    m = re.search(r"<python_question_[^>]+>([\s\S]*?)</python_question_[^>]+>", t, re.IGNORECASE)
    if m:
        body = (m.group(1) or "").strip()
        # Try to extract print("...") payload
        q = re.search(r"print\((?:r|f|fr)?([\"\'])([\s\S]*?)\1\)", body)
        if q:
            return _clip(q.group(2))
        return _clip(re.sub(r"\s+", " ", body))

    # 2) machine analysis
    m = re.search(r"<machine_[^>]+>([\s\S]*?)</machine_[^>]+>", t, re.IGNORECASE)
    if m:
        body = (m.group(1) or "").strip()
        # First non-empty line
        for ln in body.splitlines():
            ln = ln.strip()
            if ln:
                return _clip(ln)

    # 3) python code
    m = re.search(r"<python_[^>]+>([\s\S]*?)</python_[^>]+>", t, re.IGNORECASE)
    if m:
        body = (m.group(1) or "").rstrip()
        lines = body.splitlines()
        # Find top comment
        for ln in lines[:10]:
            s = ln.strip()
            if s.startswith("#") and len(s) > 1:
                return _clip(s.lstrip("# "))
        # Otherwise summarize by line count
        n = len([x for x in lines if x.strip()])
        return _clip(f"executed code: {n} lines")

    # 4) Fallback: strip tags crudely and collapse whitespace
    core = re.sub(r"<[^>]+>", " ", t)
    core = re.sub(r"\s+", " ", core).strip()
    return _clip(core)


__all__ = ["summarize_agent_output_for_memory"]
