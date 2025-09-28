from __future__ import annotations

import os
from typing import Dict, List, Optional

from jinx.micro.conversation.cont import last_agent_question, last_user_query


def build_state_frame(
    *,
    user_text: str,
    synth: str,
    anchors: Optional[Dict[str, List[str]]] = None,
    guidance: str = "",
    cont_block: str = "",
    error_summary: str = "",
) -> str:
    """Compose a compact per-turn continuity state frame.

    The frame is designed to:
      - be short and highly retrievable by embeddings,
      - store distilled intent + anchors + minimal guidance snapshot,
      - avoid large payloads (to respect RT budgets).

    Returns a plain text string; caller decides how to persist.
    """
    try:
        max_chars = max(400, int(os.getenv("JINX_STATEFRAME_MAXCHARS", "800")))
    except Exception:
        max_chars = 800

    parts: List[str] = []
    parts.append("[StateFrame v1]")

    ut = (user_text or "").strip()
    if ut:
        parts.append(f"user: {ut[:220]}")

    q = last_agent_question(synth or "")
    if q:
        parts.append(f"agent_q: {q[:220]}")

    prev_u = last_user_query(synth or "")
    if prev_u:
        parts.append(f"prev_user: {prev_u[:220]}")

    a = anchors or {}
    syms = (a.get("symbols") or [])[:5]
    if syms:
        parts.append("symbols: " + ", ".join(syms))
    paths = (a.get("paths") or [])[:3]
    if paths:
        parts.append("paths: " + ", ".join(paths))
    qs = (a.get("questions") or [])[:3]
    if qs:
        parts.append("questions: " + " | ".join(qs))

    # Include short guidance snapshot without tags to keep it retrievable
    g = (guidance or "").strip()
    if g:
        # Strip tags if present; keep the essence lines up to ~200 chars
        gs = [ln.strip() for ln in g.splitlines() if ln.strip() and not ln.startswith("<")]
        if gs:
            parts.append("guidance: " + " | ".join(gs)[:220])

    cb = (cont_block or "").strip()
    if cb:
        cs = [ln.strip() for ln in cb.splitlines() if ln.strip() and not ln.startswith("<")]
        if cs:
            parts.append("continuity: " + " | ".join(cs)[:220])

    es = (error_summary or "").strip()
    if es:
        # Keep a single-line error summary to avoid noisy retrieval
        es1 = es.splitlines()[0][:220]
        if es1:
            parts.append("error: " + es1)

    body = "\n".join(parts)
    if len(body) > max_chars:
        body = body[:max_chars]
    return body
