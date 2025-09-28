from __future__ import annotations

from typing import Dict, List


def render_continuity_block(anchors: Dict[str, List[str]] | None, last_q: str, last_u: str, short_followup: bool) -> str:
    """Render a compact continuity header block to guide the main brain.

    Kept intentionally minimal to respect real-time budgets.
    """
    lines: List[str] = []
    if last_q:
        lines.append(f"q: {last_q[:220]}")
    if last_u:
        lines.append(f"prev_user: {last_u[:220]}")
    a = anchors or {}
    syms = (a.get("symbols") or [])[:3]
    paths = (a.get("paths") or [])[:2]
    if syms:
        lines.append("symbols: " + ", ".join(syms))
    if paths:
        lines.append("paths: " + ", ".join(paths))
    if short_followup:
        lines.append("note: short follow-up detected; continuity cache applied where safe")
    if not lines:
        return ""
    return "<continuity_anchors>\n" + "\n".join(lines) + "\n</continuity_anchors>"
