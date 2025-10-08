from __future__ import annotations

from typing import Dict, List


def render_continuity_block(anchors: Dict[str, List[str]] | None, last_q: str, last_u: str, short_followup: bool) -> str:
    """Deprecated: continuity anchors tag is no longer used. Return empty string."""
    return ""
