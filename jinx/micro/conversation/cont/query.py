from __future__ import annotations

from typing import Dict, List, Optional

from .util import _is_short_reply
from .anchors import _last_question_from_agent, _last_user_query


def augment_query_for_retrieval(x: str, synth: str, anchors: Optional[Dict[str, List[str]]] = None) -> str:
    """If current input is a short clarification, blend it with last user question.

    Returns the effective query to feed into retrieval components.
    """
    t = (x or "").strip()
    if not _is_short_reply(t):
        return t
    last_q = _last_question_from_agent(synth)
    last_u = _last_user_query(synth)
    bonus: List[str] = []
    if anchors:
        # include top symbol and path hints to stabilize retrieval
        sym = (anchors.get("symbols") or [])[:2]
        pth = (anchors.get("paths") or [])[:1]
        bonus = list(sym) + list(pth)
    if last_q or last_u or bonus:
        combo = " ".join([p for p in [last_u, t] + bonus if p])
        return combo[:1200]
    return t
