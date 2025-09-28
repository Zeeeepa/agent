from __future__ import annotations

from .util import _is_short_reply
from .anchors import _last_user_query


def augment_task_text(x: str, synth: str) -> str:
    """Compose a more complete task when the user reply is a short follow-up.

    If the current input is short, prepend the last longer user question
    to preserve context for the task block.
    """
    t = (x or "").strip()
    if not _is_short_reply(t):
        return t
    last_u = _last_user_query(synth)
    if last_u:
        combo = f"{last_u}\n\nFollow-up: {t}"
        return combo[:1500]
    return t
