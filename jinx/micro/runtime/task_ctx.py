from __future__ import annotations

from contextvars import ContextVar

# Current logical task group for a conversation turn (e.g., session/thread id)
current_group: ContextVar[str] = ContextVar("jinx_current_group", default="main")


def get_current_group() -> str:
    try:
        v = current_group.get()
        return (v or "main").strip() or "main"
    except Exception:
        return "main"
