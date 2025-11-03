from __future__ import annotations

import os
from typing import Optional

from jinx.async_utils.fs import read_text_raw, write_text
from jinx.state import shard_lock
from jinx.micro.runtime.task_ctx import get_current_group as _get_group

# Directory for per-group rolling summaries
try:
    _MEM_DIR = os.getenv("JINX_MEMORY_DIR", os.path.join(".jinx", "memory"))
except Exception:
    _MEM_DIR = os.path.join(".jinx", "memory")
_GROUPS_DIR = os.path.join(_MEM_DIR, "groups")


def get_group_id() -> str:
    """Return current group/session id.

    Defaults to 'main' when JINX_SESSION is not set.
    """
    try:
        gid = _get_group()
        if gid:
            return gid
    except Exception:
        pass
    try:
        return (os.getenv("JINX_SESSION", "") or "main").strip() or "main"
    except Exception:
        return "main"


async def read_group_summary(group: Optional[str] = None, *, max_chars: int = 1200) -> str:
    """Read a compact group summary and wrap in <group_context> if present.

    Trims to 'max_chars' characters from the end to bias towards the latest turns.
    """
    gid = (group or get_group_id()).strip() or "main"
    path = os.path.join(_GROUPS_DIR, f"{gid}.md")
    async with shard_lock:
        try:
            os.makedirs(_GROUPS_DIR, exist_ok=True)
        except Exception:
            pass
        try:
            txt = await read_text_raw(path)
        except Exception:
            txt = ""
    body = (txt or "").strip()
    if not body:
        return ""
    if max_chars and len(body) > max_chars:
        body = body[-max_chars:]
    return f"<group_context id=\"{gid}\">\n{body}\n</group_context>"


async def append_group_summary(user_text: str, agent_text: str, *, group: Optional[str] = None, keep_chars: int = 4000) -> None:
    """Append a tiny two-line summary to the group's rolling memory and trim size.

    The file remains small (<= keep_chars) by trimming from the front.
    """
    gid = (group or get_group_id()).strip() or "main"
    path = os.path.join(_GROUPS_DIR, f"{gid}.md")
    # Build minimal append block
    u = (user_text or "").strip()
    a = (agent_text or "").strip()
    if not (u or a):
        return
    block = []
    if u:
        block.append(f"User: {u}")
    if a:
        block.append(f"Jinx: {a}")
    block.append("")
    to_add = "\n".join(block)
    async with shard_lock:
        try:
            os.makedirs(_GROUPS_DIR, exist_ok=True)
        except Exception:
            pass
        try:
            prev = await read_text_raw(path)
        except Exception:
            prev = ""
        new = (prev or "").strip()
        new = (new + ("\n" if new else "") + to_add).strip()
        if keep_chars and len(new) > keep_chars:
            new = new[-keep_chars:]
        try:
            await write_text(path, new + "\n")
        except Exception:
            pass
