from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple

from jinx.state import shard_lock
from jinx.async_utils.fs import read_text_raw
from jinx.micro.memory.storage import memory_dir


def _split_blocks(text: str) -> List[str]:
    # Split by blank lines (one or more)
    parts = re.split(r"\n\s*\n", text or "")
    return [p for p in parts if (p or "").strip()]


def _extract_turn(block: str) -> Tuple[str, str]:
    user = ""
    jinx = ""
    if not block:
        return user, jinx
    lines = block.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("User:"):
            user = ln[len("User:") :].strip()
            # Historically, user content is single-line; still, collect following indented lines if any
            j = i + 1
            while j < n and (lines[j].startswith("  ") or lines[j].startswith("\t")):
                user += "\n" + lines[j]
                j += 1
            i = j
            continue
        if ln.startswith("Jinx:"):
            jinx = ln[len("Jinx:") :].strip()
            # Collect the remainder of the block into the same Jinx message
            j = i + 1
            while j < n:
                jinx += "\n" + lines[j]
                j += 1
            break
        i += 1
    return user.strip(), jinx.strip()


async def parse_active_turns() -> List[Dict[str, str]]:
    """Parse .jinx/memory/active.md into a list of turns: [{turn, user, jinx}]."""
    path = os.path.join(memory_dir(), "active.md")
    async with shard_lock:
        try:
            txt = await read_text_raw(path) if os.path.exists(path) else ""
        except Exception:
            txt = ""
    if not txt:
        return []
    out: List[Dict[str, str]] = []
    blocks = _split_blocks(txt)
    for idx, blk in enumerate(blocks, start=1):
        u, a = _extract_turn(blk)
        if not u and not a:
            continue
        out.append({"turn": str(idx), "user": u, "jinx": a})
    return out


async def get_user_message(n: int) -> str:
    turns = await parse_active_turns()
    if n <= 0 or n > len(turns):
        return ""
    return turns[n - 1].get("user", "") or ""


async def get_jinx_reply_to(n: int) -> str:
    turns = await parse_active_turns()
    if n <= 0 or n > len(turns):
        return ""
    return turns[n - 1].get("jinx", "") or ""


__all__ = [
    "parse_active_turns",
    "get_user_message",
    "get_jinx_reply_to",
]
