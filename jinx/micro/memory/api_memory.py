from __future__ import annotations

import os
import re
import hashlib
from typing import Tuple, List

from jinx.state import shard_lock
from jinx.async_utils.fs import read_text_raw, write_text
from jinx.micro.memory.storage import memory_dir, ensure_nl, read_compact
from jinx.micro.conversation.memory_sanitize import sanitize_transcript_for_memory
from jinx.micro.conversation.memory_render import summarize_agent_output_for_memory as _summarize_agent


def _paths() -> Tuple[str, str]:
    mdir = memory_dir()
    active = os.path.join(mdir, "active.md")
    compact = os.path.join(mdir, "active.compact.md")
    try:
        os.makedirs(mdir, exist_ok=True)
    except Exception:
        pass
    return active, compact


def _trim_to_chars(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    # Try to cut on line boundary
    cut = s[:limit]
    i = cut.rfind("\n")
    if i > 0 and i > limit * 7 // 10:
        return cut[:i]
    return cut


def _summarize_code_blocks(text: str, head: int, tail: int, max_lines: int) -> str:
    """Replace large code blocks with compact head/tail and sha note.
    Code blocks are fenced with ```lang ... ``` or ``` ... ```.
    """
    if not text:
        return ""
    out: List[str] = []
    pos = 0
    n = len(text)
    fence_pat = re.compile(r"```[a-zA-Z0-9_+-]*\n")
    while pos < n:
        m = fence_pat.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        start = m.start()
        out.append(text[pos:start])
        # find closing fence
        close = text.find("```", m.end())
        if close == -1:
            # malformed; append rest
            out.append(text[m.start():])
            break
        block = text[m.end():close]
        lines = block.splitlines()
        if len(lines) > max_lines:
            head_lines = lines[:head]
            tail_lines = lines[-tail:] if tail > 0 else []
            code_join = "\n".join(lines)
            sha = hashlib.sha256(code_join.encode("utf-8", errors="ignore")).hexdigest()[:12]
            note = f"[code omitted {len(lines)} lines sha={sha}]"
            out.append(
                "```\n"
                + "\n".join(head_lines)
                + ("\n...\n" if tail_lines else "\n...")
                + "\n"
                + "\n".join(tail_lines)
                + "\n```\n"
                + note
                + "\n"
            )
        else:
            out.append(text[m.start():close+3])
        pos = close + 3
    return "".join(out)


async def append_turn(user_text: str, jinx_text: str) -> None:
    """Append the latest Q/A pair into active and compact memory files.

    - active.md stores the raw (sanitized) Q/A pair.
    - active.compact.md stores a compacted version to save tokens.
    """
    u = (user_text or "").strip()
    a = (jinx_text or "").strip()
    if not u and not a:
        return
    active_path, compact_path = _paths()

    # Sanitize to remove tool tags from stored agent text; if it erases content,
    # fallback to a compact summary derived from tool/code blocks so Jinx line isn't lost.
    a_s = sanitize_transcript_for_memory(a, last_user_line="") if a else ""
    if a and not (a_s or "").strip():
        try:
            a_s = _summarize_agent(a)
        except Exception:
            a_s = ""

    # Build entries
    entry_active = []
    if u:
        entry_active.append(f"User: {u}")
    if a_s:
        entry_active.append(f"Jinx: {a_s}")
    entry_active_text = "\n".join(entry_active) + "\n\n"

    try:
        # Compact agent text by truncating big code blocks
        head = max(5, int(os.getenv("JINX_API_MEM_CODE_HEAD", "20")))
        tail = max(0, int(os.getenv("JINX_API_MEM_CODE_TAIL", "10")))
        max_lines = max(50, int(os.getenv("JINX_API_MEM_CODE_MAX_LINES", "200")))
        base = a_s
        if (not base) and a:
            # If sanitation removed everything, use a compact summary
            base = _summarize_agent(a)
        a_compact = _summarize_code_blocks(base, head, tail, max_lines) if base else ""
        # Hard cap per-turn length for compact
        pt_cap = max(500, int(os.getenv("JINX_API_MEM_PER_TURN_CHARS", "4000")))
        a_compact = _trim_to_chars(a_compact, pt_cap)
    except Exception:
        a_compact = a_s or ""

    entry_compact = []
    if u:
        entry_compact.append(f"User: {u}")
    if a_compact:
        entry_compact.append(f"Jinx: {a_compact}")
    entry_compact_text = "\n".join(entry_compact) + "\n\n"

    async with shard_lock:
        try:
            prev = await read_text_raw(active_path) if os.path.exists(active_path) else ""
        except Exception:
            prev = ""
        new_active = (prev or "") + entry_active_text
        try:
            await write_text(active_path, ensure_nl(new_active))
        except Exception:
            pass
        try:
            prev_c = await read_text_raw(compact_path) if os.path.exists(compact_path) else ""
        except Exception:
            prev_c = ""
        new_compact = (prev_c or "") + entry_compact_text
        try:
            await write_text(compact_path, ensure_nl(new_compact))
        except Exception:
            pass


async def build_api_memory_block(is_followup: bool, topic_shifted: bool) -> str:
    """Return a <memory>...</memory> block from file-based views.

    - If `is_followup` and not `topic_shifted`: use active.md (fuller), else use active.compact.md.
    - Apply character budgets to bound prompt size.
    """
    active_path, compact_path = _paths()
    use_active = bool(is_followup and (not topic_shifted))
    try:
        if use_active and os.path.exists(active_path):
            txt = await read_text_raw(active_path)
        elif os.path.exists(compact_path):
            txt = await read_text_raw(compact_path)
        else:
            txt = ""
    except Exception:
        txt = ""

    # Fallback: use legacy compact memory if active files are empty
    if not txt:
        try:
            txt = await read_compact()
        except Exception:
            txt = ""

    if not txt:
        return ""

    try:
        if use_active:
            limit = max(4000, int(os.getenv("JINX_API_MEM_FOLLOWUP_MAX_CHARS", "16000")))
        else:
            limit = max(1000, int(os.getenv("JINX_API_MEM_MAX_CHARS", "4000")))
    except Exception:
        limit = 4000

    body = _trim_to_chars(txt.strip(), limit)
    if not body:
        return ""
    return f"<memory>\n{body}\n</memory>"


__all__ = ["append_turn", "build_api_memory_block"]
