from __future__ import annotations

import os
from typing import List

from jinx.state import shard_lock
from jinx.async_utils.fs import read_text_raw
from jinx.micro.memory.storage import memory_dir
from jinx.micro.memory.pin_store import load_pins as _pins_load

_TOPICS_DIR = os.path.join(memory_dir(), "topics")
_CH_PATHS = os.path.join(memory_dir(), "paths.md")
_CH_SYMBOLS = os.path.join(memory_dir(), "symbols.md")
_CH_PREFS = os.path.join(memory_dir(), "prefs.md")
_CH_DECS = os.path.join(memory_dir(), "decisions.md")
_OPEN_BUFFERS = os.path.join(memory_dir(), "open_buffers.jsonl")


def _read_file(path: str) -> str:
    try:
        if os.path.exists(path):
            # async read via fs util under shard lock
            # but here we keep a fast sync path for topics listing to minimize locks
            pass
    except Exception:
        pass
    try:
        # best-effort async read under shard lock
        # call read_text_raw in a separate awaiter in caller if needed
        # for simplicity in snapshot we do sync open with errors ignored
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _clamp(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return s
    i = s.rfind("\n", 0, n)
    return s[: i if i > 0 and i > n * 7 // 10 else n]


def _topic_head(name: str, text: str, head_lines: int) -> str:
    if not text:
        return f"# {name}\n"
    lines = text.splitlines()
    head = "\n".join(lines[: max(1, head_lines)])
    return f"# {name}\n{head}\n"


async def build_memory_snapshot(max_chars: int = 4000, *, head_lines: int = 6, buffers_chars: int = 200) -> str:
    """Assemble a compact, program-friendly snapshot of durable memory state.

    Sections:
    - PINS
    - CHANNELS (paths/symbols/prefs/decisions)
    - TOPICS (filenames and head lines)
    - OPEN_BUFFERS (name + clipped text)
    """
    parts: List[str] = []

    # PINS
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    if pins:
        parts.append("[PINS]\n" + "\n".join(pins[:16]))

    # CHANNELS
    def _add_channel(label: str, path: str) -> None:
        body = _read_file(path)
        body = body.strip()
        if body:
            parts.append(f"[CHANNEL {label}]\n{body}")
    _add_channel("paths", _CH_PATHS)
    _add_channel("symbols", _CH_SYMBOLS)
    _add_channel("prefs", _CH_PREFS)
    _add_channel("decisions", _CH_DECS)

    # TOPICS (list a few files, show head lines)
    try:
        files = [fn for fn in os.listdir(_TOPICS_DIR) if fn.endswith(".md")]
    except Exception:
        files = []
    for fn in sorted(files)[-10:]:
        p = os.path.join(_TOPICS_DIR, fn)
        txt = _read_file(p)
        parts.append(_topic_head(fn, txt, head_lines))

    # OPEN BUFFERS (jsonl)
    try:
        if os.path.exists(_OPEN_BUFFERS):
            raw = _read_file(_OPEN_BUFFERS)
            if raw:
                import json as _json
                out_lines: List[str] = []
                for line in raw.splitlines():
                    try:
                        obj = _json.loads(line)
                    except Exception:
                        continue
                    name = str(obj.get("name") or obj.get("path") or "buffer")
                    text = str(obj.get("text") or "")
                    if not text:
                        continue
                    snip = _clamp(text.strip(), max(24, buffers_chars))
                    out_lines.append(f"- {name}: {snip}")
                if out_lines:
                    parts.append("[OPEN_BUFFERS]\n" + "\n".join(out_lines[:12]))
    except Exception:
        pass

    body = "\n\n".join(parts).strip()
    if max_chars > 0 and len(body) > max_chars:
        body = _clamp(body, max_chars)
    return body


__all__ = ["build_memory_snapshot"]
