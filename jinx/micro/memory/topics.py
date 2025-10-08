from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple

from jinx.micro.memory.storage import memory_dir
from jinx.async_utils.fs import read_text_raw, write_text
from jinx.state import shard_lock

_TOPICS_DIR = os.path.join(memory_dir(), "topics")
_STAMP = os.path.join(memory_dir(), ".topics_last_run")

_SEG_RE = re.compile(r"[\\/]+")


def _ensure_dir() -> None:
    try:
        os.makedirs(_TOPICS_DIR, exist_ok=True)
    except Exception:
        pass


def _split_path(p: str) -> List[str]:
    p2 = _SEG_RE.split(p.strip())
    return [s for s in p2 if s]


def _topic_from_path(p: str) -> str:
    segs = _split_path(p.lower())
    if not segs:
        return "misc"
    # Heuristics: prefer 'jinx' subdir if present, else the first segment
    if "jinx" in segs and len(segs) > segs.index("jinx") + 1:
        return segs[segs.index("jinx") + 1]
    return segs[0]


def _topic_from_symbol(s: str) -> str:
    # e.g., module.Class.method -> module
    base = s.lower()
    if "." in base:
        return base.split(".", 1)[0]
    return "symbols"


async def update_topics(evergreen: str | None) -> None:
    """Split evergreen lines into topic files based on path/symbol heuristics.

    Throttled via JINX_MEM_TOPICS_MIN_INTERVAL_MS (default 45000).
    Max lines per topic: JINX_MEM_TOPICS_MAX_LINES (default 120).
    """
    if not evergreen:
        return
    try:
        min_interval = int(os.getenv("JINX_MEM_TOPICS_MIN_INTERVAL_MS", "45000"))
    except Exception:
        min_interval = 45000
    try:
        max_lines = int(os.getenv("JINX_MEM_TOPICS_MAX_LINES", "120"))
    except Exception:
        max_lines = 120

    try:
        st = os.stat(_STAMP)
        last = int(st.st_mtime * 1000)
    except Exception:
        last = 0
    now = int(time.time() * 1000)
    if min_interval > 0 and (now - last) < min_interval:
        return

    _ensure_dir()
    buckets: Dict[str, List[str]] = {}
    for raw in (evergreen or "").splitlines():
        ln = (raw or "").strip()
        if not ln:
            continue
        low = ln.lower()
        if low.startswith("path: "):
            p = ln[6:].strip()
            topic = _topic_from_path(p)
        elif low.startswith("symbol: "):
            s = ln[8:].strip()
            topic = _topic_from_symbol(s)
        elif low.startswith("pref: "):
            topic = "prefs"
        elif low.startswith("decision: "):
            topic = "decisions"
        elif low.startswith("setting: "):
            topic = "settings"
        else:
            topic = "misc"
        buckets.setdefault(topic, []).append(ln)

    # Write topics with merge and cap
    async with shard_lock:
        for topic, lines in buckets.items():
            path = os.path.join(_TOPICS_DIR, f"{topic}.md")
            try:
                prev = await read_text_raw(path)
            except Exception:
                prev = ""
            prev_lines = [l.strip() for l in (prev or "").splitlines() if l.strip()]
            merged = prev_lines + [l for l in lines if l not in prev_lines]
            if max_lines > 0 and len(merged) > max_lines:
                merged = merged[-max_lines:]
            body = "\n".join(merged) + ("\n" if merged and not merged[-1].endswith("\n") else "")
            try:
                await write_text(path, body)
            except Exception:
                pass
    # update stamp
    try:
        with open(_STAMP, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception:
        pass
