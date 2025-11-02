from __future__ import annotations

import os
import time
from typing import List, Optional

from jinx.micro.memory.storage import memory_dir, write_summary_snapshot


def _lines(s: str) -> List[str]:
    return [ln.strip() for ln in (s or "").splitlines() if (ln or "").strip()]


def _pick_recent(lines: List[str], n: int, max_chars: int) -> List[str]:
    if not lines:
        return []
    picked: List[str] = []
    for ln in lines[-max(1, n):]:
        if not ln:
            continue
        # Basic noise trim
        t = " ".join(ln.split())[:max_chars]
        if t:
            picked.append(t)
    return picked


def _pick_facts(evergreen: str, max_items: int, max_chars: int) -> List[str]:
    out: List[str] = []
    for ln in _lines(evergreen):
        low = ln.lower()
        if low.startswith("path: ") or low.startswith("symbol: ") or low.startswith("pref: ") or low.startswith("decision: ") or low.startswith("setting: "):
            out.append(ln[:max_chars])
        if len(out) >= max_items:
            break
    return out


async def summarize_if_needed(compact: str, evergreen: Optional[str]) -> None:
    """Best-effort periodic summary writer under strict throttling.

    Env controls:
    - JINX_MEM_SUMMARY_ENABLE: enable/disable (default 0)
    - JINX_MEM_SUMMARY_MIN_INTERVAL_MS: min interval between writes (default 300000)
    - JINX_MEM_SUMMARY_MIN_LINES: minimum compact line count (default 180)
    - JINX_MEM_SUMMARY_RECENT_LINES: number of recent lines to include (default 60)
    - JINX_MEM_SUMMARY_MAX_CHARS_PER_LINE: trim per line (default 240)
    - JINX_MEM_SUMMARY_MAX_FACTS: facts to include from evergreen (default 64)
    """
    try:
        en = str(os.getenv("JINX_MEM_SUMMARY_ENABLE", "0")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        en = False
    if not en:
        return

    try:
        min_interval = int(os.getenv("JINX_MEM_SUMMARY_MIN_INTERVAL_MS", "300000"))
    except Exception:
        min_interval = 300000
    try:
        min_lines = int(os.getenv("JINX_MEM_SUMMARY_MIN_LINES", "180"))
    except Exception:
        min_lines = 180
    try:
        recent_n = int(os.getenv("JINX_MEM_SUMMARY_RECENT_LINES", "60"))
    except Exception:
        recent_n = 60
    try:
        max_line = int(os.getenv("JINX_MEM_SUMMARY_MAX_CHARS_PER_LINE", "240"))
    except Exception:
        max_line = 240
    try:
        max_facts = int(os.getenv("JINX_MEM_SUMMARY_MAX_FACTS", "64"))
    except Exception:
        max_facts = 64

    stamp = os.path.join(memory_dir(), ".summary_last_run")
    now = int(time.time() * 1000)
    try:
        st = os.stat(stamp)
        last = int(st.st_mtime * 1000)
    except Exception:
        last = 0
    if min_interval > 0 and (now - last) < min_interval:
        return

    comp_lines = _lines(compact)
    if len(comp_lines) < min_lines:
        # Still update stamp to avoid tight loops when lines grow slowly
        try:
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(str(now))
        except Exception:
            pass
        return

    recent = _pick_recent(comp_lines, recent_n, max_line)
    facts = _pick_facts(evergreen or "", max_facts, max_line)

    if not recent and not facts:
        try:
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(str(now))
        except Exception:
            pass
        return

    body_parts: List[str] = []
    if recent:
        body_parts.append("## Recent highlights\n" + "\n".join(recent))
    if facts:
        body_parts.append("\n## Key facts snapshot\n" + "\n".join(facts))
    body = "\n\n".join(body_parts).strip()

    # Best-effort write
    try:
        await write_summary_snapshot(body, title="session")
    except Exception:
        pass

    # Update stamp
    try:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception:
        pass
