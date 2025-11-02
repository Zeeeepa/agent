from __future__ import annotations

import os
import time
from typing import List

from jinx.micro.memory.storage import memory_dir
from jinx.micro.memory.kb_extract import extract_triplets
from jinx.micro.memory.kb_store import upsert_triplets


def _lines(s: str) -> List[str]:
    return [ln.strip() for ln in (s or "").splitlines() if (ln or "").strip()]


async def update_kb_from_memory(compact: str | None, evergreen: str | None) -> None:
    """Background KB update: extract triplets and upsert.

    Env:
    - JINX_MEM_KB_ENABLE (default 0)
    - JINX_MEM_KB_MIN_INTERVAL_MS (default 45000)
    - JINX_MEM_KB_MAX_TIME_MS (default 200)
    - JINX_MEM_KB_MAX_ITEMS (default 120)
    """
    try:
        en = str(os.getenv("JINX_MEM_KB_ENABLE", "0")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        en = False
    if not en:
        return

    try:
        min_int = int(os.getenv("JINX_MEM_KB_MIN_INTERVAL_MS", "45000"))
    except Exception:
        min_int = 45000
    try:
        max_ms = int(os.getenv("JINX_MEM_KB_MAX_TIME_MS", "200"))
    except Exception:
        max_ms = 200
    try:
        max_items = int(os.getenv("JINX_MEM_KB_MAX_ITEMS", "120"))
    except Exception:
        max_items = 120

    stamp = os.path.join(memory_dir(), ".kb_last_run")
    try:
        st = os.stat(stamp)
        last = int(st.st_mtime * 1000)
    except Exception:
        last = 0
    now = int(time.time() * 1000)
    if min_int > 0 and (now - last) < min_int:
        return

    lines: List[str] = []
    lines.extend(_lines(compact or ""))
    # Evergreen lines are often already normalized facts
    lines.extend(_lines(evergreen or ""))
    if not lines:
        try:
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(str(now))
        except Exception:
            pass
        return

    # Extract with bounded budget and upsert
    trips = extract_triplets(lines, max_items=max_items, max_time_ms=max_ms)
    if trips:
        try:
            await upsert_triplets(trips[:max_items])
        except Exception:
            pass

    # Update stamp
    try:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception:
        pass
