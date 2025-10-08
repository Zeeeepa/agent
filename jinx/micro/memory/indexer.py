from __future__ import annotations

import os
import time
from typing import List

from jinx.micro.embeddings.pipeline import embed_text as _embed_text
from jinx.micro.embeddings.text_clean import is_noise_text as _is_noise
from jinx.micro.memory.storage import memory_dir


def _lines(txt: str) -> List[str]:
    return [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]


async def ingest_memory(compact: str | None, evergreen: str | None) -> None:
    """Embed a small selection of memory lines into the embeddings runtime store.

    - Source is tagged as 'state' to benefit the retrieval boost for short queries.
    - The function is budgeted and best-effort; failures are swallowed.
    Controls:
    - JINX_MEM_EMB_K: total lines to embed per run (default 12)
    - JINX_MEM_EMB_MINLEN: minimum preview length (default 12)
    - JINX_MEM_EMB_INCLUDE_EVERGREEN: include evergreen lines (default 1)
    """
    try:
        k_total = max(1, int(os.getenv("JINX_MEM_EMB_K", "12")))
    except Exception:
        k_total = 12
    try:
        minlen = max(8, int(os.getenv("JINX_MEM_EMB_MINLEN", "12")))
    except Exception:
        minlen = 12
    try:
        inc_ever = str(os.getenv("JINX_MEM_EMB_INCLUDE_EVERGREEN", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        inc_ever = True

    cand: List[str] = []
    c_lines = _lines(compact or "")
    if c_lines:
        # prefer tail from compact (recency)
        cand.extend(c_lines[-(k_total * 2):])
    if inc_ever:
        e_lines = _lines(evergreen or "")
        if e_lines:
            # take head and a bit of tail to mix stability and recency
            head = e_lines[: (k_total // 2)]
            tail = e_lines[-(k_total // 2):]
            cand.extend(head + tail)
    # filter
    seen: set[str] = set()
    picked: List[str] = []
    for ln in cand:
        s = (ln or "").strip()
        if len(s) < minlen:
            continue
        if _is_noise(s):
            continue
        if s in seen:
            continue
        seen.add(s)
        picked.append(s)
        if len(picked) >= k_total:
            break

    # Throttle to avoid embedding too frequently
    try:
        min_interval = int(os.getenv("JINX_MEM_EMB_MIN_INTERVAL_MS", "30000"))
    except Exception:
        min_interval = 30000
    stamp = os.path.join(memory_dir(), ".emb_last_run")
    try:
        st = os.stat(stamp)
        last = int(st.st_mtime * 1000)
    except Exception:
        last = 0
    now = int(time.time() * 1000)
    if min_interval > 0 and (now - last) < min_interval:
        return

    # embed
    for s in picked:
        try:
            await _embed_text(s, source="state", kind="mem")
        except Exception:
            continue
    # update stamp
    try:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception:
        pass
