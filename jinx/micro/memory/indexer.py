from __future__ import annotations

import os
import time
import asyncio
from typing import List

from jinx.micro.embeddings.pipeline import embed_text as _embed_text
from jinx.micro.memory.storage import memory_dir
from jinx.micro.memory.ingest_ranker import build_candidates as _build_candidates
from jinx.micro.memory.ingest_dedup import filter_new_lines as _filter_new_lines, update_ledger as _update_ledger


async def ingest_memory(compact: str | None, evergreen: str | None) -> None:
    """Advanced memory ingestion under strict RT and dedup.

    - Ranks salient lines from compact/evergreen (code-aware, noise-filtered).
    - Deduplicates via TTL ledger to avoid re-embedding the same content.
    - Embeds with bounded concurrency and per-call timeouts.

    Env controls:
    - JINX_MEM_EMB_K: total target lines per run (default 12)
    - JINX_MEM_EMB_MINLEN: minimum line length (default 12)
    - JINX_MEM_EMB_INCLUDE_EVERGREEN: include evergreen (default 1)
    - JINX_MEM_EMB_MIN_INTERVAL_MS: throttle between runs (default 30000)
    - JINX_MEM_EMB_LEDGER_TTL_MS: dedup ledger TTL (default 86400000)
    - JINX_MEM_EMB_MAX_TIME_MS: overall RT budget (default 600)
    - JINX_MEM_EMB_PER_CALL_TIMEOUT_MS: per-embed timeout (default 200)
    - JINX_MEM_EMB_CONC: embed concurrency (default 3)
    """
    # Read config
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
    try:
        min_interval = int(os.getenv("JINX_MEM_EMB_MIN_INTERVAL_MS", "30000"))
    except Exception:
        min_interval = 30000
    try:
        ledger_ttl = int(os.getenv("JINX_MEM_EMB_LEDGER_TTL_MS", "86400000"))  # 1 day
    except Exception:
        ledger_ttl = 86400000
    try:
        max_time_ms = int(os.getenv("JINX_MEM_EMB_MAX_TIME_MS", "600"))
    except Exception:
        max_time_ms = 600
    try:
        per_call_timeout = int(os.getenv("JINX_MEM_EMB_PER_CALL_TIMEOUT_MS", "200"))
    except Exception:
        per_call_timeout = 200
    try:
        conc = max(1, int(os.getenv("JINX_MEM_EMB_CONC", "3")))
    except Exception:
        conc = 3

    # Throttle between runs
    stamp = os.path.join(memory_dir(), ".emb_last_run")
    try:
        st = os.stat(stamp)
        last = int(st.st_mtime * 1000)
    except Exception:
        last = 0
    now = int(time.time() * 1000)
    if min_interval > 0 and (now - last) < min_interval:
        return

    # Build candidates (light heuristics, already noise-aware)
    picked: List[str] = _build_candidates(compact or "", evergreen or "", k_total=k_total, minlen=minlen, include_evergreen=inc_ever)
    if not picked:
        # update stamp anyway to avoid tight loops when memory is empty/noisy
        try:
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(str(now))
        except Exception:
            pass
        return

    # Deduplicate using TTL ledger
    new_lines, new_entries = _filter_new_lines(picked, ttl_ms=ledger_ttl)
    if not new_lines:
        try:
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(str(now))
        except Exception:
            pass
        return

    # Enforce overall budget while embedding with bounded concurrency
    t0 = time.perf_counter()
    sem = asyncio.Semaphore(conc)

    async def _embed_one(s: str) -> None:
        # Check global budget before starting
        if max_time_ms > 0 and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            return
        async with sem:
            try:
                if per_call_timeout > 0:
                    await asyncio.wait_for(_embed_text(s, source="state", kind="mem"), timeout=per_call_timeout / 1000.0)
                else:
                    await _embed_text(s, source="state", kind="mem")
            except Exception:
                return

    # Launch tasks up to k_total or global budget
    tasks = [asyncio.create_task(_embed_one(s)) for s in new_lines[:k_total]]
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass

    # Update ledger and stamp (best-effort)
    try:
        _update_ledger(new_entries, ttl_ms=ledger_ttl)
    except Exception:
        pass
    try:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception:
        pass
