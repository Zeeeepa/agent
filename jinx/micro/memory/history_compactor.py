from __future__ import annotations

import os
import time
import datetime as _dt
from collections import Counter
from typing import Dict, List, Tuple

from jinx.micro.memory.storage import memory_dir
from jinx.async_utils.fs import read_text_raw, write_text
from jinx.state import shard_lock

_HIST_DIR = os.path.join(memory_dir(), "history")
_WEEKLY_DIR = os.path.join(_HIST_DIR, "weekly")
_STAMP = os.path.join(_WEEKLY_DIR, ".weekly_last_run")


def _ensure_dirs() -> None:
    try:
        os.makedirs(_HIST_DIR, exist_ok=True)
        os.makedirs(_WEEKLY_DIR, exist_ok=True)
    except Exception:
        pass


def _week_key(ts_ms: int) -> str:
    try:
        dt = _dt.datetime.utcfromtimestamp(ts_ms / 1000.0)
        y, w, _ = dt.isocalendar()
        return f"{y}-W{int(w):02d}"
    except Exception:
        return "unknown"


async def compact_weekly(max_files: int = 500) -> None:
    """Compact history snapshots into weekly summaries.

    - Scans .jinx/memory/history/*_state.md (bounded by max_files most recent)
    - Aggregates counts for path:/symbol:/pref:/decision:/setting:
    - Writes .jinx/memory/history/weekly/<year-W##>.md
    Controlled by env gates in optimizer; function is idempotent and throttled upstream.
    """
    _ensure_dirs()
    # Throttle by last-run stamp
    try:
        min_interval = int(os.getenv("JINX_MEM_HISTORY_COMPACT_MIN_INTERVAL_MS", "180000"))
    except Exception:
        min_interval = 180000
    try:
        st = os.stat(_STAMP)
        last = int(st.st_mtime * 1000)
    except Exception:
        last = 0
    now = int(time.time() * 1000)
    if min_interval > 0 and (now - last) < min_interval:
        return
    try:
        names = [n for n in os.listdir(_HIST_DIR) if n.endswith("_state.md")]  # files only
    except Exception:
        names = []
    if not names:
        return
    names.sort()  # ts prefix ordering
    if max_files > 0:
        names = names[-max_files:]

    agg: Dict[str, Counter] = {}
    async with shard_lock:
        for fn in names:
            path = os.path.join(_HIST_DIR, fn)
            try:
                # parse ts from filename prefix
                ts_str = fn.split("_")[0]
                ts_ms = int(ts_str)
            except Exception:
                ts_ms = 0
            wk = _week_key(ts_ms)
            try:
                body = await read_text_raw(path)
            except Exception:
                body = ""
            if not body:
                continue
            c = agg.setdefault(wk, Counter())
            for raw in (body or "").splitlines():
                ln = (raw or "").strip().lower()
                if not ln:
                    continue
                # crude buckets
                if ln.startswith("path: "):
                    c[ln] += 1
                elif ln.startswith("symbol: "):
                    c[ln] += 1
                elif ln.startswith("pref: "):
                    c[ln] += 1
                elif ln.startswith("decision: "):
                    c[ln] += 1
                elif ln.startswith("setting: "):
                    c[ln] += 1
    # write weekly files
    async with shard_lock:
        for wk, ctr in agg.items():
            outp = os.path.join(_WEEKLY_DIR, f"{wk}.md")
            try:
                items = ctr.most_common(300)
                lines = [f"{k}" for k, _ in items]
                body = "\n".join(lines)
                if body and not body.endswith("\n"):
                    body += "\n"
                await write_text(outp, body)
            except Exception:
                continue
    # stamp
    try:
        with open(_STAMP, "w", encoding="utf-8") as f:
            f.write(str(int(time.time() * 1000)))
    except Exception:
        pass
