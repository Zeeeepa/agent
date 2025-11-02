from __future__ import annotations

import os
import json
import time
from typing import Dict, List

from jinx.micro.memory.storage import memory_dir
from jinx.state import shard_lock

# Files
_USAGE_COUNTS = os.path.join(memory_dir(), ".usage_counts.json")

# Cache
_WEIGHTS: Dict[str, float] = {}
_WEIGHTS_TS: int = 0
_META: Dict[str, Dict[str, int]] = {}
_META_TS: int = 0


def _now_ms() -> int:
    try:
        return int(time.time() * 1000)
    except Exception:
        return 0


def _sha(s: str) -> str:
    try:
        import hashlib
        return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return str(len(s or ""))


def _to_weight(count: int) -> float:
    # Map count -> weight in [1.0, ~2.0) via log scaling
    try:
        import math
        return 1.0 + min(1.0, math.log10(1.0 + max(0, int(count))))
    except Exception:
        return 1.0


def get_weights_cached() -> Dict[str, float]:
    """Return cached usage weights keyed by sha(line). TTL controlled by env.

    This is synchronous and cheap; safe to call inside RT-critical paths.
    """
    global _WEIGHTS, _WEIGHTS_TS
    try:
        ttl = int(os.getenv("JINX_MEM_USAGE_TTL_MS", "1200"))
    except Exception:
        ttl = 1200
    now = _now_ms()
    if ttl > 0 and (now - _WEIGHTS_TS) <= ttl and _WEIGHTS:
        return _WEIGHTS
    # Reload file best-effort
    try:
        with open(_USAGE_COUNTS, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    ws: Dict[str, float] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                c = int((v or {}).get("count") if isinstance(v, dict) else int(v))
            except Exception:
                c = 0
            ws[str(k)] = _to_weight(c)
    _WEIGHTS = ws
    _WEIGHTS_TS = now
    return _WEIGHTS


def weight_for(line: str, weights: Dict[str, float] | None = None) -> float:
    wmap = weights if isinstance(weights, dict) else get_weights_cached()
    return wmap.get(_sha(line or ""), 1.0)


def get_meta_cached() -> Dict[str, Dict[str, int]]:
    """Return cached usage meta map {sha: {count:int, last_ts:int}} with TTL.

    Intended for read paths; safe and cheap.
    """
    global _META, _META_TS
    try:
        ttl = int(os.getenv("JINX_MEM_USAGE_TTL_MS", "1200"))
    except Exception:
        ttl = 1200
    now = _now_ms()
    if ttl > 0 and (now - _META_TS) <= ttl and _META:
        return _META
    try:
        with open(_USAGE_COUNTS, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    meta: Dict[str, Dict[str, int]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                try:
                    c = int(v.get("count") or 0)
                except Exception:
                    c = 0
                try:
                    ts = int(v.get("last_ts") or 0)
                except Exception:
                    ts = 0
            else:
                try:
                    c = int(v)
                except Exception:
                    c = 0
                ts = 0
            meta[str(k)] = {"count": c, "last_ts": ts}
    _META = meta
    _META_TS = now
    return _META


def last_used_ms(line: str) -> int:
    """Return last usage timestamp (ms) or 0 if unknown."""
    h = _sha(line or "")
    return int(get_meta_cached().get(h, {}).get("last_ts") or 0)


def count_for(line: str) -> int:
    """Return usage count for the line or 0 if unknown."""
    h = _sha(line or "")
    return int(get_meta_cached().get(h, {}).get("count") or 0)


async def bump_usage(lines: List[str]) -> None:
    """Increment usage counters for the given lines (trimmed content)."""
    if not lines:
        return
    # Build batch updates
    updates: Dict[str, int] = {}
    for ln in lines:
        if not ln:
            continue
        h = _sha(ln)
        updates[h] = updates.get(h, 0) + 1
    if not updates:
        return
    # Update counts file under shard lock
    async with shard_lock:
        try:
            try:
                with open(_USAGE_COUNTS, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            now = _now_ms()
            for h, inc in updates.items():
                ent = data.get(h)
                if isinstance(ent, dict):
                    c = int(ent.get("count") or 0) + int(inc)
                elif isinstance(ent, int):
                    c = int(ent) + int(inc)
                else:
                    c = int(inc)
                data[h] = {"count": c, "last_ts": now}
            # Write back atomically best-effort
            tmp = _USAGE_COUNTS + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, _USAGE_COUNTS)
        except Exception:
            pass
    # Invalidate cache timestamp to pick up changes soon
    try:
        global _WEIGHTS_TS
        _WEIGHTS_TS = 0
    except Exception:
        pass
