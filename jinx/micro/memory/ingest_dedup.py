from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Iterable, List, Set, Tuple

from jinx.micro.memory.storage import memory_dir

_LEDGER = ".mem_ingest_ledger.jsonl"


def _now_ms() -> int:
    try:
        return int(time.time() * 1000)
    except Exception:
        return 0


def _sha(s: str) -> str:
    try:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return str(len(s or ""))


def _ledger_path() -> str:
    return os.path.join(memory_dir(), _LEDGER)


def load_ledger(ttl_ms: int) -> Tuple[Set[str], List[dict]]:
    """Load recent ingested hashes with TTL pruning.

    Returns (seen_hashes, entries) where entries is pruned list of dicts.
    """
    path = _ledger_path()
    now = _now_ms()
    seen: Set[str] = set()
    out: List[dict] = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ts = int(obj.get("ts") or 0)
                    if ttl_ms > 0 and (now - ts) > ttl_ms:
                        continue
                    h = str(obj.get("sha") or "")
                    if h:
                        seen.add(h)
                        out.append(obj)
    except Exception:
        return (set(), [])
    return (seen, out)


def filter_new_lines(lines: Iterable[str], *, ttl_ms: int, max_entries: int = 5000) -> Tuple[List[str], List[dict]]:
    """Filter out lines seen in ledger within TTL. Returns (new_lines, new_entries).

    new_entries are ledger records to append for the new lines.
    """
    seen, cur = load_ledger(ttl_ms)
    now = _now_ms()
    new_lines: List[str] = []
    new_entries: List[dict] = []
    for s in lines:
        try:
            h = _sha(s)
        except Exception:
            h = str(len(s or ""))
        if h in seen:
            continue
        new_lines.append(s)
        new_entries.append({"sha": h, "ts": now, "len": len(s or "")})
        if len(new_entries) >= max_entries:
            break
    # keep only last max_entries in memory; write will prune if needed
    return (new_lines, new_entries)


def update_ledger(new_entries: List[dict], *, ttl_ms: int, max_entries: int = 5000) -> None:
    path = _ledger_path()
    now = _now_ms()
    # Load existing (pruned) and append
    try:
        seen, cur = load_ledger(ttl_ms)
    except Exception:
        cur = []
    cur.extend(new_entries or [])
    # Drop excess oldest
    try:
        cur = sorted(cur, key=lambda o: int(o.get("ts") or 0))[-max_entries:]
    except Exception:
        cur = cur[-max_entries:]
    try:
        os.makedirs(memory_dir(), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for obj in cur:
                f.write(json.dumps(obj) + "\n")
    except Exception:
        pass


__all__ = ["filter_new_lines", "update_ledger"]
