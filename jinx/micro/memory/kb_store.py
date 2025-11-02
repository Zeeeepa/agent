from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Tuple

from jinx.micro.memory.storage import memory_dir
from jinx.state import shard_lock

_KB_DIR = os.path.join(memory_dir(), "kb")
_TRIPLETS_JSON = os.path.join(_KB_DIR, "triplets.json")


def _ensure_dir() -> None:
    try:
        os.makedirs(_KB_DIR, exist_ok=True)
    except Exception:
        pass


def _key_of(s: str, p: str, o: str) -> str:
    s = (s or "").strip()
    p = (p or "").strip()
    o = (o or "").strip()
    return f"{s}|{p}|{o}"


def _tokens(q: str) -> List[str]:
    out: List[str] = []
    cur = []
    for ch in (q or "").lower():
        if ch.isalnum() or ch in ("_", ".", "/", "-", ":"):
            cur.append(ch)
        else:
            if cur:
                t = "".join(cur)
                if len(t) >= 3:
                    out.append(t)
                cur = []
    if cur:
        t = "".join(cur)
        if len(t) >= 3:
            out.append(t)
    # Dedup preserve order
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _load_all() -> Dict[str, Dict[str, int]]:
    try:
        with open(_TRIPLETS_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj  # type: ignore[return-value]
    except Exception:
        pass
    return {}


def _save_all(data: Dict[str, Dict[str, int]]) -> None:
    try:
        _ensure_dir()
        tmp = _TRIPLETS_JSON + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, _TRIPLETS_JSON)
    except Exception:
        pass


async def upsert_triplets(trips: List[Tuple[str, str, str]]) -> None:
    if not trips:
        return
    # Sanitize
    payload: List[Tuple[str, str, str]] = []
    for s, p, o in trips:
        s = (s or "").strip()
        p = (p or "").strip()
        o = (o or "").strip()
        if not s or not p or not o:
            continue
        payload.append((s[:200], p[:200], o[:200]))
    if not payload:
        return
    now = int(time.time() * 1000)
    async with shard_lock:
        data = _load_all()
        for s, p, o in payload:
            k = _key_of(s, p, o)
            ent = data.get(k)
            if isinstance(ent, dict):
                try:
                    c = int(ent.get("count") or 0) + 1
                except Exception:
                    c = 1
            else:
                c = 1
            data[k] = {"count": c, "last_ts": now}
        _save_all(data)


def _score_for(qtoks: List[str], k: str, meta: Dict[str, int], now: float, recency_win_s: float) -> float:
    low = k.lower()
    score = 0.0
    for t in qtoks:
        if t in low:
            score += 1.0
    # recency factor
    try:
        ts = float(meta.get("last_ts") or 0.0) / 1000.0
    except Exception:
        ts = 0.0
    age = max(0.0, now - ts)
    rec = 0.0 if recency_win_s <= 0 else max(0.0, 1.0 - (age / recency_win_s))
    # count factor (log-like)
    try:
        import math
        c = float(meta.get("count") or 0.0)
        cnt = 1.0 + min(1.2, math.log1p(max(0.0, c)))
    except Exception:
        cnt = 1.0
    return 0.8 * score + 0.15 * rec + 0.05 * cnt


def _fmt_line(k: str) -> str:
    # Expect k as 's|p|o'
    try:
        s, p, o = k.split("|", 2)
    except Exception:
        s, p, o = k, "is", ""
    return f"kb: {s} | {p} | {o}"


async def search_lines(query: str, *, k: int = 6, max_time_ms: int = 25, preview_chars: int = 160) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []
    qtoks = _tokens(q)
    # Load once outside lock (read-only)
    try:
        data = _load_all()
    except Exception:
        data = {}
    if not data:
        return []
    try:
        rec_win = float(os.getenv("JINX_MEM_KB_RECENCY_SEC", str(14*24*3600)))
    except Exception:
        rec_win = 14 * 24 * 3600.0
    t0 = time.perf_counter()
    now = time.time()
    scored: List[Tuple[float, str]] = []
    for k_trip, meta in data.items():
        sc = _score_for(qtoks, k_trip, meta or {}, now, rec_win)
        if sc > 0:
            scored.append((sc, k_trip))
        if max_time_ms > 0 and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[str] = []
    seen: set[str] = set()
    for _sc, key in scored:
        line = _fmt_line(key)[:preview_chars]
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= max(1, k):
            break
    return out
