from __future__ import annotations

import time
import os
from typing import Dict, List, Tuple
import asyncio

# Lightweight, in-memory attention store with exponential decay.
# Acts as short-term working memory to bias retrieval toward recently touched concepts.
# Concept keys are expected as canonical strings like:
#   - "term: tokenizer"
#   - "symbol: ClassName.method"
#   - "path: src/module/file.py"

_ATTENTION_BUF: List[Tuple[str, float, float]] = []  # (key, ts_sec, weight)
_LOCK = asyncio.Lock()


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _half_life_sec() -> float:
    try:
        return max(10.0, float(os.getenv("JINX_BRAIN_ATTEN_HALF_SEC", "120")))
    except Exception:
        return 120.0


def _decay_factor(dt_sec: float) -> float:
    try:
        hl = _half_life_sec()
        if hl <= 0:
            return 0.0
        # Exponential decay: 0.5 ** (dt/hl)
        return 0.5 ** (max(0.0, dt_sec) / hl)
    except Exception:
        return 0.0


def _canonize(key: str) -> str:
    k = (key or "").strip().lower()
    if not k:
        return ""
    # If no concept prefix, treat as term
    if ":" not in k:
        return f"term: {k}"
    return k


async def record_attention(keys: List[str], weight: float = 1.0) -> None:
    """Record attention to concept keys with a base weight at current time.

    Non-blocking; drops empty keys. Keys are canonized.
    """
    if not keys:
        return
    ts = _now()
    items = [(_canonize(k), ts, float(weight)) for k in keys if _canonize(k)]
    if not items:
        return
    async with _LOCK:
        _ATTENTION_BUF.extend(items)
        # Bound buffer to avoid unbounded growth
        try:
            cap = max(100, int(os.getenv("JINX_BRAIN_ATTEN_BUF_CAP", "2000")))
        except Exception:
            cap = 2000
        if len(_ATTENTION_BUF) > cap:
            _ATTENTION_BUF[:] = _ATTENTION_BUF[-cap:]


def get_attention_weights() -> Dict[str, float]:
    """Return current decayed attention weights per concept key.

    Pure function wrt current time and buffer snapshot. No locking required for reads.
    """
    # Snapshot to avoid holding lock while computing
    buf = list(_ATTENTION_BUF)
    now = _now()
    acc: Dict[str, float] = {}
    for k, ts, w in buf:
        if not k:
            continue
        df = _decay_factor(now - float(ts or 0.0))
        if df <= 0.0:
            continue
        try:
            acc[k] = float(acc.get(k, 0.0)) + float(w or 0.0) * df
        except Exception:
            continue
    # Optional global scale
    try:
        scale = float(os.getenv("JINX_BRAIN_ATTEN_SCALE", "1.0"))
    except Exception:
        scale = 1.0
    if scale != 1.0:
        for k in list(acc.keys()):
            acc[k] = acc[k] * scale
    return acc
