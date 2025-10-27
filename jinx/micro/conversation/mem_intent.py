from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Tuple, Optional

from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.memory.storage import memory_dir as _memory_dir

# Language-agnostic memory intent via character n-grams + structural signals.
# No lexical word lists. Zero-IO fast path; optional cached centroid file if present.

try:
    _TTL_SEC = float(os.getenv("JINX_MEMINT_TTL_SEC", "8"))
except Exception:
    _TTL_SEC = 8.0
try:
    _THRESH = float(os.getenv("JINX_MEMINT_THRESH", "0.58"))
except Exception:
    _THRESH = 0.58

_CACHE: Dict[str, Tuple[float, float]] = {}
_CENTROID: Tuple[float, Dict[str, float]] | None = None  # (exp_ts, ngram->weight)


def _now() -> float:
    try:
        import time as _t
        return _t.time()
    except Exception:
        return 0.0


def _centroid_path() -> str:
    return os.path.join(_memory_dir(), "mem_intent", "centroid.json")


def _load_centroid() -> Optional[Dict[str, float]]:
    global _CENTROID
    now = _now()
    if _CENTROID and _CENTROID[0] > now:
        return _CENTROID[1]
    path = _centroid_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            vec = {str(k): float(v) for k, v in (obj or {}).items()}
            _CENTROID = (now + max(1.0, _TTL_SEC), vec)
            return vec
    except Exception:
        pass
    _CENTROID = (now + max(1.0, _TTL_SEC), {})
    return None


def _chargrams(text: str, n: int = 3, limit: int = 512) -> Dict[str, float]:
    t = (text or "")
    # Normalize minimal: lowercase, collapse spaces
    t = " ".join(t.lower().split())
    counts: Dict[str, int] = {}
    for i in range(0, max(0, len(t) - n + 1)):
        g = t[i : i + n]
        counts[g] = counts.get(g, 0) + 1
    # L2 normalize and keep top-N by count
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    tot = math.sqrt(sum(v * v for _, v in items)) or 1.0
    return {k: (v / tot) for k, v in items}


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            s += va * vb
    return float(max(0.0, min(1.0, s)))


def _struct_score(text: str) -> float:
    # Language-agnostic structure cues: question marks, quotes, bracketed snippets, punctuation density
    t = (text or "")
    if not t:
        return 0.0
    L = len(t)
    q = t.count("?")
    q_score = min(0.12, 0.04 * q)
    quotes = t.count('"') + t.count("'")
    qu_score = min(0.12, 0.03 * quotes)
    # bracketed segments like [ ... ] / ( ... ) often indicate quoted search targets
    br = t.count("[") + t.count("]") + t.count("(") + t.count(")")
    br_score = min(0.12, 0.03 * br)
    # punctuation density excluding letters/digits/space
    import string
    punct = sum(1 for ch in t if (not ch.isalnum()) and (not ch.isspace()))
    pd_score = min(0.2, (punct / max(10.0, L)) * 0.8)
    # shorter non-empty queries are more often retrieval intents; add mild inverse length
    len_score = 0.0
    if L <= 160:
        len_score = max(0.0, 0.14 * (1.0 - (L / 160.0)))
    return q_score + qu_score + br_score + pd_score + len_score


def memory_intent_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    # TTL cache
    ent = _CACHE.get(t)
    now = _now()
    if ent and ent[0] > now:
        return float(ent[1])

    # Char-gram similarity to optional centroid
    cg = _chargrams(t, n=3)
    cen = _load_centroid() or {}
    sim = _cos(cg, cen) if cen else 0.0

    # Structure signals
    ss = _struct_score(t)

    # Code-like penalty
    try:
        penalty = 0.22 if _is_code_like(t) else 0.0
    except Exception:
        penalty = 0.0

    score = max(0.0, min(1.0, 0.55 * ss + 0.45 * sim - penalty))

    _CACHE[t] = (now + max(1.0, _TTL_SEC), score)
    return score


def likely_memory_action(text: str, *, threshold: Optional[float] = None) -> bool:
    thr = float(threshold) if threshold is not None else _THRESH
    return memory_intent_score(text) >= thr
