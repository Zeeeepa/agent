from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple

# Vector-only, language-agnostic locator scoring via project retrieval scores
# Integrated with adaptive learning system

try:
    from jinx.micro.embeddings.retrieval_core import retrieve_project_top_k as _retr
except Exception:  # pragma: no cover
    _retr = None  # type: ignore

try:
    from jinx.micro.brain.threshold_learner import select_threshold as _select_thresh, record_threshold_outcome as _record_thresh
except Exception:
    _select_thresh = None  # type: ignore
    _record_thresh = None  # type: ignore

# Score cache: norm_key -> (score, ts)
_CACHE: Dict[str, Tuple[float, float]] = {}


def _norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    if len(s) > 512:
        s = s[:512]
    return s


def _aggregate_scores(hits: List[Tuple[float, str, Dict[str, object]]]) -> float:
    if not hits:
        return 0.0
    # Use top-N scores aggregate; clamp to [0, 1] as a soft normalization
    topn = min(3, len(hits))
    vals = [float(hits[i][0] or 0.0) for i in range(topn)]
    # Robust mean: average of top-n
    m = sum(vals) / max(1, len(vals))
    # Soft clamp: map large scores into [0,1] via logistic-like squashing
    try:
        import math
        return 1.0 / (1.0 + math.exp(-m))
    except Exception:
        return m


async def get_locator_score(text: str) -> float:
    """Get locator score using adaptive retrieval parameters."""
    if _retr is None:
        return 0.0
    q = (text or "").strip()
    if not q:
        return 0.0
    
    # Use adaptive parameters if available, otherwise fall back to env/defaults
    try:
        from jinx.micro.brain.adaptive_retrieval import select_retrieval_params as _sel_params
        kk, budget = await _sel_params(q)
    except Exception:
        try:
            kk = max(1, int(os.getenv("JINX_LOCATOR_VEC_K", "6")))
        except Exception:
            kk = 6
        try:
            budget = max(40, int(os.getenv("JINX_LOCATOR_VEC_MS", "120")))
        except Exception:
            budget = 120
    
    t0 = time.perf_counter()
    try:
        hits = await _retr(q, k=kk, max_time_ms=budget)
    except Exception:
        hits = []
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    
    # Record outcome for adaptive learning (useful_ratio estimated as hit presence)
    try:
        from jinx.micro.brain.adaptive_retrieval import record_retrieval_outcome as _rec_out
        useful = 0.8 if len(hits) >= kk * 0.5 else 0.4
        asyncio.create_task(_rec_out(q, kk, elapsed_ms, len(hits), useful))
    except Exception:
        pass
    
    return _aggregate_scores(hits or [])


def get_locator_score_cached(text: str) -> Optional[float]:
    ttl = float(os.getenv("JINX_LOCATOR_CACHE_TTL", "60"))
    q = _norm_key(text)
    ent = _CACHE.get(q)
    if not ent:
        return None
    sc, ts = ent
    if (time.perf_counter() - ts) > ttl:
        _CACHE.pop(q, None)
        return None
    return float(sc)


async def classify_and_store(text: str) -> float:
    sc = await get_locator_score(text)
    _CACHE[_norm_key(text)] = (sc, float(time.perf_counter()))
    return sc


def schedule_classify(text: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(classify_and_store(text))
    except Exception:
        pass


__all__ = [
    "get_locator_score",
    "get_locator_score_cached",
    "classify_and_store",
    "schedule_classify",
]
