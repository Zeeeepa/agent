from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List, Tuple

from jinx.micro.embeddings.embed_cache import embed_text_cached, embed_texts_cached

# Cache file for prototype embeddings (per model)
_CACHE_PATH = os.path.join(".jinx", "tmp", "cont_proto.json")
_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Positive and negative semantic prototypes (language-agnostic intent)
_POS_TEXTS = [
    "a clarifying question that requests missing information",
    "a question asking for more details from the user",
    "a question that requires an answer",
    "request for clarification"
]
_NEG_TEXTS = [
    "final answer statement",
    "code snippet or program output",
    "log line or stack trace",
    "directive step or plan item"
]


async def _embed(text: str) -> List[float]:
    if not text:
        return []
    try:
        # Use cached/coalesced embedding call
        return await embed_text_cached(text, model=_MODEL)
    except Exception:
        return []


async def _embed_many(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    # Filter empties and preserve index mapping
    items = [(i, (t or "").strip()) for i, t in enumerate(texts)]
    if not any(t for _, t in items):
        return [[] for _ in texts]
    try:
        batch = [t for _, t in items]
        vecs = await embed_texts_cached(batch, model=_MODEL)
        out = [[] for _ in texts]
        for (i, _), v in zip(items, vecs):
            if i < len(out):
                out[i] = v
        return out
    except Exception:
        return [[] for _ in texts]


def _load_cache() -> Dict:
    try:
        if not os.path.exists(_CACHE_PATH):
            return {}
        with open(_CACHE_PATH, "r", encoding="utf-8") as r:
            return json.load(r)
    except Exception:
        return {}


def _save_cache(obj: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as w:
            json.dump(obj, w, ensure_ascii=False)
    except Exception:
        pass


async def _ensure_protos() -> Tuple[List[float], List[float]]:
    """Return (pos_mean, neg_mean) prototype vectors, computing and caching if needed."""
    cache = _load_cache()
    key = f"{_MODEL}::proto_v1"
    if key in cache:
        obj = cache.get(key) or {}
        pos = obj.get("pos") or []
        neg = obj.get("neg") or []
        if pos and neg:
            return pos, neg
    # Compute
    pos_vecs: List[List[float]] = []
    neg_vecs: List[List[float]] = []
    for t in _POS_TEXTS:
        pos_vecs.append(await _embed(t))
    for t in _NEG_TEXTS:
        neg_vecs.append(await _embed(t))
    def _mean(vs: List[List[float]]) -> List[float]:
        if not vs:
            return []
        n = max(len(vs[0]), 1)
        out = [0.0] * n
        cnt = 0
        for v in vs:
            if not v:
                continue
            if len(v) != n:
                continue
            cnt += 1
            for i in range(n):
                out[i] += float(v[i])
        if cnt:
            for i in range(n):
                out[i] /= cnt
        return out
    pos_mean = _mean(pos_vecs)
    neg_mean = _mean(neg_vecs)
    cache[key] = {"pos": pos_mean, "neg": neg_mean}
    _save_cache(cache)
    return pos_mean, neg_mean


def _cos(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    s = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        s += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    import math
    return s / (math.sqrt(na) * math.sqrt(nb))


async def score_question_semantics(text: str) -> float:
    """Return a scalar score: higher => more likely a question needing an answer."""
    pos, neg = await _ensure_protos()
    vec = await _embed((text or "").strip())
    if not vec:
        return 0.0
    return _cos(vec, pos) - _cos(vec, neg)


async def find_semantic_question(synth: str, *, max_lines: int = 120, threshold: float | None = None) -> str:
    """Scan recent transcript and return the most question-like candidate by semantics.

    - Uses embedding-based scoring against lightweight prototypes.
    - Returns empty string if nothing crosses threshold.
    """
    t = (synth or "").strip()
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    # Scan last N lines only for RT
    cand_lines = lines[-max_lines:]
    best = (0.0, "")
    thr = threshold
    if thr is None:
        try:
            thr = float(os.getenv("JINX_QSEM_THRESHOLD", "0.18"))
        except Exception:
            thr = 0.18
    # Batch-embed to keep latency low
    try:
        # Pre-filter obvious code/log lines but keep order
        pairs = []
        for ln in cand_lines:
            if any(tok in ln for tok in ("def ", "class ", "import ", "from ", "return ", "Traceback", "File ")):
                pairs.append((ln, None))
            else:
                pairs.append((ln, ln))
        texts = [p[1] or "" for p in pairs]
        vecs = await _embed_many(texts)
        # Compute scores pos-neg
        pos, neg = await _ensure_protos()
        scores: List[float] = []
        for v in vecs:
            if not v:
                scores.append(0.0)
            else:
                scores.append(_cos(v, pos) - _cos(v, neg))
        # Iterate latest-first to prefer recency
        for ln, sc in zip(reversed(cand_lines), reversed(scores)):
            if sc > best[0]:
                best = (sc, ln)
    except Exception:
        # Fallback: no semantic winner
        pass
    return best[1] if best[0] >= thr else ""
