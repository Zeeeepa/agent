from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

from jinx.micro.embeddings.embed_cache import embed_text_cached, embed_texts_cached

# Cache file for prototype embeddings (per model)
_CACHE_PATH = os.path.join(".jinx", "tmp", "cont_proto.json")
_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# Precompiled code-like patterns and special character set for fast checks
_CODE_PATTERNS = [
    re.compile(r'^\s*(def|class|import|from|return|if|for|while|try|except)\s+\w+', re.MULTILINE),
    re.compile(r'Traceback\s*\(most recent call last\)'),
    re.compile(r'File\s+"[^"]+"\s*,\s*line\s+\d+'),
    re.compile(r'^\s*[a-zA-Z_]\w*\s*=\s*.+$', re.MULTILINE),
    re.compile(r'^\s*\w+\([^)]*\)\s*$', re.MULTILINE),
    re.compile(r'\{[\s\S]*:[\s\S]*\}'),
    re.compile(r'</?\w+[^>]*>'),
]
# Build special character set safely (include both quote types, backslash, pipe, etc.)
_SPECIAL_CHARS = set("(){}[]<>=;,.:" + "'`|\\/")


@dataclass
class SemanticPrototype:
    """Configurable semantic prototype for classification."""
    label: str
    examples: List[str] = field(default_factory=list)
    weight: float = 1.0


class SemanticClassifier:
    """Advanced semantic classification with extensible prototypes."""
    
    def __init__(self):
        self._positive: SemanticPrototype = SemanticPrototype(
            label="question",
            examples=[
                "a clarifying question that requests missing information",
                "a question asking for more details from the user",
                "a question that requires an answer",
                "request for clarification",
                "asking what the user needs",
                "seeking user input or confirmation"
            ]
        )
        
        self._negative: SemanticPrototype = SemanticPrototype(
            label="non_question",
            examples=[
                "final answer statement",
                "code snippet or program output",
                "log line or stack trace",
                "directive step or plan item",
                "command execution result",
                "status update or progress report"
            ]
        )
    
    def get_positive_texts(self) -> List[str]:
        """Get positive prototype examples."""
        return self._positive.examples
    
    def get_negative_texts(self) -> List[str]:
        """Get negative prototype examples."""
        return self._negative.examples
    
    def add_positive_example(self, text: str) -> None:
        """Add custom positive example."""
        if text and text not in self._positive.examples:
            self._positive.examples.append(text)
    
    def add_negative_example(self, text: str) -> None:
        """Add custom negative example."""
        if text and text not in self._negative.examples:
            self._negative.examples.append(text)


# Global classifier instance
_classifier = SemanticClassifier()

# Legacy compatibility
_POS_TEXTS = _classifier.get_positive_texts()
_NEG_TEXTS = _classifier.get_negative_texts()


@lru_cache(maxsize=1024)
def _is_code_like(text: str) -> bool:
    """Detect code-like content using structural patterns with caching.
    
    Uses structure-based detection instead of keyword matching.
    """
    if not text or len(text) < 3:
        return False
    
    # Structural indicators of code (precompiled)
    for rx in _CODE_PATTERNS:
        if rx.search(text):
            return True
    
    # Statistical indicators: High ratio of special chars suggests code
    special_chars = sum(1 for c in text if c in _SPECIAL_CHARS)
    if len(text) > 10 and special_chars / len(text) > 0.3:
        return True
    
    return False


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
    non_empty = [(i, t) for i, t in items if t]
    if not non_empty:
        return [[] for _ in texts]
    try:
        batch = [t for _, t in non_empty]
        vecs = await embed_texts_cached(batch, model=_MODEL)
        out = [[] for _ in texts]
        for (i, _), v in zip(non_empty, vecs):
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
    key = f"{_MODEL}::proto_v2"  # Incremented version for new examples
    if key in cache:
        obj = cache.get(key) or {}
        pos = obj.get("pos") or []
        neg = obj.get("neg") or []
        if pos and neg:
            return pos, neg
    # Compute using current classifier examples (batch for RT efficiency)
    pos_texts = _classifier.get_positive_texts()
    neg_texts = _classifier.get_negative_texts()
    all_texts: List[str] = [t for t in (pos_texts + neg_texts) if t]
    try:
        vecs_all = await embed_texts_cached(all_texts, model=_MODEL)
    except Exception:
        vecs_all = [[] for _ in all_texts]
    # Ensure result length matches input length to keep splitting stable
    if len(vecs_all) < len(all_texts):
        vecs_all = vecs_all + ([[]] * (len(all_texts) - len(vecs_all)))
    elif len(vecs_all) > len(all_texts):
        vecs_all = vecs_all[:len(all_texts)]
    # Split back into pos/neg
    pN = len(pos_texts)
    pos_vecs: List[List[float]] = vecs_all[:pN]
    neg_vecs: List[List[float]] = vecs_all[pN:]

    def _mean(vs: List[List[float]]) -> List[float]:
        flat = [v for v in (vs or []) if v]
        if not flat:
            return []
        n = len(flat[0])
        out = [0.0] * n
        cnt = 0
        for v in flat:
            if len(v) != n:
                continue
            cnt += 1
            for i in range(n):
                out[i] += float(v[i])
        if cnt:
            out = [x / cnt for x in out]
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
        # Pre-filter obvious code/log lines using structural detection
        pairs = []
        for ln in cand_lines:
            if _is_code_like(ln):
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
