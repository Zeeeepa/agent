from __future__ import annotations

import asyncio
import os
from typing import Any, List, Tuple

# Cross-encoder backends are optional and env-gated. Two modes:
# - sentence-transformers (local) if installed (fast, no external calls)
# - OpenAI scoring (costly, but available) if enabled via env
# Falls back to identity (no rerank) on any error.

_ST_MODEL = None  # lazy singleton
_ST_LOCK = asyncio.Lock()


def _ce_enabled() -> bool:
    try:
        return os.getenv("EMBED_PROJECT_CE_ENABLE", "0").strip().lower() in {"1", "true", "on", "yes"}
    except Exception:
        return False


def _backend() -> str:
    return (os.getenv("EMBED_PROJECT_CE_BACKEND", "auto").strip().lower() or "auto")


def _st_model_name() -> str:
    return os.getenv("EMBED_PROJECT_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


async def _score_openai(query: str, docs: List[str], *, timeout_ms: int, conc: int) -> List[float]:
    from jinx.micro.llm.openai_caller import call_openai

    sem = asyncio.Semaphore(max(1, conc))
    scores: List[float] = [0.0] * len(docs)

    async def _score_one(i: int, text: str) -> None:
        if not text:
            return
        try:
            from jinx.prompts import get_prompt as _get_prompt
            _tmpl = _get_prompt("cross_rerank")
            prompt = _tmpl.format(query=query[:512], candidate=text[:1200])
        except Exception:
            prompt = f"Query: {query[:512]}\nCandidate: {text[:1200]}\n"
        try:
            async with sem:
                if timeout_ms > 0:
                    out = await asyncio.wait_for(call_openai("single_score", os.getenv("JINX_RERANK_MODEL", "gpt-4o-mini"), prompt), timeout=timeout_ms/1000.0)
                else:
                    out = await call_openai("single_score", os.getenv("JINX_RERANK_MODEL", "gpt-4o-mini"), prompt)
            s = (out or "").strip()
            # Parse first float in the response
            import re
            m = re.search(r"([01]?(?:\.\d+)?)(?!\d)", s)
            val = float(m.group(1)) if m else 0.0
            # Clamp to [0,1]
            if val < 0.0:
                val = 0.0
            if val > 1.0:
                val = 1.0
            scores[i] = val
        except Exception:
            scores[i] = 0.0

    tasks = [asyncio.create_task(_score_one(i, d)) for i, d in enumerate(docs)]
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass
    return scores


async def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    async with _ST_LOCK:
        if _ST_MODEL is not None:
            return _ST_MODEL
        try:
            # Load in thread because it does IO and heavy init
            def _load():
                from sentence_transformers import CrossEncoder  # type: ignore
                return CrossEncoder(_st_model_name())
            _ST_MODEL = await asyncio.to_thread(_load)
        except Exception:
            _ST_MODEL = None
        return _ST_MODEL


async def _score_st(query: str, docs: List[str]) -> List[float]:
    model = await _get_st_model()
    if model is None:
        return [0.0 for _ in docs]
    try:
        pairs = [(query, d) for d in docs]
        def _predict():
            return model.predict(pairs)  # type: ignore[attr-defined]
        preds = await asyncio.to_thread(_predict)
        # Normalize to [0,1] if necessary
        out: List[float] = []
        try:
            import numpy as np  # type: ignore
            arr = np.array(preds, dtype=float)
            # Common CE models output is already [0,1] or logits; apply sigmoid if it looks logit-like
            if arr.min() < 0.0 or arr.max() > 1.0:
                out = (1.0 / (1.0 + (-arr)))  # type: ignore[operator]
                out = [float(x) for x in out]
            else:
                out = [float(x) for x in arr.tolist()]
        except Exception:
            out = [float(x) for x in preds]  # type: ignore[iteration-over-annotation]
        # Clamp
        out = [0.0 if x < 0.0 else (1.0 if x > 1.0 else x) for x in out]
        return out
    except Exception:
        return [0.0 for _ in docs]


async def cross_encoder_rerank(query: str, texts: List[str], *, max_time_ms: int | None, top_n: int) -> List[float]:
    """Return cross-encoder scores for given texts, or zeros if disabled/unavailable.

    The caller is responsible for combining with the original scores and sorting.
    """
    if not _ce_enabled() or not texts:
        return [0.0 for _ in texts]
    backend = _backend()
    # Trim to top_n
    texts = texts[: top_n]
    # Distribute a rough per-item time budget for OpenAI backend
    if backend in ("openai", "auto"):
        use_openai = True
        # Prefer local ST if available when backend=auto
        if backend == "auto":
            try:
                import importlib  # noqa: F401
                importlib.import_module("sentence_transformers")
                use_openai = False
            except Exception:
                use_openai = True
        if use_openai:
            try:
                conc = max(1, int(os.getenv("EMBED_PROJECT_CE_CONC", "3")))
            except Exception:
                conc = 3
            tout = int(os.getenv("EMBED_PROJECT_CE_TIMEOUT_MS", "120")) if max_time_ms is None else max(50, min(int(max_time_ms // max(1, conc)), int(os.getenv("EMBED_PROJECT_CE_TIMEOUT_MS", "120"))))
            return await _score_openai(query, texts, timeout_ms=tout, conc=conc)
    # Fall back to sentence-transformers
    return await _score_st(query, texts)
