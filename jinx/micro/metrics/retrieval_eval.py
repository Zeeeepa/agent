from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Dict, List, Tuple

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None  # type: ignore


def _norm(s: str) -> str:
    t = (s or "").strip().lower()
    # collapse whitespace
    t = " ".join(t.split())
    # cheap punctuation trim
    for ch in "`'\"()[]{}<>;:,":
        t = t.replace(ch, " ")
    t = " ".join(t.split())
    return t


def _best_match(pred: str, gts: List[str]) -> float:
    if not pred or not gts:
        return 0.0
    p = _norm(pred)
    best = 0.0
    if fuzz is None:
        # fallback: substring or token overlap heuristic
        for g in gts:
            g2 = _norm(g)
            if not g2:
                continue
            if p in g2 or g2 in p:
                return 1.0
            # token overlap
            ps = set(p.split())
            gs = set(g2.split())
            if ps and gs:
                inter = len(ps & gs)
                union = len(ps | gs)
                sim = inter / union if union > 0 else 0.0
                best = max(best, sim)
        return float(best)
    # rapidfuzz ratio in [0, 100]
    for g in gts:
        g2 = _norm(g)
        if not g2:
            continue
        try:
            r = float(fuzz.token_set_ratio(p, g2)) / 100.0
        except Exception:
            r = 0.0
        if r > best:
            best = r
    return float(best)


def _precision_recall_mrr(all_preds: List[List[str]], all_gts: List[List[str]], k: int) -> Dict[str, float]:
    eps = 1e-9
    total = max(1, len(all_preds))
    p_at_k = 0.0
    r_at_k = 0.0
    mrr = 0.0
    for preds, gts in zip(all_preds, all_gts):
        if not gts:
            continue
        # Precision@k: fraction of top-k that match
        hits = 0
        first_rank = None
        for i, pred in enumerate(preds[:k]):
            sim = _best_match(pred, gts)
            if sim >= 0.8:  # high-confidence match
                hits += 1
                if first_rank is None:
                    first_rank = i + 1
        p_at_k += (hits / max(1, min(k, len(preds))))
        r_at_k += (hits / max(1, len(gts)))
        if first_rank is not None:
            mrr += 1.0 / float(first_rank)
    return {
        "precision@k": p_at_k / total,
        "recall@k": r_at_k / total,
        "mrr": mrr / total,
    }


async def evaluate_dataset(dataset: List[Dict[str, Any]], *, k: int, query_runner: Callable[[str, int], asyncio.Future]) -> Dict[str, float]:
    """Evaluate a retrieval runner on a small dataset.

    dataset: list of {query: str, ground_truth: list[str]}
    query_runner: async function(query, k) -> list[str]
    Returns metrics: precision@k, recall@k, mrr, latency_ms
    """
    if not dataset:
        return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0, "latency_ms": 0.0}

    try:
        conc = max(1, int(os.getenv("JINX_EVAL_CONC", "4")))
    except Exception:
        conc = 4
    sem = asyncio.Semaphore(conc)

    preds: List[List[str]] = [None] * len(dataset)  # type: ignore[assignment]
    truths: List[List[str]] = [None] * len(dataset)  # type: ignore[assignment]
    lats: List[float] = [0.0] * len(dataset)

    async def _one(idx: int, q: str, gts: List[str]) -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await query_runner(q, k)
            except Exception:
                out = []
            lats[idx] = (time.perf_counter() - t0) * 1000.0
            preds[idx] = list(out or [])
            truths[idx] = list(gts or [])

    tasks = []
    for i, item in enumerate(dataset):
        q = str(item.get("query") or "").strip()
        gts = [str(x) for x in (item.get("ground_truth") or [])]
        tasks.append(asyncio.create_task(_one(i, q, gts)))
    await asyncio.gather(*tasks, return_exceptions=True)

    stats = _precision_recall_mrr(preds, truths, k)
    try:
        lat = sum(lats) / max(1, len(lats))
    except Exception:
        lat = 0.0
    stats["latency_ms"] = lat
    return stats
