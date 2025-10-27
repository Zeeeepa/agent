from __future__ import annotations

import asyncio
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from jinx.micro.embeddings.retrieval import retrieve_top_k as _retrieve_top_k
from jinx.micro.memory.search import rank_memory as _rank_memory
from jinx.micro.memory.optimizer import submit as _opt_submit
from jinx.micro.memory.storage import memory_dir as _memory_dir
from jinx.micro.memory.indexer import ingest_memory as _ingest_memory
from jinx.micro.embeddings.rerankers.cross_encoder import cross_encoder_rerank as _ce_rerank
from jinx.micro.metrics.retrieval_eval import evaluate_dataset as _eval_ds
from jinx.micro.memory.graph_reasoner import activate as _graph_activate


class MemoryService:
    """Memory Service (micro-module).

    Contract (initial skeleton):
    - ingest(chunks, metadata) -> int
    - query(query_text, k, policy) -> List[Dict]
    - compact(context_window, policy) -> bool
    - snapshot(version_name) -> str (path)
    - evaluate(dataset, k) -> Dict[str, float]
    """

    async def ingest(self, chunks: List[Dict[str, Any]] | List[str], metadata: Optional[Dict[str, Any]] = None) -> int:
        """Ingest memory using the advanced ingestion pipeline (rank + dedup + TTL).

        Preferred path: delegate to `jinx.micro.memory.indexer.ingest_memory`, which already performs
        candidate ranking (`ingest_ranker`) and deduplication (`ingest_dedup`) under time and conc limits.
        """
        meta = metadata or {}
        # Support explicit compact/evergreen payloads
        compact = str(meta.get("compact") or "").strip()
        evergreen = str(meta.get("evergreen") or "").strip() if meta.get("evergreen") is not None else None
        if not compact and chunks:
            # Build a temporary compact text from provided items
            lines: List[str] = []
            seen: set[str] = set()
            for it in chunks:
                s = it if isinstance(it, str) else str((it or {}).get("text") or "")
                s = (s or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                lines.append(s)
            compact = "\n".join(lines)
        # Delegate (throttled and deduped inside)
        await _ingest_memory(compact or None, evergreen)
        # Return a best-effort count (lines fed)
        try:
            return len((compact or "").splitlines())
        except Exception:
            return 0

    async def query(self, query_text: str, *, k: int = 8, policy: str = "any", preview_chars: int = 200) -> List[Dict[str, Any]]:
        """Hybrid memory retrieval: ranker + vector + optional cross-encoder rerank.

        Returns a list of dicts: {text, source, scores:{vec, ce, base, final}, meta:{...}}.
        """
        q = (query_text or "").strip()
        if not q:
            return []
        # Allocation between ranker and vector results
        try:
            frac_vec = float(os.getenv("JINX_MEM_Q_VEC_FRAC", "0.5"))
        except Exception:
            frac_vec = 0.5
        k = max(1, int(k))
        k_vec = max(1, int(round(k * max(0.0, min(1.0, frac_vec)))))
        k_rank = max(1, k - k_vec)

        # Run in parallel
        vec_task = asyncio.create_task(_retrieve_top_k(q, k=k_vec, max_time_ms=int(os.getenv("JINX_MEM_Q_VEC_MS", "200"))))
        rank_task = asyncio.create_task(_rank_memory(q, scope=policy, k=k_rank, preview_chars=preview_chars))
        vec_hits, rank_lines = await asyncio.gather(vec_task, rank_task, return_exceptions=True)
        if isinstance(vec_hits, Exception):
            vec_hits = []
        if isinstance(rank_lines, Exception):
            rank_lines = []

        # Normalize candidates
        cands: List[Dict[str, Any]] = []
        seen_txt: set[str] = set()
        # Vector candidates
        for sc, src, obj in (vec_hits or []):
            meta = (obj or {}).get("meta", {})
            pv = (meta.get("text_preview") or "").strip()
            if not pv or pv in seen_txt:
                continue
            seen_txt.add(pv)
            cands.append({
                "text": pv[:preview_chars],
                "source": str(meta.get("source") or src or "vector"),
                "scores": {"vec": float(sc or 0.0), "ce": 0.0, "base": 0.15},
                "meta": {**meta},
            })
            if len(cands) >= k:
                break
        # Ranker candidates
        for ln in (rank_lines or []):
            if not ln or ln in seen_txt:
                continue
            seen_txt.add(ln)
            cands.append({
                "text": ln[:preview_chars],
                "source": f"rank:{policy}",
                "scores": {"vec": 0.0, "ce": 0.0, "base": 0.35},
                "meta": {"scope": policy},
            })
            if len(cands) >= k * 3:  # keep a bounded pool before CE
                break

        # Optional cross-encoder reranking
        try:
            ce_on = os.getenv("JINX_MEM_Q_CE_ENABLE", "1").strip().lower() not in ("", "0", "false", "off", "no")
        except Exception:
            ce_on = True
        if ce_on and cands:
            try:
                topn = max(1, int(os.getenv("JINX_MEM_Q_CE_TOPN", "100")))
            except Exception:
                topn = 100
            docs = [c["text"] for c in cands[:topn]]
            try:
                ce_scores = await _ce_rerank(q, docs, max_time_ms=int(os.getenv("JINX_MEM_Q_CE_MS", "200")), top_n=len(docs))
            except Exception:
                ce_scores = [0.0 for _ in docs]
            for i, s in enumerate(ce_scores):
                try:
                    cands[i]["scores"]["ce"] = float(s or 0.0)
                except Exception:
                    continue

        # Graph boost (activates nodes/keys and boosts candidates containing them)
        try:
            graph_on = os.getenv("JINX_MEM_Q_GRAPH", "1").strip().lower() not in ("", "0", "false", "off", "no")
        except Exception:
            graph_on = True
        g_pairs: List[Tuple[str, float]] = []
        if graph_on:
            try:
                gk = max(1, int(os.getenv("JINX_MEM_Q_GRAPH_K", "10")))
            except Exception:
                gk = 10
            try:
                g_pairs = await _graph_activate(q, k=gk, steps=2)
            except Exception:
                g_pairs = []

        # Combine scores
        try:
            alpha = float(os.getenv("JINX_MEM_Q_CE_ALPHA", "0.6"))
        except Exception:
            alpha = 0.6
        try:
            beta = float(os.getenv("JINX_MEM_Q_VEC_ALPHA", "0.3"))
        except Exception:
            beta = 0.3
        try:
            gamma = float(os.getenv("JINX_MEM_Q_REC_ALPHA", "0.15"))
        except Exception:
            gamma = 0.15
        try:
            delta = float(os.getenv("JINX_MEM_Q_GRAPH_ALPHA", "0.2"))
        except Exception:
            delta = 0.2
        try:
            rec_window = float(os.getenv("JINX_MEM_Q_REC_WINDOW_SEC", "86400"))  # 1 day
        except Exception:
            rec_window = 86400.0
        now = time.time()
        keys_lower = [(str(k or "").strip().lower(), float(sc or 0.0)) for k, sc in (g_pairs or []) if (k or "").strip()]
        for c in cands:
            sc = c.get("scores", {})
            ce = float(sc.get("ce", 0.0) or 0.0)
            vec = float(sc.get("vec", 0.0) or 0.0)
            base = float(sc.get("base", 0.0) or 0.0)
            # Recency (only for vector candidates with ts)
            rec = 0.0
            try:
                ts = float((c.get("meta", {}) or {}).get("ts") or 0.0)
                if rec_window > 0 and ts > 0:
                    age = max(0.0, now - ts)
                    rec = max(0.0, 1.0 - (age / rec_window))
            except Exception:
                rec = 0.0
            # Graph boost: if any activated key appears in text
            gboost = 0.0
            if keys_lower:
                tlow = (c.get("text", "") or "").lower()
                try:
                    for klow, ksc in keys_lower:
                        if klow and klow in tlow:
                            gboost += float(ksc)
                except Exception:
                    pass
                # normalize roughly
                if gboost > 0.0:
                    try:
                        gboost = min(1.0, gboost / max(1.0, sum(x[1] for x in keys_lower)))
                    except Exception:
                        gboost = min(1.0, gboost)

            final = alpha * ce + beta * vec + gamma * rec + delta * gboost + max(0.0, 1.0 - (alpha + beta + gamma + delta)) * base
            sc["final"] = final
            # Explain breakdown
            c["explain"] = {
                "alpha_ce": alpha * ce,
                "beta_vec": beta * vec,
                "gamma_recency": gamma * rec,
                "delta_graph": delta * gboost,
                "base": max(0.0, 1.0 - (alpha + beta + gamma + delta)) * base,
            }
        cands.sort(key=lambda x: float(x.get("scores", {}).get("final", 0.0) or 0.0), reverse=True)
        return cands[:k]

    async def feedback(self, query_text: str, chosen: List[str], *, success: bool) -> bool:
        """Record retrieval feedback for online learning (best-effort append-only log)."""
        base = _memory_dir()
        path = os.path.join(base, "feedback.jsonl")
        rec = {
            "ts": int(time.time() * 1000),
            "query": (query_text or "").strip(),
            "chosen": list(chosen or []),
            "success": bool(success),
        }
        try:
            os.makedirs(base, exist_ok=True)
            line = __import__("json").dumps(rec)
            def _append():
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            await asyncio.to_thread(_append)
            return True
        except Exception:
            return False

    async def compact(self, context_window: Optional[int] = None, policy: Optional[str] = None) -> bool:
        """Trigger memory optimization (compaction)."""
        try:
            await _opt_submit(None)
            return True
        except Exception:
            return False

    async def snapshot(self, version_name: str) -> str:
        """Create a filesystem snapshot of the memory directory under snapshots/<version_name>."""
        base = _memory_dir()
        snap_dir = os.path.join(base, "snapshots")
        out_dir = os.path.join(snap_dir, version_name or "unnamed")
        os.makedirs(snap_dir, exist_ok=True)
        # Copytree best-effort; replace if exists
        def _copytree(src: str, dst: str) -> None:
            if os.path.exists(dst):
                try:
                    shutil.rmtree(dst)
                except Exception:
                    pass
            shutil.copytree(src, dst, dirs_exist_ok=False)
        try:
            await asyncio.to_thread(_copytree, base, out_dir)
        except Exception:
            # Return even on failure; callers can inspect
            pass
        return out_dir

    async def evaluate(self, dataset: List[Dict[str, Any]], *, k: int = 5) -> Dict[str, float]:
        """Evaluate retrieval on a dataset of {query, ground_truth: [strings]}.

        Returns: {precision@k, recall@k, mrr, latency_ms}
        """
        async def _runner(q: str, kx: int) -> List[str]:
            res = await self.query(q, k=kx)
            return [r.get("text", "") for r in res]
        return await _eval_ds(dataset, k=k, query_runner=_runner)


async def start_memory_service_task() -> asyncio.Task[None]:
    svc = MemoryService()

    async def _noop_forever() -> None:
        while True:
            await asyncio.sleep(3600)

    return asyncio.create_task(_noop_forever(), name="memory-service")
