from __future__ import annotations

import os
import time
import asyncio
import re
from typing import Any, Dict, List, Tuple

# TTL caches for retrieval results
_PRJ_CACHE: Dict[str, Tuple[int, List[Tuple[float, str, Dict[str, Any]]]]] = {}
_PRJ_MULTI_CACHE: Dict[str, Tuple[int, List[Tuple[float, str, Dict[str, Any]]]]] = {}
try:
    _PRJ_TTL_MS = int(os.getenv("JINX_PROJ_RETR_TTL_MS", "800"))
except Exception:
    _PRJ_TTL_MS = 800

from .project_retrieval_config import (
    PROJ_DEFAULT_TOP_K,
    PROJ_EXHAUSTIVE_MODE,
    PROJ_NO_STAGE_BUDGETS,
    PROJ_STAGE_PYAST_MS,
    PROJ_STAGE_JEDI_MS,
    PROJ_STAGE_PYDOC_MS,
    PROJ_STAGE_REGEX_MS,
    PROJ_STAGE_PYFLOW_MS,
    PROJ_STAGE_LIBCST_MS,
    PROJ_STAGE_TB_MS,
    PROJ_STAGE_PYLITERALS_MS,
    PROJ_STAGE_LINEEXACT_MS,
    PROJ_STAGE_ASTMATCH_MS,
    PROJ_STAGE_RAPIDFUZZ_MS,
    PROJ_STAGE_TOKENMATCH_MS,
    PROJ_STAGE_PRE_MS,
    PROJ_STAGE_EXACT_MS,
    PROJ_STAGE_VECTOR_MS,
    PROJ_STAGE_KEYWORD_MS,
    PROJ_STAGE_LITERAL_MS,
    PROJ_LITERAL_BURST_MS,
    PROJ_STAGE_COOCCUR_MS,
    PROJ_STAGE_ASTCONTAINS_MS,
)
from .project_stage_exact import stage_exact_hits
from .project_stage_vector import stage_vector_hits
from .project_stage_keyword import stage_keyword_hits
from .project_stage_textscan import stage_textscan_hits
from .project_stage_jedi import stage_jedi_hits
from .project_stage_pyast import stage_pyast_hits
from .project_stage_pydoc import stage_pydoc_hits
from .project_stage_regex import stage_regex_hits
from .project_stage_pyflow import stage_pyflow_hits
from .project_stage_libcst import stage_libcst_hits
from .project_stage_traceback import stage_traceback_hits
from .project_stage_pyliterals import stage_pyliterals_hits
from .project_stage_lineexact import stage_lineexact_hits
from .project_stage_literal import stage_literal_hits
from .project_stage_cooccur import stage_cooccur_hits
from .project_stage_astmatch import stage_astmatch_hits
from .project_stage_astcontains import stage_astcontains_hits
from .project_stage_rapidfuzz import stage_rapidfuzz_hits
from .project_stage_tokenmatch import stage_tokenmatch_hits
from .project_stage_openbuffer import stage_openbuffer_hits
from .project_query_core import extract_code_core
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from .rerankers.cross_encoder import cross_encoder_rerank as _ce_rerank


async def retrieve_project_top_k(query: str, k: int | None = None, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    q = (query or "").strip()
    if not q:
        return []
    k_eff = k or PROJ_DEFAULT_TOP_K
    # TTL cache
    try:
        now_ms = int(time.time() * 1000)
    except Exception:
        now_ms = 0
    ck = f"{k_eff}|{q}"
    ent = _PRJ_CACHE.get(ck)
    if ent and (_PRJ_TTL_MS <= 0 or (now_ms - ent[0]) <= _PRJ_TTL_MS):
        return list(ent[1])[:k_eff]
    t0 = time.perf_counter()
    accumulate = bool(PROJ_EXHAUSTIVE_MODE)
    # Extract Python code-core from natural language query if present
    q_core = extract_code_core(q) or q

    # Accumulator for exhaustive mode
    collected: List[Tuple[float, str, Dict[str, Any]]] = []
    seen_keys: set[tuple] = set()

    def _key_of(hit: Tuple[float, str, Dict[str, Any]]) -> tuple:
        _score, _rel, _obj = hit
        m = (_obj.get("meta") or {})
        return (str(m.get("file_rel") or _rel), int(m.get("line_start") or 0), int(m.get("line_end") or 0))

    def _merge(hits: List[Tuple[float, str, Dict[str, Any]]] | None) -> None:
        if not hits:
            return
        for h in hits:
            kx = _key_of(h)
            if kx in seen_keys:
                continue
            seen_keys.add(kx)
            collected.append(h)

    # Helper closures to minimize repetition across stages
    def _time_left() -> int | None:
        # Enforce overall budget even in exhaustive mode; per-stage caps are skipped via _bounded
        if max_time_ms is None:
            return None
        rem = int(max(1, max_time_ms - (time.perf_counter() - t0) * 1000.0))
        return rem

    def _bounded(rem: int | None, cap_ms: int) -> int | None:
        if rem is None:
            return None
        # If configured, skip per-stage caps
        if PROJ_NO_STAGE_BUDGETS or PROJ_EXHAUSTIVE_MODE:
            return rem
        return max(1, min(rem, cap_ms))

    async def _maybe_ce_rerank(hits: List[Tuple[float, str, Dict[str, Any]]]) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Optionally apply cross-encoder reranking to top-N hits and combine scores.

        final = alpha * ce + (1 - alpha) * original
        """
        import os as _os
        try:
            gate = _os.getenv("EMBED_PROJECT_CE_ENABLE", "0").strip().lower() not in ("", "0", "false", "off", "no")
        except Exception:
            gate = False
        if not gate or not hits:
            return hits
        try:
            topn = max(1, int(_os.getenv("EMBED_PROJECT_CE_TOPN", "100")))
        except Exception:
            topn = 100
        try:
            alpha = float(_os.getenv("EMBED_PROJECT_CE_ALPHA", "0.7"))
        except Exception:
            alpha = 0.7
        # Build docs from preview text
        docs: List[str] = []
        idxs: List[int] = []
        for i, (_sc, _rel, obj) in enumerate(hits[:topn]):
            try:
                pv = (obj.get("meta", {}).get("text_preview") or "").strip()
            except Exception:
                pv = ""
            if not pv:
                continue
            docs.append(pv)
            idxs.append(i)
        if not docs:
            return hits
        # Respect remaining overall budget if any
        rem_ms = _time_left()
        try:
            scores = await _ce_rerank(q_core, docs, max_time_ms=rem_ms, top_n=len(docs))
        except Exception:
            return hits
        new_hits = list(hits)
        for pos, ce_sc in enumerate(scores):
            i = idxs[pos] if pos < len(idxs) else None
            if i is None or i >= len(new_hits):
                continue
            sc0, rel0, obj0 = new_hits[i]
            try:
                scn = float(alpha) * float(ce_sc or 0.0) + (1.0 - float(alpha)) * float(sc0 or 0.0)
            except Exception:
                scn = float(sc0 or 0.0)
            new_hits[i] = (scn, rel0, obj0)
        new_hits.sort(key=lambda h: float(h[0] or 0.0), reverse=True)
        return new_hits

    async def _run_sync_stage(stage_fn, query: str, k_arg: int, cap_ms: int):
        rem = _bounded(_time_left(), cap_ms)
        def _call():
            try:
                return stage_fn(query, k_arg, max_time_ms=rem)
            except Exception:
                return []
        hits = await asyncio.to_thread(_call)
        return (hits[:k_arg]) if hits else None

    async def _run_sync_stage_forced(stage_fn, query: str, k_arg: int, cap_ms: int):
        """Run a sync stage with its own cap, ignoring overall remaining time."""
        def _call():
            try:
                return stage_fn(query, k_arg, max_time_ms=cap_ms)
            except Exception:
                return []
        hits = await asyncio.to_thread(_call)
        return (hits[:k_arg]) if hits else None

    # Prefer embeddings by default: start vector similarity immediately (in parallel)
    try:
        rem_vec0 = _bounded(_time_left(), PROJ_STAGE_VECTOR_MS)
        vec_task: asyncio.Task[List[Tuple[float, str, Dict[str, Any]]]] = asyncio.create_task(stage_vector_hits(q_core, k_eff, max_time_ms=rem_vec0))
    except Exception:
        vec_task = asyncio.create_task(asyncio.sleep(0.0))  # type: ignore

    # Early precise stages: run concurrently when accumulating (exhaustive mode)
    if accumulate:
        codey_early = _is_code_like(q or "")
        cap_lineexact = _bounded(_time_left(), int(PROJ_STAGE_LINEEXACT_MS * (1.5 if codey_early else 1.0))) or PROJ_STAGE_LINEEXACT_MS
        cap_literal = _bounded(_time_left(), int(PROJ_STAGE_LITERAL_MS * (1.5 if codey_early else 1.0))) or PROJ_STAGE_LITERAL_MS
        tasks = [
            _run_sync_stage(stage_tokenmatch_hits, q_core, k_eff, PROJ_STAGE_TOKENMATCH_MS),
            _run_sync_stage(stage_lineexact_hits, q_core, k_eff, cap_lineexact),
            _run_sync_stage(stage_astmatch_hits, q_core, k_eff, PROJ_STAGE_ASTMATCH_MS),
            _run_sync_stage(stage_rapidfuzz_hits, q_core, k_eff, PROJ_STAGE_RAPIDFUZZ_MS),
            _run_sync_stage(stage_literal_hits, (q_core or q), k_eff, cap_literal),
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            results = []
        for hits in results:
            if isinstance(hits, list):
                _merge(hits)
        await asyncio.sleep(0)
    else:
        # Sequential short-circuit path when not accumulating
        tm_hits = await _run_sync_stage(stage_tokenmatch_hits, q_core, 1, PROJ_STAGE_TOKENMATCH_MS)
        if tm_hits:
            return tm_hits[:1]
        await asyncio.sleep(0)

        codey_seq = _is_code_like(q or "")
        cap_lineexact2 = _bounded(_time_left(), int(PROJ_STAGE_LINEEXACT_MS * (1.5 if codey_seq else 1.0))) or PROJ_STAGE_LINEEXACT_MS
        le_hits = await _run_sync_stage(stage_lineexact_hits, q_core, 1, cap_lineexact2)
        if le_hits:
            return le_hits[:1]
        await asyncio.sleep(0)

        # Literal exact/flex immediately after line-exact for code-like queries
        cap_literal2 = _bounded(_time_left(), int(PROJ_STAGE_LITERAL_MS * (1.5 if codey_seq else 1.0))) or PROJ_STAGE_LITERAL_MS
        lit_early = await _run_sync_stage(stage_literal_hits, (q_core or q), 1, cap_literal2)
        if lit_early:
            return lit_early[:1]
        await asyncio.sleep(0)

        # Open-buffer search to catch unsaved code in editors
        ob_hits = await _run_sync_stage(stage_openbuffer_hits, (q_core or q), 1, 140 if codey_seq else 100)
        if ob_hits:
            return ob_hits[:1]
        await asyncio.sleep(0)

        am_hits = await _run_sync_stage(stage_astmatch_hits, q_core, 1, PROJ_STAGE_ASTMATCH_MS)
        if am_hits:
            return am_hits[:1]
        await asyncio.sleep(0)

        # AST structural contains (e.g., isinstance(..., ast.Type))
        ac_hits = await _run_sync_stage(stage_astcontains_hits, q, 1, PROJ_STAGE_ASTCONTAINS_MS)
        if ac_hits:
            return ac_hits[:1]
        await asyncio.sleep(0)

        rf_hits = await _run_sync_stage(stage_rapidfuzz_hits, q_core, 1, PROJ_STAGE_RAPIDFUZZ_MS)
        if rf_hits:
            return rf_hits[:1]
        await asyncio.sleep(0)

        # Co-occurrence of multiple query tokens within short distance
        co_hits = await _run_sync_stage(stage_cooccur_hits, q, 1, PROJ_STAGE_COOCCUR_MS)
        if co_hits:
            return co_hits[:1]
        await asyncio.sleep(0)

        # Removed primitive fast substring and line-token stages to reduce overhead

    # Quick router: if query looks like an assignment/comprehension, try PyFlow early
    try:
        _qr = q_core
        qlow = _qr.lower()
        assign_like = bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*\((?s).*?\bfor\b", _qr))
        comp_like = (" for " in qlow and " in " in qlow) or any(sym in _qr for sym in [":=", "=>"])
    except Exception:
        assign_like = False
        comp_like = False
    if assign_like or comp_like:
        pf0 = await _run_sync_stage(stage_pyflow_hits, q_core, (k_eff if accumulate else 1), PROJ_STAGE_PYFLOW_MS)
        if pf0:
            if accumulate:
                _merge(pf0)
            else:
                return pf0[:1]
        await asyncio.sleep(0)

    # Await embeddings vector search and prefer its hits if present
    try:
        vec_hits0 = await vec_task
    except Exception:
        vec_hits0 = []
    if vec_hits0:
        if accumulate:
            _merge(vec_hits0)
        else:
            return vec_hits0[:k_eff]
    await asyncio.sleep(0)

    if accumulate:
        # Grouped concurrency under overall budget
        codey = _is_code_like(q or "")
        cap_pre = int(PROJ_STAGE_PRE_MS * 2) if codey else PROJ_STAGE_PRE_MS

        # Group A: traceback, pyast, pydoc, pyliterals
        try:
            res_a = await asyncio.gather(
                _run_sync_stage(stage_traceback_hits, q, k_eff, PROJ_STAGE_TB_MS),
                _run_sync_stage(stage_pyast_hits, q, k_eff, PROJ_STAGE_PYAST_MS),
                _run_sync_stage(stage_pydoc_hits, q, k_eff, PROJ_STAGE_PYDOC_MS),
                _run_sync_stage(stage_pyliterals_hits, q, k_eff, PROJ_STAGE_PYLITERALS_MS),
                return_exceptions=True,
            )
        except Exception:
            res_a = []
        for hits in res_a:
            if isinstance(hits, list):
                _merge(hits)
        await asyncio.sleep(0)

        # Group B: pyflow, libcst, jedi, regex, ast-contains
        try:
            res_b = await asyncio.gather(
                _run_sync_stage(stage_pyflow_hits, q, k_eff, PROJ_STAGE_PYFLOW_MS),
                _run_sync_stage(stage_libcst_hits, q, k_eff, PROJ_STAGE_LIBCST_MS),
                _run_sync_stage(stage_jedi_hits, q, k_eff, PROJ_STAGE_JEDI_MS),
                _run_sync_stage(stage_regex_hits, q, k_eff, PROJ_STAGE_REGEX_MS),
                _run_sync_stage(stage_astcontains_hits, q, k_eff, PROJ_STAGE_ASTCONTAINS_MS),
                return_exceptions=True,
            )
        except Exception:
            res_b = []
        for hits in res_b:
            if isinstance(hits, list):
                _merge(hits)
        await asyncio.sleep(0)

        # Group C: text pre-scan, exact, literal, co-occurrence, open-buffer
        try:
            res_c = await asyncio.gather(
                _run_sync_stage(stage_textscan_hits, q, k_eff, cap_pre),
                _run_sync_stage(stage_exact_hits, q, k_eff, PROJ_STAGE_EXACT_MS),
                _run_sync_stage(stage_literal_hits, (q_core or q), k_eff, PROJ_STAGE_LITERAL_MS),
                _run_sync_stage(stage_cooccur_hits, q, k_eff, PROJ_STAGE_COOCCUR_MS),
                _run_sync_stage(stage_openbuffer_hits, (q_core or q), k_eff, 140),
                return_exceptions=True,
            )
        except Exception:
            res_c = []
        for hits in res_c:
            if isinstance(hits, list):
                _merge(hits)
        await asyncio.sleep(0)

        # Final keyword stage
        kw = (await _run_sync_stage(stage_keyword_hits, q, k_eff, PROJ_STAGE_KEYWORD_MS)) or []
        _merge(kw)
        # Literal burst if still empty (give a bit more time once)
        if not collected:
            try:
                burst_ms = max(50, int(PROJ_LITERAL_BURST_MS))
            except Exception:
                burst_ms = 800
            lit_res = await _run_sync_stage_forced(stage_literal_hits, q, k_eff, burst_ms)
            if isinstance(lit_res, list):
                _merge(lit_res)

        # Return deduped, score-sorted
        out_hits = sorted(collected, key=lambda h: float(h[0] or 0.0), reverse=True)[:k_eff]
        # Optional cross-encoder reranking of the final shortlist
        try:
            out_hits = await _maybe_ce_rerank(out_hits)
        except Exception:
            pass
        try:
            _PRJ_CACHE[ck] = (now_ms, list(out_hits))
        except Exception:
            pass
        return out_hits
    else:
        # Stage -3: traceback
        tb_hits = await _run_sync_stage(stage_traceback_hits, q, k_eff, PROJ_STAGE_TB_MS)
        if tb_hits:
            return tb_hits
        await asyncio.sleep(0)

        # Stage -2: pyast
        ast_hits = await _run_sync_stage(stage_pyast_hits, q, k_eff, PROJ_STAGE_PYAST_MS)
        if ast_hits:
            return ast_hits
        await asyncio.sleep(0)

        # Stage -1.8: pydoc
        pydoc_hits = await _run_sync_stage(stage_pydoc_hits, q, k_eff, PROJ_STAGE_PYDOC_MS)
        if pydoc_hits:
            return pydoc_hits
        await asyncio.sleep(0)

        # Stage -1.75: pyliterals
        pl_hits = await _run_sync_stage(stage_pyliterals_hits, q, k_eff, PROJ_STAGE_PYLITERALS_MS)
        if pl_hits:
            return pl_hits
        await asyncio.sleep(0)

        # Stage -1.7: pyflow
        pyflow_hits = await _run_sync_stage(stage_pyflow_hits, q, k_eff, PROJ_STAGE_PYFLOW_MS)
        if pyflow_hits:
            return pyflow_hits
        await asyncio.sleep(0)

        # Stage -1.6: libcst
        cst_hits = await _run_sync_stage(stage_libcst_hits, q, k_eff, PROJ_STAGE_LIBCST_MS)
        if cst_hits:
            return cst_hits
        await asyncio.sleep(0)

        # Stage -1.5: jedi
        jedi_hits = await _run_sync_stage(stage_jedi_hits, q, k_eff, PROJ_STAGE_JEDI_MS)
        if jedi_hits:
            return jedi_hits
        await asyncio.sleep(0)

        # Stage -1.4: regex
        rx_hits = await _run_sync_stage(stage_regex_hits, q, k_eff, PROJ_STAGE_REGEX_MS)
        if rx_hits:
            return rx_hits
        await asyncio.sleep(0)

        # Stage -1: textscan
        codey = _is_code_like(q or "")
        cap_pre = int(PROJ_STAGE_PRE_MS * 2) if codey else PROJ_STAGE_PRE_MS
        txt_hits = await _run_sync_stage(stage_textscan_hits, q, k_eff, cap_pre)
        if txt_hits:
            return txt_hits
        await asyncio.sleep(0)

        # Stage 0: exact
        exact = await _run_sync_stage(stage_exact_hits, q, k_eff, PROJ_STAGE_EXACT_MS)
        if exact:
            return exact
        await asyncio.sleep(0)

        # Stage 2: keyword
        kw = (await _run_sync_stage(stage_keyword_hits, q, k_eff, PROJ_STAGE_KEYWORD_MS)) or []
        out_hits = kw[:k_eff]
        # Final literal pass
        if not out_hits:
            lit_hits = await _run_sync_stage(stage_literal_hits, (q_core or q), k_eff, PROJ_STAGE_LITERAL_MS)
            if lit_hits:
                out_hits = lit_hits
        try:
            _PRJ_CACHE[ck] = (now_ms, list(out_hits))
        except Exception:
            pass
        return out_hits


async def retrieve_project_multi_top_k(queries: List[str], *, per_query_k: int, max_time_ms: int | None = 300) -> List[Tuple[float, str, Dict[str, Any]]]:
    qs = [q.strip() for q in (queries or []) if (q or "").strip()]
    if not qs:
        return []
    # TTL cache
    try:
        now_ms = int(time.time() * 1000)
    except Exception:
        now_ms = 0
    ck = f"{per_query_k}|{' || '.join(qs)}"
    ent = _PRJ_MULTI_CACHE.get(ck)
    if ent and (_PRJ_TTL_MS <= 0 or (now_ms - ent[0]) <= _PRJ_TTL_MS):
        return list(ent[1])
    # Conservative per-query budget
    if max_time_ms is None:
        per_budget = None
    else:
        per_budget = max(50, int(max_time_ms // max(1, len(qs))))
    sem = asyncio.Semaphore(3)
    results: List[Tuple[float, str, Dict[str, Any]]] = []

    async def _run_one(q: str) -> None:
        async with sem:
            try:
                hits = await retrieve_project_top_k(q, k=per_query_k, max_time_ms=per_budget)
            except Exception:
                hits = []
            if hits:
                results.extend(hits)

    await asyncio.gather(*[asyncio.create_task(_run_one(q)) for q in qs])
    # Dedupe by (file_rel, ls, le)
    seen: set[tuple] = set()
    merged: List[Tuple[float, str, Dict[str, Any]]] = []
    for sc, rel, obj in sorted(results, key=lambda h: float(h[0] or 0.0), reverse=True):
        m = (obj.get("meta") or {})
        key = (str(m.get("file_rel") or rel), int(m.get("line_start") or 0), int(m.get("line_end") or 0))
        if key in seen:
            continue
        seen.add(key)
        merged.append((sc, rel, obj))
    out = merged[: (per_query_k * len(qs))]
    try:
        _PRJ_MULTI_CACHE[ck] = (now_ms, list(out))
    except Exception:
        pass
    return out


__all__ = [
    "retrieve_project_top_k",
    "retrieve_project_multi_top_k",
]
