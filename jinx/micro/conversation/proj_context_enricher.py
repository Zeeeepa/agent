from __future__ import annotations

import asyncio
import os
import time
from typing import List, Dict, Tuple

from jinx.micro.memory.router import assemble_memroute as _memroute
from jinx.micro.embeddings.project_retrieval import (
    build_project_context_for as _build_proj_ctx_single,
    build_project_context_multi_for as _build_proj_ctx_multi,
)
from jinx.micro.memory.storage import read_channel as _read_channel
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.embeddings.query_subqueries import build_codecentric_subqueries as _build_code_subqs
from jinx.micro.memory.evergreen_hints import build_evergreen_hints as _build_evg_hints


async def build_project_context_enriched(query: str, user_text: str = "", synth: str = "") -> str:
    """ML-driven project context builder with intelligent subquery generation.

    Advanced features:
    - Query intent classification for adaptive strategy
    - ML-based subquery scoring and selection
    - Adaptive k/timeout/budget from brain systems
    - Multi-source fusion with learned weights
    - Outcome tracking for continuous improvement
    """
    t0 = time.time()
    _q = (query or "").strip()
    if not _q:
        return ""
    
    # Step 1: Search memories for relevant context
    memory_hints = []
    try:
        from jinx.micro.brain import search_all_memories
        
        # Search all memory systems
        memories = await search_all_memories(_q, k=5)
        
        # Extract hints from memories
        for mem in memories:
            if mem.importance > 0.6:
                memory_hints.append(mem.content[:100])
    except Exception:
        pass
    
    # Step 2: Expand query with memory context
    expanded_query = None
    try:
        from jinx.micro.brain import expand_query
        
        # Build context with memories
        expansion_context = {
            'memories': memory_hints,
            'memory_count': len(memory_hints)
        }
        
        expanded_query = await expand_query(_q, expansion_context)
        
        # Use expanded query if confident
        if expanded_query and expanded_query.confidence > 0.5:
            _q_enriched = expanded_query.expanded
        else:
            _q_enriched = _q
    except Exception:
        _q_enriched = _q
    
    # Step 2: Use FULL INTELLIGENCE pipeline from orchestrator
    intelligence_results = None
    query_intent = None
    try:
        from jinx.micro.brain import process_with_full_intelligence
        # This runs through ALL 20 brain systems automatically!
        intelligence_results = await process_with_full_intelligence(_q_enriched)
        
        # Extract results
        if intelligence_results:
            # Already have route, intent, k, timeout, budget, plan!
            route_name = intelligence_results.get('route', 'explain')
            intent_name = intelligence_results.get('intent', 'code_exec')
            
            # Create intent object
            from jinx.micro.brain.query_classifier import QueryIntent
            query_intent = QueryIntent(
                intent=intent_name,
                confidence=intelligence_results.get('intent_confidence', 0.7),
                sub_intents=[],
                features={}
            )
    except Exception:
        # Fallback to individual systems
        try:
            from jinx.micro.brain import classify_query
            query_intent = await classify_query(_q_enriched)
        except Exception:
            pass
    # Routed memory hints
    try:
        mem_hints: List[str] = await _memroute(_q, k=8, preview_chars=120)
    except Exception:
        mem_hints = []
    # Caps and dynamics
    try:
        max_q_chars = int(os.getenv("JINX_PROJ_QUERY_MAX_CHARS", "800"))
    except Exception:
        max_q_chars = 800
    _q_proj = (" ".join([_q] + mem_hints)).strip()
    if len(_q_proj) > max_q_chars:
        _q_proj = _q_proj[:max_q_chars]
    codey = _is_code_like(_q_proj or "")
    first_budget = 1200 if codey else None
    # Get adaptive parameters - prioritize from intelligence_results
    proj_k, first_budget = None, None
    
    if intelligence_results:
        # Use results from full intelligence pipeline
        proj_k = intelligence_results.get('k')
        first_budget = intelligence_results.get('timeout')
        
        # Use context budget if available
        budget = intelligence_results.get('context_budget')
        if budget:
            proj_k = min(30, max(5, int(budget / 400)))
    
    # Fallback to individual system calls if not available
    if proj_k is None or first_budget is None:
        try:
            from jinx.micro.brain import select_retrieval_params, allocate_context_budget
            
            # Adaptive k and timeout
            if proj_k is None:
                proj_k, timeout_adaptive = await select_retrieval_params(_q_proj)
            else:
                _, timeout_adaptive = await select_retrieval_params(_q_proj)
            
            if first_budget is None:
                first_budget = timeout_adaptive if timeout_adaptive else (1200 if codey else 800)
            
            # Adaptive context budget allocation
            allocation = await allocate_context_budget(_q_proj)
            if allocation and allocation.code_budget and proj_k:
                # Use ML-optimized budget
                proj_k = min(30, max(5, int(allocation.code_budget / 400)))  # ~400 chars per hit
        except Exception:
            pass
    
    # Fallback to heuristics
    if proj_k is None:
        try:
            proj_k = int(os.getenv("JINX_PROJ_CTX_K", ("10" if codey else "6")))
        except Exception:
            proj_k = 10 if codey else 6
    
    if first_budget is None:
        first_budget = 1200 if codey else 800
    try:
        subq_cap = int(os.getenv("JINX_PROJ_SUBQ_MAX", "3"))
    except Exception:
        subq_cap = 3
    subqs: List[str] = [_q_proj]
    # 0) optional evergreen-derived hints (tokens/paths/symbols) — not sent to LLM directly
    try:
        evg_on = str(os.getenv("JINX_EVG_HINTS_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        evg_on = True
    if evg_on and subq_cap > 0:
        try:
            evg = await _build_evg_hints(_q_proj)
        except Exception:
            evg = {"tokens": [], "paths": [], "symbols": [], "prefs": [], "decisions": []}
        # tokens phrase first
        try:
            max_tok = max(1, int(os.getenv("JINX_EVG_HINT_TOKS", "8")))
        except Exception:
            max_tok = 8
        tok_phrase = " ".join((evg.get("tokens") or [])[:max_tok]).strip()
        if tok_phrase and tok_phrase not in subqs:
            subqs.append(tok_phrase)
        # a couple of path/symbol-based subqs if space remains
        add_more = max(0, subq_cap - max(0, len(subqs) - 1))
        for ln in (evg.get("paths") or [])[:2]:
            if add_more <= 0:
                break
            p = (ln or "").strip()
            if not p:
                continue
            qh = f"{_q} {p}".strip()
            if len(qh) > max_q_chars:
                qh = qh[:max_q_chars]
            if qh not in subqs:
                subqs.append(qh)
                add_more -= 1
        for ln in (evg.get("symbols") or [])[:2]:
            if add_more <= 0:
                break
            s = (ln or "").strip()
            if not s:
                continue
            qh = f"{_q} {s}".strip()
            if len(qh) > max_q_chars:
                qh = qh[:max_q_chars]
            if qh not in subqs:
                subqs.append(qh)
                add_more -= 1
    # 1) task + top memory hints
    for h in (mem_hints or [])[: max(0, subq_cap)]:
        qh = f"{_q} {h}".strip()
        if len(qh) > max_q_chars:
            qh = qh[:max_q_chars]
        if qh not in subqs:
            subqs.append(qh)
    # Insert code-centric sub-queries early (delegated to micro-module)
    try:
        for s in _build_code_subqs(_q_proj or ""):
            if s and s not in subqs:
                subqs.append(s)
    except Exception:
        pass

    # 2) task + top channels (paths/symbols)
    try:
        ch_paths = (await _read_channel("paths") or "").splitlines()
    except Exception:
        ch_paths = []
    try:
        ch_syms = (await _read_channel("symbols") or "").splitlines()
    except Exception:
        ch_syms = []

    def _after(prefix: str, ln: str) -> str:
        low = (ln or "").lower()
        return ln[len(prefix) :].strip() if low.startswith(prefix) else ""

    add_more = max(0, subq_cap - max(0, len(subqs) - 1))
    for ln in ch_paths[:2]:
        if add_more <= 0:
            break
        p = _after("path: ", ln)
        if not p:
            continue
        qh = f"{_q} {p}".strip()
        if len(qh) > max_q_chars:
            qh = qh[:max_q_chars]
        if qh not in subqs:
            subqs.append(qh)
            add_more -= 1
    for ln in ch_syms[:2]:
        if add_more <= 0:
            break
        s = _after("symbol: ", ln)
        if not s:
            continue
        qh = f"{_q} {s}".strip()
        if len(qh) > max_q_chars:
            qh = qh[:max_q_chars]
        if qh not in subqs:
            subqs.append(qh)
            add_more -= 1

    # Intelligent subquery scoring and selection
    scored_subqs = await _score_subqueries(subqs, _q, query_intent)
    
    # Select top subqueries based on ML scores
    top_subqs = [sq for sq, score in scored_subqs[:min(5, len(scored_subqs))]]
    
    # Build context with adaptive strategy
    ctx = ""
    success = False
    try:
        if len(top_subqs) > 1:
            ctx = await _build_proj_ctx_multi(top_subqs, k=proj_k, max_time_ms=first_budget)
        else:
            ctx = await _build_proj_ctx_single(_q_proj, k=proj_k, max_time_ms=first_budget)
        success = len(ctx) > 100
    except Exception:
        ctx = ""
    
    # Fallback with larger budget if needed
    if not success:
        try:
            fallback_budget = min(3000, first_budget * 2)
            if len(top_subqs) > 1:
                ctx = await _build_proj_ctx_multi(top_subqs, k=proj_k, max_time_ms=fallback_budget)
            else:
                ctx = await _build_proj_ctx_single(_q_proj, k=proj_k, max_time_ms=fallback_budget)
            success = len(ctx) > 100
        except Exception:
            ctx = ""
    
    # Record outcome for learning
    elapsed_ms = (time.time() - t0) * 1000
    try:
        from jinx.micro.brain import record_outcome
        quality = 1.0 if success and len(ctx) > 500 else (0.6 if len(ctx) > 100 else 0.2)
        asyncio.create_task(record_outcome(
            'context_build',
            success,
            {
                'query_intent': query_intent.intent if query_intent else 'unknown',
                'subqs_count': len(top_subqs),
                'k': proj_k,
                'codey': codey,
                'chars': len(ctx),
                'query_expanded': expanded_query is not None,
            },
            latency_ms=elapsed_ms
        ))
        
        # Record query expansion outcome
        if expanded_query:
            from jinx.micro.brain import record_expansion_outcome
            asyncio.create_task(record_expansion_outcome(
                _q,
                expanded_query.expanded,
                expanded_query.method,
                success and len(ctx) > 100
            ))
    except Exception:
        pass
    
    return ctx or ""


async def _score_subqueries(
    subqs: List[str],
    original_query: str,
    query_intent
) -> List[Tuple[str, float]]:
    """Score and rank subqueries using ML-based relevance estimation."""
    if not subqs:
        return []
    
    scored: List[Tuple[str, float]] = []
    
    # Base scoring: length, overlap with original, uniqueness
    for sq in subqs:
        score = 0.0
        
        # Length penalty/bonus
        if 10 <= len(sq) <= 200:
            score += 1.0
        elif len(sq) > 200:
            score += 0.5
        else:
            score += 0.3
        
        # Overlap with original query (term intersection)
        orig_terms = set(original_query.lower().split())
        sq_terms = set(sq.lower().split())
        if orig_terms and sq_terms:
            overlap = len(orig_terms & sq_terms) / len(orig_terms | sq_terms)
            score += overlap * 2.0
        
        # Uniqueness bonus (not too similar to original)
        if sq.lower() != original_query.lower():
            score += 0.5
        
        # Intent-based boosting
        if query_intent:
            if query_intent.intent == 'code_exec' and ('def ' in sq or 'class ' in sq or 'import ' in sq):
                score += 1.5
            elif query_intent.intent == 'debug' and ('error' in sq.lower() or 'exception' in sq.lower()):
                score += 1.2
            elif query_intent.intent == 'refactor' and ('function' in sq.lower() or 'method' in sq.lower()):
                score += 1.0
        
        scored.append((sq, score))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return scored
