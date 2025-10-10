from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Tuple

from .project_rerank import rerank_hits
from .project_config import ROOT
from jinx.micro.common.internal_paths import is_restricted_path
from .project_retrieval_config import (
    PROJ_DEFAULT_TOP_K,
    PROJ_SNIPPET_AROUND,
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_TOTAL_CODE_BUDGET,
    PROJ_ALWAYS_FULL_PY_SCOPE,
    PROJ_FULL_SCOPE_TOP_N,
    PROJ_NO_CODE_BUDGET,
    PROJ_CALLGRAPH_ENABLED,
    PROJ_CALLGRAPH_TOP_HITS,
    PROJ_CALLGRAPH_CALLERS_LIMIT,
    PROJ_CALLGRAPH_CALLEES_LIMIT,
    PROJ_CALLGRAPH_TIME_MS,
    PROJ_MAX_FILES,
    PROJ_CONSOLIDATE_PER_FILE,
    PROJ_USAGE_REFS_LIMIT,
)
from .project_snippet import build_snippet
from .snippet_cache import make_snippet_cache_key, get_cached_snippet, put_cached_snippet
from .graph_cache import get_symbol_graph_cached, find_usages_cached
from .project_py_scope import get_python_symbol_at_line
from .project_stage_literal import stage_literal_hits
from .project_lang import lang_for_file
from .refs_format import format_usage_ref, format_literal_ref
from jinx.micro.text.heuristics import is_code_like as _is_code_like

from .retrieval_core import (
    retrieve_project_top_k,
    retrieve_project_multi_top_k,
)


async def build_project_context_for(query: str, *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    k = k or PROJ_DEFAULT_TOP_K
    hits = await retrieve_project_top_k(query, k=k, max_time_ms=max_time_ms)
    if not hits:
        return ""
    # Rerank to prioritize path/preview token matches with the query
    hits_sorted = rerank_hits(hits, query)
    parts: List[str] = []
    refs_parts: List[str] = []
    graph_parts: List[str] = []
    seen: set[str] = set()  # dedupe by preview text
    headers_seen: set[str] = set()  # dedupe by [file:ls-le]
    refs_headers_seen: set[str] = set()  # dedupe refs by header
    graph_headers_seen: set[str] = set()  # dedupe graph entries by header
    included_files: set[str] = set()

    # Build per-file centers from all hits to allow multi-segment snippets to include other hotspots
    file_hit_centers: Dict[str, List[int]] = {}
    for sc, fr, obj in hits_sorted:
        try:
            m = (obj.get("meta") or {})
            ls = int(m.get("line_start") or 0)
            le = int(m.get("line_end") or 0)
            c = int((ls + le) // 2) if (ls and le) else int(ls or le or 0)
            if c > 0:
                file_hit_centers.setdefault(fr, []).append(c)
        except Exception:
            continue

    # Disable total code budget if configured
    budget = None if PROJ_NO_CODE_BUDGET else (PROJ_TOTAL_CODE_BUDGET if (max_chars is None) else max_chars)
    total_len = 0

    full_scope_used = 0
    codey_query = _is_code_like(query or "")  # currently informational; future heuristics may use it
    # Parallel snippet building with bounded concurrency
    try:
        _SNIP_CONC = max(1, int(os.getenv("EMBED_PROJECT_SNIPPET_CONC", "4")))
    except Exception:
        _SNIP_CONC = 4
    sem = asyncio.Semaphore(_SNIP_CONC)

    prepared: List[Tuple[int, str, Dict[str, Any], bool, List[int]]] = []  # (idx, file_rel, meta, prefer_full, extra_centers_abs)
    for idx, (score, file_rel, obj) in enumerate(hits_sorted):
        # Skip restricted files defensively (.jinx, log, etc.) and dedupe by preview text
        try:
            if is_restricted_path(str(file_rel or "")):
                continue
        except Exception:
            pass
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        if pv and pv in seen:
            continue
        if pv:
            seen.add(pv)
        prefer_full = PROJ_ALWAYS_FULL_PY_SCOPE and (
            PROJ_FULL_SCOPE_TOP_N <= 0 or (full_scope_used < PROJ_FULL_SCOPE_TOP_N)
        )
        try:
            extra_centers_abs = sorted({int(x) for x in (file_hit_centers.get(file_rel) or []) if int(x) > 0})
        except Exception:
            extra_centers_abs = []
        prepared.append((idx, file_rel, meta, prefer_full, extra_centers_abs))

    async def _build(idx_i: int, file_rel_i: str, meta_i: Dict[str, Any], prefer_full_i: bool, centers_i: List[int]):
        async with sem:
            # Try cache first inside the worker to avoid main loop blocking
            def _run():
                key = make_snippet_cache_key(
                    file_rel_i,
                    meta_i,
                    query,
                    prefer_full_scope=prefer_full_i,
                    expand_callees=True,
                    extra_centers_abs=centers_i,
                )
                cached = get_cached_snippet(key)
                if cached is not None:
                    hdr_c, code_c, ls_c, le_c, is_full_c = cached
                    return (hdr_c, code_c, ls_c, le_c, is_full_c)
                res = build_snippet(
                    file_rel_i,
                    meta_i,
                    query,
                    max_chars=PROJ_SNIPPET_PER_HIT_CHARS,
                    prefer_full_scope=prefer_full_i,
                    expand_callees=True,
                    extra_centers_abs=centers_i,
                )
                try:
                    put_cached_snippet(key, res)
                except Exception:
                    pass
                return res
            hdr, code, ls, le, is_full = await asyncio.to_thread(_run)
            return (idx_i, file_rel_i, meta_i, hdr, code, ls, le, is_full)

    tasks = [asyncio.create_task(_build(*args)) for args in prepared]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Assemble in original order, enforcing budget and per-file consolidation
    for r in sorted([x for x in results if not isinstance(x, Exception)], key=lambda t: t[0]):
        idx, file_rel, meta, header, code_block, use_ls, use_le, is_full_scope = r
        if PROJ_CONSOLIDATE_PER_FILE and file_rel in included_files:
            continue
        snippet_text = f"{header}\n{code_block}"
        if header in headers_seen:
            continue
        headers_seen.add(header)
        if budget is not None:
            would = total_len + len(snippet_text)
            if (not is_full_scope or not PROJ_ALWAYS_FULL_PY_SCOPE) and would > budget:
                if not parts:
                    parts.append(snippet_text)
                break
            total_len = would
        parts.append(snippet_text)
        if PROJ_CONSOLIDATE_PER_FILE:
            included_files.add(file_rel)
        if is_full_scope:
            full_scope_used += 1
        # Optional callgraph enrichment for top hits (Python only)
        try:
            if PROJ_CALLGRAPH_ENABLED and file_rel.endswith('.py') and idx < max(0, PROJ_CALLGRAPH_TOP_HITS):
                pairs = await get_symbol_graph_cached(
                    file_rel,
                    use_ls or 0,
                    use_le or 0,
                    callers_limit=PROJ_CALLGRAPH_CALLERS_LIMIT,
                    callees_limit=PROJ_CALLGRAPH_CALLEES_LIMIT,
                    around=PROJ_SNIPPET_AROUND,
                    scan_cap_files=PROJ_MAX_FILES,
                    time_budget_ms=PROJ_CALLGRAPH_TIME_MS,
                )
                for hdr2, block in (pairs or []):
                    if hdr2 in graph_headers_seen:
                        continue
                    graph_headers_seen.add(hdr2)
                    graph_parts.append(f"{hdr2}\n{block}")
        except Exception:
            pass
        # Optionally add a couple of usage references for the enclosing symbol (Python only)
        try:
            # Allow env override for usage references limit
            try:
                _usage_lim_env = os.getenv("JINX_REFS_USAGE_LIMIT", "")
                usage_limit = int(_usage_lim_env) if _usage_lim_env.strip() else PROJ_USAGE_REFS_LIMIT
            except Exception:
                usage_limit = PROJ_USAGE_REFS_LIMIT

            async def _collect_usages() -> list[tuple[str, str]]:
                out: list[tuple[str, str]] = []
                try:
                    file_text = ""
                    try:
                        with open(os.path.join(ROOT, file_rel), 'r', encoding='utf-8', errors='ignore') as _f:
                            file_text = _f.read()
                    except Exception:
                        file_text = ""
                    if file_rel.endswith('.py') and file_text:
                        cand_line = int((use_ls + use_le) // 2) if (use_ls and use_le) else int(use_ls or use_le or 0)
                        sym_name, sym_kind = get_python_symbol_at_line(file_text, cand_line)
                        if sym_name:
                            usages = await find_usages_cached(sym_name, file_rel, limit=usage_limit, around=PROJ_SNIPPET_AROUND)
                            for fr, ua, ub, usnip, ulang in usages:
                                try:
                                    hdrx, blockx = format_usage_ref(
                                        sym_name,
                                        sym_kind,
                                        fr,
                                        int(ua or 0),
                                        int(ub or 0),
                                        usnip or "",
                                        ulang,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    langx = ulang
                                    hdrx = f"[{fr}:{ua}-{ub}]"
                                    blockx = f"```{langx}\n{usnip}\n```" if langx else f"```\n{usnip}\n```"
                                out.append((hdrx, blockx))
                    # Fallback: literal-occurrences refs when no symbol usages were found
                    if not out and (query or "").strip():
                        # Literal refs collection tuning via env
                        try:
                            _lim_env = os.getenv("JINX_REFS_LIT_LIMIT", "")
                            _lim = int(_lim_env) if _lim_env.strip() else (6 if _is_code_like(query or "") else 3)
                        except Exception:
                            _lim = 6 if _is_code_like(query or "") else 3
                        try:
                            _ms_env = os.getenv("JINX_REFS_LIT_MS", "")
                            _ms = int(_ms_env) if _ms_env.strip() else (300 if _is_code_like(query or "") else 200)
                        except Exception:
                            _ms = 300 if _is_code_like(query or "") else 200
                        def _lit_call():
                            try:
                                return stage_literal_hits(query, _lim, max_time_ms=_ms)
                            except Exception:
                                return []
                        lit_hits = await asyncio.to_thread(_lit_call)
                        for _sc2, rel2, obj2 in (lit_hits or [])[:_lim]:
                            try:
                                meta2 = (obj2.get("meta") or {})
                                ls2 = int(meta2.get("line_start") or 0)
                                le2 = int(meta2.get("line_end") or 0)
                                if str(rel2) == str(file_rel) and ls2 == int(use_ls or 0) and le2 == int(use_le or 0):
                                    continue
                                prev = (meta2.get("text_preview") or "").strip()
                                if not prev:
                                    continue
                                lang2 = lang_for_file(rel2)
                                try:
                                    hdrx, blockx = format_literal_ref(
                                        query,
                                        str(rel2),
                                        int(ls2 or 0),
                                        int(le2 or 0),
                                        prev,
                                        lang2,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    hdrx = f"[{rel2}:{ls2}-{le2}]"
                                    blockx = f"```{lang2}\n{prev}\n```" if lang2 else f"```\n{prev}\n```"
                                out.append((hdrx, blockx))
                            except Exception:
                                continue
                except Exception:
                    return out
                return out

            pairs = await _collect_usages()
            for hdr3, block3 in pairs:
                if hdr3 in refs_headers_seen:
                    continue
                refs_headers_seen.add(hdr3)
                refs_parts.append(f"{hdr3}\n{block3}")
        except Exception:
            pass

    if not parts:
        return ""
    body = "\n".join(parts)
    out_blocks: List[str] = [f"<embeddings_code>\n{body}\n</embeddings_code>"]
    # Refs policy gating and size budget to avoid unnecessary tokens
    # Default to 'always' so references are visible by default; can be tuned via env
    refs_policy = os.getenv("JINX_REFS_POLICY", "always").strip().lower()
    try:
        refs_min = max(1, int(os.getenv("JINX_REFS_AUTO_MIN", "2")))
    except Exception:
        refs_min = 2
    try:
        refs_max_chars = max(200, int(os.getenv("JINX_REFS_MAX_CHARS", "1600")))
    except Exception:
        refs_max_chars = 1600

    def _should_send_refs(codey: bool, count: int) -> bool:
        if refs_policy in ("never", "0", "off", "false", ""):
            return False
        if refs_policy in ("always", "1", "on", "true"):
            return True
        return bool(codey) or (count >= refs_min)

    if refs_parts and _should_send_refs(codey_query, len(refs_parts)):
        # Trim refs to the configured character budget
        acc: List[str] = []
        total = 0
        for p in refs_parts:
            plen = len(p) + 1
            if total + plen > refs_max_chars:
                break
            acc.append(p)
            total += plen
        if acc:
            rbody = "\n".join(acc)
            out_blocks.append(f"<embeddings_refs>\n{rbody}\n</embeddings_refs>")
    if graph_parts:
        gbody = "\n".join(graph_parts)
        out_blocks.append(f"<embeddings_graph>\n{gbody}\n</embeddings_graph>")
    return "\n\n".join(out_blocks)


__all__ = [
    "build_project_context_for",
    "build_project_context_multi_for",
]


async def build_project_context_multi_for(queries: List[str], *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    k_eff = k or PROJ_DEFAULT_TOP_K
    per_query_k = max(1, int((k_eff + max(1, len(queries)) - 1) // max(1, len(queries))))
    hits = await retrieve_project_multi_top_k(queries, per_query_k=per_query_k, max_time_ms=max_time_ms)
    if not hits:
        return ""
    # Re-rank across all hits by combined query string
    hits_sorted = rerank_hits(hits, " ".join(queries))
    parts: List[str] = []
    refs_parts: List[str] = []
    graph_parts: List[str] = []
    seen: set[str] = set()
    headers_seen: set[str] = set()
    refs_headers_seen: set[str] = set()
    graph_headers_seen: set[str] = set()
    included_files: set[str] = set()
    budget = None if PROJ_NO_CODE_BUDGET else (PROJ_TOTAL_CODE_BUDGET if (max_chars is None) else max_chars)
    total_len = 0

    full_scope_used = 0
    # Build per-file centers from all hits to allow multi-segment snippets to include other hotspots
    file_hit_centers: Dict[str, List[int]] = {}
    for sc, fr, obj in hits_sorted:
        try:
            m = (obj.get("meta") or {})
            ls = int(m.get("line_start") or 0)
            le = int(m.get("line_end") or 0)
            c = int((ls + le) // 2) if (ls and le) else int(ls or le or 0)
            if c > 0:
                file_hit_centers.setdefault(fr, []).append(c)
        except Exception:
            continue

    # Parallel snippet building with bounded semaphore
    try:
        _SNIP_CONC = max(1, int(os.getenv("EMBED_PROJECT_SNIPPET_CONC", "4")))
    except Exception:
        _SNIP_CONC = 4
    sem = asyncio.Semaphore(_SNIP_CONC)

    q_join = " ".join(queries)[:512]
    codey_join = _is_code_like(q_join or "")
    prepared: List[Tuple[int, str, Dict[str, Any], bool, List[int]]] = []
    for idx, (score, file_rel, obj) in enumerate(hits_sorted):
        try:
            if is_restricted_path(str(file_rel or "")):
                continue
        except Exception:
            pass
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        if pv and pv in seen:
            continue
        if pv:
            seen.add(pv)
        prefer_full = PROJ_ALWAYS_FULL_PY_SCOPE and (
            PROJ_FULL_SCOPE_TOP_N <= 0 or (full_scope_used < PROJ_FULL_SCOPE_TOP_N)
        )
        try:
            extra_centers_abs = sorted({int(x) for x in (file_hit_centers.get(file_rel) or []) if int(x) > 0})
        except Exception:
            extra_centers_abs = []
        prepared.append((idx, file_rel, meta, prefer_full, extra_centers_abs))

    async def _build(idx_i: int, file_rel_i: str, meta_i: Dict[str, Any], prefer_full_i: bool, centers_i: List[int]):
        async with sem:
            def _run():
                return build_snippet(
                    file_rel_i,
                    meta_i,
                    q_join,
                    max_chars=PROJ_SNIPPET_PER_HIT_CHARS,
                    prefer_full_scope=prefer_full_i,
                    expand_callees=True,
                    extra_centers_abs=centers_i,
                )
            hdr, code, ls, le, is_full = await asyncio.to_thread(_run)
            return (idx_i, file_rel_i, meta_i, hdr, code, ls, le, is_full)

    tasks = [asyncio.create_task(_build(*args)) for args in prepared]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in sorted([x for x in results if not isinstance(x, Exception)], key=lambda t: t[0]):
        idx, file_rel, meta, header, code_block, use_ls, use_le, is_full_scope = r
        if PROJ_CONSOLIDATE_PER_FILE and file_rel in included_files:
            continue
        snippet_text = f"{header}\n{code_block}"
        if header in headers_seen:
            continue
        headers_seen.add(header)
        if budget is not None:
            would = total_len + len(snippet_text)
            if (not is_full_scope or not PROJ_ALWAYS_FULL_PY_SCOPE) and would > budget:
                if not parts:
                    parts.append(snippet_text)
                break
            total_len = would
        parts.append(snippet_text)
        if PROJ_CONSOLIDATE_PER_FILE:
            included_files.add(file_rel)
        if is_full_scope:
            full_scope_used += 1
        # Optional callgraph enrichment for top hits (Python only)
        try:
            if PROJ_CALLGRAPH_ENABLED and file_rel.endswith('.py') and idx < max(0, PROJ_CALLGRAPH_TOP_HITS):
                pairs = await get_symbol_graph_cached(
                    file_rel,
                    use_ls or 0,
                    use_le or 0,
                    callers_limit=PROJ_CALLGRAPH_CALLERS_LIMIT,
                    callees_limit=PROJ_CALLGRAPH_CALLEES_LIMIT,
                    around=PROJ_SNIPPET_AROUND,
                    scan_cap_files=PROJ_MAX_FILES,
                    time_budget_ms=PROJ_CALLGRAPH_TIME_MS,
                )
                for hdr2, block in (pairs or []):
                    if hdr2 in graph_headers_seen:
                        continue
                    graph_headers_seen.add(hdr2)
                    graph_parts.append(f"{hdr2}\n{block}")
        except Exception:
            pass
        # Optionally add a couple of usage references for the enclosing symbol (Python only)
        try:
            async def _collect_usages() -> list[tuple[str, str]]:
                out: list[tuple[str, str]] = []
                try:
                    file_text = ""
                    try:
                        with open(os.path.join(ROOT, file_rel), 'r', encoding='utf-8', errors='ignore') as _f:
                            file_text = _f.read()
                    except Exception:
                        file_text = ""
                    if file_rel.endswith('.py') and file_text:
                        cand_line = int((use_ls + use_le) // 2) if (use_ls and use_le) else int(use_ls or use_le or 0)
                        sym_name, sym_kind = get_python_symbol_at_line(file_text, cand_line)
                        if sym_name:
                            usages = await find_usages_cached(sym_name, file_rel, limit=PROJ_USAGE_REFS_LIMIT, around=PROJ_SNIPPET_AROUND)
                            for fr, ua, ub, usnip, ulang in usages:
                                try:
                                    hdrx, blockx = format_usage_ref(
                                        sym_name,
                                        sym_kind,
                                        fr,
                                        int(ua or 0),
                                        int(ub or 0),
                                        usnip or "",
                                        ulang,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    langx = ulang
                                    hdrx = f"[{fr}:{ua}-{ub}]"
                                    blockx = f"```{langx}\n{usnip}\n```" if langx else f"```\n{usnip}\n```"
                                out.append((hdrx, blockx))
                    # Fallback: literal-occurrences refs when no symbol usages were found
                    if not out and (q_join or "").strip():
                        # Literal refs collection tuning via env
                        try:
                            _lim_env = os.getenv("JINX_REFS_LIT_LIMIT", "")
                            _lim = int(_lim_env) if _lim_env.strip() else (6 if _is_code_like(q_join or "") else 3)
                        except Exception:
                            _lim = 6 if _is_code_like(q_join or "") else 3
                        try:
                            _ms_env = os.getenv("JINX_REFS_LIT_MS", "")
                            _ms = int(_ms_env) if _ms_env.strip() else (300 if _is_code_like(q_join or "") else 200)
                        except Exception:
                            _ms = 300 if _is_code_like(q_join or "") else 200
                        def _lit_call():
                            try:
                                return stage_literal_hits(q_join, _lim, max_time_ms=_ms)
                            except Exception:
                                return []
                        lit_hits = await asyncio.to_thread(_lit_call)
                        for _sc2, rel2, obj2 in (lit_hits or [])[:_lim]:
                            try:
                                meta2 = (obj2.get("meta") or {})
                                ls2 = int(meta2.get("line_start") or 0)
                                le2 = int(meta2.get("line_end") or 0)
                                prev = (meta2.get("text_preview") or "").strip()
                                if not prev:
                                    continue
                                lang2 = lang_for_file(rel2)
                                try:
                                    hdrx, blockx = format_literal_ref(
                                        q_join,
                                        str(rel2),
                                        int(ls2 or 0),
                                        int(le2 or 0),
                                        prev,
                                        lang2,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    hdrx = f"[{rel2}:{ls2}-{le2}]"
                                    blockx = f"```{lang2}\n{prev}\n```" if lang2 else f"```\n{prev}\n```"
                                out.append((hdrx, blockx))
                            except Exception:
                                continue
                except Exception:
                    return out
                return out

            pairs = await _collect_usages()
            for hdr3, block3 in pairs:
                if hdr3 in refs_headers_seen:
                    continue
                refs_headers_seen.add(hdr3)
                refs_parts.append(f"{hdr3}\n{block3}")
        except Exception:
            pass

    if not parts:
        return ""
    body = "\n".join(parts)
    out_blocks: List[str] = [f"<embeddings_code>\n{body}\n</embeddings_code>"]
    # Refs policy gating and size budget (multi-query). Default to 'always' so refs are visible by default.
    refs_policy = os.getenv("JINX_REFS_POLICY", "always").strip().lower()
    try:
        refs_min = max(1, int(os.getenv("JINX_REFS_AUTO_MIN", "2")))
    except Exception:
        refs_min = 2
    try:
        refs_max_chars = max(200, int(os.getenv("JINX_REFS_MAX_CHARS", "1600")))
    except Exception:
        refs_max_chars = 1600

    def _should_send_refs_multi(codey: bool, count: int) -> bool:
        if refs_policy in ("never", "0", "off", "false", ""):
            return False
        if refs_policy in ("always", "1", "on", "true"):
            return True
        return bool(codey) or (count >= refs_min)

    if refs_parts and _should_send_refs_multi(codey_join, len(refs_parts)):
        acc: List[str] = []
        total = 0
        for p in refs_parts:
            plen = len(p) + 1
            if total + plen > refs_max_chars:
                break
            acc.append(p)
            total += plen
        if acc:
            rbody = "\n".join(acc)
            out_blocks.append(f"<embeddings_refs>\n{rbody}\n</embeddings_refs>")
    if graph_parts:
        gbody = "\n".join(graph_parts)
        out_blocks.append(f"<embeddings_graph>\n{gbody}\n</embeddings_graph>")
    return "\n\n".join(out_blocks)
