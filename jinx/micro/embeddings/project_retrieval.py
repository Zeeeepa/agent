from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Dict, List, Tuple

from .project_config import ROOT
from .project_retrieval_config import (
    PROJ_DEFAULT_TOP_K,
    PROJ_SNIPPET_AROUND,
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_TOTAL_CODE_BUDGET,
    PROJ_ALWAYS_FULL_PY_SCOPE,
    PROJ_FULL_SCOPE_TOP_N,
    PROJ_EXHAUSTIVE_MODE,
    PROJ_NO_STAGE_BUDGETS,
    PROJ_NO_CODE_BUDGET,
    PROJ_USAGE_REFS_LIMIT,
    PROJ_STAGE_PYAST_MS,
    PROJ_STAGE_JEDI_MS,
    PROJ_STAGE_PYDOC_MS,
    PROJ_STAGE_REGEX_MS,
    PROJ_STAGE_PYFLOW_MS,
    PROJ_STAGE_LIBCST_MS,
    PROJ_STAGE_TB_MS,
    PROJ_STAGE_PYLITERALS_MS,
    PROJ_STAGE_FASTSUBSTR_MS,
    PROJ_STAGE_LINETOKENS_MS,
    PROJ_STAGE_LINEEXACT_MS,
    PROJ_STAGE_ASTMATCH_MS,
    PROJ_STAGE_RAPIDFUZZ_MS,
    PROJ_STAGE_TOKENMATCH_MS,
    PROJ_STAGE_PRE_MS,
    PROJ_STAGE_EXACT_MS,
    PROJ_STAGE_VECTOR_MS,
    PROJ_STAGE_KEYWORD_MS,
)
from .project_py_scope import get_python_symbol_at_line
from .project_refs import find_usages_in_project
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
from .project_stage_fastsubstr import stage_fastsubstr_hits
from .project_stage_linetokens import stage_linetokens_hits
from .project_stage_lineexact import stage_lineexact_hits
from .project_stage_astmatch import stage_astmatch_hits
from .project_stage_rapidfuzz import stage_rapidfuzz_hits
from .project_stage_tokenmatch import stage_tokenmatch_hits
from .project_snippet import build_snippet
from .project_query_core import extract_code_core

# Tunables moved to project_retrieval_config.py


async def retrieve_project_top_k(query: str, k: int | None = None, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    q = (query or "").strip()
    if not q:
        return []
    k_eff = k or PROJ_DEFAULT_TOP_K
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
        # In exhaustive or no-stage-budget mode, do not enforce a time limit per stage
        if PROJ_EXHAUSTIVE_MODE or PROJ_NO_STAGE_BUDGETS or max_time_ms is None:
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

    def _run(stage_fn, cap_ms: int):
        rem = _bounded(_time_left(), cap_ms)
        try:
            hits = stage_fn(q, k_eff, max_time_ms=rem)
        except Exception:
            hits = []
        return (hits[:k_eff]) if hits else None

    async def _run_async(stage_fn, cap_ms: int):
        rem = _bounded(_time_left(), cap_ms)
        try:
            hits = await stage_fn(q, k_eff, max_time_ms=rem)
        except Exception:
            hits = []
        return (hits[:k_eff]) if hits else None

    async def _run_sync_stage(stage_fn, query: str, k_arg: int, cap_ms: int):
        rem = _bounded(_time_left(), cap_ms)
        def _call():
            try:
                return stage_fn(query, k_arg, max_time_ms=rem)
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

    # Stage -2.5: Python token-sequence exact subsequence match (very precise)
    tm_hits = await _run_sync_stage(stage_tokenmatch_hits, q_core, (k_eff if accumulate else 1), PROJ_STAGE_TOKENMATCH_MS)
    if tm_hits:
        if accumulate:
            _merge(tm_hits)
        else:
            return tm_hits[:1]
    await asyncio.sleep(0)

    # Stage -2.4: whitespace-insensitive literal line match (precise)
    le_hits = await _run_sync_stage(stage_lineexact_hits, q_core, (k_eff if accumulate else 1), PROJ_STAGE_LINEEXACT_MS)
    if le_hits:
        if accumulate:
            _merge(le_hits)
        else:
            return le_hits[:1]
    await asyncio.sleep(0)

    # Stage -2.3: AST shape matching (precise structure)
    am_hits = await _run_sync_stage(stage_astmatch_hits, q_core, (k_eff if accumulate else 1), PROJ_STAGE_ASTMATCH_MS)
    if am_hits:
        if accumulate:
            _merge(am_hits)
        else:
            return am_hits[:1]
    await asyncio.sleep(0)

    # Stage -2.25: RapidFuzz approximate match (robust, still precise-ish)
    rf_hits = await _run_sync_stage(stage_rapidfuzz_hits, q_core, (k_eff if accumulate else 1), PROJ_STAGE_RAPIDFUZZ_MS)
    if rf_hits:
        if accumulate:
            _merge(rf_hits)
        else:
            return rf_hits[:1]
    await asyncio.sleep(0)

    # Stage -2.2: fast substring search using code-core/anchors (very cheap)
    # Use k=1 normally; allow more in exhaustive mode
    fs_hits = await _run_sync_stage(stage_fastsubstr_hits, q, (k_eff if accumulate else 1), PROJ_STAGE_FASTSUBSTR_MS)
    if fs_hits:
        if accumulate:
            _merge(fs_hits)
        else:
            return fs_hits[:1]
    await asyncio.sleep(0)

    # Stage -2.15: line-level all-anchors match (very cheap)
    lt_hits = await _run_sync_stage(stage_linetokens_hits, q, (k_eff if accumulate else 1), PROJ_STAGE_LINETOKENS_MS)
    if lt_hits:
        if accumulate:
            _merge(lt_hits)
        else:
            return lt_hits[:1]
    await asyncio.sleep(0)

    # Quick router: if query looks like an assignment/comprehension, try PyFlow early
    try:
        import re as _rtr_re
        _qr = q_core
        qlow = _qr.lower()
        assign_like = bool(_rtr_re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*\((?s).*?\bfor\b", _qr))
        comp_like = (" for " in qlow and " in " in qlow) or any(sym in _qr for sym in [":=", "=>"])  # broad
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

    # Stage -3: parse traceback-like text for exact file/line windows (fast and precise)
    tb_hits = await _run_sync_stage(stage_traceback_hits, q, k_eff, PROJ_STAGE_TB_MS)
    if tb_hits:
        if accumulate:
            _merge(tb_hits)
        else:
            return tb_hits
    await asyncio.sleep(0)

    # Stage -2: Python AST-driven call-site detection (very precise for Python queries)
    ast_hits = await _run_sync_stage(stage_pyast_hits, q, k_eff, PROJ_STAGE_PYAST_MS)
    if ast_hits:
        if accumulate:
            _merge(ast_hits)
        else:
            return ast_hits
    await asyncio.sleep(0)

    # Stage -1.8: Python docstring scan (precise for doc/comment queries)
    pydoc_hits = await _run_sync_stage(stage_pydoc_hits, q, k_eff, PROJ_STAGE_PYDOC_MS)
    if pydoc_hits:
        if accumulate:
            _merge(pydoc_hits)
        else:
            return pydoc_hits
    await asyncio.sleep(0)

    # Stage -1.75: Python string literals (error/log/message fragments, f-strings)
    pl_hits = await _run_sync_stage(stage_pyliterals_hits, q, k_eff, PROJ_STAGE_PYLITERALS_MS)
    if pl_hits:
        if accumulate:
            _merge(pl_hits)
        else:
            return pl_hits
    await asyncio.sleep(0)

    # Stage -1.7: Python return/call flow patterns (e.g., 'return db.get(rel_path)')
    pyflow_hits = await _run_sync_stage(stage_pyflow_hits, q, k_eff, PROJ_STAGE_PYFLOW_MS)
    if pyflow_hits:
        if accumulate:
            _merge(pyflow_hits)
        else:
            return pyflow_hits
    await asyncio.sleep(0)

    # Stage -1.6: libcst structural patterns (broad Python constructs)
    cst_hits = await _run_sync_stage(stage_libcst_hits, q, k_eff, PROJ_STAGE_LIBCST_MS)
    if cst_hits:
        if accumulate:
            _merge(cst_hits)
        else:
            return cst_hits
    await asyncio.sleep(0)

    # Stage -1.5: optional Jedi-driven identifier/reference discovery (Python)
    jedi_hits = await _run_sync_stage(stage_jedi_hits, q, k_eff, PROJ_STAGE_JEDI_MS)
    if jedi_hits:
        if accumulate:
            _merge(jedi_hits)
        else:
            return jedi_hits
    await asyncio.sleep(0)

    # Stage -1.4: fuzzy regex phrase matching (optional dependency 'regex')
    rx_hits = await _run_sync_stage(stage_regex_hits, q, k_eff, PROJ_STAGE_REGEX_MS)
    if rx_hits:
        if accumulate:
            _merge(rx_hits)
        else:
            return rx_hits
    await asyncio.sleep(0)

    # Stage -1: direct text scan in files (fast path for identifiers not present in previews/terms)
    # Dynamic cap for pre-scan based on code-ish query
    low_q = q.lower()
    codey = any(sym in q for sym in "=[](){}.:,") or any(kw in low_q for kw in ["def ", "class ", "import ", "from ", "return ", "async ", "await "])
    cap_pre = int(PROJ_STAGE_PRE_MS * 2) if codey else PROJ_STAGE_PRE_MS
    txt_hits = await _run_sync_stage(stage_textscan_hits, q, k_eff, cap_pre)
    if txt_hits:
        if accumulate:
            _merge(txt_hits)
        else:
            return txt_hits
    await asyncio.sleep(0)

    # Stage 0: exact/identifier substring hits (fast)
    exact = await _run_sync_stage(stage_exact_hits, q, k_eff, PROJ_STAGE_EXACT_MS)
    if exact:
        if accumulate:
            _merge(exact)
        else:
            return exact
    await asyncio.sleep(0)

    # (Vector similarity already awaited above.)

    # Stage 2: keyword/substring fallback
    kw = (await _run_sync_stage(stage_keyword_hits, q, k_eff, PROJ_STAGE_KEYWORD_MS)) or []
    if accumulate:
        _merge(kw)
        # Return deduped, score-sorted
        return sorted(collected, key=lambda h: float(h[0] or 0.0), reverse=True)[:k_eff]
    return kw[:k_eff]


async def build_project_context_for(query: str, *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    k = k or PROJ_DEFAULT_TOP_K
    hits = await retrieve_project_top_k(query, k=k, max_time_ms=max_time_ms)
    if not hits:
        return ""
    # Sort by score desc to surface the most relevant first
    hits_sorted = sorted(hits, key=lambda h: float(h[0] or 0.0), reverse=True)
    parts: List[str] = []
    refs_parts: List[str] = []
    seen: set[str] = set()  # dedupe by preview text
    headers_seen: set[str] = set()  # dedupe by [file:ls-le]
    refs_headers_seen: set[str] = set()  # dedupe refs by header
    # Language detection moved to project_lang.py

    # Disable total code budget if configured
    budget = None if PROJ_NO_CODE_BUDGET else (PROJ_TOTAL_CODE_BUDGET if (max_chars is None) else max_chars)
    total_len = 0

    full_scope_used = 0
    qlow = (query or "").lower()
    codey_query = any(sym in (query or "") for sym in "=[](){}.:,") or any(kw in qlow for kw in ["def ", "class ", "import ", "from ", "return ", "async ", "await ", " for ", " in ", " = "])
    for idx, (score, file_rel, obj) in enumerate(hits_sorted):
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        if pv and pv in seen:
            continue
        if pv:
            seen.add(pv)
        # Build minimal header + code block snippet via micro-module
        prefer_full = PROJ_ALWAYS_FULL_PY_SCOPE and (
            PROJ_FULL_SCOPE_TOP_N <= 0 or (full_scope_used < PROJ_FULL_SCOPE_TOP_N)
        )
        # Build snippet off the event loop (file reads, scope detection)
        header, code_block, use_ls, use_le, is_full_scope = await asyncio.to_thread(
            lambda: build_snippet(
                file_rel,
                meta,
                query,
                max_chars=PROJ_SNIPPET_PER_HIT_CHARS,
                prefer_full_scope=prefer_full,
                expand_callees=True,
            )
        )
        snippet_text = f"{header}\n{code_block}"
        # Dedupe snippets by header only (permit identical code bodies from different files)
        if header in headers_seen:
            continue
        headers_seen.add(header)
        # Enforce total budget unless full scope must be included
        if budget is not None:
            would = total_len + len(snippet_text)
            if (not is_full_scope or not PROJ_ALWAYS_FULL_PY_SCOPE) and would > budget:
                if not parts:
                    # Ensure at least one snippet is returned
                    parts.append(snippet_text)
                break
            total_len = would
        parts.append(snippet_text)
        if is_full_scope:
            full_scope_used += 1
        # Periodically yield while assembling snippets
        if (idx % 2) == 1:
            await asyncio.sleep(0)

        # Optionally add a couple of usage references for the enclosing symbol (Python only), minimal format
        try:
            async def _collect_usages() -> list[tuple[str, str]]:
                def _work() -> list[tuple[str, str]]:
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
                            sym_name, sym_kind = get_python_symbol_at_line(file_text, cand_line)  # noqa: F841
                            if sym_name:
                                usages = find_usages_in_project(sym_name, file_rel, limit=PROJ_USAGE_REFS_LIMIT, around=PROJ_SNIPPET_AROUND)
                                for fr, ua, ub, usnip, ulang in usages:
                                    hdr = f"[{fr}:{ua}-{ub}]"
                                    block = f"```{ulang}\n{usnip}\n```" if ulang else f"```\n{usnip}\n```"
                                    out.append((hdr, block))
                    except Exception:
                        return out
                    return out
                return await asyncio.to_thread(_work)

            pairs = await _collect_usages()
            for hdr, block in pairs:
                if hdr in refs_headers_seen:
                    continue
                refs_headers_seen.add(hdr)
                refs_parts.append(f"{hdr}\n{block}")
        except Exception:
            pass
        # Do not hard-break here; total budget enforced above
    if not parts:
        return ""
    body = "\n".join(parts)
    if refs_parts:
        rbody = "\n".join(refs_parts)
        return f"<embeddings_code>\n{body}\n</embeddings_code>\n\n<embeddings_refs>\n{rbody}\n</embeddings_refs>"
    return f"<embeddings_code>\n{body}\n</embeddings_code>"
