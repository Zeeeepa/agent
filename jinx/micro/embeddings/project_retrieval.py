from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT
from .project_retrieval_config import (
    PROJ_DEFAULT_TOP_K,
    PROJ_SNIPPET_AROUND,
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_TOTAL_CODE_BUDGET,
    PROJ_ALWAYS_FULL_PY_SCOPE,
    PROJ_FULL_SCOPE_TOP_N,
)
from .project_py_scope import get_python_symbol_at_line
from .project_refs import find_usages_in_project
from .project_stage_exact import stage_exact_hits
from .project_stage_vector import stage_vector_hits
from .project_stage_keyword import stage_keyword_hits
from .project_stage_textscan import stage_textscan_hits
from .project_snippet import build_snippet

# Tunables moved to project_retrieval_config.py


async def retrieve_project_top_k(query: str, k: int | None = None, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    q = (query or "").strip()
    if not q:
        return []
    k_eff = k or PROJ_DEFAULT_TOP_K
    t0 = time.perf_counter()

    # Stage -1: direct text scan in files (fast path for identifiers not present in previews/terms)
    rem_pre = None if max_time_ms is None else max(1, int(max_time_ms - (time.perf_counter() - t0) * 1000.0))
    txt_hits = stage_textscan_hits(q, k_eff, max_time_ms=rem_pre)
    if txt_hits:
        return txt_hits[:k_eff]

    # Stage 0: exact/identifier substring hits (fast)
    rem0 = None if max_time_ms is None else max(1, int(max_time_ms - (time.perf_counter() - t0) * 1000.0))
    exact = stage_exact_hits(q, k_eff, max_time_ms=rem0)
    if exact:
        return exact[:k_eff]

    # Stage 1: vector similarity (await)
    rem1 = None if max_time_ms is None else max(1, int(max_time_ms - (time.perf_counter() - t0) * 1000.0))
    vec_hits = await stage_vector_hits(q, k_eff, max_time_ms=rem1)
    if vec_hits:
        return vec_hits[:k_eff]

    # Stage 2: keyword/substring fallback
    rem2 = None if max_time_ms is None else max(1, int(max_time_ms - (time.perf_counter() - t0) * 1000.0))
    kw = stage_keyword_hits(q, k_eff, max_time_ms=rem2)
    return kw[:k_eff]


async def build_project_context_for(query: str, *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    k = k or PROJ_DEFAULT_TOP_K
    hits = await retrieve_project_top_k(query, k=k, max_time_ms=max_time_ms)
    if not hits:
        return ""
    # Sort by ts if present to keep chronological order
    hits_sorted = sorted(
        hits,
        key=lambda h: float((h[2].get("meta", {}).get("ts") or 0.0)),
    )
    parts: List[str] = []
    refs_parts: List[str] = []
    seen: set[str] = set()  # dedupe by preview text
    headers_seen: set[str] = set()  # dedupe by [file:ls-le]
    refs_headers_seen: set[str] = set()  # dedupe refs by header
    # Language detection moved to project_lang.py

    budget = PROJ_TOTAL_CODE_BUDGET if (max_chars is None) else max_chars
    total_len = 0

    full_scope_used = 0
    for score, file_rel, obj in hits_sorted:
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
        header, code_block, use_ls, use_le, is_full_scope = build_snippet(
            file_rel, meta, query, max_chars=PROJ_SNIPPET_PER_HIT_CHARS, prefer_full_scope=prefer_full
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

        # Optionally add a couple of usage references for the enclosing symbol (Python only), minimal format
        try:
            # Load file text for symbol resolution
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
                    usages = find_usages_in_project(sym_name, file_rel, limit=2, around=PROJ_SNIPPET_AROUND)
                    for fr, ua, ub, usnip, ulang in usages:
                        hdr = f"[{fr}:{ua}-{ub}]"
                        if hdr in refs_headers_seen:
                            continue
                        refs_headers_seen.add(hdr)
                        block = f"```{ulang}\n{usnip}\n```" if ulang else f"```\n{usnip}\n```"
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
