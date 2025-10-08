from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
import re

from .project_config import ROOT
from .project_retrieval_config import (
    PROJ_SNIPPET_AROUND,
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_SCOPE_MAX_CHARS,
    PROJ_EXPAND_CALLEES_TOP_N,
    PROJ_EXPAND_CALLEE_MAX_CHARS,
    PROJ_MULTI_SEGMENT_ENABLE,
    PROJ_SEGMENT_HEAD_LINES,
    PROJ_SEGMENT_TAIL_LINES,
    PROJ_SEGMENT_MID_WINDOWS,
    PROJ_SEGMENT_MID_AROUND,
    PROJ_SEGMENT_STRIP_COMMENTS,
)
from .project_line_window import find_line_window
from .project_identifiers import extract_identifiers
from .project_lang import lang_for_file
from .project_py_scope import find_python_scope, get_python_symbol_at_line
from .project_callees import extract_callees_from_scope, find_def_scope_in_project
from .project_query_tokens import expand_strong_tokens, codeish_tokens
from .project_query_core import extract_code_core
from .snippet_segments import build_multi_segment_python
from .snippet_cache import (
    make_snippet_cache_key,
    get_cached_snippet,
    put_cached_snippet,
    file_signature,
    coalesce_enter,
    coalesce_exit,
    coalesce_wait_ms,
)


def _read_file(rel_path: str) -> str:
    try:
        abs_path = os.path.join(ROOT, rel_path)
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


def build_snippet(
    file_rel: str,
    meta: Dict[str, Any],
    query: str,
    *,
    max_chars: int,
    prefer_full_scope: bool = True,
    expand_callees: bool = True,
    extra_centers_abs: List[int] | None = None,
) -> Tuple[str, str, int, int, bool]:
    """Build a minimal header + code block snippet for a hit.

    Returns (header, code_block, ls, le, is_full_scope), where header is like "[file:ls-le]".
    """
    pv = (meta.get("text_preview") or "").strip()
    ls = int(meta.get("line_start") or 0)
    le = int(meta.get("line_end") or 0)
    local_ls = ls
    local_le = le
    is_full_scope = False
    did_segment = False

    # Snippet TTL cache (default-on)
    _sig0 = None
    try:
        _sig0 = file_signature(file_rel)
    except Exception:
        _sig0 = None
    _cache_key = make_snippet_cache_key(
        file_rel,
        meta,
        query,
        prefer_full_scope=prefer_full_scope,
        expand_callees=expand_callees,
        extra_centers_abs=extra_centers_abs,
        file_sig=_sig0,
    )
    try:
        _cached = get_cached_snippet(_cache_key)
    except Exception:
        _cached = None
    if _cached:
        return _cached

    # Coalesce concurrent builds for the same key
    _leader = False
    _wait_ev = None
    try:
        mode, ev = coalesce_enter(_cache_key)
        _leader = (mode == "leader")
        _wait_ev = ev
    except Exception:
        _leader = False
        _wait_ev = None
    if (not _leader) and _wait_ev is not None:
        try:
            _wait_ev.wait(max(0.0, float(coalesce_wait_ms()) / 1000.0))
        except Exception:
            pass
        try:
            _cached2 = get_cached_snippet(_cache_key)
        except Exception:
            _cached2 = None
        if _cached2:
            return _cached2

    header = f"[{file_rel}:{ls}-{le}]" if (ls or le) else f"[{file_rel}]"
    body = pv

    file_text = _read_file(file_rel)
    if file_text:
        lines_all = file_text.splitlines()
        # Token helpers provided by micro-module
        # If meta already points to the entire file, honor it and skip shaping
        if (ls == 1 and le == len(lines_all)):
            body = file_text
            local_ls, local_le = 1, len(lines_all)
            is_full_scope = True
        elif ls or le:
            a = max(1, ls)
            b = le if le > 0 else a
            a_i = min(len(lines_all), a) - 1
            b_i = min(len(lines_all), b) - 1
            if a_i <= b_i:
                span = "\n".join(lines_all[a_i:b_i+1]).strip()
                if span:
                    body = span
        else:
            # Prefer an exact window by code-core if present
            core = extract_code_core(query or "")
            a = b = 0
            snip = ""
            if core:
                try:
                    # Build flexible regex: escape and normalize spaces
                    esc = re.escape(core)
                    esc = esc.replace(r"\ ", r"\s+")
                    pat = re.compile(esc, re.DOTALL)
                    m = pat.search(file_text)
                    if m:
                        pre = file_text[: m.start()]
                        ls = pre.count("\n") + 1
                        le = ls + max(1, file_text[m.start(): m.end()].count("\n"))
                        lines_all = file_text.splitlines()
                        a = max(1, ls - PROJ_SNIPPET_AROUND)
                        b = min(len(lines_all), le + PROJ_SNIPPET_AROUND)
                        snip = "\n".join(lines_all[a-1:b]).strip()
                except Exception:
                    pass
            if not (a or b):
                # Fallback: locate by identifiers from query
                q_toks = sorted(extract_identifiers(query or "", max_items=24), key=len, reverse=True)
                a, b, snip = find_line_window(file_text, q_toks, around=PROJ_SNIPPET_AROUND)
            if a or b:
                body = snip or body
                local_ls, local_le = a, b

        # Prefer full Python scope if it fits budget
        use_ls = local_ls or ls
        use_le = local_le or le
        if file_rel.endswith('.py') and (use_ls or use_le) and not (local_ls == 1 and local_le == len(lines_all)):
            try:
                cand_line = int((use_ls + use_le) // 2) if (use_ls and use_le) else int(use_ls or use_le or 0)
                s_scope, e_scope = find_python_scope(file_text, cand_line)
                if s_scope and e_scope:
                    s_idx = max(1, s_scope) - 1
                    e_idx = min(len(lines_all), e_scope) - 1
                    scope_text = "\n".join(lines_all[s_idx:e_idx+1]).strip()
                    if scope_text:
                        if prefer_full_scope:
                            # Prefer entire function/class scope; if too large, use multi-segment composite
                            too_large_scope = (PROJ_SCOPE_MAX_CHARS > 0 and len(scope_text) > PROJ_SCOPE_MAX_CHARS) or (len(scope_text) > PROJ_SNIPPET_PER_HIT_CHARS)
                            if too_large_scope and PROJ_MULTI_SEGMENT_ENABLE:
                                body = build_multi_segment_python(
                                    lines_all,
                                    s_scope,
                                    e_scope,
                                    query,
                                    per_hit_chars=PROJ_SNIPPET_PER_HIT_CHARS,
                                    head_lines=PROJ_SEGMENT_HEAD_LINES,
                                    tail_lines=PROJ_SEGMENT_TAIL_LINES,
                                    mid_windows=PROJ_SEGMENT_MID_WINDOWS,
                                    mid_around=PROJ_SEGMENT_MID_AROUND,
                                    strip_comments=PROJ_SEGMENT_STRIP_COMMENTS,
                                    extra_centers=extra_centers_abs,
                                )
                                did_segment = True
                                is_full_scope = False
                            elif PROJ_SCOPE_MAX_CHARS > 0 and len(scope_text) > PROJ_SCOPE_MAX_CHARS:
                                body = scope_text[:PROJ_SCOPE_MAX_CHARS]
                                is_full_scope = False
                            else:
                                body = scope_text
                                is_full_scope = True
                            local_ls, local_le = s_scope, e_scope
                        elif len(scope_text) <= PROJ_SNIPPET_PER_HIT_CHARS:
                            body = scope_text
                            local_ls, local_le = s_scope, e_scope
                            is_full_scope = True
                        else:
                            # Window around candidate line
                            c = max(1, cand_line)
                            a = max(1, c - PROJ_SNIPPET_AROUND)
                            b = min(len(lines_all), c + PROJ_SNIPPET_AROUND)
                            body = "\n".join(lines_all[a-1:b]).strip() or body
                            local_ls, local_le = a, b
            except Exception:
                pass

    # Final cap per hit (skip if we intentionally included full scope under policy)
    if (not is_full_scope) and (not did_segment) and len(body) > PROJ_SNIPPET_PER_HIT_CHARS:
        body = body[:PROJ_SNIPPET_PER_HIT_CHARS]

    # Final header (optionally enriched with Python symbol name/kind)
    if local_ls or local_le:
        header = f"[{file_rel}:{local_ls}-{local_le}]"
    else:
        header = f"[{file_rel}]"

    # Enrich with Python symbol info if available
    try:
        if file_rel.endswith('.py') and file_text:
            cand_line = int((local_ls + local_le) // 2) if (local_ls and local_le) else int(local_ls or local_le or 0)
            sym_name, sym_kind = get_python_symbol_at_line(file_text, cand_line)
            if sym_name:
                header = f"[{file_rel}:{local_ls}-{local_le} {sym_kind or ''} {sym_name}]".rstrip()
    except Exception:
        pass

    # Optional: expand a couple of direct callees for Python full-scope snippets
    lang = lang_for_file(file_rel)
    final_body = body
    try:
        if expand_callees and file_rel.endswith('.py') and PROJ_EXPAND_CALLEES_TOP_N > 0 and (is_full_scope or did_segment):
            callees = extract_callees_from_scope(body, max_items=PROJ_EXPAND_CALLEES_TOP_N * 2)
            appended: list[str] = []
            used = 0
            for nm in callees:
                if used >= PROJ_EXPAND_CALLEES_TOP_N:
                    break
                defs = find_def_scope_in_project(nm, prefer_rel=file_rel, limit=1)
                if not defs:
                    continue
                fr, s, e = defs[0]
                try:
                    abs_p = os.path.join(ROOT, fr)
                    with open(abs_p, 'r', encoding='utf-8', errors='ignore') as _cf:
                        src = _cf.read()
                    lines_all = src.splitlines()
                    s_i = max(1, s) - 1
                    e_i = min(len(lines_all), e) - 1
                    seg = "\n".join(lines_all[s_i:e_i+1]).strip()
                    if not seg:
                        continue
                    if PROJ_EXPAND_CALLEE_MAX_CHARS > 0 and len(seg) > PROJ_EXPAND_CALLEE_MAX_CHARS:
                        seg = seg[:PROJ_EXPAND_CALLEE_MAX_CHARS]
                    header_line = f"# callee: {nm} [{fr}:{s}-{e}]"
                    appended.append(f"{header_line}\n{seg}")
                    used += 1
                except Exception:
                    continue
            if appended:
                final_body = f"{body}\n\n# ---- expanded callees ----\n" + "\n\n".join(appended)
    except Exception:
        pass

    code_block = f"```{lang}\n{final_body}\n```" if lang else f"```\n{final_body}\n```"
    _result = (header, code_block, int(local_ls or 0), int(local_le or 0), bool(is_full_scope))
    try:
        _sig1 = file_signature(file_rel)
        if (_sig0 is not None) and (_sig1 == _sig0):
            put_cached_snippet(_cache_key, _result)
    except Exception:
        pass
    finally:
        if _leader:
            try:
                coalesce_exit(_cache_key)
            except Exception:
                pass
    return _result
