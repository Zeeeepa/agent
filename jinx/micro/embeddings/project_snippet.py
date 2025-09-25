from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .project_config import ROOT
from .project_retrieval_config import (
    PROJ_SNIPPET_AROUND,
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_SCOPE_MAX_CHARS,
)
from .project_line_window import find_line_window
from .project_identifiers import extract_identifiers
from .project_lang import lang_for_file
from .project_py_scope import find_python_scope, get_python_symbol_at_line
from .project_query_tokens import expand_strong_tokens, codeish_tokens


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
            # Locate by identifiers from query
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
                            # Prefer entire function/class scope; if PROJ_SCOPE_MAX_CHARS <= 0 treat as unlimited
                            if PROJ_SCOPE_MAX_CHARS > 0 and len(scope_text) > PROJ_SCOPE_MAX_CHARS:
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
    if not is_full_scope and len(body) > PROJ_SNIPPET_PER_HIT_CHARS:
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

    lang = lang_for_file(file_rel)
    code_block = f"```{lang}\n{body}\n```" if lang else f"```\n{body}\n```"
    return header, code_block, int(local_ls or 0), int(local_le or 0), bool(is_full_scope)
