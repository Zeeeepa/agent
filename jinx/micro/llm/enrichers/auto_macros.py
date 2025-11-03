from __future__ import annotations

import os
from typing import List
import re
from jinx.micro.text.heuristics import is_code_like as _is_code_like


async def auto_context_lines(input_text: str) -> List[str]:
    """Return fallback context macro lines when unified context is unavailable.

    Produces dialogue/project/memory automacros with dynamic K and simple heuristics.
    """
    lines: List[str] = []
    txt = (input_text or "").strip()
    codey = _is_code_like(txt)
    # feature flags
    try:
        use_dlg = str(os.getenv("JINX_AUTOMACRO_DIALOGUE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        use_dlg = True
    try:
        use_proj = str(os.getenv("JINX_AUTOMACRO_PROJECT", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        use_proj = True
    try:
        dlg_k = int(os.getenv("JINX_AUTOMACRO_DIALOGUE_K", "3"))
    except Exception:
        dlg_k = 3
    try:
        proj_k = int(os.getenv("JINX_AUTOMACRO_PROJECT_K", "3"))
    except Exception:
        proj_k = 3
    # Memory automacros
    try:
        use_mem = str(os.getenv("JINX_AUTOMACRO_MEMORY", "1").lower()) not in ("", "0", "false", "off", "no")
    except Exception:
        use_mem = True
    try:
        mem_comp_k = int(os.getenv("JINX_AUTOMACRO_MEM_COMPACT_K", "8"))
    except Exception:
        mem_comp_k = 8
    try:
        mem_ever_k = int(os.getenv("JINX_AUTOMACRO_MEM_EVERGREEN_K", "8"))
    except Exception:
        mem_ever_k = 8

    # Heuristic preference: project for code-like, dialogue for natural text
    if use_dlg:
        if codey and not use_proj:
            lines.append(f"Context (dialogue): {{{{m:emb:dialogue:{dlg_k}}}}}")
        elif not codey:
            lines.append(f"Context (dialogue): {{{{m:emb:dialogue:{dlg_k}}}}}")
    if use_proj:
        if codey or not use_dlg:
            lines.append(f"Context (code): {{{{m:emb:project:{proj_k}}}}}")
    if use_mem:
        # Inject routed memory (pins + graph-aligned + ranker)
        lines.append(f"Memory (routed): {{{{m:memroute:{max(mem_comp_k, mem_ever_k)}}}}}")
    return lines


async def auto_code_lines(input_text: str) -> List[str]:
    """Return code intelligence macro lines (usage/def) inferred from input.

    Gate with JINX_AUTOMACRO_CODE (default ON). Extract the most salient token
    from code-like input or from the first callable pattern in text.
    """
    try:
        on = str(os.getenv("JINX_AUTOMACRO_CODE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        on = True
    if not on:
        return []
    txt = (input_text or "").strip()
    token = ""
    # Prefer last identifier before '(' in a simple assignment/call line
    # e.g., "tk = brain_topk(default_topk)" -> brain_topk
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", txt)
    if m:
        token = m.group(1)
    # fallback: longest identifier-like word
    if not token:
        ids = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", txt)
        ids = sorted(ids, key=len, reverse=True)
        if ids:
            token = ids[0]
    if not token or len(token) < 3:
        return []
    try:
        topk = max(1, int(os.getenv("JINX_MACRO_CODE_TOPK", "8")))
    except Exception:
        topk = 8
    # Build usage + def lines; def is small by default
    lines = [
        f"Code usage: {{{{m:code:usage:{token}:{topk}}}}}",
        f"Code def: {{{{m:code:def:{token}:3}}}}",
    ]
    return lines


__all__ = ["auto_context_lines", "auto_code_lines"]
