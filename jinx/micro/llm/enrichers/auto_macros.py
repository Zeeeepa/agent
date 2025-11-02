from __future__ import annotations

import os
from typing import List
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


__all__ = ["auto_context_lines"]
