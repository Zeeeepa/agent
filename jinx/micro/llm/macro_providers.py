from __future__ import annotations

import os
from typing import List

from jinx.micro.llm.macro_registry import register_macro, MacroContext
from jinx.micro.embeddings.retrieval import retrieve_top_k as _dlg_topk
from jinx.micro.embeddings.project_retrieval import retrieve_project_top_k as _proj_topk

_registered = False


def _norm_preview(x: str, lim: int) -> str:
    s = " ".join((x or "").split())
    return s[:lim]


async def _emb_handler(args: List[str], ctx: MacroContext) -> str:
    try:
        scope = (args[0] if args else "dialogue").strip().lower()
    except Exception:
        scope = "dialogue"
    n = 0
    q = ""
    # parse args like [scope, N, q=...]
    for a in (args[1:] if len(args) > 1 else []):
        aa = a.strip()
        if aa.startswith("q="):
            q = aa[2:]
            continue
        try:
            n = int(aa)
        except Exception:
            pass
    if n <= 0:
        try:
            n = max(1, int(os.getenv("JINX_MACRO_EMB_TOPK", "3")))
        except Exception:
            n = 3
    if not q:
        q = (ctx.input_text or "").strip()
    if not q:
        # fallback: last question anchor
        try:
            q = (ctx.anchors.get("questions") or [""])[-1].strip()
        except Exception:
            q = ""
    if not q:
        return ""
    try:
        ms = max(50, int(os.getenv("JINX_MACRO_EMB_MS", "180")))
    except Exception:
        ms = 180
    try:
        lim = max(24, int(os.getenv("JINX_MACRO_EMB_PREVIEW_CHARS", "160")))
    except Exception:
        lim = 160

    out: List[str] = []
    if scope in ("dialogue", "dlg"):
        hits = await _dlg_topk(q, k=n, max_time_ms=ms)
        for _score, _src, obj in hits:
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or "").strip()
            if not pv:
                continue
            out.append(_norm_preview(pv, lim))
    elif scope in ("project", "proj"):
        hits = await _proj_topk(q, k=n, max_time_ms=ms)
        for _score, file_rel, obj in hits:
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or "").strip()
            if pv:
                out.append(_norm_preview(pv, lim))
                continue
            ls = int(meta.get("line_start") or 0)
            le = int(meta.get("line_end") or 0)
            if file_rel:
                if ls or le:
                    out.append(f"[{file_rel}:{ls}-{le}]")
                else:
                    out.append(f"[{file_rel}]")
    else:
        return ""

    # Compact single-line result for inline prompt usage
    out = [s for s in out if s]
    return " | ".join(out[:n])


async def register_builtin_macros() -> None:
    global _registered
    if _registered:
        return
    await register_macro("emb", _emb_handler)
    _registered = True
