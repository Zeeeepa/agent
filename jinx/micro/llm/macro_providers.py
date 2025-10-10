from __future__ import annotations

import os
import re
from typing import List

from jinx.micro.llm.macro_registry import register_macro, MacroContext
from jinx.micro.embeddings.retrieval import retrieve_top_k as _dlg_topk
from jinx.micro.embeddings.project_retrieval import retrieve_project_top_k as _proj_topk
from jinx.micro.memory.storage import read_compact as _read_compact, read_evergreen as _read_evergreen, read_channel as _read_channel, read_topic as _read_topic
from jinx.micro.memory.search import rank_memory as _rank_memory
from jinx.micro.memory.graph import query_graph as _query_graph
from jinx.micro.memory.pin_store import load_pins as _pins_load, save_pins as _pins_save
from jinx.micro.memory.router import assemble_memroute as _memroute
from jinx.micro.exec.run_exports import read_last_stdout as _run_stdout, read_last_stderr as _run_stderr, read_last_status as _run_status
from jinx.micro.llm.macro_cache import memoized_call
from jinx.micro.memory.turns import parse_active_turns as _parse_turns, get_user_message as _turn_user, get_jinx_reply_to as _turn_jinx

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


async def _memfacts_handler(args: List[str], ctx: MacroContext) -> str:
    """Facts provider: {{m:memfacts:kind[:N]}}

    kind: paths|symbols|prefs|decisions
    N: number of lines (default 8)
    """
    kind = (args[0] if args else "").strip().lower()
    if kind not in ("paths","symbols","prefs","decisions"):
        return ""
    n = 0
    if len(args) > 1:
        try:
            n = int(args[1])
        except Exception:
            n = 0
    if n <= 0:
        try:
            n = max(1, int(os.getenv("JINX_MACRO_MEM_TOPK", "8")))
        except Exception:
            n = 8
    try:
        lim = max(24, int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
    except Exception:
        lim = 160
    try:
        txt = await _read_channel(kind)
    except Exception:
        txt = ""
    if not txt:
        return ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    out = [ln[:lim] for ln in lines[:n]]
    return " | ".join(out)


async def _memgraph_handler(args: List[str], ctx: MacroContext) -> str:
    """Knowledge graph neighbors: {{m:memgraph:term[:K]}}

    term: substring to match node keys (e.g., 'symbol: my_func' or 'path: utils.py') or any token
    K: number of neighbors (default 8)
    """
    term = (args[0] if args else "").strip()
    if not term:
        # fallback to input_text
        term = (ctx.input_text or "").strip()
    n = 0
    if len(args) > 1:
        try:
            n = int(args[1])
        except Exception:
            n = 0
    if n <= 0:
        try:
            n = max(1, int(os.getenv("JINX_MACRO_MEM_TOPK", "8")))
        except Exception:
            n = 8
    if not term:
        return ""
    try:
        items = await _query_graph(term, k=n)
    except Exception:
        items = []
    if not items:
        return ""
    # already formatted as 'key (score)'
    return " | ".join(items[:n])


async def _memtopic_handler(args: List[str], ctx: MacroContext) -> str:
    """Topic memory: {{m:memtopic:name[:N]}}

    Reads from .jinx/memory/topics/<name>.md
    """
    name = (args[0] if args else "").strip().lower()
    if not name:
        return ""
    n = 0
    if len(args) > 1:
        try:
            n = int(args[1])
        except Exception:
            n = 0
    if n <= 0:
        try:
            n = max(1, int(os.getenv("JINX_MACRO_MEM_TOPK", "8")))
        except Exception:
            n = 8
    try:
        lim = max(24, int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
    except Exception:
        lim = 160
    try:
        txt = await _read_topic(name)
    except Exception:
        txt = ""
    if not txt:
        return ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return " | ".join([ln[:lim] for ln in lines[:n]])


async def _memroute_handler(args: List[str], ctx: MacroContext) -> str:
    """Assemble routed memory: {{m:memroute[:K]}} using pins+graph+ranker.

    K default 12, preview via JINX_MACRO_MEM_PREVIEW_CHARS.
    """
    n = 0
    if len(args) > 0:
        try:
            n = int(args[0])
        except Exception:
            n = 0
    if n <= 0:
        try:
            n = max(1, int(os.getenv("JINX_MACRO_MEM_TOPK", "12")))
        except Exception:
            n = 12
    try:
        lim = max(24, int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
    except Exception:
        lim = 160
    q = (ctx.input_text or "").strip()
    if not q:
        try:
            q = (ctx.anchors.get("questions") or [""])[-1].strip()
        except Exception:
            q = ""
    # TTL memoization to avoid recomputation within a short window
    try:
        ttl_ms = int(os.getenv("JINX_MACRO_PROVIDER_TTL_MS", "1500"))
    except Exception:
        ttl_ms = 1500
    key = f"memroute|{n}|{lim}|{q}"
    async def _call() -> str:
        try:
            lines = await _memroute(q, k=n, preview_chars=lim)
        except Exception:
            lines = []
        return " | ".join(lines[:n])
    return await memoized_call(key, ttl_ms, _call)


async def _turns_handler(args: List[str], ctx: MacroContext) -> str:
    """Turns provider: {{m:turns:kind:n[:chars=lim]}}

    kind: user|jinx|pair (default user)
    n: 1-based turn index
    chars: optional clamp (defaults to JINX_MACRO_TURNS_PREVIEW_CHARS or JINX_MACRO_MEM_PREVIEW_CHARS)
    """
    try:
        kind = (args[0] if args else "user").strip().lower()
    except Exception:
        kind = "user"
    n = 0
    clamp = None
    for a in (args[1:] if len(args) > 1 else []):
        aa = (a or "").strip()
        if not aa:
            continue
        if aa.startswith("chars="):
            try:
                clamp = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        try:
            n = int(aa)
        except Exception:
            pass
    if n <= 0:
        return ""
    try:
        lim = int(os.getenv("JINX_MACRO_TURNS_PREVIEW_CHARS", os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
        lim = max(24, lim)
    except Exception:
        lim = 160
    if clamp is not None:
        try:
            lim = max(24, int(clamp))
        except Exception:
            pass
    if kind == "user":
        s = await _turn_user(n)
        return (s or "")[:lim]
    if kind == "jinx":
        s = await _turn_jinx(n)
        return (s or "")[:lim]
    if kind == "pair":
        turns = await _parse_turns()
        if n <= 0 or n > len(turns):
            return ""
        t = turns[n-1]
        u = (t.get("user") or "").strip()
        a = (t.get("jinx") or "").strip()
        out = (f"User: {u}\nJinx: {a}").strip()
        return out[:lim]
    return ""


async def _run_handler(args: List[str], ctx: MacroContext) -> str:
    """Last run artifacts: {{m:run:kind[:N][:ttl=ms][:chars=lim]}}

    kind: stdout|stderr|status (default stdout)
    N: number of tail lines for stdout/stderr (default 3)
    ttl: freshness window in ms (default JINX_RUN_EXPORT_TTL_MS or 120000)
    chars: preview clamp (default JINX_MACRO_MEM_PREVIEW_CHARS or 160)
    """
    kind = (args[0] if args else "stdout").strip().lower()
    n = 0
    ttl_ms = None
    lim = None
    # parse optional numeric and kv args
    for a in (args[1:] if len(args) > 1 else []):
        aa = a.strip()
        if not aa:
            continue
        if aa.startswith("ttl="):
            try:
                ttl_ms = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        if aa.startswith("chars="):
            try:
                lim = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        try:
            n = int(aa)
        except Exception:
            pass
    if n <= 0:
        n = 3 if kind in ("stdout","stderr") else 1
    if ttl_ms is None:
        try:
            ttl_ms = int(os.getenv("JINX_RUN_EXPORT_TTL_MS", "120000"))
        except Exception:
            ttl_ms = 120000
    if lim is None:
        try:
            lim = max(24, int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
        except Exception:
            lim = 160
    # TTL memoization across identical macro invocations
    try:
        pttl = int(os.getenv("JINX_MACRO_PROVIDER_TTL_MS", "1500"))
    except Exception:
        pttl = 1500
    key = f"run|{kind}|{n}|{ttl_ms}|{lim}"
    async def _call() -> str:
        if kind == "stdout":
            return _run_stdout(n, lim, ttl_ms)
        if kind == "stderr":
            return _run_stderr(n, lim, ttl_ms)
        if kind == "status":
            return _run_status(ttl_ms)
        return ""
    return await memoized_call(key, pttl, _call)


def _pins_enabled() -> bool:
    try:
        return str(os.getenv("JINX_MEM_PINS_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


async def _pins_handler(args: List[str], ctx: MacroContext) -> str:
    """List pinned lines: {{m:pins[:N]}}"""
    n = 0
    if len(args) > 0:
        try:
            n = int(args[0])
        except Exception:
            n = 0
    if n <= 0:
        try:
            n = max(1, int(os.getenv("JINX_MACRO_MEM_TOPK", "8")))
        except Exception:
            n = 8
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    out = [p for p in pins[:n] if p]
    return " | ".join(out)


async def _pinadd_handler(args: List[str], ctx: MacroContext) -> str:
    """Add a pinned line: {{m:pinadd:line...}} (uses input_text if empty)."""
    if not _pins_enabled():
        return ""
    line = " ".join(args).strip()
    if not line:
        line = (ctx.input_text or "").strip()
    if not line:
        return ""
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    if line not in pins:
        pins.insert(0, line)
        try:
            _pins_save(pins)
        except Exception:
            pass
    return line


async def _pindel_handler(args: List[str], ctx: MacroContext) -> str:
    """Delete a pinned line by exact match: {{m:pindel:line...}}"""
    if not _pins_enabled():
        return ""
    line = " ".join(args).strip()
    if not line:
        return ""
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    pins = [p for p in pins if p != line]
    try:
        _pins_save(pins)
    except Exception:
        pass
    return ""


def _tokens(s: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"(?u)[\w\.]+", s or ""):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 3:
            out.append(t)
    return out


async def _mem_handler(args: List[str], ctx: MacroContext) -> str:
    """Memory provider: {{m:mem:scope[:N][:q=...]}}

    scope: compact|evergreen|any (default: compact)
    N: number of snippets (default: JINX_MACRO_MEM_TOPK or 6)
    q=: optional query to filter/select relevant lines
    """
    scope = (args[0] if args else "compact").strip().lower()
    n = 0
    q = ""
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
            n = max(1, int(os.getenv("JINX_MACRO_MEM_TOPK", "6")))
        except Exception:
            n = 6
    try:
        lim = max(24, int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
    except Exception:
        lim = 160

    # Load memory texts
    try:
        comp = await _read_compact()
    except Exception:
        comp = ""
    try:
        ever = await _read_evergreen()
    except Exception:
        ever = ""

    def _lines_of(txt: str) -> List[str]:
        return [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]

    c_lines = _lines_of(comp)
    e_lines = _lines_of(ever)
    # Build candidate pool
    if scope == "evergreen":
        pool = e_lines
    elif scope == "any":
        pool = e_lines + c_lines
    else:
        pool = c_lines

    if not pool:
        return ""

    # Selection
    out: List[str] = []
    if q:
        # Use async ranker for better relevance
        ranked = await _rank_memory(q, scope=scope if scope in ("compact","evergreen","any") else "compact", k=n, preview_chars=lim)
        out.extend(ranked)
    else:
        # No query: take most recent for compact, otherwise head for evergreen
        if scope == "evergreen":
            for ln in e_lines[-n:]:
                out.append(ln[:lim])
        elif scope == "any":
            # interleave last few from both (favor compact recency)
            tail_c = c_lines[-(n*2):]
            tail_e = e_lines[-n:]
            merged = (tail_c + tail_e)[-n:]
            out.extend([ln[:lim] for ln in merged])
        else:
            for ln in c_lines[-n:]:
                out.append(ln[:lim])

    out = [s for s in out if s]
    return " | ".join(out[:n])


async def register_builtin_macros() -> None:
    global _registered
    if _registered:
        return
    await register_macro("emb", _emb_handler)
    await register_macro("mem", _mem_handler)
    await register_macro("memfacts", _memfacts_handler)
    await register_macro("memgraph", _memgraph_handler)
    await register_macro("memtopic", _memtopic_handler)
    await register_macro("memroute", _memroute_handler)
    await register_macro("turns", _turns_handler)
    await register_macro("run", _run_handler)
    await register_macro("pins", _pins_handler)
    await register_macro("pinadd", _pinadd_handler)
    await register_macro("pindel", _pindel_handler)
    _registered = True
