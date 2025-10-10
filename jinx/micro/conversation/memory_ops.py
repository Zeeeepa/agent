from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from jinx.state import shard_lock
from jinx.async_utils.fs import write_text, read_text_raw
from jinx.micro.memory.storage import memory_dir, ensure_nl
from jinx.micro.memory.pin_store import load_pins as _pins_load, save_pins as _pins_save
from jinx.micro.memory.router import assemble_memroute as _memroute

_TOPICS_DIR = os.path.join(memory_dir(), "topics")
_CH_PATHS = os.path.join(memory_dir(), "paths.md")
_CH_SYMBOLS = os.path.join(memory_dir(), "symbols.md")
_CH_PREFS = os.path.join(memory_dir(), "prefs.md")
_CH_DECS = os.path.join(memory_dir(), "decisions.md")


def _safe_topic_name(name: str) -> str:
    n = (name or "").strip().lower()
    import re
    n = re.sub(r"[^a-z0-9_\-\.]+", "_", n)
    if not n:
        n = "topic"
    if not n.endswith(".md"):
        n = n + ".md"
    return n


async def _append_lines(path: str, lines: List[str]) -> None:
    if not lines:
        return
    body = "\n".join([ln.rstrip("\n") for ln in lines if (ln or "").strip()])
    if not body:
        return
    async with shard_lock:
        try:
            prev = await read_text_raw(path) if os.path.exists(path) else ""
        except Exception:
            prev = ""
        try:
            await write_text(path, ensure_nl((prev or "") + ensure_nl(body)))
        except Exception:
            pass


async def exec_ops(ops: List[Dict[str, Any]]) -> Dict[str, str]:
    """Execute memory ops. Returns blocks to inject: {'memory_selected','pins'} if any.

    Supported ops:
    - {action:'memroute', params:{query:str, k:int}}
    - {action:'pins', params:{op:'add'|'remove'|'list', line?:str}}
    - {action:'write_topic', params:{name:str, lines:List[str], mode:'append'|'replace'}}
    - {action:'append_channel', params:{kind:'paths'|'symbols'|'prefs'|'decisions', lines:List[str]}}
    """
    out_blocks: Dict[str, str] = {}
    if not ops:
        return out_blocks

    for op in ops:
        try:
            action = str(op.get("action") or "").strip().lower()
            params = op.get("params") or {}
        except Exception:
            continue
        if action == "memroute":
            q = str(params.get("query") or "")
            try:
                k = int(params.get("k") or 8)
            except Exception:
                k = 8
            k = max(1, min(16, k))
            try:
                pv = int(os.getenv("JINX_MEMSEL_PREVIEW_CHARS", os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
                pv = max(24, pv)
            except Exception:
                pv = 160
            lines = await _memroute(q, k=k, preview_chars=pv)
            body = "\n".join([ln for ln in (lines or [])[:k] if ln])
            if body:
                try:
                    cap = int(os.getenv("JINX_MEMSEL_MAX_CHARS", "1200"))
                except Exception:
                    cap = 1200
                if len(body) > cap:
                    body = body[:cap]
                out_blocks["memory_selected"] = f"<memory_selected>\n{body}\n</memory_selected>"
        elif action == "pins":
            opx = str(params.get("op") or "list").strip().lower()
            if opx == "list":
                pins = _pins_load()
                if pins:
                    out_blocks["pins"] = "<pins>\n" + "\n".join(pins[:8]) + "\n</pins>"
            elif opx in ("add","remove"):
                line = (params.get("line") or "").strip()
                if not line:
                    continue
                pins = _pins_load()
                if opx == "add":
                    if line not in pins:
                        pins.insert(0, line)
                else:
                    pins = [p for p in pins if p != line]
                _pins_save(pins)
        elif action == "write_topic":
            name = _safe_topic_name(str(params.get("name") or ""))
            mode = str(params.get("mode") or "append").strip().lower()
            lines = [str(x) for x in (params.get("lines") or []) if str(x).strip()]
            try:
                os.makedirs(_TOPICS_DIR, exist_ok=True)
            except Exception:
                pass
            path = os.path.join(_TOPICS_DIR, name)
            if mode == "replace":
                body = "\n".join(lines)
                async with shard_lock:
                    try:
                        await write_text(path, ensure_nl(body))
                    except Exception:
                        pass
            else:
                await _append_lines(path, lines)
        elif action == "append_channel":
            kind = str(params.get("kind") or "").strip().lower()
            if kind not in ("paths","symbols","prefs","decisions"):
                continue
            lines = [str(x) for x in (params.get("lines") or []) if str(x).strip()]
            p = {"paths": _CH_PATHS, "symbols": _CH_SYMBOLS, "prefs": _CH_PREFS, "decisions": _CH_DECS}[kind]
            await _append_lines(p, lines)
        else:
            continue
    return out_blocks


__all__ = ["exec_ops"]
