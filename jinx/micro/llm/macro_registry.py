from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

try:
    from jinx.logger.file_logger import append_line as _append
    from jinx.log_paths import BLUE_WHISPERS
except Exception:  # pragma: no cover
    _append = None
    BLUE_WHISPERS = ""


@dataclass
class MacroContext:
    key: str
    anchors: Dict[str, List[str]]
    programs: List[str]
    os_name: str
    py_ver: str
    cwd: str
    now_iso: str
    now_epoch: str
    input_text: str = ""

    def env(self, name: str, default: str = "") -> str:
        return os.getenv(name, default)


_Handler = Callable[[List[str], MacroContext], Awaitable[str]]

_REGISTRY: Dict[str, _Handler] = {}
_LOCK = asyncio.Lock()
_GEN_RE = re.compile(r"\{\{m:([a-zA-Z0-9_]+)((?::[^{}:\s]+)*)\}\}")


async def register_macro(namespace: str, handler: _Handler) -> None:
    ns = (namespace or "").strip().lower()
    if not ns:
        return
    async with _LOCK:
        _REGISTRY[ns] = handler


async def list_namespaces() -> List[str]:
    async with _LOCK:
        return sorted(_REGISTRY.keys())


async def expand_dynamic_macros(text: str, ctx: MacroContext, *, max_expansions: int = 50) -> str:
    if not text or not isinstance(text, str):
        return text
    # fast path
    if "{{m:" not in text:
        return text
    out = []
    pos = 0
    expands = 0
    while True:
        m = _GEN_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos:m.start()])
        ns = m.group(1).strip().lower()
        args_blob = m.group(2) or ""
        args = [a for a in args_blob.split(":") if a]
        val = ""
        try:
            async with _LOCK:
                h = _REGISTRY.get(ns)
            if h:
                val = await h(args, ctx)
        except Exception as e:
            # do not propagate provider errors
            val = ""
            if _append and os.getenv("JINX_PROMPT_MACRO_TRACE", "").lower() not in ("", "0", "false", "off", "no"):
                try:
                    await _append(BLUE_WHISPERS, f"[MACRO:{ns}] error: {e}")
                except Exception:
                    pass
        out.append(val or "")
        pos = m.end()
        expands += 1
        if expands >= max_expansions:
            out.append(text[pos:])
            break
    return "".join(out)
