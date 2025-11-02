from __future__ import annotations

import os
import asyncio
import re as _re
from typing import Dict

from jinx.log_paths import TRIGGER_ECHOES, BLUE_WHISPERS


async def _tail_file(path: str, max_chars: int = 40000) -> str:
    try:
        def _read() -> str:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    t = f.read()
                    return t[-max_chars:]
            except Exception:
                return ''
        return await asyncio.to_thread(_read)
    except Exception:
        return ''


async def scan_error_classes() -> Dict[str, float]:
    nodes: Dict[str, float] = {}
    for fp in (TRIGGER_ECHOES, BLUE_WHISPERS):
        try:
            if not fp or not os.path.exists(fp):
                continue
            tail = await _tail_file(fp)
            if not tail:
                continue
            for m in _re.finditer(r"(?m)\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b", tail):
                name = (m.group(1) or '').strip()
                if name:
                    nodes[f"error: {name}"] = nodes.get(f"error: {name}", 0.0) + 1.5
        except Exception:
            continue
    return nodes


__all__ = ["scan_error_classes"]
