from __future__ import annotations

from typing import Dict

from jinx.micro.memory.storage import read_compact, read_evergreen


def _add(node_map: Dict[str, float], key: str, w: float) -> None:
    try:
        node_map[key] = float(node_map.get(key, 0.0)) + float(w)
    except Exception:
        node_map[key] = float(w)


async def scan_memory_concepts() -> Dict[str, float]:
    nodes: Dict[str, float] = {}
    comp, ever = "", ""
    try:
        ever = await read_evergreen()
    except Exception:
        ever = ""
    try:
        comp = await read_compact()
    except Exception:
        comp = ""
    # Evergreen: curated keys
    for raw in (ever or "").splitlines():
        line = (raw or "").strip()
        low = line.lower()
        if low.startswith("path: "):
            _add(nodes, line.lower(), 3.0)
        elif low.startswith("symbol: "):
            _add(nodes, line.lower(), 2.0)
        elif low.startswith("pref: "):
            _add(nodes, line.lower(), 1.0)
        elif low.startswith("decision: "):
            _add(nodes, line.lower(), 1.0)
    # Compact: loose terms
    import re as _re
    for raw in (comp or "").splitlines()[-800:]:
        s = (raw or "").strip().lower()
        for m in _re.finditer(r"(?u)[\w\.]{4,}", s):
            tok = (m.group(0) or "").lower()
            if tok:
                _add(nodes, f"term: {tok}", 0.2)
    return nodes


__all__ = ["scan_memory_concepts"]
