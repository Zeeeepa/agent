from __future__ import annotations

import os
import asyncio
from typing import Dict, Any

from jinx.micro.embeddings.project_paths import PROJECT_INDEX_DIR


async def _read_json(path: str) -> Dict[str, Any]:
    try:
        def _load() -> Dict[str, Any]:
            import json
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
        return await asyncio.to_thread(_load)
    except Exception:
        return {}


def _add(node_map: Dict[str, float], key: str, w: float) -> None:
    try:
        node_map[key] = float(node_map.get(key, 0.0)) + float(w)
    except Exception:
        node_map[key] = float(w)


async def scan_project_concepts() -> Dict[str, float]:
    """Extract concepts from project indexes: terms and path markers.

    Bounded reading under RT: cap number of files, chunks, and terms.
    """
    nodes: Dict[str, float] = {}
    try:
        entries = [os.path.join(PROJECT_INDEX_DIR, f) for f in os.listdir(PROJECT_INDEX_DIR) if f.endswith(".json")]
    except Exception:
        entries = []
    for p in entries[:800]:
        try:
            obj = await _read_json(p)
            chunks = obj.get("chunks") or []
            for ch in chunks[:128]:
                try:
                    terms = ch.get("terms") or []
                    for t in terms[:96]:
                        _add(nodes, f"term: {str(t).lower()}", 1.0)
                    pr = (ch.get("text_preview") or "").strip().lower()
                    if pr:
                        import re as _re
                        for m in _re.finditer(r"(?u)[\w\.]{3,}", pr):
                            tok = (m.group(0) or "").lower()
                            if tok and len(tok) >= 3:
                                _add(nodes, f"term: {tok}", 0.2)
                except Exception:
                    continue
            file_rel = (obj.get("file_rel") or "").strip().lower()
            if file_rel:
                _add(nodes, f"path: {file_rel}", 2.0)
        except Exception:
            continue
    return nodes


__all__ = ["scan_project_concepts"]
