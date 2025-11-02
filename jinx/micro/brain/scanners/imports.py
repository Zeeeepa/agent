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


async def scan_import_graph() -> Dict[str, float]:
    nodes: Dict[str, float] = {}
    try:
        entries = [os.path.join(PROJECT_INDEX_DIR, f) for f in os.listdir(PROJECT_INDEX_DIR) if f.endswith('.json')]
    except Exception:
        entries = []
    cnt = 0
    for p in entries:
        cnt += 1
        if cnt > 400:
            break
        try:
            obj = await _read_json(p)
            imports = obj.get('imports') or []
            if isinstance(imports, dict):
                imports = list(imports.keys())
            imp_list = list(imports)[:64]
            for imp in imp_list:
                try:
                    mod = str(imp or '').strip().split('.')[0].lower()
                    if mod:
                        nodes[f"import: {mod}"] = nodes.get(f"import: {mod}", 0.0) + 0.5
                except Exception:
                    continue
        except Exception:
            continue
    return nodes


__all__ = ["scan_import_graph"]
