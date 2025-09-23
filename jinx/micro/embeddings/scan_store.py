from __future__ import annotations

import json
import os
from typing import List, Tuple, Dict, Any


def iter_items(root: str, max_files_per_source: int, max_sources: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Yield (source, payload) pairs from the on-disk embeddings store.

    - Scans per-source directories under `root`, skipping the special 'index' dir.
    - Limits number of sources and files per source to bound I/O.
    - Swallows JSON and file errors to preserve best-effort semantics.
    """
    items: List[Tuple[str, Dict[str, Any]]] = []
    if not os.path.isdir(root):
        return items
    sources = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != "index"]
    sources = sources[:max_sources]
    for src in sources:
        src_dir = os.path.join(root, src)
        try:
            files = [f for f in os.listdir(src_dir) if f.endswith(".json")]
        except FileNotFoundError:
            continue
        files = files[:max_files_per_source]
        for fn in files:
            p = os.path.join(src_dir, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    items.append((src, obj))
            except Exception:
                continue
    return items
