from __future__ import annotations

import os
from .project_paths import PROJECT_INDEX_DIR, PROJECT_FILES_DIR, safe_rel_path


def _has_any_chunk_file(safe_rel: str) -> bool:
    d = os.path.join(PROJECT_FILES_DIR, safe_rel)
    try:
        if not os.path.isdir(d):
            return False
        for fn in os.listdir(d):
            if fn.endswith('.json'):
                return True
        return False
    except Exception:
        return False


def artifacts_exist_for_rel(rel_path: str) -> bool:
    """Return True if both index exists and at least one chunk file exists."""
    safe = safe_rel_path(rel_path)
    index_path = os.path.join(PROJECT_INDEX_DIR, f"{safe}.json")
    if not os.path.exists(index_path):
        return False
    return _has_any_chunk_file(safe)
