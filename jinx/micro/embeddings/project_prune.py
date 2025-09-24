from __future__ import annotations

import os
import shutil
from typing import Dict

from .project_paths import PROJECT_FILES_DIR, PROJECT_INDEX_DIR, safe_rel_path
from .project_hashdb import del_record


def prune_deleted(root: str, db: Dict[str, Dict[str, object]]) -> bool:
    """Remove artifacts for files that no longer exist. Returns True if db changed."""
    changed = False
    to_delete: list[str] = []
    for rel_p in list(db.keys()):
        abs_p = os.path.join(root, rel_p)
        if not os.path.exists(abs_p):
            to_delete.append(rel_p)
    for rel_p in to_delete:
        changed |= prune_single(root, db, rel_p)
    return changed


def prune_single(root: str, db: Dict[str, Dict[str, object]], rel_p: str) -> bool:
    """Prune artifacts for a single file rel path. Returns True if db changed."""
    abs_p = os.path.join(root, rel_p)
    if os.path.exists(abs_p):
        return False
    safe = safe_rel_path(rel_p)
    file_dir = os.path.join(PROJECT_FILES_DIR, safe)
    index_path = os.path.join(PROJECT_INDEX_DIR, f"{safe}.json")
    try:
        if os.path.isdir(file_dir):
            shutil.rmtree(file_dir, ignore_errors=True)
    except Exception:
        pass
    try:
        if os.path.exists(index_path):
            os.remove(index_path)
    except Exception:
        pass
    del_record(db, rel_p)
    return True
