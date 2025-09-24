from __future__ import annotations

import json
import os
from typing import Dict, Any

from .project_paths import ensure_project_dirs, PROJECT_HASH_DB_PATH
from .project_io import write_json_atomic


def load_hash_db() -> Dict[str, Dict[str, Any]]:
    ensure_project_dirs()
    try:
        if not os.path.exists(PROJECT_HASH_DB_PATH):
            return {}
        with open(PROJECT_HASH_DB_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj  # type: ignore[return-value]
    except Exception:
        pass
    return {}


def save_hash_db(db: Dict[str, Dict[str, Any]]) -> None:
    ensure_project_dirs()
    try:
        write_json_atomic(PROJECT_HASH_DB_PATH, db)
    except Exception:
        # Best-effort: leave previous DB
        pass


def set_record(db: Dict[str, Dict[str, Any]], rel_path: str, *, sha: str, mtime: float) -> None:
    db[rel_path] = {"sha": sha, "mtime": mtime}


def get_record(db: Dict[str, Dict[str, Any]], rel_path: str) -> Dict[str, Any] | None:
    return db.get(rel_path)


def del_record(db: Dict[str, Dict[str, Any]], rel_path: str) -> None:
    if rel_path in db:
        del db[rel_path]
