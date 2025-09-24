from __future__ import annotations

import os
import hashlib

# Root directory for project code embeddings (separate from log/embeddings)
PROJECT_EMBED_ROOT = os.path.join("emb")
# Per-file chunks will be stored under: emb/files/<safe_rel_path>/*.json
PROJECT_FILES_DIR = os.path.join(PROJECT_EMBED_ROOT, "files")
# Per-file index lines: emb/index/<safe_rel_path>.jsonl (overwritten on update)
PROJECT_INDEX_DIR = os.path.join(PROJECT_EMBED_ROOT, "index")
# Internal state (hash db, etc.)
PROJECT_STATE_DIR = os.path.join(PROJECT_EMBED_ROOT, "_state")
PROJECT_HASH_DB_PATH = os.path.join(PROJECT_STATE_DIR, "hashes.json")


def ensure_project_dirs() -> None:
    os.makedirs(PROJECT_EMBED_ROOT, exist_ok=True)
    os.makedirs(PROJECT_FILES_DIR, exist_ok=True)
    os.makedirs(PROJECT_INDEX_DIR, exist_ok=True)
    os.makedirs(PROJECT_STATE_DIR, exist_ok=True)


def safe_rel_path(rel_path: str) -> str:
    """Make a relative path safe for use as a single directory name.

    We flatten the path by replacing both os.sep and '/' with '__'.
    """
    rel_path = rel_path.strip().lstrip("./\\")
    flattened = rel_path.replace(os.sep, "__").replace("/", "__")
    # Add a short stable hash prefix to avoid collisions
    h = hashlib.sha1(rel_path.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{h}__{flattened}"
