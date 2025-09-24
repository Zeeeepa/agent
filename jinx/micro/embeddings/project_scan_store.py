from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Iterator, Tuple, Any, List

from .project_paths import PROJECT_FILES_DIR


def iter_project_chunks(max_files: int = 2000, max_chunks_per_file: int = 500) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (file_rel, chunk_payload) from emb/files structure.

    Scans directories under emb/files/<safe_rel_path>/ and reads *.json payloads.
    Yields up to `max_files` files and up to `max_chunks_per_file` chunks per file.
    Best-effort: skips files on JSON errors.
    """
    if not os.path.isdir(PROJECT_FILES_DIR):
        return iter(())

    count_files = 0
    # Each directory under PROJECT_FILES_DIR corresponds to a single original file (safe_rel_path)
    for d in os.listdir(PROJECT_FILES_DIR):
        dir_path = os.path.join(PROJECT_FILES_DIR, d)
        if not os.path.isdir(dir_path):
            continue
        count_files += 1
        if count_files > max_files:
            break

        # Reconstruct file_rel from chunk payload meta, don't rely on directory name only
        try:
            files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        except FileNotFoundError:
            continue
        files = files[:max_chunks_per_file]
        for fn in files:
            p = os.path.join(dir_path, fn)
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                meta = obj.get('meta', {})
                file_rel = meta.get('file_rel') or ''
                # as a fallback, keep empty; retrieval can still use meta
                yield file_rel, obj
            except Exception:
                continue
