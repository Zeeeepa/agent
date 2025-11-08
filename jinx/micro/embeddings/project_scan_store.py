from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Iterator, Tuple, Any, List

from .project_paths import PROJECT_FILES_DIR, PROJECT_INDEX_DIR


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
        # Try to read per-file index to recover file_rel for legacy chunks missing meta.file_rel
        index_file_rel = ""
        try:
            # Prefer JSON (current format)
            idx_json = os.path.join(PROJECT_INDEX_DIR, f"{d}.json")
            if os.path.isfile(idx_json):
                with open(idx_json, 'r', encoding='utf-8') as f:
                    _idx = json.load(f)
                    index_file_rel = str((_idx or {}).get('file_rel') or '')
            # Fallback: legacy JSONL (use last non-empty line)
            if not index_file_rel:
                idx_jsonl = os.path.join(PROJECT_INDEX_DIR, f"{d}.jsonl")
                if os.path.isfile(idx_jsonl):
                    with open(idx_jsonl, 'r', encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    if lines:
                        try:
                            _idxl = json.loads(lines[-1])
                            index_file_rel = str((_idxl or {}).get('file_rel') or '')
                        except Exception:
                            index_file_rel = index_file_rel or ''
        except Exception:
            index_file_rel = ""
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
                file_rel = meta.get('file_rel') or index_file_rel or ''
                # as a fallback, keep empty; retrieval can still use meta
                yield file_rel, obj
            except Exception:
                continue
