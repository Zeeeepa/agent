from __future__ import annotations

import asyncio
import os
from typing import Dict

from .project_paths import PROJECT_INDEX_DIR, safe_rel_path
from .project_hashdb import set_record, get_record
from .project_util import sha256_path
from .project_pipeline import embed_file
from .project_artifacts import artifacts_exist_for_rel


async def embed_if_changed(
    db: Dict[str, Dict[str, object]],
    abs_p: str,
    rel_p: str,
    *,
    sem: asyncio.Semaphore,
) -> bool:
    """Embed a file if it's changed or artifacts are missing.

    Returns True if the hash DB was mutated (embedded or metadata updated).
    """
    try:
        st = os.stat(abs_p)
    except FileNotFoundError:
        return False
    mtime = float(st.st_mtime)
    rec = get_record(db, rel_p) or {}
    prev_m = float(rec.get("mtime", 0.0) or 0.0)

    # Ensure artifacts exist; if not, force embed even if unchanged
    artifacts_ok = artifacts_exist_for_rel(rel_p)
    if prev_m >= mtime and rec.get("sha") and artifacts_ok:
        return False

    sha = sha256_path(abs_p)
    if not sha:
        return False

    prev_sha = str(rec.get("sha") or "")
    need_embed = (prev_sha != sha) or (not artifacts_ok)
    if need_embed:
        async with sem:
            await embed_file(abs_p, rel_p, file_sha=sha)
        set_record(db, rel_p, sha=sha, mtime=mtime)
        return True
    # No embedding needed; update mtimes for accuracy
    set_record(db, rel_p, sha=sha, mtime=mtime)
    return True
