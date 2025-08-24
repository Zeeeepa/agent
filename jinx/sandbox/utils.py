from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional
import json
import aiofiles
from jinx.log_paths import SANDBOX_DIR, SANDBOX_INDEX


def make_run_log_path(base_dir: str = SANDBOX_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log")


async def index_run(
    log_path: str,
    status: str,
    code_id: Optional[str] = None,
    error: Optional[str] = None,
    index_path: str = SANDBOX_INDEX,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Append a compact record about a sandbox run to an index file.

    Parameters
    ----------
    log_path : str
        Path to the streamed log file for this run.
    status : str
        "ok" | "error" | "cancelled" etc.
    code_id : Optional[str]
        Optional code identifier.
    error : Optional[str]
        Optional error message.
    index_path : str
        JSONL index file path.
    extra : Optional[dict]
        Any extra fields to record.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    rec = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "log_path": log_path,
        "status": status,
        "code_id": code_id,
    }
    if error:
        rec["error"] = error
    if extra:
        rec.update(extra)
    try:
        async with aiofiles.open(index_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort: indexing must not break execution
        pass
