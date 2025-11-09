from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _repro_dir() -> str:
    d = os.path.join("log", "repro")
    os.makedirs(d, exist_ok=True)
    return d


def start_run_record(context: Optional[Dict[str, Any]] = None) -> str:
    """Create a reproducibility bundle directory and write initial metadata.

    Returns a run_id (directory path) that should be passed to finalize.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = _repro_dir()
    run_id = os.path.join(base, ts)
    os.makedirs(run_id, exist_ok=True)
    meta = {
        "ts": time.time(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "context": context or {},
    }
    try:
        with open(os.path.join(run_id, "meta.start.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return run_id


def write_blob(run_id: str, name: str, content: str) -> None:
    try:
        with open(os.path.join(run_id, name), "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


def finalize_run_record(run_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write final metadata and seal the bundle."""
    if not run_id:
        return
    meta = {
        "ts": time.time(),
        "pid": os.getpid(),
        "extra": extra or {},
    }
    try:
        with open(os.path.join(run_id, "meta.end.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
