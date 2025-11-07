from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def _journal_dir() -> str:
    d = os.path.join(os.getcwd(), ".jinx", "selfupdate")
    os.makedirs(d, exist_ok=True)
    return d


essential_keys = ("event", "ts", "stage", "ok")


def append(event: str, *, stage: str, ok: bool | None = None, **meta: Any) -> None:
    """Append a compact JSON line to the self-update journal.
    Never raises.
    """
    try:
        rec: Dict[str, Any] = {
            "event": str(event),
            "ts": time.time(),
            "stage": stage,
        }
        if ok is not None:
            rec["ok"] = bool(ok)
        if meta:
            # Keep JSON serializable; fall back to string
            safe_meta: Dict[str, Any] = {}
            for k, v in meta.items():
                try:
                    json.dumps(v)
                    safe_meta[k] = v
                except Exception:
                    safe_meta[k] = str(v)
            rec.update(safe_meta)
        path = os.path.join(_journal_dir(), "journal.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Update pointer to last record
        with open(os.path.join(_journal_dir(), "last.json"), "w", encoding="utf-8") as f2:
            json.dump(rec, f2, ensure_ascii=False)
    except Exception:
        return


__all__ = ["append"]
