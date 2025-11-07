from __future__ import annotations

import json
import os
from typing import Optional


def _path() -> str | None:
    p = (os.getenv("JINX_SELFUPDATE_HANDSHAKE_FILE") or "").strip()
    return p or None


def set_status(*, online: Optional[bool] = None, healthy: Optional[bool] = None) -> None:
    path = _path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if online is not None:
            data["online"] = bool(online)
        if healthy is not None:
            data["healthy"] = bool(healthy)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        # Never crash caller
        return


def set_online() -> None:
    set_status(online=True)


def set_healthy() -> None:
    set_status(healthy=True)


__all__ = [
    "set_status",
    "set_online",
    "set_healthy",
]
