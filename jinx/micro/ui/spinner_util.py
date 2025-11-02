from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Any


def parse_env_bool(name: str, default: bool = True) -> bool:
    try:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        v = raw.strip().lower()
        return v not in ("", "0", "false", "off", "no")
    except Exception:
        return bool(default)


def parse_env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        s = raw.strip()
        return int(s or str(default))
    except Exception:
        return int(default)


def _max_detail_keys() -> int:
    return parse_env_int("JINX_SPINNER_ACTIVITY_DETAIL_KEYS", 4)


def format_activity_detail(det: Dict[str, Any] | None) -> Tuple[str, Optional[str], Optional[str]]:
    """Format activity_detail dict into a compact string and extract stage/tasks.

    Returns (detail_string, stage, tasks)
    - detail_string is empty if no items
    - stage/tasks are extracted if present among keys
    """
    if not isinstance(det, dict) or not det:
        return "", None, None
    items = []
    stage = None
    tasks = None
    max_keys = _max_detail_keys()
    for i, (k, v) in enumerate(det.items()):
        if i >= max_keys:
            break
        items.append(f"{k}:{v}")
        if k == "retr_stage":
            stage = v
        if k == "tasks":
            tasks = v
    if not items:
        return "", stage, tasks
    return " [" + ", ".join(items) + "]", stage, tasks


__all__ = [
    "format_activity_detail",
    "parse_env_bool",
    "parse_env_int",
]
