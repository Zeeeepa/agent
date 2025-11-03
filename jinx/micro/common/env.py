from __future__ import annotations

import os
from typing import Optional

__all__ = [
    "truthy",
    "get_int",
    "get_float",
    "get_str",
]


def truthy(name: str, default: str | int | float = "1") -> bool:
    try:
        val = os.getenv(name, str(default))
        return str(val).strip().lower() not in ("", "0", "false", "off", "no", "none")
    except Exception:
        return True


def get_int(name: str, default: int | str, *, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except Exception:
        v = int(default) if not isinstance(default, str) else int(default or 0)
    if min_val is not None and v < min_val:
        v = min_val
    if max_val is not None and v > max_val:
        v = max_val
    return v


def get_float(name: str, default: float | str, *, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    try:
        v = float(os.getenv(name, str(default)))
    except Exception:
        v = float(default) if not isinstance(default, str) else float(default or 0.0)
    if min_val is not None and v < min_val:
        v = min_val
    if max_val is not None and v > max_val:
        v = max_val
    return v


def get_str(name: str, default: str = "") -> str:
    try:
        return str(os.getenv(name, default))
    except Exception:
        return str(default or "")
