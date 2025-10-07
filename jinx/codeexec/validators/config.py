from __future__ import annotations

import os
from typing import Optional

_TRUE_SET = {"1", "true", "on", "yes", "y"}
_FALSE_SET = {"0", "false", "off", "no", "n", ""}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def env_bool(name: str, default: bool) -> bool:
    v = _norm(os.getenv(name))
    if v in _TRUE_SET:
        return True
    if v in _FALSE_SET:
        return False
    return bool(default)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(v.strip())
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(v.strip())
    except Exception:
        return float(default)


# Global enable switch (default ON)
_DEFAULT_ENABLED = env_bool("JINX_VALIDATORS_ENABLE", True)


def is_enabled(validator_key: str, default: Optional[bool] = None) -> bool:
    """Check whether a specific validator is enabled.

    Uses env var: JINX_VALIDATOR_<KEY>
    - KEY should be uppercase with non-alnum replaced by underscores.
    - If not set, falls back to `default` if provided, otherwise global default.
    """
    key = "JINX_VALIDATOR_" + "".join(ch if ch.isalnum() else "_" for ch in validator_key.upper())
    if default is None:
        default = _DEFAULT_ENABLED
    return env_bool(key, default)


# Threshold helpers with env overrides (provide conservative defaults)
IO_MAX_LOOP_BODY_LINES = env_int("JINX_IO_MAX_LOOP_BODY_LINES", 400)
IO_MAX_LITERAL_ELEMS = env_int("JINX_IO_MAX_LITERAL_ELEMS", 1000)
RT_MAX_SLEEP_SECONDS = env_float("JINX_RT_MAX_SLEEP_SECONDS", 2.0)
RT_MAX_RANGE_CONST = env_int("JINX_RT_MAX_RANGE_CONST", 100000)
HTTP_MAX_TIMEOUT = env_float("JINX_HTTP_MAX_TIMEOUT", 10.0)
