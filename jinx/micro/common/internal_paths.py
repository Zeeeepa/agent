from __future__ import annotations

import re

_INTERNAL_PATH_PAT = re.compile(r"(^|[\\/])\.jinx([\\/]|$)", re.IGNORECASE)
_LOG_PATH_PAT = re.compile(r"(^|[\\/])log([\\/]|$)", re.IGNORECASE)


def is_internal_path(path_or_text: str) -> bool:
    """True if string contains a '.jinx' path segment."""
    if not path_or_text:
        return False
    return bool(_INTERNAL_PATH_PAT.search(path_or_text))


def is_log_path(path_or_text: str) -> bool:
    """True if string contains a 'log' path segment (likely project log dir)."""
    if not path_or_text:
        return False
    return bool(_LOG_PATH_PAT.search(path_or_text))


def is_restricted_path(path_or_text: str) -> bool:
    """True if path refers to a restricted location ('.jinx' or 'log')."""
    return is_internal_path(path_or_text) or is_log_path(path_or_text)
