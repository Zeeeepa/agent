from __future__ import annotations

"""
Risk policies for self-reprogramming and autopatch.

Env variables (comma/semicolon-separated globs; match against relative path from repo root, fallback to substring):
- JINX_RISK_DENY
- JINX_RISK_ALLOW

API:
- is_allowed_path(path: str) -> bool
- deny_patterns() -> list[str]
"""

import fnmatch
import os
from functools import lru_cache
from typing import List


def _split_list(s: str) -> List[str]:
    if not s:
        return []
    out: List[str] = []
    for part in s.replace(";", ",").split(","):
        x = part.strip()
        if x:
            out.append(x)
    return out


@lru_cache(maxsize=1)
def _deny() -> List[str]:
    return _split_list(os.getenv("JINX_RISK_DENY", "") or "")


@lru_cache(maxsize=1)
def _allow() -> List[str]:
    return _split_list(os.getenv("JINX_RISK_ALLOW", "") or "")


def deny_patterns() -> List[str]:
    return list(_deny())


def _match_any(path: str, pats: List[str]) -> bool:
    for p in pats:
        try:
            if fnmatch.fnmatch(path, p):
                return True
        except Exception:
            pass
        # fallback: substring
        try:
            if p in path:
                return True
        except Exception:
            pass
    return False


def is_allowed_path(path: str) -> bool:
    # Normalize to forward slashes for glob matching
    rel = path.replace("\\", "/")
    # Allowlist overrides deny
    if _match_any(rel, _allow()):
        return True
    if _match_any(rel, _deny()):
        return False
    return True
