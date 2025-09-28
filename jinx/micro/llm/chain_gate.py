from __future__ import annotations

import os
import re

from .chain_utils import truthy_env


_CODEY_RE = re.compile(r"(def\s+|class\s+|import\s+|from\s+|return\b|async\b|await\b|traceback|exception|error\b|\(|\)|\{|\}|\[|\]|\.|:)")


def _is_codey(text: str) -> bool:
    t = (text or "").lower()
    if _CODEY_RE.search(t):
        return True
    # simple heuristics for identifiers with underscores or camelCase
    if "_" in t or any(ch.isupper() for ch in text):
        return True
    return False


def should_run_planner(user_text: str) -> bool:
    """Gate for running the planner.

    Autonomy is always on: run for any non-empty input. No env gating.
    """
    q = (user_text or "").strip()
    return bool(q)
