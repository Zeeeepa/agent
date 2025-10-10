from __future__ import annotations

import os
import re

from .chain_utils import truthy_env
from jinx.micro.text.heuristics import is_code_like as _is_code_like


_CODEY_RE = re.compile(r"(def\s+|class\s+|import\s+|from\s+|return\b|async\b|await\b|traceback|exception|error\b|\(|\)|\{|\}|\[|\]|\.|:)")


def _is_codey(text: str) -> bool:
    # Backward-compatible wrapper; delegate to language-agnostic heuristic
    return _is_code_like(text or "")


def should_run_planner(user_text: str) -> bool:
    """Gate for running the planner.

    Autonomy is always on: run for any non-empty input. No env gating.
    """
    q = (user_text or "").strip()
    return bool(q)
