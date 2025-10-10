from __future__ import annotations

import ast
import hashlib
import re
from typing import Tuple

__all__ = ["code_key", "canonicalize"]

_NEWLINES_RE = re.compile(r"\r\n|\r|\n")
_WS_TAIL_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def _stable_ast_dump(code: str) -> str | None:
    try:
        tree = ast.parse(code)
        # Exclude attributes so positions/line numbers don't affect the digest
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        return None


def canonicalize(code: str) -> str:
    """Return a language-aware canonical form for Python code for hashing.

    - Normalizes newlines to \n
    - Strips trailing whitespace

    - Tries AST dump to ignore comments/whitespace differences
    """
    s = code or ""
    # Normalize newlines and trailing spaces
    s = _NEWLINES_RE.sub("\n", s)
    s = _WS_TAIL_RE.sub("", s)
    # Prefer AST structure when available
    adump = _stable_ast_dump(s)
    if adump is not None:
        return adump
    return s


def code_key(code: str) -> str:
    """Stable key for sandbox coalescing and result caching.

    Uses AST-based canonicalization when possible; falls back to normalized text.
    """
    can = canonicalize(code)
    h = hashlib.sha256(can.encode("utf-8", errors="ignore")).hexdigest()
    return h
