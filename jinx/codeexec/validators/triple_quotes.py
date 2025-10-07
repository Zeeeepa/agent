from __future__ import annotations

from typing import Optional
import io
import tokenize
from .config import is_enabled


def check_triple_quotes(code: str) -> Optional[str]:
    """Return a violation message if triple-quoted strings are present.

    Uses tokenization to avoid false positives and reliably catches docstrings
    and any triple-quoted literals anywhere in the code.
    """
    if not is_enabled("triple_quotes", True):
        return None
    src = code or ""
    try:
        g = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in g:
            if tok.type == tokenize.STRING:
                s = tok.string
                # skip literal prefixes (r, u, f, b, combinations)
                i = 0
                while i < len(s) and s[i] in "rRuUfFbB":
                    i += 1
                if s[i : i + 3] in ("'''", '"""'):
                    return "Triple quotes are not allowed by prompt"
    except Exception:
        # Fallback to simple textual check
        if "'''" in src or '"""' in src:
            return "Triple quotes are not allowed by prompt"
    return None
