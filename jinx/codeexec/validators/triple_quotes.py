from __future__ import annotations

from typing import Optional


def check_triple_quotes(code: str) -> Optional[str]:
    """Return a violation message if triple quotes are found, else None."""
    if "'''" in code or '"""' in code:
        return "Triple quotes are not allowed by prompt"
    return None
