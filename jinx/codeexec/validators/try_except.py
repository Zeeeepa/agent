from __future__ import annotations

from typing import Optional


def check_try_except(code: str) -> Optional[str]:
    """Return a violation message if try/except is found, else None."""
    if "try:" in code or " except" in code or "\nexcept" in code:
        return "Usage of try/except is not allowed by prompt"
    return None
