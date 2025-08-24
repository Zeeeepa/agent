from __future__ import annotations

from jinx.bootstrap import ensure_optional

black = ensure_optional(["black"])["black"]  # type: ignore


def black_format(code: str) -> str:
    """Apply Black formatting. Best-effort."""
    try:
        return black.format_str(code, mode=black.Mode())
    except Exception:
        return code
