from __future__ import annotations

from jinx.bootstrap import ensure_optional

autopep8 = ensure_optional(["autopep8"])["autopep8"]  # type: ignore


def pep8_format(code: str) -> str:
    """Apply autopep8 formatting. Best-effort."""
    try:
        return autopep8.fix_code(code)
    except Exception:
        return code
