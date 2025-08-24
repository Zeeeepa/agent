from __future__ import annotations

from jinx.bootstrap import ensure_optional

libcst = ensure_optional(["libcst"])["libcst"]  # type: ignore


def cst_format(code: str) -> str:
    """Format code using LibCST round-trip pretty printing.

    Best-effort: returns original code on failure.
    """
    try:
        return libcst.cst.parse_module(code).code
    except Exception:
        return code
