"""Code formatting utilities for generated snippets."""

from __future__ import annotations

from jinx.formatters import chain_format


def warp_blk(code: str) -> str:
    """Normalize code by parsing and formatting with several tools.

    Best-effort: each step may fail independently and is safely ignored to avoid
    blocking execution on formatting issues.
    """
    return chain_format(code)
