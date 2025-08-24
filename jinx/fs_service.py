"""Filesystem helpers."""

from __future__ import annotations

from jinx.fs import read_text


def wire(f: str) -> str:
    """Read entire file if it exists, else return an empty string."""
    return read_text(f)
