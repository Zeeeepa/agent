from __future__ import annotations

from jinx.config import CODE_TAGS


def is_code_tag(tag: str) -> bool:
    """Return True if tag is one of the configured code tags."""
    return tag in CODE_TAGS
