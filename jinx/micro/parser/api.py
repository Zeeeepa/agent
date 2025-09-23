from __future__ import annotations

from typing import List, Tuple

from jinx.parser import parse_tagged_blocks as _parse_tagged_blocks, is_code_tag as _is_code_tag


def parse_tagged_blocks(out: str, code_id: str) -> List[Tuple[str, str]]:
    """Extract pairs of (tag, content) for the given code id.

    Tolerates CRLF and surrounding whitespace and captures minimal content.
    """
    return _parse_tagged_blocks(out, code_id)


def is_code_tag(tag: str) -> bool:
    """Return True if tag is one of the configured code tags."""
    return _is_code_tag(tag)
