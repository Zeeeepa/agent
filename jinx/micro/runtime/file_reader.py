from __future__ import annotations

import os
from typing import Optional, Tuple


def _is_binary(data: bytes) -> bool:
    # Heuristic: if NUL byte present or high fraction of non-text bytes
    if b"\x00" in data:
        return True
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    if not data:
        return False
    nontext = data.translate(None, text_chars)  # type: ignore[arg-type]
    return float(len(nontext)) / float(len(data)) > 0.30


def read_file_preview(
    path: str,
    *,
    max_chars: int = 4000,
    head_lines: int = 200,
    tail_lines: int = 80,
    encoding: str = "utf-8"
) -> Tuple[str, bool]:
    """
    Read a safe preview of a text file.

    Returns (text, truncated). If binary or unreadable, returns ("", False).
    """
    try:
        if not path or not os.path.isfile(path):
            return ("", False)
        # Quick binary check on first 1KB
        with open(path, "rb") as f:
            head = f.read(1024)
            if _is_binary(head):
                return ("", False)
        # Read text lines
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            lines = f.readlines()
        total = len(lines)
        # Slice head and tail if too many lines
        if total > (head_lines + tail_lines):
            head_part = lines[:head_lines]
            tail_part = lines[-tail_lines:]
            body = head_part + ["\n...\n"] + tail_part
            truncated = True
        else:
            body = lines
            truncated = False
        text = "".join(body)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars] + "\n...\n"
            truncated = True
        return (text, truncated)
    except Exception:
        return ("", False)


__all__ = [
    "read_file_preview",
]
