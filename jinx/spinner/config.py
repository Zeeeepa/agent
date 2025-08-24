from __future__ import annotations

import os, sys
from typing import Optional


def ascii_mode() -> bool:
    """Return True if ASCII-only rendering is forced via env.
    Env: JINX_ASCII in {"1","true","yes"} (case-insensitive).
    """
    return (os.getenv("JINX_ASCII", "").strip().lower() in {"1", "true", "yes"})


def can_render(s: str, encoding: Optional[str] = None) -> bool:
    """Check if string can be encoded in the current terminal encoding.
    Falls back to sys.stdout.encoding or utf-8.
    """
    enc = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        s.encode(enc)
        return True
    except Exception:
        return False
