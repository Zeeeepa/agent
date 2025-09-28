from __future__ import annotations

import os
import re
from typing import Optional

_FORBIDDEN_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"sys\.exit\b",
    r"os\._exit\b",
    r"subprocess\.Popen\b",
    r"subprocess\.run\b.*shell\s*=\s*True",
    r"pip\b",
]

_DEF_TRIPLE = ("'''", '"""')


def sanitize_kernels(code: str) -> str:
    """Return code if it passes basic safety/size checks; else return empty string.

    Rules:
    - Forbid triple quotes.
    - Enforce max char length from env JINX_KERNEL_MAXCHARS (default 3000).
    - Reject if obvious forbidden tokens are present.
    - This is a best-effort hygiene gate; not a sandbox replacement.
    """
    body = (code or "").strip()
    if not body:
        return ""
    try:
        max_chars = max(256, int(os.getenv("JINX_KERNEL_MAXCHARS", "3000")))
    except Exception:
        max_chars = 3000
    if len(body) > max_chars:
        return ""
    if any(t in body for t in _DEF_TRIPLE):
        return ""
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, body, flags=re.IGNORECASE):
            return ""
    return body
