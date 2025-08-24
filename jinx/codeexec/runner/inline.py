from __future__ import annotations

import io
import sys
from typing import Tuple


def run_inline(code: str) -> str:
    """Execute code in current globals and capture stdout.

    Returns the captured stdout string. Raises on exceptions from exec.
    """
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        exec(code, globals())
        return buf.getvalue()
    finally:
        sys.stdout = old_stdout
