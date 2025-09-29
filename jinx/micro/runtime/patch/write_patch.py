from __future__ import annotations

import ast
import asyncio
from typing import Tuple
import os

from jinx.async_utils.fs import read_text_raw, write_text
from .utils import unified_diff, syntax_check_enabled


async def patch_write(path: str, text: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Create/overwrite file contents atomically (best-effort). If preview, returns diff only."""
    cur = await read_text_raw(path)
    new = text or ""
    if preview:
        return True, unified_diff(cur or "", new, path=path)
    # Optional syntax check for Python files
    if str(path).endswith(".py") and syntax_check_enabled():
        try:
            await asyncio.to_thread(ast.parse, new or "")
        except Exception as e:
            return False, f"syntax error: {e}"
    # Ensure directory exists
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    await write_text(path, new)
    return True, unified_diff(cur or "", new, path=path)
