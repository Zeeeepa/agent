from __future__ import annotations

import ast
import asyncio
import difflib
from typing import Tuple

from jinx.async_utils.fs import read_text_raw, write_text
from .utils import (
    unified_diff,
    syntax_check_enabled,
    detect_eol,
    has_trailing_newline,
    join_lines,
    normalize_indentation,
    leading_ws,
    trim_trailing_ws_enabled,
    trim_trailing_ws_lines,
    auto_indent_enabled,
)


async def patch_context_replace(path: str, before_block: str, replacement: str, *, preview: bool = False, tolerance: float = 0.72) -> Tuple[bool, str]:
    """Replace the first segment matching before_block (exact or fuzzy) with replacement.

    Fuzzy mode scans for the best-matching window by line-count with SequenceMatcher and
    applies the change if ratio >= tolerance.
    """
    cur = await read_text_raw(path)
    if cur == "":
        return False, "file read error or empty"
    eol = detect_eol(cur)
    trailing_nl = has_trailing_newline(cur)
    lines = cur.splitlines()
    src = (before_block or "")
    if not src.strip():
        return False, "empty context"
    src_lines = src.splitlines()
    m = len(src_lines)
    n = len(lines)
    if m <= 0 or n <= 0 or m > n:
        return False, "context size invalid"
    # 1) exact match first
    match_i = -1
    for i in range(0, n - m + 1):
        if lines[i : i + m] == src_lines:
            match_i = i
            break
    # 2) fuzzy match
    if match_i < 0:
        def _best_index() -> tuple[int, float]:
            best_i_local = -1
            best_r_local = 0.0
            for i in range(0, n - m + 1):
                win = lines[i : i + m]
                r = difflib.SequenceMatcher(None, "\n".join(win), "\n".join(src_lines)).ratio()
                if r > best_r_local:
                    best_r_local = r
                    best_i_local = i
            return best_i_local, best_r_local
        best_i, best_r = await asyncio.to_thread(_best_index)
        if best_i >= 0 and best_r >= max(0.0, float(tolerance)):
            match_i = best_i
        else:
            return False, "context not found"
    rep_lines = (replacement or "").splitlines()
    if trim_trailing_ws_enabled():
        rep_lines = trim_trailing_ws_lines(rep_lines)
    # Auto-align indentation to the matched block's first line indentation
    if auto_indent_enabled():
        base_indent = leading_ws(lines[match_i])
        rep_lines = normalize_indentation(rep_lines)
        rep_lines = [(base_indent + ln) if ln.strip() else ln for ln in rep_lines]
    out_lines = lines[: match_i] + rep_lines + lines[match_i + m :]
    out = join_lines(out_lines, eol, trailing_nl)
    if preview:
        return True, unified_diff(cur, out, path=path)
    # Optional syntax check for Python files
    if str(path).endswith(".py") and syntax_check_enabled():
        try:
            await asyncio.to_thread(ast.parse, out or "")
        except Exception as e:
            return False, f"syntax error: {e}"
    await write_text(path, out)
    return True, unified_diff(cur, out, path=path)
