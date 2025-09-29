from __future__ import annotations

import ast
import asyncio
from typing import Optional, Tuple

from jinx.async_utils.fs import read_text_raw, write_text
from .utils import (
    unified_diff,
    syntax_check_enabled,
    detect_eol,
    has_trailing_newline,
    join_lines,
    normalize_indentation,
    leading_ws,
    auto_indent_enabled,
    trim_trailing_ws_enabled,
    trim_trailing_ws_lines,
)


async def patch_line_range(path: str, ls: int, le: int, replacement: str, *, preview: bool = False, max_span: Optional[int] = None) -> Tuple[bool, str]:
    cur = await read_text_raw(path)
    if cur == "":
        return False, "file read error or empty"
    if ls <= 0 or le <= 0 or le < ls:
        return False, "invalid line range"
    eol = detect_eol(cur)
    trailing_nl = has_trailing_newline(cur)
    lines = cur.splitlines()
    n = len(lines)
    if ls > n:
        return False, "start beyond EOF"
    le_eff = min(le, n)
    span = le_eff - ls + 1
    # Prepare replacement lines with optional formatting policies
    rep_lines = (replacement or "").splitlines()
    # Optional trim trailing whitespace
    if trim_trailing_ws_enabled():
        rep_lines = trim_trailing_ws_lines(rep_lines)
    # Optional auto-indent to match the indentation of the first replaced line
    if auto_indent_enabled() and ls - 1 < len(lines) and ls - 1 >= 0:
        base_indent = leading_ws(lines[ls - 1])
        rep_lines = normalize_indentation(rep_lines)
        rep_lines = [(base_indent + ln) if ln.strip() else ln for ln in rep_lines]
    lines[ls - 1 : le_eff] = rep_lines
    out = join_lines(lines, eol, trailing_nl)
    if max_span is not None and span > max_span and not preview:
        return False, f"span {span} exceeds guard {max_span}; preview required"
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
