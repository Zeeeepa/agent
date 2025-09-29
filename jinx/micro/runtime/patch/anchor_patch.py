from __future__ import annotations

import re
import ast
import asyncio
from typing import Tuple

from jinx.async_utils.fs import read_text_raw, write_text
from .utils import (
    unified_diff,
    detect_eol,
    has_trailing_newline,
    join_lines,
    anchor_regex_enabled,
    anchor_last_enabled,
    leading_ws,
    normalize_indentation,
    trim_trailing_ws_enabled,
    trim_trailing_ws_lines,
    syntax_check_enabled,
)


async def patch_anchor_insert_after(path: str, anchor: str, replacement: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Insert replacement lines immediately after the (first/last) anchor match.

    Enhancements:
    - Optional regex anchors (JINX_PATCH_ANCHOR_REGEX)
    - Choose first or last match (JINX_PATCH_ANCHOR_LAST)
    - Preserve original EOL and trailing newline
    - Auto-align replacement indentation to anchor line indentation
    """
    cur = await read_text_raw(path)
    if cur == "":
        return False, "file read error or empty"
    eol = detect_eol(cur)
    trailing_nl = has_trailing_newline(cur)
    lines = cur.splitlines()
    idx = -1
    needle = (anchor or "").strip()
    use_regex = anchor_regex_enabled()
    take_last = anchor_last_enabled()
    if not needle:
        return False, "empty anchor"
    if use_regex:
        try:
            rx = re.compile(needle)
        except Exception as e:
            return False, f"bad anchor regex: {e}"
        def _scan_regex() -> int:
            found = [i for i, line in enumerate(lines) if rx.search(line)]
            if not found:
                return -1
            return found[-1] if take_last else found[0]
        idx = await asyncio.to_thread(_scan_regex)
    else:
        def _scan_substr() -> int:
            first = -1
            last = -1
            for i, line in enumerate(lines):
                if needle in line:
                    if first < 0:
                        first = i
                    last = i
            if take_last:
                return last
            return first
        idx = await asyncio.to_thread(_scan_substr)
    if idx < 0:
        return False, "anchor not found"
    # Prepare replacement with indentation aligned to anchor line
    rep_lines = (replacement or "").splitlines()
    if trim_trailing_ws_enabled():
        rep_lines = trim_trailing_ws_lines(rep_lines)
    base_indent = leading_ws(lines[idx])
    rep_lines = normalize_indentation(rep_lines)
    rep_lines = [(base_indent + ln) if ln.strip() else ln for ln in rep_lines]
    out_lines = lines[: idx + 1] + rep_lines + lines[idx + 1 :]
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
