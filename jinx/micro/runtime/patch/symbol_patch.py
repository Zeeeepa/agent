from __future__ import annotations

import ast
import asyncio
from typing import Tuple

from jinx.async_utils.fs import read_text_raw, write_text
from .utils import (
    unified_diff,
    syntax_check_enabled,
    detect_eol,
    has_trailing_newline,
    join_lines,
)


async def patch_symbol_python(path: str, symbol: str, replacement: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Replace a Python function or class block by name with replacement text.

    If symbol not found and replacement starts with proper def/class, append at EOF.
    """
    cur = await read_text_raw(path)
    if cur == "":
        # treat missing file as write
        if preview:
            return True, unified_diff("", replacement or "", path=path)
        await write_text(path, replacement or "")
        return True, unified_diff("", replacement or "", path=path)
    try:
        tree = await asyncio.to_thread(ast.parse, cur)
    except Exception as e:
        return False, f"ast parse failed: {e}"
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if getattr(node, "name", "") == symbol:
                target = node
                break
    if not target:
        # append at end if replacement defines the symbol
        if replacement.lstrip().startswith((f"def {symbol}", f"class {symbol}")):
            out = cur
            if not out.endswith("\n"):
                out += "\n"
            out += (replacement or "")
            if not out.endswith("\n"):
                out += "\n"
            if preview:
                return True, unified_diff(cur, out, path=path)
            if syntax_check_enabled():
                try:
                    await asyncio.to_thread(ast.parse, out or "")
                except Exception as e:
                    return False, f"syntax error: {e}"
            await write_text(path, out)
            return True, unified_diff(cur, out, path=path)
        return False, "symbol not found"
    ls = getattr(target, "lineno", None)
    le = getattr(target, "end_lineno", None)
    if not (ls and le):
        return False, "symbol has no line info"
    eol = detect_eol(cur)
    trailing_nl = has_trailing_newline(cur)
    lines = cur.splitlines()
    lines[ls - 1 : le] = (replacement or "").splitlines()
    out = join_lines(lines, eol, trailing_nl)
    if preview:
        return True, unified_diff(cur, out, path=path)
    if syntax_check_enabled():
        try:
            await asyncio.to_thread(ast.parse, out or "")
        except Exception as e:
            return False, f"syntax error: {e}"
    await write_text(path, out)
    return True, unified_diff(cur, out, path=path)
