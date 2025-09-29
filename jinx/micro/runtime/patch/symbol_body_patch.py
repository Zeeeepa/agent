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
    leading_ws,
    normalize_indentation,
    preserve_docstring_enabled,
    trim_trailing_ws_enabled,
    trim_trailing_ws_lines,
)


async def patch_symbol_body_python(path: str, symbol: str, body: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Replace only the body of a Python function or class by name.

    - If a docstring exists and JINX_PATCH_PRESERVE_DOCSTRING is enabled,
      it will be preserved unless the provided body begins with a triple-quoted string.
    - Indentation and EOL are preserved.
    """
    cur = await read_text_raw(path)
    if cur == "":
        return False, "file read error or empty"
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
        return False, "symbol not found"
    # calculate body region lines
    eol = detect_eol(cur)
    trailing_nl = has_trailing_newline(cur)
    lines = cur.splitlines()
    # Determine docstring bounds if present
    keep_doc = preserve_docstring_enabled()
    doc_start = doc_end = None
    if getattr(target, "body", None):
        first_stmt = target.body[0]
        # docstring detection
        is_doc = (
            isinstance(first_stmt, ast.Expr)
            and isinstance(getattr(first_stmt, "value", None), (ast.Str, ast.Constant))
            and isinstance(getattr(first_stmt, "value", None).s if isinstance(first_stmt.value, ast.Str) else getattr(first_stmt.value, "value", None), str)
        )
        if is_doc:
            doc_start = getattr(first_stmt, "lineno", None)
            doc_end = getattr(first_stmt, "end_lineno", doc_start)
    # docstring override if new body begins with a triple-quoted string
    new_body_stripped = (body or "").lstrip()
    override_doc = False
    if keep_doc and new_body_stripped.startswith(('"""', "'''")):
        override_doc = True

    # compute replacement boundaries (inside the block)
    # start after signature line; if docstring to be preserved, start after docstring
    start_line = None
    if doc_start and doc_end and keep_doc and not override_doc:
        start_line = int(doc_end) + 1
    elif getattr(target, "body", None):
        # if overriding docstring, start from doc_start to replace it
        if doc_start and override_doc:
            start_line = int(doc_start)
        else:
            start_line = int(target.body[0].lineno)
    else:
        start_line = int(getattr(target, "lineno", 0)) + 1
    end_line = int(getattr(target, "end_lineno", start_line))
    if start_line <= 0 or end_line <= 0 or end_line < start_line - 1:
        return False, "invalid body range"
    # Build replacement lines, normalized to block indentation
    new_lines = (body or "").splitlines()
    if trim_trailing_ws_enabled():
        new_lines = trim_trailing_ws_lines(new_lines)
    # choose indentation baseline
    if start_line - 1 < len(lines) and start_line - 1 >= 0:
        base_indent = leading_ws(lines[start_line - 1])
    else:
        # fallback: signature indent + 4 spaces
        base_indent = leading_ws(lines[getattr(target, "lineno", 1) - 1]) + "    "
    new_lines = normalize_indentation(new_lines)
    new_lines = [(base_indent + ln) if ln.strip() else ln for ln in new_lines]
    # splice into original lines
    pre = lines[: start_line - 1]
    post = lines[end_line:]
    out_lines = pre + new_lines + post
    out = join_lines(out_lines, eol, trailing_nl)
    if preview:
        return True, unified_diff(cur, out, path=path)
    if str(path).endswith(".py") and syntax_check_enabled():
        try:
            await asyncio.to_thread(ast.parse, out or "")
        except Exception as e:
            return False, f"syntax error: {e}"
    await write_text(path, out)
    return True, unified_diff(cur, out, path=path)
