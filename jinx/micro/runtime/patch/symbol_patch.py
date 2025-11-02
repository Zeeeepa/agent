from __future__ import annotations

import ast
import asyncio
from typing import Tuple, List, Optional

from jinx.async_utils.fs import read_text_raw, write_text
from .utils import (
    unified_diff,
    syntax_check_enabled,
    detect_eol,
    has_trailing_newline,
    join_lines,
)


def _indent_block(text: str, spaces: int) -> str:
    pad = " " * max(0, spaces)
    lines = (text or "").splitlines()
    return "\n".join([(pad + ln if ln.strip() else ln) for ln in lines])


def _parts(sym: str) -> List[str]:
    return [p for p in (sym or "").split(".") if p.strip()]


def _find_nested(tree: ast.AST, chain: List[str]) -> Tuple[Optional[ast.AST], Optional[ast.AST]]:
    """Return (target_node, parent_node) for a dotted chain like [Class, Inner, func]."""
    cur: ast.AST = tree
    parent: Optional[ast.AST] = None
    for i, name in enumerate(chain):
        found = None
        for node in getattr(cur, "body", []) or []:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if getattr(node, "name", "") == name:
                    found = node
                    break
        if found is None:
            return None, cur if cur is not tree else None
        parent = cur
        cur = found
    return cur, parent


def _lines_for(node: ast.AST) -> Tuple[Optional[int], Optional[int]]:
    return getattr(node, "lineno", None), getattr(node, "end_lineno", None)


async def patch_symbol_python(path: str, symbol: str, replacement: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Replace Python function/class by dotted name; insert if missing in existing parent.

    - Supports dotted paths: "Class.method", "Outer.Inner.func".
    - If last symbol not found but parent exists, inserts replacement into parent body with proper indentation.
    - If top-level symbol missing and replacement begins with def/class <name>, appends at EOF.
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
    chain = _parts(symbol)
    target, parent = _find_nested(tree, chain)
    # Not found: try inserting into parent if exists
    if target is None:
        last = chain[-1] if chain else symbol
        if parent is not None:
            # Insert into parent's body with indentation
            # Compute indent: use first child indent if available, else parent.col_offset + 4
            lines = cur.splitlines()
            try:
                base_col = int(getattr(parent, "col_offset", 0))
            except Exception:
                base_col = 0
            # prefer 4 spaces indent
            ind = base_col + 4
            block = replacement or ""
            # If replacement is top-level def/class text (no indent), indent it into parent
            block_i = _indent_block(block, ind)
            # Find insertion line: after last child of parent or after parent header
            body_nodes = list(getattr(parent, "body", []) or [])
            insert_after = None
            if body_nodes:
                _, end_ln = _lines_for(body_nodes[-1])
                insert_after = (end_ln or getattr(parent, "lineno", 0))
            else:
                insert_after = getattr(parent, "lineno", 0)
            # splice
            eol = detect_eol(cur)
            trailing_nl = has_trailing_newline(cur)
            before = lines[:insert_after]
            after = lines[insert_after:]
            new_lines = (block_i or "").splitlines()
            out = "\n".join(before + new_lines + after)
            if trailing_nl and not out.endswith("\n"):
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
        # Fallback: append if replacement defines the final symbol name
        last = chain[-1] if chain else symbol
        if replacement.lstrip().startswith((f"def {last}", f"class {last}")):
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

    ls, le = _lines_for(target)
    if not (ls and le):
        return False, "symbol has no line info"
    eol = detect_eol(cur)
    trailing_nl = has_trailing_newline(cur)
    lines = cur.splitlines()
    # If replacement does not start with decorator/def/class but target has them, treat as body replace
    rep = (replacement or "").lstrip()
    if not rep.startswith(("def ", "class ", "@", "async def ")) and isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # preserve header and decorators; replace body lines only, handle docstring
        header_start = ls - 1
        deco_count = len(getattr(target, "decorator_list", []) or [])
        header_start = max(0, header_start - deco_count)
        header = lines[header_start: ls]  # decorators + def line
        indent_size = len(header[-1]) - len(header[-1].lstrip(' ')) + 4
        # Docstring detection
        doc_start = doc_end = None
        if getattr(target, "body", None):
            first_stmt = target.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(getattr(first_stmt, "value", None), ast.Constant) and isinstance(first_stmt.value.value, str):
                doc_start = getattr(first_stmt, "lineno", None)
                doc_end = getattr(first_stmt, "end_lineno", None)
        doc_lines_new: list[str] = []
        body_core_lines: list[str] = []
        # If replacement begins with indented triple-quote, treat as new docstring
        if rep.startswith(('"""', "'''")):
            # place new docstring with proper indent
            doci = _indent_block(rep, indent_size)
            doc_lines_new = doci.splitlines()
        else:
            # preserve existing docstring if present
            if doc_start and doc_end and (doc_start >= ls and doc_end <= le):
                doc_lines_new = lines[doc_start - 1: doc_end]
            # remaining replacement as body
            body_core_lines = rep.splitlines()
        # indent body core
        if body_core_lines:
            body_i = _indent_block("\n".join(body_core_lines), indent_size).splitlines()
        else:
            body_i = []
        # Replace the entire function range with header + doc + body
        lines[header_start: le] = header + doc_lines_new + body_i
    else:
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
