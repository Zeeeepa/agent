from __future__ import annotations

import os
import re
import asyncio
import ast
from typing import Tuple, Dict, Any, Optional

from jinx.async_utils.fs import read_text_raw
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT


def _abs_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    base = PROJECT_ROOT or os.getcwd()
    return os.path.normpath(os.path.join(base, p))


async def _parse_ast(text: str) -> ast.AST:
    return await asyncio.to_thread(ast.parse, text)


def _node_range_with_decorators(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    decos = getattr(node, "decorator_list", None)
    if decos:
        try:
            dmin = min(int(getattr(d, "lineno", start)) for d in decos if hasattr(d, "lineno"))
            if dmin > 0:
                start = min(start, dmin)
        except Exception:
            pass
    return int(start), int(end)


def _contains_lineno(node: ast.AST, ln: int) -> bool:
    s = int(getattr(node, "lineno", 0) or 0)
    e = int(getattr(node, "end_lineno", 0) or 0)
    return s and e and (s <= ln <= e)


async def extract_symbol_source(path: str, symbol: str, *, include_decorators: bool = True, include_docstring: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """Extract the exact source of a function/class by name from a file.

    Returns: (ok, code, meta) where meta has {kind, start, end} line numbers (1-based inclusive).
    """
    apath = _abs_path(path)
    text = await read_text_raw(apath)
    if text == "":
        return False, "", {"error": "file read error or empty"}
    # First try AST-precise extraction
    try:
        tree = await _parse_ast(text)
        target: Optional[ast.AST] = None
        kind: str = ""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if getattr(node, "name", "") == symbol:
                    target = node
                    kind = node.__class__.__name__
                    break
        if target is not None:
            s, e = _node_range_with_decorators(target) if include_decorators else (
                int(getattr(target, "lineno", 1)), int(getattr(target, "end_lineno", 1))
            )
            # If we intend to exclude docstring, shift start past docstring when present
            if not include_docstring:
                try:
                    body = getattr(target, "body", None)
                    if body:
                        first_stmt = body[0]
                        is_doc = (
                            isinstance(first_stmt, ast.Expr)
                            and isinstance(getattr(first_stmt, "value", None), (ast.Str, ast.Constant))
                            and isinstance(getattr(first_stmt, "value", None).s if isinstance(first_stmt.value, ast.Str) else getattr(first_stmt.value, "value", None), str)
                        )
                        if is_doc:
                            doc_end = int(getattr(first_stmt, "end_lineno", getattr(first_stmt, "lineno", s)))
                            s = max(s, doc_end + 1)
                except Exception:
                    pass
            lines = text.splitlines()
            n = len(lines)
            if s <= 0 or e <= 0 or s > e or s > n:
                return False, "", {"error": f"invalid range s={s} e={e} n={n}"}
            e_eff = min(e, n)
            code = "\n".join(lines[s - 1 : e_eff])
            return True, code, {"kind": kind, "start": s, "end": e_eff, "path": apath}
    except Exception:
        # fall through to text fallback
        pass

    # Text-based fallback if AST path failed or symbol not found
    lines = text.splitlines()
    header_re = re.compile(r"^\s*(?:async\s+def|def|class)\s+" + re.escape(symbol) + r"\b")
    start_idx = -1
    kind = ""
    for i, ln in enumerate(lines):
        if header_re.match(ln):
            start_idx = i
            kind = "FunctionDef" if ln.lstrip().startswith(("def", "async def")) else ("ClassDef" if ln.lstrip().startswith("class") else "")
            break
    if start_idx < 0:
        return False, "", {"error": "symbol not found (text fallback)"}
    # include decorators if requested
    if include_decorators:
        j = start_idx - 1
        while j >= 0:
            prev = lines[j]
            if prev.lstrip().startswith("@"):
                start_idx = j
                j -= 1
                continue
            # stop on blank line or other code
            break
    # determine end by indentation or next def/class at same or lower indent
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    end_idx = start_idx
    for k in range(start_idx + 1, len(lines)):
        l = lines[k]
        if not l.strip():
            end_idx = k
            continue
        indent = len(l) - len(l.lstrip())
        if indent <= base_indent and re.match(r"^\s*(?:async\s+def|def|class)\s+\w+", l):
            end_idx = k - 1
            break
        end_idx = k
    # Optionally drop leading docstring in fallback mode (best-effort)
    s = start_idx + 1
    if not include_docstring:
        if s <= end_idx:
            first_nonempty = s
            while first_nonempty <= end_idx and not lines[first_nonempty].strip():
                first_nonempty += 1
            if first_nonempty <= end_idx:
                l0 = lines[first_nonempty].lstrip()
                if l0.startswith('"""') or l0.startswith("'''"):
                    quote = '"""' if l0.startswith('"""') else "'''"
                    # advance until closing triple quote
                    m = first_nonempty
                    closed = False
                    while m <= end_idx:
                        if quote in lines[m] and (m != first_nonempty or lines[m].count(quote) >= 2):
                            s = m + 1
                            closed = True
                            break
                        m += 1
                    if not closed:
                        s = first_nonempty + 1
    code = "\n".join(lines[start_idx : end_idx + 1]) if include_docstring else "\n".join(lines[s : end_idx + 1])
    return True, code, {"kind": kind or "", "start": (start_idx + 1 if include_docstring else s + 1), "end": end_idx + 1, "path": apath}


async def find_enclosing_symbol(path: str, query: str) -> Tuple[bool, Dict[str, Any]]:
    """Find the nearest enclosing function/class for the first occurrence of query in a file.

    Returns: (ok, data) where data has {symbol, kind, start, end, match_line}.
    """
    apath = _abs_path(path)
    text = await read_text_raw(apath)
    if text == "":
        return False, {"error": "file read error or empty"}
    pos = text.find(query)
    if pos < 0:
        return False, {"error": "query not found"}
    # compute 1-based line of the first match
    match_line = text.count("\n", 0, pos) + 1
    # Try AST-based first
    try:
        tree = await _parse_ast(text)
        candidates: list[tuple[int, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _contains_lineno(node, match_line):
                    span = int(getattr(node, "end_lineno", match_line) or match_line) - int(getattr(node, "lineno", match_line) or match_line)
                    candidates.append((span, node))
        if candidates:
            _, node = sorted(candidates, key=lambda x: x[0])[0]
            s, e = _node_range_with_decorators(node)
            return True, {
                "symbol": str(getattr(node, "name", "")),
                "kind": node.__class__.__name__,
                "start": s,
                "end": e,
                "match_line": match_line,
                "path": apath,
            }
    except Exception:
        pass
    # Text fallback: search upwards for a def/class header
    lines = text.splitlines()
    i = min(max(1, match_line), len(lines)) - 1
    header_re = re.compile(r"^\s*(?:async\s+def|def|class)\s+(\w+)\b")
    while i >= 0:
        ln = lines[i]
        m = header_re.match(ln)
        if m:
            sym = m.group(1)
            # find end similar to extract fallback
            base_indent = len(ln) - len(ln.lstrip())
            end_idx = i
            for k in range(i + 1, len(lines)):
                l = lines[k]
                if not l.strip():
                    end_idx = k
                    continue
                indent = len(l) - len(l.lstrip())
                if indent <= base_indent and header_re.match(l):
                    end_idx = k - 1
                    break
                end_idx = k
            return True, {
                "symbol": sym,
                "kind": "",
                "start": i + 1,
                "end": end_idx + 1,
                "match_line": match_line,
                "path": apath,
            }
        i -= 1
    return False, {"error": "no enclosing symbol (text fallback)"}
