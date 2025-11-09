from __future__ import annotations

import ast
import os
import time
from typing import Dict, List, Tuple, Optional

from .project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_line_window import find_line_window
from .project_lang import lang_for_file
from .project_py_scope import get_python_symbol_at_line


def _parse_ast_safe(text: str) -> Optional[ast.AST]:
    try:
        return ast.parse(text or "")
    except Exception:
        return None


def _read_text(abs_path: str) -> str:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _def_node_span(node: ast.AST) -> Tuple[int, int]:
    s = int(getattr(node, "lineno", 1) or 1)
    e = int(getattr(node, "end_lineno", s) or s)
    return s, max(e, s)


def _find_def_node(tree: ast.AST, symbol: str) -> Optional[ast.AST]:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(n, "name", "") == symbol:
            return n
    return None


def _find_callees_in_def(node: ast.AST) -> List[str]:
    names: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = getattr(sub, "func", None)
            if isinstance(fn, ast.Name):
                nm = fn.id
            elif isinstance(fn, ast.Attribute):
                nm = fn.attr
            else:
                nm = ""
            if nm and nm not in names:
                names.append(nm)
    return names


def _iter_project_py_files(limit: int, time_budget_ms: Optional[int]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    t0 = time.perf_counter()
    for abs_p, rel_p in iter_candidate_files(
        ROOT,
        include_exts=INCLUDE_EXTS,
        exclude_dirs=EXCLUDE_DIRS,
        max_file_bytes=MAX_FILE_BYTES,
    ):
        if time_budget_ms is not None and (time.perf_counter() - t0) * 1000.0 > time_budget_ms:
            break
        if not rel_p.endswith(".py"):
            continue
        out.append((abs_p, rel_p))
        if len(out) >= max(1, limit):
            break
    return out


def _find_defs_by_name(symbol: str, *, scan_cap_files: int, time_budget_ms: Optional[int]) -> List[Tuple[str, int, int, str, str]]:
    """Find definitions of symbol across project.

    Returns list of (file_rel, line_start, line_end, snippet, lang)
    """
    if not symbol:
        return []
    t0 = time.perf_counter()
    got: List[Tuple[str, int, int, str, str]] = []
    for abs_p, rel_p in _iter_project_py_files(scan_cap_files, time_budget_ms):
        if time_budget_ms is not None and (time.perf_counter() - t0) * 1000.0 > time_budget_ms:
            break
        text = _read_text(abs_p)
        if not text:
            continue
        tree = _parse_ast_safe(text)
        if not tree:
            continue
        dn = _find_def_node(tree, symbol)
        if not dn:
            continue
        s, e = _def_node_span(dn)
        a, b = s, e
        # Reuse line-window to build compact snippet around def
        _a, _b, snip = find_line_window(text, [f"def {symbol}", f"class {symbol}"], around=8)
        if not (_a or _b):
            # fallback to exact span if window not found
            lines = text.splitlines()
            snip = "\n".join(lines[max(0, a - 1) : min(len(lines), b)])
        lang = lang_for_file(rel_p)
        got.append((rel_p, a, b, snip, lang))
    return got


def _find_callers_ast(symbol: str, *, exclude_rel: str, around: int, scan_cap_files: int, time_budget_ms: Optional[int]) -> List[Tuple[str, int, int, str, str]]:
    """Find call sites to symbol across project via AST.

    Returns list of (file_rel, line_start, line_end, snippet, lang)
    """
    if not symbol:
        return []
    t0 = time.perf_counter()
    out: List[Tuple[str, int, int, str, str]] = []
    seen_files: set[str] = set()
    for abs_p, rel_p in _iter_project_py_files(scan_cap_files, time_budget_ms):
        if rel_p == exclude_rel or rel_p in seen_files:
            continue
        if time_budget_ms is not None and (time.perf_counter() - t0) * 1000.0 > time_budget_ms:
            break
        text = _read_text(abs_p)
        if not text:
            continue
        tree = _parse_ast_safe(text)
        if not tree:
            continue
        hit_lines: List[int] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = getattr(n, "func", None)
                nm = None
                if isinstance(fn, ast.Name):
                    nm = fn.id
                elif isinstance(fn, ast.Attribute):
                    nm = fn.attr
                if nm == symbol:
                    ln = int(getattr(n, "lineno", 0) or 0)
                    if ln and ln not in hit_lines:
                        hit_lines.append(ln)
        if hit_lines:
            seen_files.add(rel_p)
            # Build one snippet around the first call line
            ln = hit_lines[0]
            a, b, snip = find_line_window(text, [symbol], around=around)
            if not (a or b):
                # fallback to small window around lineno
                lines = text.splitlines()
                a = max(1, ln - around)
                b = min(len(lines), ln + around)
                snip = "\n".join(lines[a - 1 : b])
            lang = lang_for_file(rel_p)
            out.append((rel_p, a or ln, b or ln, snip, lang))
    return out


def build_symbol_graph(
    file_rel: str,
    use_ls: int,
    use_le: int,
    *,
    callers_limit: int,
    callees_limit: int,
    around: int,
    scan_cap_files: int,
    time_budget_ms: Optional[int],
) -> List[Tuple[str, str]]:
    """Build a small callgraph slice: callers of the symbol and definitions of its callees.

    Returns pairs of (header, code_block) strings ready to include in context.
    Header format: "[file_rel:ls-le] <kind>"; code_block is fenced with language.
    """
    abs_path = os.path.join(ROOT, file_rel)
    text = _read_text(abs_path)
    if not text:
        return []
    # Determine symbol at mid-line of the snippet
    mid = int((use_ls + use_le) // 2) if (use_ls and use_le) else int(use_ls or use_le or 0)
    sym_name, sym_kind = get_python_symbol_at_line(text, mid)
    if not sym_name:
        return []
    out_pairs: List[Tuple[str, str]] = []

    # 1) Callers
    callers = _find_callers_ast(
        sym_name,
        exclude_rel=file_rel,
        around=around,
        scan_cap_files=scan_cap_files,
        time_budget_ms=time_budget_ms,
    )
    for fr, a, b, snip, lang in callers[: max(0, callers_limit)]:
        hdr = f"[CALLER] [{fr}:{a}-{b}]"
        block = f"```{lang}\n{snip}\n```" if lang else f"```\n{snip}\n```"
        out_pairs.append((hdr, block))

    # 2) Callees: within this function, which names are called; then find their defs in project
    tree = _parse_ast_safe(text)
    if tree:
        dn = _find_def_node(tree, sym_name)
        if dn:
            callees = _find_callees_in_def(dn)
            seen_defs: set[Tuple[str, int, int]] = set()
            for nm in callees[: max(0, callees_limit)]:
                defs = _find_defs_by_name(nm, scan_cap_files=scan_cap_files, time_budget_ms=time_budget_ms)
                for fr, a, b, snip, lang in defs:
                    key = (fr, a, b)
                    if key in seen_defs:
                        continue
                    seen_defs.add(key)
                    hdr = f"[CALLEE DEF {nm}] [{fr}:{a}-{b}]"
                    block = f"```{lang}\n{snip}\n```" if lang else f"```\n{snip}\n```"
                    out_pairs.append((hdr, block))
    return out_pairs


def windows_for_symbol(
    symbol: str,
    *,
    prefer_rel: Optional[str] = None,
    callers_limit: int = 4,
    callees_limit: int = 4,
    around: int = 10,
    scan_cap_files: int = 200,
    time_budget_ms: Optional[int] = 600,
) -> List[Tuple[str, int, int, str]]:
    """Return list of (file_rel, line_start, line_end, kind) windows for a symbol.

    Kinds: "DEF", "CALLER", "CALLEE_DEF".
    Only Python files are considered.
    """
    out: List[Tuple[str, int, int, str]] = []
    if not symbol:
        return out
    # 1) If we know preferred file, include its DEF scope
    def _append_def_scope(rel: str) -> None:
        abs_p = os.path.join(ROOT, rel)
        text = _read_text(abs_p)
        if not text:
            return
        tree = _parse_ast_safe(text)
        if not tree:
            return
        dn = _find_def_node(tree, symbol)
        if not dn:
            return
        a, b = _def_node_span(dn)
        out.append((rel, a, b, "DEF"))

    if prefer_rel:
        _append_def_scope(prefer_rel)

    # 2) Callers across project
    callers = _find_callers_ast(
        symbol,
        exclude_rel=prefer_rel or "",
        around=around,
        scan_cap_files=scan_cap_files,
        time_budget_ms=time_budget_ms,
    )
    for rel, a, b, _snip, _lang in callers[: max(0, callers_limit)]:
        out.append((rel, a, b, "CALLER"))

    # 3) Callee definitions if we have a DEF scope to inspect
    base_rel = prefer_rel
    if not base_rel:
        # Try to discover one DEF quickly
        defs = _find_defs_by_name(symbol, scan_cap_files=scan_cap_files, time_budget_ms=time_budget_ms)
        if defs:
            base_rel = defs[0][0]
            out.append((defs[0][0], defs[0][1], defs[0][2], "DEF"))

    if base_rel:
        abs_p = os.path.join(ROOT, base_rel)
        text = _read_text(abs_p)
        tree = _parse_ast_safe(text)
        if tree:
            dn = _find_def_node(tree, symbol)
            if dn:
                callees = _find_callees_in_def(dn)
                seen_keys: set[Tuple[str, int, int]] = set()
                for nm in callees[: max(0, callees_limit)]:
                    defs = _find_defs_by_name(nm, scan_cap_files=scan_cap_files, time_budget_ms=time_budget_ms)
                    for rel, a, b, _snip, _lang in defs:
                        key = (rel, a, b)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        out.append((rel, a, b, "CALLEE_DEF"))
    return out
