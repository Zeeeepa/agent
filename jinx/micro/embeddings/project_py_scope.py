from __future__ import annotations

import ast
from typing import Tuple, Optional
import functools


@functools.lru_cache(maxsize=128)
def _parse_ast_cached(source: str):
    try:
        return ast.parse(source)
    except Exception:
        return None


def find_python_scope(source: str, line: int) -> Tuple[int, int]:
    """Return (start_line, end_line) of the smallest Python def/class containing 'line'.

    If not found or AST lacks end positions, returns (0, 0).
    """
    tree = _parse_ast_cached(source)
    if tree is None:
        return 0, 0

    best_span = (0, 10**9)

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[name-defined]
            _check(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[name-defined]
            _check(node)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[name-defined]
            _check(node)
            self.generic_visit(node)

    def _check(node: ast.AST) -> None:
        nonlocal best_span
        ln = getattr(node, "lineno", None)
        en = getattr(node, "end_lineno", None)
        if ln is None or en is None:
            return
        if ln <= line <= en:
            # choose the smallest span that still contains the line
            cur_span = (ln, en)
            cur_size = en - ln
            best_size = best_span[1] - best_span[0]
            if best_span == (0, 10**9) or cur_size < best_size:
                best_span = cur_span

    _Visitor().visit(tree)
    if best_span == (0, 10**9):
        return 0, 0
    return best_span


def get_python_symbol_at_line(source: str, line: int) -> Tuple[Optional[str], Optional[str]]:
    """Return (name, kind) for the smallest def/class containing 'line'.

    kind is one of: 'def', 'async def', 'class'. Returns (None, None) if not found.
    """
    tree = _parse_ast_cached(source)
    if tree is None:
        return None, None

    best: Tuple[Optional[str], Optional[str], int, int] = (None, None, 0, 10**9)

    def _try(node: ast.AST, kind: str, name: str) -> None:
        nonlocal best
        ln = getattr(node, "lineno", None)
        en = getattr(node, "end_lineno", None)
        if ln is None or en is None:
            return
        if ln <= line <= en:
            cur_size = en - ln
            _, _, b_s, b_e = best
            best_size = b_e - b_s
            if best[0] is None or cur_size < best_size:
                best = (name, kind, ln, en)

    class _V(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[name-defined]
            _try(node, 'def', getattr(node, 'name', ''))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[name-defined]
            _try(node, 'async def', getattr(node, 'name', ''))
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[name-defined]
            _try(node, 'class', getattr(node, 'name', ''))
            self.generic_visit(node)

    _V().visit(tree)
    return best[0], best[1]
