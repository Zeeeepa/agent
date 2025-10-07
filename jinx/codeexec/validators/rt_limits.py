from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import (
    is_enabled,
    RT_MAX_SLEEP_SECONDS as _MAX_SLEEP_SECONDS,
    RT_MAX_RANGE_CONST as _MAX_RANGE_CONST,
)


def _const_num(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def check_rt_limits(code: str) -> Optional[str]:
    """Enforce basic RT limits: avoid long sleeps, huge constant loops, and infinite loops.

    Flags:
    - time.sleep(X) where X is a numeric literal > _MAX_SLEEP_SECONDS
    - while True (literal True) without an obvious break (best-effort)
    - for i in range(N) where N is a large numeric literal > _MAX_RANGE_CONST
    """
    if not is_enabled("rt_limits", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    # Track while True nodes and whether they contain a 'break'
    infinite_while_nodes: list[ast.While] = []
    breaks_in_while: set[int] = set()

    for n in ast.walk(t):
        # time.sleep(...) clamp
        if isinstance(n, ast.Call):
            fn = getattr(n, "func", None)
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "time" and fn.attr == "sleep":
                if n.args:
                    v = _const_num(n.args[0])
                    if v is not None and v > _MAX_SLEEP_SECONDS:
                        return f"sleep too long: {v}s (> {_MAX_SLEEP_SECONDS}s limit)"
        # range(N) clamp for constant loop bounds
        if isinstance(n, ast.For):
            it = n.iter
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
                if it.args:
                    # Consider single-arg or stop argument as bound when constant
                    bound = _const_num(it.args[0]) if len(it.args) == 1 else _const_num(it.args[-1])
                    if bound is not None and bound > _MAX_RANGE_CONST:
                        return f"range bound too large: {int(bound)} (> {_MAX_RANGE_CONST})"
        # while True detection
        if isinstance(n, ast.While):
            test = n.test
            if isinstance(test, ast.Constant) and test.value is True:
                infinite_while_nodes.append(n)
            # Collect breaks inside this while
            for b in ast.walk(n):
                if isinstance(b, ast.Break):
                    breaks_in_while.add(id(n))

    for w in infinite_while_nodes:
        if id(w) not in breaks_in_while:
            return "infinite while True without break is disallowed under RT constraints"

    return None
