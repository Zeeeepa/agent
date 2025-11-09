from __future__ import annotations

import ast
from typing import Tuple


def syntax_ok(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    except Exception as e:
        return False, f"parse error: {e}"


def libcst_ok(code: str) -> Tuple[bool, str]:
    try:
        import libcst as cst  # type: ignore
    except Exception:
        return True, "libcst not available"
    try:
        _ = cst.parse_module(code)
        return True, ""
    except Exception as e:
        return False, f"libcst parse error: {e}"
