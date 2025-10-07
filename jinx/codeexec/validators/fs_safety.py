from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled

_WRITE_MODES = {"w", "a", "x"}


def _is_writing_mode(node: ast.AST) -> bool:
    # mode can be positional second arg or keyword 'mode'
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        m = node.value
        return any(ch in m for ch in _WRITE_MODES) or "+" in m
    return False


def check_fs_safety(code: str) -> Optional[str]:
    """Disallow direct filesystem writes/deletions; require patcher usage.

    Blocks:
    - open(path, mode=... with write/append/create)
    - Path(...).write_text / write_bytes
    - os.remove/unlink/rename; shutil.rmtree
    """
    if not is_enabled("fs_safety", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Call):
            fn = getattr(n, "func", None)
            # open(..., 'w') etc
            if isinstance(fn, ast.Name) and fn.id == "open":
                # positional
                if len(n.args) >= 2 and _is_writing_mode(n.args[1]):
                    return "direct file write is disallowed; use jinx.micro.runtime.patcher"
                # keyword
                for kw in (n.keywords or []):
                    if (kw.arg or "") == "mode" and _is_writing_mode(kw.value):
                        return "direct file write is disallowed; use jinx.micro.runtime.patcher"
            # Path(...).write_text / write_bytes
            if isinstance(fn, ast.Attribute) and fn.attr in {"write_text", "write_bytes"}:
                return "direct file write is disallowed; use jinx.micro.runtime.patcher"
            # os.remove/unlink/rename
            if isinstance(fn, ast.Attribute) and fn.attr in {"remove", "unlink", "rename"}:
                val = getattr(fn, "value", None)
                if isinstance(val, ast.Name) and val.id == "os":
                    return f"destructive filesystem op 'os.{fn.attr}(...)' is disallowed"
            # shutil.rmtree
            if isinstance(fn, ast.Attribute) and fn.attr in {"rmtree"}:
                val = getattr(fn, "value", None)
                if isinstance(val, ast.Name) and val.id == "shutil":
                    return "destructive filesystem op 'shutil.rmtree(...)' is disallowed"
    return None
