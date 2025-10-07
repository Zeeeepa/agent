from __future__ import annotations

from typing import Optional
import ast

from .ast_cache import get_ast
from .config import is_enabled
from .policy import (
    BANNED_NET_MODS as _BANNED_MODS,
    BANNED_NET_FUNCS as _BANNED_FUNCS,
    BANNED_NET_FROM as _BANNED_FROM,
)


def _is_attr_name(node: ast.AST, mod_name: str, attr_name: str) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == mod_name and node.attr == attr_name


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_safe_pip_install_call(call: ast.Call) -> bool:
    """Allow subprocess.check_call([sys.executable,'-m','pip','install', X])."""
    fn = call.func
    if not _is_attr_name(fn, "subprocess", "check_call"):
        return False
    if not call.args:
        return False
    arg0 = call.args[0]
    # Expect a list or tuple literal of specific structure
    seq = None
    if isinstance(arg0, (ast.List, ast.Tuple)):
        seq = arg0.elts
    if not seq or len(seq) < 4:
        return False
    # [sys.executable, '-m', 'pip', 'install', ...]
    a0 = seq[0]
    if not (isinstance(a0, ast.Attribute) and isinstance(a0.value, ast.Name) and a0.value.id == "sys" and a0.attr == "executable"):
        return False
    a1 = _const_str(seq[1])
    a2 = _const_str(seq[2])
    a3 = _const_str(seq[3])
    if not (a1 == "-m" and a2 == "pip" and a3 == "install"):
        return False
    return True


def _is_safe_open_popen(call: ast.Call) -> bool:
    """Allow subprocess.Popen(['xdg-open'|'open', url]) or Windows 'cmd /c start url'."""
    fn = call.func
    if not _is_attr_name(fn, "subprocess", "Popen"):
        return False
    if not call.args:
        return False
    arg0 = call.args[0]
    seq = None
    if isinstance(arg0, (ast.List, ast.Tuple)):
        seq = arg0.elts
    if not seq or len(seq) < 2:
        return False
    h = _const_str(seq[0]) or ""
    if h in {"xdg-open", "open"}:
        return True
    # Windows cmd start
    if h == "cmd" and len(seq) >= 3:
        a1 = _const_str(seq[1]) or ""
        a2 = _const_str(seq[2]) or ""
        if a1.lower() == "/c" and a2.lower() == "start":
            return True
    return False


def check_net_system_safety(code: str) -> Optional[str]:
    if not is_enabled("net_system_safety", True):
        return None
    t = get_ast(code)
    if not t:
        return None
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in _BANNED_MODS:
                    return f"import of '{a.name}' is disallowed"
        if isinstance(n, ast.ImportFrom):
            mod = (n.module or '').split('.')[0]
            for a in n.names:
                if (mod, a.name) in _BANNED_FROM:
                    return f"from {mod} import {a.name} is disallowed"
        if isinstance(n, ast.Call):
            fn = getattr(n, "func", None)
            # Allow safe patterns before applying bans
            if isinstance(n.func, ast.Attribute) and _is_safe_pip_install_call(n):
                continue
            if isinstance(n.func, ast.Attribute) and _is_safe_open_popen(n):
                continue
            # Otherwise enforce bans
            if isinstance(fn, ast.Name) and fn.id in _BANNED_FUNCS:
                return f"call '{fn.id}(...)' is disallowed"
            if isinstance(fn, ast.Attribute) and fn.attr in _BANNED_FUNCS:
                return f"call '...{fn.attr}(...)' is disallowed"
    return None
