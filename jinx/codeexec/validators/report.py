from __future__ import annotations

from typing import Any, Dict, List, Optional
import ast
import io
import tokenize

from .ast_cache import get_ast

# Reuse policy details from sibling validators without importing them to avoid cycles
_BANNED_DYN = {"eval", "exec", "compile", "__import__"}
_BANNED_NET_MODS = {"socket", "ftplib", "telnetlib"}
_BANNED_NET_FUNCS = {"system", "popen", "Popen", "call", "check_call", "check_output"}
_BANNED_NET_FROM = {("os", "system"), ("subprocess", "Popen"), ("subprocess", "call"), ("subprocess", "check_call"), ("subprocess", "check_output")}
_MAX_SLEEP_SECONDS = 2.0
_MAX_RANGE_CONST = 100000
_MAX_LOOP_BODY_LINES = 400
_MAX_LIST_LITERAL_ELEMS = 1000


def _const_num(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _tokenize(src: str):
    try:
        return list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return []


def _find_triple_quote_line(src: str) -> Optional[int]:
    for tok in _tokenize(src):
        if tok.type == tokenize.STRING:
            s = tok.string
            i = 0
            while i < len(s) and s[i] in "rRuUfFbB":
                i += 1
            if s[i : i + 3] in ("'''", '"""'):
                return int(tok.start[0])
    # fallback textual
    if "'''" in src or '"""' in src:
        return 1
    return None


def _first(node_iter, pred):
    for n in node_iter:
        if pred(n):
            return n
    return None


def _iter_ast(tree: Optional[ast.AST]):
    return ast.walk(tree) if tree is not None else []


def collect_violations_detailed(code: str) -> List[Dict[str, Any]]:
    """Analyze code and return structured violations with line hints.

    This complements the simple string list returned by collect_violations and is
    safe to consume by logs/telemetry. It does not raise.
    """
    src = code or ""
    out: List[Dict[str, Any]] = []

    # Syntax error pass (use direct parse to capture error location)
    try:
        ast.parse(src)
    except SyntaxError as e:
        out.append({
            "id": "syntax",
            "category": "structure",
            "msg": f"syntax error: {e.msg}",
            "line": int(getattr(e, 'lineno', 1) or 1),
        })
        # Continue; other checks may still add useful hints

    t = get_ast(src)

    # try/except
    tr = _first(_iter_ast(t), lambda n: isinstance(n, ast.Try))
    if tr is not None:
        out.append({
            "id": "try_except",
            "category": "policy",
            "msg": "Usage of try/except/finally is not allowed by prompt",
            "line": int(getattr(tr, 'lineno', 1) or 1),
        })

    # triple quotes
    tq_line = _find_triple_quote_line(src)
    if tq_line is not None:
        out.append({
            "id": "triple_quotes",
            "category": "policy",
            "msg": "Triple quotes are not allowed by prompt",
            "line": tq_line,
        })

    # banned dynamic calls
    dyn = _first(
        (n for n in _iter_ast(t) if isinstance(n, ast.Call)),
        lambda c: isinstance(getattr(c, 'func', None), ast.Name) and getattr(c.func, 'id', '') in _BANNED_DYN
                or (isinstance(getattr(c, 'func', None), ast.Attribute) and isinstance(getattr(c.func, 'value', None), ast.Name) and c.func.value.id == 'importlib' and c.func.attr == 'import_module')
    )
    if dyn is not None:
        fn = dyn.func
        name = getattr(fn, 'id', None) or (f"{getattr(getattr(fn,'value', None),'id', '?')}.{getattr(fn,'attr','?')}" if isinstance(fn, ast.Attribute) else 'call')
        out.append({
            "id": "dynamic",
            "category": "safety",
            "msg": f"dynamic call '{name}(...)' is disallowed",
            "line": int(getattr(dyn, 'lineno', 1) or 1),
        })

    # net/system safety
    for n in _iter_ast(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                if (a.name or '').split('.')[0] in _BANNED_NET_MODS:
                    out.append({
                        "id": "net_import",
                        "category": "safety",
                        "msg": f"import of '{a.name}' is disallowed",
                        "line": int(getattr(n, 'lineno', 1) or 1),
                    })
                    break
        if isinstance(n, ast.ImportFrom):
            mod = (n.module or '').split('.')[0]
            for a in n.names:
                if (mod, a.name) in _BANNED_NET_FROM:
                    out.append({
                        "id": "net_from",
                        "category": "safety",
                        "msg": f"from {mod} import {a.name} is disallowed",
                        "line": int(getattr(n, 'lineno', 1) or 1),
                    })
                    break
        if isinstance(n, ast.Call):
            fn = getattr(n, 'func', None)
            if isinstance(fn, ast.Name) and fn.id in _BANNED_NET_FUNCS:
                out.append({
                    "id": "net_call",
                    "category": "safety",
                    "msg": f"call '{fn.id}(...)' is disallowed",
                    "line": int(getattr(n, 'lineno', 1) or 1),
                })
            if isinstance(fn, ast.Attribute) and fn.attr in _BANNED_NET_FUNCS:
                out.append({
                    "id": "net_call",
                    "category": "safety",
                    "msg": f"call '...{fn.attr}(...)' is disallowed",
                    "line": int(getattr(n, 'lineno', 1) or 1),
                })

    # filesystem writes/deletes
    for n in _iter_ast(t):
        if isinstance(n, ast.Call):
            fn = getattr(n, 'func', None)
            if isinstance(fn, ast.Name) and fn.id == 'open':
                if len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
                    m = n.args[1].value
                    if any(ch in m for ch in ('w','a','x','+')):
                        out.append({
                            "id": "fs_write",
                            "category": "safety",
                            "msg": "direct file write is disallowed; use jinx.micro.runtime.patcher",
                            "line": int(getattr(n, 'lineno', 1) or 1),
                        })
            if isinstance(fn, ast.Attribute) and fn.attr in {'write_text','write_bytes'}:
                out.append({
                    "id": "fs_write",
                    "category": "safety",
                    "msg": "direct file write is disallowed; use jinx.micro.runtime.patcher",
                    "line": int(getattr(n, 'lineno', 1) or 1),
                })

    # blocking input
    bl = _first(
        (n for n in _iter_ast(t) if isinstance(n, ast.Call)),
        lambda c: (isinstance(getattr(c,'func', None), ast.Name) and getattr(c.func,'id','') == 'input')
               or (isinstance(getattr(c,'func', None), ast.Attribute) and c.func.attr == 'readline' and isinstance(getattr(c.func, 'value', None), ast.Attribute) and c.func.value.attr == 'stdin')
    )
    if bl is not None:
        out.append({
            "id": "blocking_io",
            "category": "rt",
            "msg": "blocking input/readline is disallowed under RT constraints",
            "line": int(getattr(bl, 'lineno', 1) or 1),
        })

    # RT limits
    for n in _iter_ast(t):
        if isinstance(n, ast.Call):
            fn = getattr(n, 'func', None)
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == 'time' and fn.attr == 'sleep':
                if n.args:
                    v = _const_num(n.args[0])
                    if v is not None and v > _MAX_SLEEP_SECONDS:
                        out.append({
                            "id": "sleep_long",
                            "category": "rt",
                            "msg": f"sleep too long: {v}s (> {_MAX_SLEEP_SECONDS}s)",
                            "line": int(getattr(n, 'lineno', 1) or 1),
                        })
        if isinstance(n, ast.For):
            it = n.iter
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'range':
                if it.args:
                    bound = _const_num(it.args[0]) if len(it.args) == 1 else _const_num(it.args[-1])
                    if bound is not None and bound > _MAX_RANGE_CONST:
                        out.append({
                            "id": "range_large",
                            "category": "rt",
                            "msg": f"range bound too large: {int(bound)} (> {_MAX_RANGE_CONST})",
                            "line": int(getattr(n, 'lineno', 1) or 1),
                        })
        if isinstance(n, ast.While):
            if isinstance(n.test, ast.Constant) and n.test.value is True:
                # see if break exists inside
                has_break = any(isinstance(b, ast.Break) for b in ast.walk(n))
                if not has_break:
                    out.append({
                        "id": "while_true",
                        "category": "rt",
                        "msg": "infinite while True without break is disallowed under RT constraints",
                        "line": int(getattr(n, 'lineno', 1) or 1),
                    })

    # IO clamps (oversized bodies / huge literals)
    for n in _iter_ast(t):
        if isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
            # approximate span: end_lineno - lineno
            a = int(getattr(n, 'lineno', 0) or 0)
            b = int(getattr(n, 'end_lineno', a) or a)
            if max(0, b - a) > _MAX_LOOP_BODY_LINES:
                out.append({
                    "id": "loop_huge",
                    "category": "rt",
                    "msg": "loop body too large",
                    "line": a or 1,
                })
        if isinstance(n, (ast.List, ast.Set, ast.Tuple)):
            elts = getattr(n, 'elts', []) or []
            if len(elts) > _MAX_LIST_LITERAL_ELEMS:
                out.append({
                    "id": "literal_huge",
                    "category": "rt",
                    "msg": "literal too large",
                    "line": int(getattr(n, 'lineno', 1) or 1),
                })
        if isinstance(n, ast.Dict):
            keys = getattr(n, 'keys', []) or []
            if len(keys) > _MAX_LIST_LITERAL_ELEMS:
                out.append({
                    "id": "dict_huge",
                    "category": "rt",
                    "msg": "dict literal too large",
                    "line": int(getattr(n, 'lineno', 1) or 1),
                })

    # Side-effects sentinel policy: if code intends to open things, require sentinel prints
    intends_open = any(
        isinstance(n, ast.Call) and (
            (isinstance(getattr(n,'func',None), ast.Attribute) and getattr(n.func,'attr','') == 'open') or
            (isinstance(getattr(n,'func',None), ast.Attribute) and isinstance(getattr(n.func,'value',None), ast.Name) and n.func.value.id == 'webbrowser' and n.func.attr == 'open') or
            (isinstance(getattr(n,'func',None), ast.Attribute) and isinstance(getattr(n.func,'value',None), ast.Name) and n.func.value.id == 'os' and n.func.attr == 'startfile') or
            (isinstance(getattr(n,'func',None), ast.Attribute) and isinstance(getattr(n.func,'value',None), ast.Name) and n.func.value.id == 'subprocess' and n.func.attr == 'Popen')
        ) for n in _iter_ast(t)
    )
    if intends_open and (('OK:' not in src) and ('ERROR:' not in src) and ('<<JINX_ERROR>>' not in src)):
        out.append({
            "id": "side_effects",
            "category": "policy",
            "msg": "side-effect must verify with sentinel prints (OK:/ERROR:)",
            "line": 1,
        })

    return out
