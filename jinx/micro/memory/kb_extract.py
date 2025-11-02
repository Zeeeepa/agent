from __future__ import annotations

import re
import time
from typing import List, Tuple, Dict
import os
from jinx.micro.text.structural import (
    NAME as _NAME,
    is_camel_case as _is_camel,
    is_pathlike as _is_pathlike,
    match_paren as _match_paren,
    match_bracket as _match_bracket,
    split_top_args as _split_top_args,
)

# Lightweight, heuristic extraction of (subject, predicate, object) triplets

# Structural-only extractor (no word-based pattern rules)

# No pattern metadata in structural mode


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def _clean_entity(x: str, *, max_len: int = 200) -> str:
    """Aggressive entity sanitizer: trim quotes/backticks, collapse space, clamp length."""
    s = _norm(x)
    # Strip surrounding quotes/backticks/brackets
    s = re.sub(r"^[`'\"]+|[`'\"]+$", "", s)
    s = re.sub(r"^[\[\(\{]+|[\)\}\]]+$", "", s)
    if max_len > 0 and len(s) > max_len:
        s = s[:max_len]
    return s


def _split_entities(s: str) -> List[str]:
    """Structural split only: commas/pipes; no word-based cues."""
    t = _clean_entity(s)
    if not t:
        return []
    if any(ch in t for ch in ("/", "\\", "::")) or ("." in t and " " not in t):
        return [t]
    parts = re.split(r"\s*(?:,|;|\|)\s*", t)
    out = [p for p in (parts or []) if p]
    return out or [t]


def _chain_edges(text: str) -> List[Tuple[str, str, str]]:
    """Compatibility delegate."""
    return _extract_arrows(text)


def _trip_from_match(pat: re.Pattern, groups: tuple[int, int, int], text: str) -> List[Tuple[str, str, str]]:
    # Disabled in structural mode
    return []


# No language-based negation in structural mode


_PRED_W = {
    "->": 1.5,   # chains and mappings
    "==": 1.2,   # equality/aliasing
    "()": 1.3,   # call/association
    ".": 1.0,    # dotted membership
    "/": 1.0,    # path membership
    "@": 1.0,    # decorator association
    "<>": 1.1,   # generic/parameter-of
    "[]": 1.0,   # index/subscript
    "::": 1.0,   # scope resolution
    ":": 0.95,   # type annotation
    ">>": 0.95,  # structural chain order
    "{}": 1.0,   # object field membership
}


# _is_pathlike and _is_camel are imported from jinx.micro.text.structural


def _score(tri: Tuple[str, str, str], line: str) -> float:
    a, p, b = tri
    base = _PRED_W.get(p, 1.0)
    if base == 1.0 and p.startswith("ret#"):
        base = _PRED_W.get("()", 1.0)
    sc = 1.0 + base - 1.0
    if _is_pathlike(a) or _is_pathlike(b):
        sc += 0.2
    if _is_camel(a) or _is_camel(b):
        sc += 0.15
    if len(a) > 160 or len(b) > 160:
        sc -= 0.2
    return sc


# ---- Structural helpers (language-agnostic) ----
_LABEL_RE = re.compile(rf"^\s*{_NAME}\s*:\s*(.*)$")


def _strip_label(s: str) -> str:
    """Drop leading label like 'foo: value' and return the value part; if no label, return s."""
    m = _LABEL_RE.match(s or "")
    return (m.group(1) if m else s) or ""


_HEAD_LABEL_RE = re.compile(rf"^\s*({_NAME})\s*:\s*(.+)$")


def _extract_head_label(s: str) -> List[Tuple[str, str, str]]:
    """Extract a head label edge (value, ':', label) from lines like 'label: value'."""
    m = _HEAD_LABEL_RE.match(s or "")
    if not m:
        return []
    label = _clean_entity(m.group(1))
    val = _clean_entity(m.group(2), max_len=200)
    if not label or not val:
        return []
    # Avoid Windows drive letters like "C:\path"
    if len(label) == 1 and (val.startswith("\\") or val.startswith("/")):
        return []
    # Avoid C++ scope lines like ns::Type.method
    if "::" in (s or ""):
        return []
    return [(val, ":", label)]


def _extract_arrows(s: str) -> List[Tuple[str, str, str]]:
    if not s:
        return []
    # Skip Python/TS return type arrows; handled by return-type extractors
    try:
        if _RET_PY_RE.search(s) or _RET_RE.search(s):
            return []
    except Exception:
        pass
    if ("->" not in s and "=>" not in s and "→" not in s):
        return []
    toks = re.split(r"\s*(?:->|=>|→)\s*", s)
    toks = [_clean_entity(t) for t in toks if _clean_entity(t)]
    out: List[Tuple[str, str, str]] = []
    for i in range(len(toks) - 1):
        out.append((toks[i], "->", toks[i + 1]))
    return out


def _lhs_membership_edges(lhs: str) -> List[Tuple[str, str, str]]:
    """Produce structural membership edges from LHS like obj.field or arr[idx]."""
    edges: List[Tuple[str, str, str]] = []
    lhst = (lhs or "").strip()
    # dot membership
    if "." in lhst and " " not in lhst:
        pre, last = lhst.rsplit(".", 1)
        pre_c = _clean_entity(pre, max_len=160)
        last_c = _clean_entity(last, max_len=120)
        if pre_c and last_c:
            edges.append((last_c, "{}", pre_c))
    # bracket membership
    i = lhst.rfind("[")
    if i != -1:
        j = lhst.find("]", i + 1)
        if j != -1:
            cont = _clean_entity(lhst[:i], max_len=160)
            idx = _clean_entity(lhst[i + 1:j], max_len=120)
            if cont and idx:
                edges.append((idx, "[]", cont))
    return edges


def _extract_assignment(s: str) -> List[Tuple[str, str, str]]:
    """Extract simple assignment edges avoiding =>, ==, >=, <=, := ."""
    m = re.search(r"(?<![<>=:])=(?![=><])", s or "")
    if not m:
        return []
    lhs = (s[: m.start()] or "").strip()
    rhs = (s[m.end():] or "").strip()
    if not lhs or not rhs:
        return []
    edges: List[Tuple[str, str, str]] = []
    lhsc = _clean_entity(lhs, max_len=200)
    rhsc = _clean_entity(rhs, max_len=200)
    # Skip equality edge for destructuring LHS or object/array RHS
    is_destructuring = lhs.lstrip().startswith(("(", "[")) or "," in lhs
    rhs0 = rhs.lstrip()[:1]
    if lhsc and rhsc and (not is_destructuring) and (rhs0 not in ("{", "[")):
        edges.append((lhsc, "==", rhsc))
    edges += _lhs_membership_edges(lhs)
    return edges


def _extract_paths_backslash(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for token in re.findall(r"[^\s]+\\[^\s]+", s or ""):
        if '://' in token:
            continue
        t = token.strip('"\'')
        base = t.split('\\')[-1]
        sep_idx = t.rfind('\\')
        dirn = t[:sep_idx] if sep_idx != -1 else ''
        if dirn.endswith('\\'):
            dirn = dirn[:-1]
        if base and dirn:
            out.append((base, "/", dirn))
        segs = [seg for seg in t.split('\\') if seg]
        for i in range(len(segs) - 1):
            parent = '\\'.join(segs[: i + 1])
            child = segs[i + 1]
            if parent and child:
                out.append((child, "/", parent))
    return out


def _extract_object_array_literals(s: str) -> List[Tuple[str, str, str]]:
    """Extract field/element membership from structural literals:
    name = { key: value, ... } => (key, "{}", name)
    name = [ a, b ]           => (a,   "[]", name)
    """
    out: List[Tuple[str, str, str]] = []
    if not s or '=' not in s:
        return out
    # Try object literal
    m = re.search(r"\b([A-Za-z_][\w]*)\s*=\s*\{", s)
    if m:
        name = _clean_entity(m.group(1))
        i_open = s.find('{', m.end() - 1)
        j = _match_bracket(s, i_open, '{', '}') if i_open != -1 else -1
        if name and i_open != -1 and j != -1 and (j - i_open) <= 800:
            body = s[i_open + 1:j]
            # find top-level keys ending with ':'
            for km in re.finditer(r"\b([A-Za-z_][\w]*)\s*:\s*", body):
                key = _clean_entity(km.group(1))
                if key:
                    out.append((key, "{}", name))
                if len(out) >= 12:
                    break
    # Try array literal
    ma = re.search(r"\b([A-Za-z_][\w]*)\s*=\s*\[", s)
    if ma:
        name = _clean_entity(ma.group(1))
        i_open = s.find('[', ma.end() - 1)
        j = _match_bracket(s, i_open, '[', ']') if i_open != -1 else -1
        if name and i_open != -1 and j != -1 and (j - i_open) <= 800:
            body = s[i_open + 1:j]
            # split top-level by commas
            for part in [x.strip() for x in body.split(',')[:8] if x.strip()]:
                elem = _clean_entity(part)
                if elem:
                    out.append((elem, "[]", name))
    return out


def _next_name(s: str, pos: int) -> Tuple[str, int]:
    m = re.search(rf"{_NAME}", s[pos:])
    if not m:
        return ("", pos)
    start = pos + m.start()
    end = pos + m.end()
    return (_clean_entity(s[start:end]), end)


def _extract_chain_order(s: str) -> List[Tuple[str, str, str]]:
    """Extract structural adjacency edges between name tokens in expression chains.

    Example: a.b(c).d[e] -> (a,>>,b), (b,>>,d)
    """
    out: List[Tuple[str, str, str]] = []
    L = len(s or "")
    i = 0
    # find first name
    prev, i = _next_name(s, 0)
    if not prev:
        return out
    steps = 0
    while i < L and steps < 8:
        # skip spaces
        while i < L and s[i].isspace():
            i += 1
        if i >= L:
            break
        # handle operators that do not change current subject ((), [], etc.)
        if s[i] == '(':
            j = _match_paren(s, i)
            if j == -1:
                break
            i = j + 1
            continue
        if s[i] == '[':
            j = _match_bracket(s, i, '[', ']')
            if j == -1:
                break
            i = j + 1
            continue
        # dot
        if s[i] == '.':
            name, ni = _next_name(s, i + 1)
            if name:
                out.append((prev, ">>", name))
                prev = name
                i = ni
                steps += 1
                continue
            i += 1
            continue
        # scope '::'
        if s[i] == ':' and (i + 1) < L and s[i + 1] == ':':
            name, ni = _next_name(s, i + 2)
            if name:
                out.append((prev, ">>", name))
                prev = name
                i = ni
                steps += 1
                continue
            i += 2
            continue
        # otherwise, try to find next top-level name
        name, ni = _next_name(s, i)
        if name:
            out.append((prev, ">>", name))
            prev = name
            i = ni
            steps += 1
            continue
        i += 1
    return out


# Destructuring assignment: (a,b)=callee(...), [a,b]=callee(...)
_TUPLE_RET_PAREN_RE = re.compile(r"^\s*\(\s*([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_][\w]*)+)\s*\)\s*=\s*([^\s\(\)]+)\s*\(")
_TUPLE_RET_BRACK_RE = re.compile(r"^\s*\[\s*([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_][\w]*)+)\s*\]\s*=\s*([^\s\(\)]+)\s*\(")


def _extract_tuple_returns(s: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    m = _TUPLE_RET_PAREN_RE.match(s or "")
    mb = _TUPLE_RET_BRACK_RE.match(s or "") if not m else None
    if not m and not mb:
        return triples
    mm = m or mb
    vars_part = mm.group(1)
    callee = _clean_entity(mm.group(2))
    if not vars_part or not callee:
        return triples
    for idx, v in enumerate([_clean_entity(x) for x in vars_part.split(',')][:6]):
        if not v:
            continue
        triples.append((_clean_entity(callee), f"ret#{idx}", v))
    return triples


_RET_PY_RE = re.compile(rf"([^\s\(\)]+)\s*\([^)]*\)\s*->\s*({_NAME}[\w\.<\>\[\]]*)")


def _extract_return_types_py(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for m in _RET_PY_RE.finditer(s or ""):
        name = _clean_entity(m.group(1))
        typ = _clean_entity(m.group(2))
        if name and typ:
            out.append((name, ":", typ))
        if len(out) >= 6:
            break
    return out


_RET_RE = re.compile(r"([^\s\(\)]+)\s*\([^)]*\)\s*:\s*([A-Za-z_][\w\.\[\]<>]*)")


def _extract_return_types(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for m in _RET_RE.finditer(s or ""):
        name = _clean_entity(m.group(1))
        typ = _clean_entity(m.group(2))
        if name and typ:
            out.append((name, ":", typ))
        if len(out) >= 6:
            break
    return out


_COLON_RE = re.compile(r"\b([A-Za-z_][\w]*)\s*:\s*([A-Za-z_][\w\.\[\]<>]*)\b")


def _extract_colon_types(s: str) -> List[Tuple[str, str, str]]:
    """Extract NAME:TYPE pairs; structural only. Avoid when TYPE looks like plain scalar value with spaces."""
    out: List[Tuple[str, str, str]] = []
    # Skip when looks like object literal assignment: name = { key: value }
    if re.search(r"=\s*\{", s or ""):
        return out
    for m in _COLON_RE.finditer(s or ""):
        name = _clean_entity(m.group(1))
        typ = _clean_entity(m.group(2))
        if name and typ:
            out.append((name, ":", typ))
        if len(out) >= 8:
            break
    return out


def _extract_equals(s: str) -> List[Tuple[str, str, str]]:
    if not s or ("=" not in s):
        return []
    # Prefer '==' over '='; avoid '=>'
    m = re.search(r"(.+?)\s*==\s*(.+)$", s)
    if not m:
        # avoid => and >= <= by negative lookahead/behind
        m = re.search(r"(?<![<>:=-])\s(.+?)\s*=\s*(.+?)(?![>=])$", s)
    if not m:
        return []
    a = _clean_entity(m.group(1))
    b = _clean_entity(m.group(2))
    if not a or not b:
        return []
    return [(a, "==", b)]


# split_top_args is imported from jinx.micro.text.structural


_CALL_RE = re.compile(r"([^\s\(\)]+)\s*\(")  # sentinel, not used directly for heavy parsing


# match_paren is imported from jinx.micro.text.structural


# match_bracket is imported from jinx.micro.text.structural


def _iter_calls(s: str, *, max_calls: int = 4) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    i = 0
    L = len(s or "")
    while i < L and len(out) < max_calls:
        j = s.find('(', i)
        if j == -1:
            break
        # find token before '('
        k = j - 1
        while k >= 0 and s[k].isspace():
            k -= 1
        if k < 0:
            break
        # token chars
        t_end = k + 1
        while k >= 0 and not s[k].isspace() and s[k] not in '()':
            k -= 1
        func = _clean_entity(s[k+1:t_end])
        if not func:
            i = j + 1
            continue
        j2 = _match_paren(s, j)
        if j2 == -1:
            break
        args_body = s[j+1:j2]
        out.append((func, args_body))
        i = j2 + 1
    return out


def _extract_calls(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    try:
        max_calls = max(1, int(os.getenv("JINX_KB_EX_MAX_CALLS", "4")))
    except Exception:
        max_calls = 4
    try:
        max_args = max(1, int(os.getenv("JINX_KB_EX_MAX_ARGS", "6")))
    except Exception:
        max_args = 6
    for func, body in _iter_calls(s, max_calls=max_calls):
        for idx, a in enumerate(_split_top_args(body)[:max_args]):
            if not a:
                continue
            fn = _clean_entity(func, max_len=160)
            arg = _clean_entity(a, max_len=160)
            out.append((fn, "()", arg))
            # Positional argument edge to preserve order structurally
            out.append((fn, f"arg#{idx}", arg))
            if len(out) >= max_args * max_calls * 2:
                break
        if len(out) >= max_args * max_calls * 2:
            break
    return out


def _extract_dots(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    # Extract dotted identifiers like A.B.C
    for token in re.findall(rf"{_NAME}(?:\.{_NAME})+", s or ""):
        parts = token.split('.')
        for i in range(len(parts) - 1):
            child = parts[i + 1]
            parent = '.'.join(parts[: i + 1])
            out.append((child, ".", parent))
    # Scope resolution with '::'
    for token in re.findall(rf"{_NAME}(?:::{_NAME})+", s or ""):
        parts = token.split('::')
        for i in range(len(parts) - 1):
            child = parts[i + 1]
            parent = '::'.join(parts[: i + 1])
            out.append((child, "::", parent))
    return out


def _extract_paths(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    # Simple path-like tokens without spaces; avoid URL schemes by skipping '://'
    for token in re.findall(r"[^\s]+/[^\s]+", s or ""):
        if '://' in token:
            continue
        t = token.strip('\"\'')
        base = t.rsplit('/', 1)[-1]
        dirn = t[:-len(base)-1] if '/' in t else ''
        if base and dirn:
            out.append((base, "/", dirn))
        # chain edges across segments
        segs = [seg for seg in t.split('/') if seg]
        for i in range(len(segs) - 1):
            parent = '/'.join(segs[: i + 1])
            child = segs[i + 1]
            if parent and child:
                out.append((child, "/", parent))
    return out

def _iter_generics(s: str, *, max_hits: int = 4) -> List[Tuple[str, str]]:
    """Return list of (host, params_text) for occurrences of Host<...> with nested depth support."""
    out: List[Tuple[str, str]] = []
    i = 0
    L = len(s or "")
    while i < L and len(out) < max_hits:
        j = s.find('<', i)
        if j == -1:
            break
        # find host token before '<'
        k = j - 1
        while k >= 0 and s[k].isspace():
            k -= 1
        if k < 0:
            break
        t_end = k + 1
        while k >= 0 and not s[k].isspace() and s[k] not in '<>()[]{}':
            k -= 1
        host = _clean_entity(s[k+1:t_end])
        if not host:
            i = j + 1
            continue
        # match nested angle brackets
        depth = 0
        m = j
        while m < L:
            ch = s[m]
            if ch == '<':
                depth += 1
            elif ch == '>':
                depth -= 1
                if depth == 0:
                    break
            m += 1
        if depth != 0:
            break
        params = s[j+1:m]
        out.append((host, params))
        i = m + 1
    return out


def _extract_generics(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for host, params in _iter_generics(s, max_hits=4):
        parts = [x.strip() for x in re.split(r"\s*(?:,|\|)\s*", params) if x.strip()]
        for p in parts[:5]:
            out.append((_clean_entity(p, max_len=120), "<>", _clean_entity(host)))
        if len(out) >= 10:
            break
    return out


_BRACK_RE = re.compile(r"([^\s\[\]]+)\[([^\[\]]+)\]")


def _extract_brackets(s: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for m in _BRACK_RE.finditer(s or ""):
        container = _clean_entity(m.group(1))
        key = _clean_entity(m.group(2))
        if container and key:
            out.append((key, "[]", container))
        if len(out) >= 8:
            break
    return out


# No dynamic word-based relations in structural mode


def extract_triplets(lines: List[str], *, max_items: int = 120, max_time_ms: int = 60) -> List[Tuple[str, str, str]]:
    """Extract up to max_items triplets from provided lines within time budget.

    Heuristics focus on evergreen-style lines and simple relational hints.
    """
    t0 = time.perf_counter()
    cands: List[Tuple[float, Tuple[str, str, str]]] = []
    seen: set[Tuple[str, str, str]] = set()
    try:
        fac = max(1, int(os.getenv("JINX_KB_EX_MAX_CANDS_FACTOR", "6")))
    except Exception:
        fac = 6
    max_cands = max_items * fac
    DEC_RE = re.compile(r"^\s*@([^\s(]+)")
    pending_decorators: List[str] = []
    def _gate(name: str, default: str = "1") -> bool:
        try:
            return str(os.getenv(f"JINX_KB_EX_{name}", default)).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            return default not in ("", "0", "false", "off", "no")
    for ln in lines:
        s_raw = (ln or "").strip()
        if not s_raw or len(s_raw) < 3:
            continue
        # Decorators accumulate until a callable/entity appears
        mdec = DEC_RE.match(s_raw)
        if mdec:
            name = _clean_entity(mdec.group(1))
            if name:
                pending_decorators.append(name)
            continue
        # Head label (label: value) before stripping
        triples: List[Tuple[str, str, str]] = []
        if _gate("HEADLABEL", "1"):
            triples.extend(_extract_head_label(s_raw))
        s = _strip_label(_norm(s_raw))
        if _gate("ARROWS", "1"):
            triples.extend(_extract_arrows(s))
        if len(triples) < 1 and _gate("EQUALS", "1"):
            triples.extend(_extract_equals(s))
        # Assignment-specific structural edges (== + membership from LHS)
        if _gate("ASSIGN", "1"):
            triples.extend(_extract_assignment(s))
        # Calls may be heavy; only try if budget allows
        if (time.perf_counter() - t0) * 1000.0 <= max(10, max_time_ms - 10) and _gate("CALLS", "1"):
            triples.extend(_extract_calls(s))
        if _gate("DOTS", "1"):
            triples.extend(_extract_dots(s))
        if _gate("PATHS", "1"):
            triples.extend(_extract_paths(s))
        if _gate("PATHS_BS", "1"):
            triples.extend(_extract_paths_backslash(s))
        # Generics and brackets (structural only)
        if _gate("GENERICS", "1"):
            triples.extend(_extract_generics(s))
        if _gate("BRACKETS", "1"):
            triples.extend(_extract_brackets(s))
        # Object/array literals
        if _gate("OBJECTARRAY", "1"):
            triples.extend(_extract_object_array_literals(s))
        # Chain order edges
        if _gate("CHAIN", "1"):
            triples.extend(_extract_chain_order(s))
        # Return types and name:Type pairs
        if _gate("RET_PY", "1"):
            triples.extend(_extract_return_types_py(s))
        if _gate("RET", "1"):
            triples.extend(_extract_return_types(s))
        if _gate("COLON", "1"):
            triples.extend(_extract_colon_types(s))

        if triples:
            # Filter conflicting edges within the same line
            mem_pairs = {(a, b) for (a, p, b) in triples if p in ("{}", "[]")}
            type_pairs = {(a, b) for (a, p, b) in triples if p == ":"}
            triples = [
                tri for tri in triples
                if not (
                    (tri[1] == "." and (tri[0], tri[2]) in mem_pairs)
                    or (tri[1] == ">>" and (tri[0], tri[2]) in type_pairs)
                )
            ]
            # Optionally attach decorators to subjects that look like callable (had () edges)
            subs_for_decor: set[str] = set()
            for tri in triples:
                a, p, b = tri
                if tri in seen:
                    continue
                seen.add(tri)
                cands.append((_score(tri, s), tri))
                if p == "()":
                    subs_for_decor.add(a)
                # also derive '.' membership from dotted sides already done
                if len(cands) >= max_cands:
                    break
            if _gate("DECOR", "1") and pending_decorators and subs_for_decor:
                for subj in subs_for_decor:
                    for dec in pending_decorators:
                        tri_d = (subj, "@", dec)
                        if tri_d not in seen:
                            seen.add(tri_d)
                            cands.append((_score(tri_d, s), tri_d))
                pending_decorators = []
        if len(cands) >= max_cands:
            break
        if max_time_ms > 0 and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
    if not cands:
        return []
    cands.sort(key=lambda x: x[0], reverse=True)
    return [tri for _sc, tri in cands[:max_items]]
