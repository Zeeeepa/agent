from __future__ import annotations

import re
import asyncio
from typing import Tuple

from jinx.async_utils.fs import read_text_raw, write_text
from .utils import unified_diff


def _scan_mask(text: str, lang: str) -> list[bool]:
    """Return a mask array where True marks positions inside strings/comments.
    Handles // and /* */ for JavaScript; strings ' " and template literals `...`.
    """
    n = len(text)
    mask = [False] * n
    i = 0
    def mark(a: int, b: int):
        for k in range(max(0, a), min(n, b)):
            mask[k] = True
    while i < n:
        ch = text[i]
        ch2 = text[i:i+2]
        # line comment
        if ch2 == '//' and (lang in ('js',)):
            j = i+2
            while j < n and text[j] != '\n':
                j += 1
            mark(i, j)
            i = j
            continue
        # block comment
        if ch2 == '/*' and (lang in ('js',)):
            j = i+2
            while j < n-1 and text[j:j+2] != '*/':
                j += 1
            j = min(n, j+2)
            mark(i, j)
            i = j
            continue
        # template string (ts/js)
        if ch == '`' and (lang in ('js',)):
            j = i+1
            esc = False
            while j < n:
                c = text[j]
                if esc:
                    esc = False
                elif c == '`':
                    j += 1
                    break
                elif c == '\\':
                    esc = True
                j += 1
            mark(i, j)
            i = j
            continue
        # normal strings ' and "
        if ch in ('\'', '"'):
            q = ch
            j = i+1
            esc = False
            while j < n:
                c = text[j]
                if esc:
                    esc = False
                elif c == q:
                    j += 1
                    break
                elif c == '\\':
                    esc = True
                j += 1
            mark(i, j)
            i = j
            continue
        i += 1
    return mask


def _next_unmasked(text: str, mask: list[bool], pos: int, target: str) -> int:
    n = len(text)
    i = pos
    while i < n and (mask[i] or text[i] != target):
        i += 1
    return i if i < n else -1


def _brace_span(text: str, start_idx: int, lang: str, mask: list[bool]) -> Tuple[int, int]:
    """Find span [start, end) of a brace-delimited block starting at or after start_idx.
    Skips masked areas (strings/comments)."""
    n = len(text)
    i = start_idx
    i = _next_unmasked(text, mask, i, '{')
    if i < 0:
        return (start_idx, start_idx)
    depth = 0
    j = i
    while j < n:
        if not mask[j]:
            ch = text[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return (start_idx, j + 1)
        j += 1
    return (start_idx, start_idx)


def _arrow_expr_end(text: str, start_idx: int, mask: list[bool]) -> int:
    """Find end index for an arrow function expression body without a block.
    Scans until first unmasked ';' at zero nesting, otherwise newline fallback.
    Tracks (), {} nesting to avoid early termination inside object/paren.
    """
    n = len(text)
    i = start_idx
    depth_par = 0
    depth_brace = 0
    while i < n:
        if not mask[i]:
            ch = text[i]
            if ch == '(':
                depth_par += 1
            elif ch == ')':
                depth_par = max(0, depth_par - 1)
            elif ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace = max(0, depth_brace - 1)
            elif ch == ';' and depth_par == 0 and depth_brace == 0:
                return i + 1
            elif ch == '\n' and depth_par == 0 and depth_brace == 0:
                return i
        i += 1
    return n


def _find_header_span(lang: str, symbol: str, text: str, mask: list[bool]) -> Tuple[int, int]:
    """Locate header and block span for a symbol in TS/JS/Go/Java.
    Supports dotted Class.method for TS/JS/Java.
    Returns (start, end) indices; (-1,-1) if not found.
    """
    s = symbol.strip()
    T = lang.lower()
    parts = s.split('.')
    # Find within class if dotted path (JS classes)
    if len(parts) >= 2 and T in ('js',):
        cls = parts[0]
        meth = parts[-1]
        # find class block
        cm = re.search(rf"\bclass\s+{re.escape(cls)}\b", text)
        if cm:
            cstart = cm.start()
            st, en = _brace_span(text, cstart, T, mask)
            if en > st:
                sub = text[st:en]
                submask = mask[st:en]
                # method header: allow modifiers, generics
                mm = re.search(rf"\b{re.escape(meth)}\s*\(", sub)
                if mm:
                    mstart = st + mm.start()
                    st2, en2 = _brace_span(text, mstart, T, mask)
                    if en2 > st2:
                        return (mstart, en2)
    # Top-level patterns for JavaScript only
    if T in ("js",):
        name = re.escape(s)
        patterns: list[tuple[str, str]] = [
            (rf"\bexport\s+default\s+async\s+function\s+{name}\s*(<[^>]*>)?\s*\(", "func"),
            (rf"\bexport\s+default\s+function\s+{name}\s*(<[^>]*>)?\s*\(", "func"),
            (rf"\bexport\s+async\s+function\s+{name}\s*(<[^>]*>)?\s*\(", "func"),
            (rf"\bexport\s+function\s+{name}\s*(<[^>]*>)?\s*\(", "func"),
            (rf"\basync\s+function\s+{name}\s*(<[^>]*>)?\s*\(", "func"),
            (rf"\bfunction\s+{name}\s*(<[^>]*>)?\s*\(", "func"),
            (rf"\bexport\s+(const|let|var)\s+{name}\s*=\s*async\s*function\s*\(", "funcexpr"),
            (rf"\b(const|let|var)\s+{name}\s*=\s*async\s*function\s*\(", "funcexpr"),
            (rf"\bexport\s+(const|let|var)\s+{name}\s*=\s*function\s*\(", "funcexpr"),
            (rf"\b(const|let|var)\s+{name}\s*=\s*function\s*\(", "funcexpr"),
            (rf"\bexport\s+(const|let|var)\s+{name}\s*=\s*async\s*\([^)]*\)\s*=>", "arrow"),
            (rf"\b(const|let|var)\s+{name}\s*=\s*async\s*\([^)]*\)\s*=>", "arrow"),
            (rf"\bexport\s+(const|let|var)\s+{name}\s*=\s*\([^)]*\)\s*=>", "arrow"),
            (rf"\b(const|let|var)\s+{name}\s*=\s*\([^)]*\)\s*=>", "arrow"),
        ]
        for p, kind in patterns:
            for m in re.finditer(p, text):
                if any(mask[k] for k in range(m.start(), min(len(text), m.end()))):
                    continue
                if kind in ("func", "funcexpr"):
                    st, en = _brace_span(text, m.start(), T, mask)
                    if en > st:
                        return (m.start(), en)
                else:  # arrow
                    # after '=>' either block or expression
                    arrow_end = m.end()
                    brace_pos = _next_unmasked(text, mask, arrow_end, '{')
                    if brace_pos != -1 and brace_pos - arrow_end < 5:
                        st, en = _brace_span(text, brace_pos, T, mask)
                        if en > st:
                            return (m.start(), en)
                    # expression body: end at semicolon/newline
                    en2 = _arrow_expr_end(text, arrow_end, mask)
                    if en2 > m.start():
                        return (m.start(), en2)
    # Other languages (TS/Go/Java) are intentionally not supported.
    return (-1, -1)


async def patch_symbol_generic(path: str, lang: str, symbol: str, replacement: str, *, preview: bool = False) -> Tuple[bool, str]:
    """Replace or insert a function/class block for JavaScript files (.js/.jsx).

    - Locates the header by regex + brace matching; replaces the whole block.
    - If not found, appends replacement at EOF with required newline.
    """
    cur = await read_text_raw(path)
    if cur == "":
        if preview:
            return True, unified_diff("", replacement or "", path=path)
        await write_text(path, replacement or "")
        return True, unified_diff("", replacement or "", path=path)
    mask = _scan_mask(cur, 'js')
    st, en = _find_header_span('js', symbol, cur, mask)
    if st >= 0 and en > st:
        # Decide body-only replacement: if replacement lacks header keywords
        rep = (replacement or "").lstrip()
        hdr_keywords = {
            'ts': ("function ", "class ", "const ", "let ", "var "),
            'js': ("function ", "class ", "const ", "let ", "var "),
            'go': ("func ",),
            'java': ("class ",),
        }.get(lang.lower(), ("{" ,))
        if not any(rep.startswith(k) for k in hdr_keywords):
            # replace inside braces only, preserve header/closing
            # find opening brace and matching end within [st, en)
            open_pos = cur.find('{', st, en)
            if open_pos != -1:
                # find matching close using brace_span from open_pos
                b_st, b_en = _brace_span(cur, open_pos, lang.lower(), mask)
                if b_en > b_st:
                    # preserve header upto open_pos+1 and closing from b_en-1
                    header = cur[st:open_pos+1]
                    footer = cur[b_en-1:en]
                    # indent body by header indent + 4
                    last_nl = header.rfind('\n')
                    indent = 0
                    if last_nl != -1:
                        line = header[last_nl+1:]
                        indent = len(line) - len(line.lstrip(' ')) + 4
                    body_i = "\n".join([(" "*indent + ln if ln.strip() else ln) for ln in rep.splitlines()])
                    new_block = header + "\n" + body_i + "\n" + footer
                    out = cur[:st] + new_block + cur[en:]
                else:
                    out = cur[:st] + (replacement or "") + cur[en:]
            else:
                out = cur[:st] + (replacement or "") + cur[en:]
        else:
            out = cur[:st] + (replacement or "") + cur[en:]
        if preview:
            return True, unified_diff(cur, out, path=path)
        await write_text(path, out)
        return True, unified_diff(cur, out, path=path)
    # append
    out = cur
    if not out.endswith("\n"):
        out += "\n"
    out += (replacement or "")
    if not out.endswith("\n"):
        out += "\n"
    if preview:
        return True, unified_diff(cur, out, path=path)
    await write_text(path, out)
    return True, unified_diff(cur, out, path=path)
