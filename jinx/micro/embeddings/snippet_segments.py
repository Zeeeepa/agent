from __future__ import annotations

from typing import List, Tuple
import re

from .project_identifiers import extract_identifiers
from .project_query_tokens import expand_strong_tokens, codeish_tokens


def _strip_comments_py(lines: List[str]) -> List[str]:
    out: List[str] = []
    in_triple = False
    triple_pat = re.compile(r"^[ \t]*([rRuU]?[fF]?)(?:'''|\"\"\")")
    important_tags = ("todo", "fixme", "note", "warn", "warning", "important", "hack", "bug")
    for ln in lines:
        s = ln.rstrip("\n")
        # Track triple-quoted docstrings/strings; never strip them
        if triple_pat.search(s):
            in_triple = not in_triple
            out.append(s)
            continue
        if in_triple:
            out.append(s)
            continue
        # Strip standalone comments and empty lines
        st = s.lstrip()
        if not st:
            continue
        if st.startswith('#'):
            low = st[1:].strip().lower()
            if any(tag in low for tag in important_tags):
                out.append(s)
            # else drop comment line
            continue
        out.append(s)
    return out


def _collect_anchor_lines(lines: List[str], anchors: List[str], max_windows: int) -> List[int]:
    # returns 1-indexed line numbers inside given lines
    found: List[int] = []
    if not anchors:
        return found
    low_lines = [ln.lower() for ln in lines]
    for a in anchors:
        tok = a.strip().lower()
        if not tok or len(tok) < 2:
            continue
        for i, l in enumerate(low_lines):
            if tok in l:
                found.append(i + 1)
                if len(found) >= max_windows * 2:  # allow a few extra before dedupe/merge
                    break
        if len(found) >= max_windows * 2:
            break
    # Dedupe preserving order
    seen = set()
    uniq = []
    for x in found:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    # Fallback anchors if none were found: look for control/IO lines that often carry semantics
    if not uniq:
        ctrl_tokens = ("return", "raise", "yield", "assert", "except", "finally")
        for i, l in enumerate(low_lines):
            if any(t in l for t in ctrl_tokens):
                uniq.append(i + 1)
                if len(uniq) >= max_windows:
                    break
    return sorted(uniq)[: max_windows * 2]


def _merge_windows(points: List[int], around: int, nlines: int, limit: int) -> List[Tuple[int, int, int]]:
    # Returns list of (start, end, center) windows within [1..nlines]
    out: List[Tuple[int, int, int]] = []
    for c in sorted(points):
        a = max(1, c - around)
        b = min(nlines, c + around)
        # merge overlaps
        if out and not (a > out[-1][1] + 2):
            # extend last
            la, lb, lc = out[-1]
            out[-1] = (la, max(lb, b), lc)
        else:
            out.append((a, b, c))
        if len(out) >= limit:
            break
    return out[:limit]


def build_multi_segment_python(
    file_lines: List[str],
    scope_start: int,
    scope_end: int,
    query: str,
    *,
    per_hit_chars: int,
    head_lines: int,
    tail_lines: int,
    mid_windows: int,
    mid_around: int,
    strip_comments: bool,
    extra_centers: List[int] | None = None,
) -> str:
    """Build a composite snippet for a large Python scope.

    Includes:
    - Head lines (signature + docstring)
    - Mid windows around query anchors (identifiers)
    - Tail lines (returns/cleanup)
    Keeps within per_hit_chars budget.
    """
    s_idx = max(1, scope_start) - 1
    e_idx = min(len(file_lines), scope_end) - 1
    scope = file_lines[s_idx : e_idx + 1]

    work_scope = list(scope)
    if strip_comments:
        work_scope = _strip_comments_py(work_scope)
        if not work_scope:  # fallback
            work_scope = list(scope)

    # Head segment
    head = work_scope[: max(0, head_lines)]
    segments: List[str] = []
    if head:
        segments.append("# segment: head\n" + "\n".join(head).rstrip())

    # Mid segments around anchors
    # Build strong anchors from query: identifiers + strong tokens + code-ish tokens
    anchors = []
    try:
        anchors.extend(extract_identifiers(query or "", max_items=16))
    except Exception:
        pass
    try:
        anchors.extend(list(expand_strong_tokens(query or "", max_items=12)))
    except Exception:
        pass
    try:
        anchors.extend(list(codeish_tokens(query or "", max_items=12)))
    except Exception:
        pass
    centers = _collect_anchor_lines(work_scope, anchors, max_windows=max(1, mid_windows))
    # Include extra centers (from other hits within same file) if they fall into this scope
    if extra_centers:
        for c in extra_centers:
            try:
                c_int = int(c)
            except Exception:
                continue
            if scope_start <= c_int <= scope_end:
                # convert to scope-relative line index
                rel = c_int - scope_start + 1
                centers.append(rel)
    mid_windows_list = _merge_windows(centers, around=max(1, mid_around), nlines=len(work_scope), limit=max(1, mid_windows))
    # Avoid heavy overlap with head/tail segments
    head_span = (1, max(0, head_lines))
    tail_span = (max(1, len(work_scope) - max(0, tail_lines) + 1), len(work_scope)) if tail_lines > 0 else None
    def _overlaps(a1: int, b1: int, a2: int, b2: int) -> bool:
        return not (b1 < a2 or b2 < a1)
    filtered_mid = []
    for a, b, c in mid_windows_list:
        if head_span[1] > 0 and _overlaps(a, b, head_span[0], head_span[1]):
            continue
        if tail_span and _overlaps(a, b, tail_span[0], tail_span[1]):
            continue
        filtered_mid.append((a, b, c))
    for a, b, c in filtered_mid:
        segment = work_scope[a - 1 : b]
        segments.append(f"# segment: mid @L{c}\n" + "\n".join(segment).rstrip())

    # Tail segment
    if tail_lines > 0:
        tail = work_scope[-tail_lines:]
        if tail:
            segments.append("# segment: tail\n" + "\n".join(tail).rstrip())

    # Compose under budget
    sep = "\n\n# ---- \n\n"
    out: List[str] = []
    total = 0
    for seg in segments:
        add = len(seg) + (len(sep) if out else 0)
        if total + add > max(1, per_hit_chars):
            # try to truncate last segment to fit
            remain = max(0, per_hit_chars - total - (len(sep) if out else 0))
            if remain > 40:  # keep a minimum useful size
                out.append((sep if out else "") + seg[:remain])
                total += (len(sep) if out else 0) + remain
            break
        out.append((sep if out else "") + seg)
        total += add

    return "".join(out) if out else "\n".join(scope)[:per_hit_chars]
