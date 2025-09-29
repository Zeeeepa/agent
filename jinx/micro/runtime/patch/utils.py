from __future__ import annotations

import os
import difflib
from typing import Tuple, List
import re


def unified_diff(old: str, new: str, *, path: str = "") -> str:
    """Produce a compact unified diff for preview/logging."""
    old_lines = (old or "").splitlines(keepends=True)
    new_lines = (new or "").splitlines(keepends=True)
    fn = os.path.basename(path) if path else "file"
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{fn}", tofile=f"b/{fn}")
    return "".join(diff)


def diff_stats(diff: str) -> Tuple[int, int]:
    """Return (added_lines, removed_lines) ignoring headers."""
    add = rem = 0
    for ln in (diff or "").splitlines():
        if ln.startswith("+++") or ln.startswith("---") or ln.startswith("@@"):
            continue
        if ln.startswith("+"):
            add += 1
        elif ln.startswith("-"):
            rem += 1
    return add, rem


def should_autocommit(strategy: str, diff: str) -> Tuple[bool, str]:
    """Decide whether to auto-commit based on env thresholds and strategy.

    Returns (ok_to_commit, reason).
    """
    try:
        max_changes = int(os.getenv("JINX_PATCH_AUTOCOMMIT_MAX_CHANGES", "40"))
    except Exception:
        max_changes = 40
    allow = str(os.getenv("JINX_PATCH_AUTOCOMMIT", "1")).lower() not in ("", "0", "false", "off", "no")
    if not allow:
        return False, "autocommit disabled"
    a, r = diff_stats(diff)
    total = a + r
    if total <= 0:
        return True, "no changes"
    # Be stricter on write
    if strategy == "write" and total > max_changes // 2:
        return False, f"write changes {total} > {max_changes//2}"
    if total > max_changes:
        return False, f"changes {total} > {max_changes}"
    return True, f"changes {total} within limit"


def syntax_check_enabled() -> bool:
    try:
        return str(os.getenv("JINX_PATCH_CHECK_SYNTAX", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


# --- Formatting helpers -----------------------------------------------------

def detect_eol(text: str) -> str:
    """Detect dominant EOL: returns "\r\n" if CRLF appears more than LF, else "\n"."""
    s = text or ""
    crlf = s.count("\r\n")
    lf = s.count("\n")
    # subtract CRLF from LF to get pure LF occurrences
    pure_lf = max(0, lf - crlf)
    return "\r\n" if crlf > pure_lf else "\n"


def has_trailing_newline(text: str) -> bool:
    s = text or ""
    return s.endswith("\n")


def join_lines(lines: List[str], eol: str, trailing_newline: bool) -> str:
    out = (eol or "\n").join(lines)
    if trailing_newline and not out.endswith(eol):
        out += eol
    return out


def leading_ws(s: str) -> str:
    i = 0
    while i < len(s) and s[i] in (" ", "\t"):
        i += 1
    return s[:i]


def normalize_indentation(lines: List[str]) -> List[str]:
    """Normalize indentation by removing the common leading whitespace across non-empty lines."""
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return list(lines)
    mins = min(len(leading_ws(ln)) for ln in nonempty)
    if mins <= 0:
        return list(lines)
    return [ln[mins:] if ln.strip() else ln for ln in lines]


def preserve_docstring_enabled() -> bool:
    try:
        return str(os.getenv("JINX_PATCH_PRESERVE_DOCSTRING", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def anchor_last_enabled() -> bool:
    try:
        return str(os.getenv("JINX_PATCH_ANCHOR_LAST", "0")).lower() in ("1", "true", "on", "yes")
    except Exception:
        return False


def anchor_regex_enabled() -> bool:
    try:
        return str(os.getenv("JINX_PATCH_ANCHOR_REGEX", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def auto_indent_enabled() -> bool:
    try:
        return str(os.getenv("JINX_PATCH_AUTO_INDENT", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def trim_trailing_ws_enabled() -> bool:
    try:
        return str(os.getenv("JINX_PATCH_TRIM_TRAILING_WS", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def trim_trailing_ws_lines(lines: List[str]) -> List[str]:
    return [ln.rstrip(" \t") for ln in lines]
