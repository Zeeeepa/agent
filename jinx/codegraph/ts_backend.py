from __future__ import annotations

"""
Tree-sitter backend (optional). Provides fast cross-language symbol lookups.
If tree_sitter or tree_sitter_languages are not available, functions return empty.
"""

from typing import List, Tuple
import os

try:
    from tree_sitter import Language, Parser  # type: ignore
    import tree_sitter_languages as tsl  # type: ignore
    _TS_OK = True
except Exception:
    _TS_OK = False

_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}


def _lang_for_ext(ext: str):
    if not _TS_OK:
        return None
    name = _LANG_MAP.get(ext.lower())
    if not name:
        return None
    try:
        return tsl.get_language(name)
    except Exception:
        return None


def _read_text(p: str) -> str:
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def defs_for_token(abs_path: str, token: str, *, max_items: int = 1) -> List[Tuple[int, int]]:
    """Return list of (line_start, line_end) for definition-like nodes matching token.
    Best-effort, language-specific heuristics per grammar.
    """
    if not _TS_OK or not token:
        return []
    ext = os.path.splitext(abs_path)[1]
    lang = _lang_for_ext(ext)
    if lang is None:
        return []
    src = _read_text(abs_path)
    if not src:
        return []
    try:
        parser = Parser(lang)
        tree = parser.parse(bytes(src, "utf-8"))
    except Exception:
        return []
    root = tree.root_node
    out: List[Tuple[int, int]] = []

    def _add(node):
        ls = node.start_point[0] + 1
        le = node.end_point[0] + 1
        out.append((ls, le))

    def _walk(node):
        try:
            t = node.type
            # Rough heuristic: function/class/identifier declarations
            if token in src[node.start_byte:node.end_byte].decode("utf-8", "ignore"):
                if t in ("function_definition", "function_declaration", "class_definition", "class_declaration"):
                    _add(node)
                    return
            for ch in node.children:
                _walk(ch)
        except Exception:
            return

    _walk(root)
    return out[: max(1, max_items)]


def usages_for_token(abs_path: str, token: str, *, max_items: int = 1) -> List[int]:
    """Return list of line numbers where token appears (heuristic text scan fallback)."""
    src = _read_text(abs_path)
    if not src or not token:
        return []
    out: List[int] = []
    lines = src.splitlines()
    for i, ln in enumerate(lines, start=1):
        if token in ln:
            out.append(i)
            if len(out) >= max(1, max_items):
                break
    return out
