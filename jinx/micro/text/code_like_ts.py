from __future__ import annotations

import os
from typing import Optional

# Optional Tree-sitter signal: returns a small positive score if parsing succeeds
# with a plausible grammar. Requires a pre-built languages .so via tree_sitter.Language.

_TS_READY = False
_PARSER = None
_LANGS = {}

try:
    from tree_sitter import Language, Parser  # type: ignore
    _DLL = os.getenv("JINX_TS_LANGSO", "").strip()
    if _DLL and os.path.exists(_DLL):
        _LANGS = {
            "python": Language(_DLL, "python"),
            "javascript": Language(_DLL, "javascript"),
            "typescript": Language(_DLL, "typescript"),
            "tsx": Language(_DLL, "tsx"),
            "c": Language(_DLL, "c"),
            "cpp": Language(_DLL, "cpp"),
            "java": Language(_DLL, "java"),
            "go": Language(_DLL, "go"),
            "rust": Language(_DLL, "rust"),
            "c_sharp": Language(_DLL, "c_sharp"),
            "ruby": Language(_DLL, "ruby"),
            "php": Language(_DLL, "php"),
        }
        _PARSER = Parser()
        _TS_READY = True
except Exception:
    _TS_READY = False


def _guess_lang(text: str) -> Optional[str]:
    t = (text or "").lower()
    # Very rough cues; the exact grammar id must exist in _LANGS
    if "def " in t or "import " in t or "async " in t:
        return "python"
    if "function " in t or "=>" in t or "const " in t:
        return "javascript"
    if "package " in t and ";" in t and "import " in t:
        # Could be Java or Go; prefer Java if class appears
        if "class " in t:
            return "java"
        return "go"
    if "fn " in t or "impl " in t or "trait " in t:
        return "rust"
    if "#include" in t or "::" in t or "template<" in t:
        return "cpp"
    return None


def ts_parse_signal(text: str) -> float:
    if not _TS_READY or not text or _PARSER is None or not _LANGS:
        return 0.0
    lang = _guess_lang(text)
    if not lang:
        return 0.0
    try:
        _PARSER.set_language(_LANGS[lang])
        tree = _PARSER.parse(bytes(text[:8000], "utf-8", errors="ignore"))
        # If root has children, treat as valid parse
        root = tree.root_node  # type: ignore[attr-defined]
        if getattr(root, "child_count", 0) > 0:
            # Small positive signal; caller will weight appropriately
            return 0.15
    except Exception:
        return 0.0
    return 0.0
