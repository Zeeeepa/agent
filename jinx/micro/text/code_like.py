from __future__ import annotations

import asyncio
import os
import re
from typing import List, Tuple

# Advanced, multi-signal code-like detector with optional backends.
# - Fast structural features (fences, brackets, operators, indentation variance)
# - Generic label/hex/bitwise cues (assembly-friendly without listing mnemonics)
# - Optional parsers (Python libcst), optional lexers (Pygments) guarded by env
# - TTL cache + coalescing to remain RT-friendly

_CODE_FENCE_RE = re.compile(r"```|<python_|</python_", re.IGNORECASE)
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_CALL_RE = re.compile(rf"\b{_IDENT}\s*\(.*?\)")  # foo(...)
_ASSIGN_RE = re.compile(rf"\b{_IDENT}\s*=\s*[^=]")  # x = y (not ==)
_STRUCT = set("()[]{}:;.,=<>+-*/|&%~^!@$")
_LINE_LABEL_RE = re.compile(r"(?m)^\s*[A-Za-z_.$][\w.$]*:\s*$")
_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
_HEX_OCTETS_RE = re.compile(r"\b[0-9A-Fa-f]{2}\b(?:\s+[0-9A-Fa-f]{2}){4,}")
_COMMENT_PREFIXES = ("#", "//", "/*")

# Optional Tree-sitter signal (small positive boost on successful parse)
try:
    from jinx.micro.text.code_like_ts import ts_parse_signal as _ts_signal  # type: ignore
except Exception:
    _ts_signal = None  # type: ignore

_cache: dict[tuple[str, float], float] = {}
_inflight: dict[tuple[str, float], asyncio.Future] = {}
_TTL_SEC = float(os.getenv("JINX_CODELIKE_TTL_SEC", "60"))


def _now() -> float:
    import time
    return time.time()


def _cache_get(key: tuple[str, float]) -> float | None:
    v = _cache.get(key)
    return v


def _cache_put(key: tuple[str, float], val: float) -> None:
    _cache[key] = val


def _struct_features(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    L = len(t)
    score = 0.0
    if _CODE_FENCE_RE.search(t):
        score += 0.45
    if _CALL_RE.search(t):
        score += 0.18
    if _ASSIGN_RE.search(t):
        score += 0.12
    struct = sum(1 for ch in t if ch in _STRUCT)
    score += min(0.32, struct / max(20.0, L) * 1.1)
    if t.endswith(":"):
        score += 0.07
    # Indentation/line cues (multi-line input)
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if lines:
        indented = sum(1 for ln in lines if ln.startswith((" ", "\t")))
        if indented >= 2:
            score += min(0.1, indented / max(4.0, len(lines)) * 0.2)
        # Semicolon line termini
        semi_end = sum(1 for ln in lines if ln.rstrip().endswith(";"))
        if semi_end >= 2:
            score += min(0.08, semi_end / max(4.0, len(lines)) * 0.16)
    return score


def _asm_hex_features(text: str) -> float:
    t = (text or "")
    score = 0.0
    if _LINE_LABEL_RE.search(t):
        score += 0.12
    if _HEX_ADDR_RE.search(t):
        score += 0.04
    if _HEX_OCTETS_RE.search(t):
        score += 0.18
    # Comment-style markers common in code
    if any(ln.lstrip().startswith(_p) for _p in _COMMENT_PREFIXES for ln in t.splitlines() if ln.strip()):
        score += 0.06
    return score


async def _py_parse_score(text: str) -> float:
    # Optional Python CST parse: success -> boost
    try:
        import libcst as cst  # type: ignore
        mod = await asyncio.to_thread(cst.parse_module, text)
        # Count nodes roughly via traversal of body
        n = len(getattr(mod, "body", []) or [])
        if n >= 1:
            return 0.12
    except Exception:
        return 0.0
    return 0.0


async def _pygments_score(text: str) -> float:
    # Optional lexer attempt; guard by env as it may be heavy
    try:
        if os.getenv("JINX_CODELIKE_PYGMENTS", "0").strip().lower() in ("", "0", "false", "off", "no"):
            return 0.0
        from pygments.lexers import guess_lexer  # type: ignore
        from pygments.token import Text as TokText  # type: ignore
        def _lex():
            try:
                lex = guess_lexer(text)
                toks = list(lex.get_tokens(text))  # type: ignore[attr-defined]
                return toks
            except Exception:
                return []
        toks = await asyncio.to_thread(_lex)
        if not toks:
            return 0.0
        non_text = sum(1 for ttype, _val in toks if str(ttype) != str(TokText))
        total = max(1, len(toks))
        frac = non_text / total
        if frac >= 0.6:
            return min(0.22, (frac - 0.6) * 0.55)
        return 0.0
    except Exception:
        return 0.0


async def code_like_score_advanced(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    key = (t[:512], float(_TTL_SEC))
    c = _cache_get(key)
    if c is not None:
        return c
    # Compose features concurrently
    s1 = _struct_features(t)
    s2 = _asm_hex_features(t)
    py_task = asyncio.create_task(_py_parse_score(t))
    pyg_task = asyncio.create_task(_pygments_score(t))
    s3, s4 = await asyncio.gather(py_task, pyg_task, return_exceptions=True)
    if isinstance(s3, Exception):
        s3 = 0.0
    if isinstance(s4, Exception):
        s4 = 0.0
    score = s1 + s2 + float(s3 or 0.0) + float(s4 or 0.0)
    if score > 1.0:
        score = 1.0
    _cache_put(key, score)
    return score


async def is_code_like_advanced(text: str, threshold: float = 0.58) -> bool:
    try:
        sc = await code_like_score_advanced(text)
        return sc >= threshold
    except Exception:
        return False


def code_like_score_fast(text: str) -> float:
    """Synchronous, fast multi-signal scorer (struct + asm/hex/comments)."""
    t = (text or "").strip()
    if not t:
        return 0.0
    score = _struct_features(t) + _asm_hex_features(t)
    # Optional Tree-sitter parse signal (env-gated, small weight)
    try:
        if _ts_signal is not None and os.getenv("JINX_CODELIKE_TS", "0").strip().lower() not in ("", "0", "false", "off", "no"):
            score += min(0.2, float(_ts_signal(t) or 0.0))
    except Exception:
        pass
    if score > 1.0:
        score = 1.0
    return score


def is_code_like_fast(text: str, threshold: float = 0.58) -> bool:
    try:
        return code_like_score_fast(text) >= threshold
    except Exception:
        return False
