from __future__ import annotations

import re
from typing import List, Tuple
import os as _os

try:
    # Prefer centralized multi-signal detector (assembly/hex/comments aware)
    from jinx.micro.text.code_like import code_like_score_fast as _cl_score_fast  # type: ignore
except Exception:
    _cl_score_fast = None  # type: ignore
try:
    # Optional ML calibrated detector (weights configurable via env/file)
    from jinx.micro.text.code_like_ml import is_code_like_ml as _cl_is_ml  # type: ignore
except Exception:
    _cl_is_ml = None  # type: ignore

# --- Code-like detection ---

_CODE_FENCE_RE = re.compile(r"```|<python_|</python_", re.IGNORECASE)
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_CALL_RE = re.compile(rf"\b{_IDENT}\s*\(.*?\)")  # foo(...)
_ASSIGN_RE = re.compile(rf"\b{_IDENT}\s*=\s*[^=]" )  # x = y  (not ==)
_STRUCT_TOKENS = set("()[]{}:;.,=<>+-*/|&%~^!")


def code_like_score(s: str) -> float:
    if not s:
        return 0.0
    t = s.strip()
    if not t:
        return 0.0
    L = len(t)
    score = 0.0
    # Strong signals
    if _CODE_FENCE_RE.search(t):
        score += 0.5
    if _CALL_RE.search(t):
        score += 0.2
    if _ASSIGN_RE.search(t):
        score += 0.15
    # Bracket/structure density
    struct = sum(1 for ch in t if ch in _STRUCT_TOKENS)
    score += min(0.35, struct / max(20.0, L) * 1.2)
    # Indentation / line ending cues
    if t.endswith(":") or t.startswith(("def ", "class ", "function ", "proc ")):
        score += 0.1
    # Clamp
    if score > 1.0:
        score = 1.0
    return score


def is_code_like(s: str, threshold: float = 0.58) -> bool:
    t = s or ""
    # Threshold override from env
    try:
        thr_env = _os.getenv("JINX_CODELIKE_THRESH", "").strip()
        if thr_env:
            threshold = float(thr_env)
    except Exception:
        pass
    # Mode selection: ml | fast | basic (default: ml)
    try:
        mode = (_os.getenv("JINX_CODELIKE_MODE", "ml").strip().lower() or "ml")
    except Exception:
        mode = "ml"
    if mode == "ml" and _cl_is_ml is not None:
        try:
            return bool(_cl_is_ml(t, threshold=threshold))
        except Exception:
            # fall back to fast
            mode = "fast"
    if mode == "fast" and _cl_score_fast is not None:
        try:
            return float(_cl_score_fast(t)) >= float(threshold)
        except Exception:
            pass
    # basic fallback
    try:
        return code_like_score(t) >= threshold
    except Exception:
        return False


def is_code_like_line(s: str, threshold: float = 0.5) -> bool:
    return is_code_like(s, threshold)


# --- Preference/Decision extraction (language-agnostic leaning) ---

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\[[ xX]?\])\s*(.+)$")
_SHORT_LABEL_RE = re.compile(r"^\s*([A-Za-zА-Яа-я0-9_]{2,16})\s*:\s*(.+)$")
_ARROW_RE = re.compile(r"(.+?)\s*(?:=>|->|→)\s*(.+)")
# No language word lists: rely only on structural patterns (bullets/labels/arrows) and non-code lines.


def _clean_frag(s: str) -> str:
    s = (s or "").strip()
    # Collapse excessive spaces
    s = re.sub(r"\s+", " ", s)
    return s[:320]


def extract_preference_fragments(text: str, *, max_items: int = 40) -> List[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    # First pass: count duplicates (identical lines)
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    cand: list[tuple[float, str]] = []
    for raw in lines:
        ln = raw
        # Skip code-like lines completely
        if is_code_like_line(ln):
            continue
        score = 0.0
        frag: str | None = None
        mb = _BULLET_RE.match(ln)
        if mb:
            # Bullet or checkbox item is a strong structural preference signal
            frag = _clean_frag(mb.group(1))
            score += 1.0
            # Ticked checkbox gives a small extra weight
            if ln.lstrip().startswith("[") and any(ch in ln for ch in ("[x]", "[X]")):
                score += 0.2
        else:
            ml = _SHORT_LABEL_RE.match(ln)
            if ml:
                label, rest = ml.group(1), ml.group(2)
                frag = _clean_frag(f"{label}: {rest}")
                score += 0.8
        # Minor boosts based on structure/length (language-agnostic)
        if frag:
            L = len(frag)
            if 8 <= L <= 200:
                score += 0.1
            if frag.endswith(":"):
                score += 0.05
            # Duplicate lines in the same text suggest salience
            score += min(0.3, 0.05 * max(0, counts.get(raw, 0) - 1))
            cand.append((score, frag))
    # Sort by score descending, then by shorter fragment (stability) and return uniques
    cand.sort(key=lambda x: (-x[0], len(x[1])))
    out: List[str] = []
    seen: set[str] = set()
    for _sc, fr in cand:
        if fr and fr not in seen:
            seen.add(fr)
            out.append(fr)
            if len(out) >= max_items:
                break
    return out


def extract_decision_fragments(text: str, *, max_items: int = 40) -> List[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    cand: list[tuple[float, str]] = []
    for raw in lines:
        ln = raw
        if is_code_like_line(ln):
            continue
        score = 0.0
        frag: str | None = None
        ma = _ARROW_RE.search(ln)
        if ma:
            frag = _clean_frag(ln)
            score += 1.0
        else:
            ml = _SHORT_LABEL_RE.match(ln)
            if ml:
                label, rest = ml.group(1), ml.group(2)
                frag = _clean_frag(f"{label}: {rest}")
                score += 0.7
            else:
                # Enumerated bullet may indicate a step/decision (e.g., "1) ...")
                mb = _BULLET_RE.match(ln)
                if mb and any(ch.isdigit() for ch in ln[:4]):
                    frag = _clean_frag(mb.group(1))
                    score += 0.6
        if frag:
            L = len(frag)
            if 8 <= L <= 220:
                score += 0.1
            if frag.endswith(":") or "->" in frag or "=>" in frag or "→" in frag:
                score += 0.05
            score += min(0.3, 0.05 * max(0, counts.get(raw, 0) - 1))
            cand.append((score, frag))
    cand.sort(key=lambda x: (-x[0], len(x[1])))
    out: List[str] = []
    seen: set[str] = set()
    for _sc, fr in cand:
        if fr and fr not in seen:
            seen.add(fr)
            out.append(fr)
            if len(out) >= max_items:
                break
    return out
