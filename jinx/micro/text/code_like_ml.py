from __future__ import annotations

import math
import os
import re
from typing import Dict

from .code_like import code_like_score_fast  # lightweight base features

_WORD_RE = re.compile(r"(?u)[\w_]+")
_PAREN_BAL_RE = re.compile(r"[()]")
_BRACE_BAL_RE = re.compile(r"[{}]")
_BRACK_BAL_RE = re.compile(r"[\[\]]")
_COMMENT_PREFIXES = ("#", "//", "/*", ";")  # asm uses ';'


def _line_metrics(text: str) -> Dict[str, float]:
    lines = [ln for ln in (text or "").splitlines()]
    if not lines:
        return {"n_lines": 0.0, "avg_len": 0.0, "std_len": 0.0, "semi_frac": 0.0, "comment_frac": 0.0}
    n = len(lines)
    lens = [len(ln) for ln in lines]
    avg = sum(lens) / n
    var = sum((l - avg) * (l - avg) for l in lens) / max(1, n - 1)
    std = math.sqrt(var)
    semi = sum(1 for ln in lines if ln.rstrip().endswith(";")) / n
    comm = 0
    for ln in lines:
        s = ln.lstrip()
        if not s:
            continue
        if any(s.startswith(p) for p in _COMMENT_PREFIXES):
            comm += 1
    return {"n_lines": float(n), "avg_len": float(avg), "std_len": float(std), "semi_frac": float(semi), "comment_frac": float(comm / n)}


def _balance_score(text: str) -> float:
    def bal(s: str, open_c: str, close_c: str) -> float:
        d = 0
        bad = 0
        for ch in s:
            if ch == open_c:
                d += 1
            elif ch == close_c:
                if d == 0:
                    bad += 1
                else:
                    d -= 1
        return max(0.0, 1.0 - (bad / max(1.0, len(s))))
    t = (text or "")
    return min(1.0, (bal(t, '(', ')') + bal(t, '{', '}') + bal(t, '[', ']')) / 3.0)


def _charset_ratios(text: str) -> Dict[str, float]:
    t = (text or "")
    if not t:
        return {"alpha": 0.0, "digit": 0.0, "space": 0.0, "punct": 0.0, "other": 0.0}
    import string
    a = d = s = p = o = 0
    for ch in t:
        if ch.isalpha(): a += 1
        elif ch.isdigit(): d += 1
        elif ch.isspace(): s += 1
        elif ch in string.punctuation: p += 1
        else: o += 1
    tot = max(1, len(t))
    return {"alpha": a/tot, "digit": d/tot, "space": s/tot, "punct": p/tot, "other": o/tot}


def _token_complexity(text: str) -> Dict[str, float]:
    toks = _WORD_RE.findall(text or "")
    if not toks:
        return {"tok_ratio": 0.0, "uniq_ratio": 0.0}
    n = len(toks)
    uniq = len(set(toks))
    # code often has lower uniq ratio due to repeated identifiers/keywords
    return {"tok_ratio": n / max(1, len(text or "")), "uniq_ratio": uniq / n}


_DEFAULT_WEIGHTS = {
    # logistic regression weights (bias + features)
    "bias": -0.2,
    "score_fast": 2.1,
    "n_lines": 0.002,
    "avg_len": 0.002,
    "std_len": 0.004,
    "semi_frac": 0.7,
    "comment_frac": 0.5,
    "balance": 0.8,
    "alpha": -0.6,
    "digit": 0.4,
    "space": -0.1,
    "punct": 0.6,
    "other": 0.2,
    "tok_ratio": -0.4,
    "uniq_ratio": -0.6,
}


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.0


def _load_weights() -> Dict[str, float]:
    # Optionally load from file path in env; else defaults
    path = os.getenv("JINX_CODELIKE_WEIGHTS", "").strip()
    if not path:
        return dict(_DEFAULT_WEIGHTS)
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {**_DEFAULT_WEIGHTS, **obj}
    except Exception:
        return dict(_DEFAULT_WEIGHTS)


def code_like_prob_ml(text: str) -> float:
    t = (text or "")
    if not t.strip():
        return 0.0
    w = _load_weights()
    s_fast = code_like_score_fast(t)
    lm = _line_metrics(t)
    bal = _balance_score(t)
    cs = _charset_ratios(t)
    tk = _token_complexity(t)

    z = (
        w.get("bias", 0.0)
        + w.get("score_fast", 0.0) * s_fast
        + w.get("n_lines", 0.0) * lm.get("n_lines", 0.0)
        + w.get("avg_len", 0.0) * lm.get("avg_len", 0.0)
        + w.get("std_len", 0.0) * lm.get("std_len", 0.0)
        + w.get("semi_frac", 0.0) * lm.get("semi_frac", 0.0)
        + w.get("comment_frac", 0.0) * lm.get("comment_frac", 0.0)
        + w.get("balance", 0.0) * bal
        + w.get("alpha", 0.0) * cs.get("alpha", 0.0)
        + w.get("digit", 0.0) * cs.get("digit", 0.0)
        + w.get("space", 0.0) * cs.get("space", 0.0)
        + w.get("punct", 0.0) * cs.get("punct", 0.0)
        + w.get("other", 0.0) * cs.get("other", 0.0)
        + w.get("tok_ratio", 0.0) * tk.get("tok_ratio", 0.0)
        + w.get("uniq_ratio", 0.0) * tk.get("uniq_ratio", 0.0)
    )
    p = _sigmoid(z)
    return p


def is_code_like_ml(text: str, threshold: float = 0.58) -> bool:
    try:
        return code_like_prob_ml(text) >= threshold
    except Exception:
        return False
