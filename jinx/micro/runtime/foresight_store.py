from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

from jinx.micro.embeddings.project_config import ROOT

_PATH = os.path.join(ROOT, ".jinx", "memory", "foresight.json")
_TOK = re.compile(r"(?u)[\w\.]{3,}")


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def load_state() -> Dict[str, Any]:
    try:
        with open(_PATH, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
            return dict(obj or {})
    except Exception:
        return {"tok": {}, "bi": {}, "last": 0.0}


def save_state(st: Dict[str, Any]) -> None:
    try:
        _ensure_dir(_PATH)
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(tmp, _PATH)
    except Exception:
        pass


def _norm_tokens(text: str) -> List[str]:
    toks: List[str] = []
    seen: set[str] = set()
    for m in _TOK.finditer(text or ""):
        t = (m.group(0) or "").strip().lower()
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            toks.append(t)
    return toks


def update_tokens(st: Dict[str, Any], text: str, *, w: float = 1.0) -> List[str]:
    toks = _norm_tokens(text)
    TK: Dict[str, float] = st.setdefault("tok", {})  # type: ignore[assignment]
    BI: Dict[str, float] = st.setdefault("bi", {})   # type: ignore[assignment]
    # decay (light) to keep fresh trends — optional
    now = time.perf_counter()
    st["last"] = float(now)
    for t in toks:
        TK[t] = float(TK.get(t, 0.0)) + float(w)
    # bigrams from adjacent tokens
    for i in range(len(toks) - 1):
        k = toks[i] + "||" + toks[i + 1]
        BI[k] = float(BI.get(k, 0.0)) + float(0.7 * w)
    # prune occasionally (keep top 2k)
    if len(TK) > 4000:
        items = sorted(TK.items(), key=lambda x: -x[1])[:2000]
        st["tok"] = dict(items)
    if len(BI) > 6000:
        items = sorted(BI.items(), key=lambda x: -x[1])[:3000]
        st["bi"] = dict(items)
    return toks


def predict_next(st: Dict[str, Any], seeds: List[str] | None = None, *, top_k: int = 5) -> List[str]:
    seeds = [s.strip().lower() for s in (seeds or []) if s and len(s.strip()) >= 3]
    TK: Dict[str, float] = dict(st.get("tok", {}))
    BI: Dict[str, float] = dict(st.get("bi", {}))
    if not TK and not BI:
        return []
    # candidate tokens
    cand: Dict[str, float] = {}
    # base popularity
    for t, v in TK.items():
        cand[t] = cand.get(t, 0.0) + float(v) * 0.6
    # bigram neighbors of seeds
    if seeds:
        right: Dict[str, float] = {}
        left: Dict[str, float] = {}
        for k, v in BI.items():
            if "||" not in k:
                continue
            a, b = k.split("||", 1)
            if a in seeds:
                right[b] = right.get(b, 0.0) + float(v)
            if b in seeds:
                left[a] = left.get(a, 0.0) + float(v) * 0.8
        # blend neighbors higher
        for t, v in right.items():
            cand[t] = cand.get(t, 0.0) + float(v) * 1.2
        for t, v in left.items():
            cand[t] = cand.get(t, 0.0) + float(v) * 0.9
    # rank
    items = sorted(cand.items(), key=lambda x: -x[1])
    out = [t for t, _ in items if t not in (seeds or [])][: max(1, top_k)]
    return out


__all__ = [
    "load_state",
    "save_state",
    "update_tokens",
    "predict_next",
]
