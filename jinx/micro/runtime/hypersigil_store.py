from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Tuple

from jinx.micro.embeddings.project_config import ROOT

_PATH = os.path.join(ROOT, ".jinx", "memory", "hypersigil.json")
_TOK = re.compile(r"(?u)[\w\.]{3,}")

# Flattened n-gram counts: key "t1||t2||...||tn" -> count
NGram = Dict[str, float]


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def load_store(max_keys: int | None = None) -> NGram:
    try:
        with open(_PATH, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
            ng: NGram = obj.get("ng", {})  # type: ignore[assignment]
    except Exception:
        ng = {}
    if max_keys is not None and len(ng) > max_keys:
        items = sorted(ng.items(), key=lambda kv: -float(kv[1]))[:max_keys]
        ng = {k: float(v) for k, v in items}
    return ng


def save_store(ng: NGram) -> None:
    try:
        _ensure_dir(_PATH)
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"ng": ng, "ts": time.time()}, f, ensure_ascii=False)
        os.replace(tmp, _PATH)
    except Exception:
        pass


def _tokify(text: str, max_n: int = 64) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for m in _TOK.finditer(text or ""):
        t = (m.group(0) or "").strip().lower()
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def _join(seq: List[str]) -> str:
    return "||".join(seq)


def update_ngrams(ng: NGram, text: str, *, max_order: int = 4, w: float = 1.0) -> List[str]:
    toks = _tokify(text)
    n = len(toks)
    for i in range(n):
        for k in range(1, max_order + 1):
            if i + k > n:
                break
            key = _join(toks[i : i + k])
            ng[key] = float(ng.get(key, 0.0)) + float(w)
    # prune to keep store reasonable
    if len(ng) > 20000:
        items = sorted(ng.items(), key=lambda kv: -float(kv[1]))[:12000]
        ng.clear(); ng.update({k: float(v) for k, v in items})
    return toks


def _cand_from_history(ng: NGram, hist: List[str], *, max_order: int = 4) -> Dict[str, float]:
    # Score next-token candidates by variable-order backoff
    # s(next|h) ~ sum_{k=1..m} gamma_k * count(h_suffix_k + next)
    m = min(max_order - 1, len(hist))
    if m <= 0:
        # unigram fallback
        return {k: v for k, v in ng.items() if "||" not in k}
    suffixes: List[List[str]] = [hist[-j:] for j in range(1, m + 1)]
    # discount weights
    gammas: List[float] = [1.0, 0.6, 0.4]
    scores: Dict[str, float] = {}
    for idx, suf in enumerate(suffixes):
        w = gammas[idx] if idx < len(gammas) else 0.3
        pref = _join(suf) + "||"
        plen = len(pref)
        for key, cnt in ng.items():
            if not key.startswith(pref):
                continue
            # key is suf + next (+ maybe tail); only consider immediate next token
            rest = key[plen:]
            nxt = rest.split("||", 1)[0]
            scores[nxt] = scores.get(nxt, 0.0) + float(cnt) * w
    return scores


def predict_next_tokens(ng: NGram, seeds: List[str], *, top_k: int = 6, max_order: int = 4) -> List[str]:
    seeds = [t.strip().lower() for t in (seeds or []) if t and len(t.strip()) >= 3]
    cand = _cand_from_history(ng, seeds, max_order=max_order) if seeds else {k: v for k, v in ng.items() if "||" not in k}
    items = sorted(cand.items(), key=lambda kv: -float(kv[1]))
    return [t for t, _ in items[:max(1, top_k)]]


def predict_next_sequences(ng: NGram, seeds: List[str], *, seq_len: int = 2, top_k: int = 3, max_order: int = 4) -> List[List[str]]:
    # Greedy: pick best next token, then predict again using extended history
    best: List[List[str]] = []
    hist = list(seeds or [])
    for _ in range(max(1, top_k)):
        h = list(hist)
        seq: List[str] = []
        for _j in range(max(1, seq_len)):
            cand = _cand_from_history(ng, h, max_order=max_order)
            if not cand:
                break
            tok = max(cand.items(), key=lambda kv: kv[1])[0]
            seq.append(tok)
            h.append(tok)
        if seq:
            best.append(seq)
    # dedupe
    out: List[List[str]] = []
    seen: set[str] = set()
    for s in best:
        k = _join(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:max(1, top_k)]


__all__ = [
    "load_store",
    "save_store",
    "update_ngrams",
    "predict_next_tokens",
    "predict_next_sequences",
]
