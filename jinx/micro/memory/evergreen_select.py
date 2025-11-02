from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz.fuzz import partial_ratio as _fuzzy
except Exception:  # optional
    _fuzzy = None  # type: ignore

from jinx.micro.memory.storage import read_evergreen
from jinx.micro.memory.vector_index import search as _vec_search

# Simple in-process TTL cache
_SEL_CACHE: Dict[Tuple[str, str, str], Tuple[int, str]] = {}


def _now_ms() -> int:
    try:
        return int(time.time() * 1000)
    except Exception:
        return 0


def _hash(s: str) -> str:
    try:
        return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return str(len(s or ""))


def _tokens(s: str) -> List[str]:
    out: List[str] = []
    cur = []
    for ch in (s or "").lower():
        if ch.isalnum() or ch in ("_", ".", "/", "-"):
            cur.append(ch)
        else:
            if cur:
                t = "".join(cur)
                if len(t) >= 3:
                    out.append(t)
                cur = []
    if cur:
        t = "".join(cur)
        if len(t) >= 3:
            out.append(t)
    # dedup preserve order
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _split_blocks(md: str) -> List[str]:
    # Split by double newlines; keep moderate sized paragraphs
    raw = (md or "").strip().split("\n\n")
    blocks = [b.strip() for b in raw if (b or '').strip()]
    return blocks[:1000]


def _score_block(block: str, q_tokens: List[str], anchor_tokens: List[str]) -> float:
    s = (block or "").lower()
    if not s:
        return 0.0
    score = 0.0
    # anchor presence gets stronger weight
    for a in anchor_tokens:
        if a and a in s:
            score += 2.0
    # query token presence
    for t in q_tokens:
        if t and t in s:
            score += 1.0
    # fuzzy fallback if available
    if _fuzzy is not None:
        try:
            # Use up to 180 chars window for fuzz to keep cost bounded
            sub = s[:180]
            qjoin = " ".join(q_tokens[:8])
            if len(qjoin) >= 3 and sub:
                score += 0.01 * float(_fuzzy(qjoin, sub))
        except Exception:
            pass
    return score


async def select_evergreen_for(query: str, anchors: Optional[Dict[str, List[str]]] = None, *, max_chars: Optional[int] = None, max_blocks: Optional[int] = None, max_time_ms: Optional[int] = None, ttl_ms: Optional[int] = None) -> str:
    """Return a small, relevant evergreen snippet for the current query.

    - Reads full evergreen via read_evergreen() but only returns a compact selection
      based on query and anchors.
    - Time-bounded, with a small TTL cache keyed by (evergreen_sha, query_sha, anchors_sha).
    """
    q = (query or "").strip()
    if not q:
        return ""
    try:
        max_chars = int(os.getenv("JINX_EVG_MAX_CHARS", str(max_chars if max_chars is not None else 1200)))
    except Exception:
        max_chars = 1200
    try:
        max_blocks = int(os.getenv("JINX_EVG_MAX_BLOCKS", str(max_blocks if max_blocks is not None else 6)))
    except Exception:
        max_blocks = 6
    try:
        max_time_ms = int(os.getenv("JINX_EVG_MAX_TIME_MS", str(max_time_ms if max_time_ms is not None else 120)))
    except Exception:
        max_time_ms = 120
    try:
        ttl_ms = int(os.getenv("JINX_EVG_SEL_TTL_MS", str(ttl_ms if ttl_ms is not None else 2500)))
    except Exception:
        ttl_ms = 2500

    t0 = time.perf_counter()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    full = (await read_evergreen()) or ""
    if not full:
        return ""

    evg_sha = _hash(full)
    q_sha = _hash(q)
    anch_list: List[str] = []
    try:
        for k, vs in (anchors or {}).items():
            for v in vs or []:
                anch_list.append(str(v or ""))
    except Exception:
        pass
    anch_sha = _hash("|".join(anch_list))

    ck = (evg_sha, q_sha, anch_sha)
    now = _now_ms()
    ent = _SEL_CACHE.get(ck)
    if ent and (ttl_ms is None or (now - ent[0]) <= ttl_ms):
        return ent[1]

    q_tokens = _tokens(q)
    anchor_tokens: List[str] = []
    for v in anch_list:
        anchor_tokens.extend(_tokens(v))
    # de-dup
    anchor_tokens = list(dict.fromkeys(anchor_tokens))[:12]

    blocks = _split_blocks(full)
    scored: List[Tuple[float, str]] = []
    for b in blocks:
        if time_up():
            break
        sc = _score_block(b, q_tokens, anchor_tokens)
        if sc > 0:
            scored.append((sc, b))
    scored.sort(key=lambda x: x[0], reverse=True)

    out_blocks: List[str] = []
    used = 0
    for _sc, b in scored[: max_blocks or 6]:
        if used + len(b) + 2 > (max_chars or 1200):
            # try to truncate at block boundary; prefer fewer complete blocks
            break
        out_blocks.append(b)
        used += len(b) + 2

    out = ("\n\n".join(out_blocks)).strip()

    # Optional vector-based fallback if no blocks selected under the time budget
    if not out:
        try:
            use_vec = str(os.getenv("JINX_EVG_VEC_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            use_vec = True
        if use_vec and not time_up():
            try:
                # Leave up to half of remaining budget
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                rem_ms = int(max(1.0, (max_time_ms or 120) - elapsed_ms))
                k_vec = max(3, min(12, (max_blocks or 6) * 2))
                if rem_ms > 1:
                    lines = await _vec_search(q, k=k_vec, max_time_ms=rem_ms, preview_chars=min(200, int(os.getenv("JINX_EVG_PREVIEW_CHARS", "160"))))
                else:
                    lines = []
            except Exception:
                lines = []
            if lines:
                seen: set[str] = set()
                parts: List[str] = []
                limit = max_chars or 1200
                for ln in lines:
                    if not ln or ln in seen:
                        continue
                    if (sum(len(p) for p in parts) + len(ln) + 1) > limit:
                        break
                    parts.append(ln)
                    seen.add(ln)
                out = "\n".join(parts).strip()

    _SEL_CACHE[ck] = (now, out)
    return out


__all__ = ["select_evergreen_for"]
