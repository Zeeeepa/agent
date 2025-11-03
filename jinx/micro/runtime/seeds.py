from __future__ import annotations

import time
from typing import List
import jinx.state as jx_state

# Central aggregator for attention/prediction seeds.
# Merges cog/foresight/oracle/hypersigil with TTLs and dedupe, returns top-N.

_DEF_TTL = 30.0


def _get_list(name: str) -> List[str]:
    try:
        v = list(getattr(jx_state, name, []) or [])
        return [str(t).strip().lower() for t in v if str(t).strip()]
    except Exception:
        return []


def _alive(ts_name: str, ttl_name: str, default_ttl: float) -> bool:
    try:
        ts = float(getattr(jx_state, ts_name, 0.0) or 0.0)
        ttl = float(getattr(jx_state, ttl_name, default_ttl) or default_ttl)
        if ts <= 0.0:
            return False
        return (time.perf_counter() - ts) <= ttl
    except Exception:
        return False


def get_seeds(top_n: int = 12) -> List[str]:
    sources: List[List[str]] = []
    # order matters: most recent/focused first
    if _alive("cog_seeds_ts", "cog_seeds_ttl", _DEF_TTL):
        sources.append(_get_list("cog_seeds_terms"))
    if _alive("foresight_ts", "foresight_ttl", _DEF_TTL):
        sources.append(_get_list("foresight_terms"))
    if _alive("oracle_ts", "oracle_ttl", _DEF_TTL):
        sources.append(_get_list("oracle_terms"))
    if _alive("hsigil_ts", "hsigil_ttl", _DEF_TTL):
        sources.append(_get_list("hsigil_terms"))
        # sequences: flatten a couple of seqs as tokens
        try:
            seqs = getattr(jx_state, "hsigil_seqs", []) or []
            for s in seqs[:2]:
                if isinstance(s, (list, tuple)):
                    sources.append([str(t).strip().lower() for t in s if str(t).strip()])
        except Exception:
            pass
    # dedupe preserving order
    seen: set[str] = set()
    out: List[str] = []
    for arr in sources:
        for t in arr:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= top_n:
                    return out
    return out[:top_n]


__all__ = ["get_seeds"]
