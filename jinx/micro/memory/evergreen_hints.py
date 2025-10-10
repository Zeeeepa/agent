from __future__ import annotations

from typing import Dict, List, Optional

from jinx.micro.memory.evergreen_select import select_evergreen_for as _select_evg
from jinx.micro.embeddings.project_query_tokens import expand_strong_tokens as _expand_tokens
from jinx.micro.memory.storage import read_channel as _read_channel


async def build_evergreen_hints(query: str, anchors: Optional[Dict[str, List[str]]] = None, *, max_tokens: int = 12) -> Dict[str, List[str]]:
    """Derive retrieval hints from evergreen without sending its content to the LLM.

    Returns a dict with keys:
    - tokens: strong tokens extracted from the selected evergreen snippet and query
    - paths: lines from the 'paths' channel (already derived from durable memory)
    - symbols: lines from the 'symbols' channel
    - prefs: lines from the 'prefs' channel
    - decisions: lines from the 'decisions' channel
    """
    q = (query or "").strip()
    if not q:
        return {"tokens": [], "paths": [], "symbols": [], "prefs": [], "decisions": []}

    # Select a small relevant evergreen snippet (internal use only)
    try:
        snip = await _select_evg(q, anchors=anchors)
    except Exception:
        snip = ""

    # Expand strong tokens from both query and snippet
    toks_q: List[str]
    toks_e: List[str]
    try:
        toks_q = _expand_tokens(q, max_items=max_tokens)
    except Exception:
        toks_q = []
    try:
        toks_e = _expand_tokens(snip or "", max_items=max_tokens)
    except Exception:
        toks_e = []
    # Merge tokens, preserve order
    seen: set[str] = set()
    tokens: List[str] = []
    for t in toks_q + toks_e:
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)

    # Read durable channels (already parsed from evergreen by storage.write_state)
    async def _safe_read(kind: str) -> List[str]:
        try:
            raw = (await _read_channel(kind) or "").splitlines()
        except Exception:
            raw = []
        # Keep non-empty trimmed lines
        return [ln.strip() for ln in raw if (ln or "").strip()]

    paths = await _safe_read("paths")
    symbols = await _safe_read("symbols")
    prefs = await _safe_read("prefs")
    decisions = await _safe_read("decisions")

    return {
        "tokens": tokens,
        "paths": paths,
        "symbols": symbols,
        "prefs": prefs,
        "decisions": decisions,
    }


__all__ = ["build_evergreen_hints"]
