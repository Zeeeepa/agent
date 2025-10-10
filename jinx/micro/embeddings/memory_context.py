from __future__ import annotations

import hashlib
import os
from typing import List

from .retrieval import retrieve_top_k
from jinx.micro.embeddings.text_clean import is_noise_text
from jinx.micro.memory.storage import read_compact


async def build_memory_context_for(query: str, *, k: int | None = None, max_chars: int = 1500, max_time_ms: int | None = 220) -> str:
    """Build a memory context block similar to <embeddings_code>, without sending evergreen.

    Prefer top-k runtime-embedded memory hits (dialogue/state/sandbox). If none are available
    under the time budget, fall back to a compact selection from .jinx/memory/compact.md.
    Returns a block tagged as <embeddings_memory>...</embeddings_memory> or an empty string.
    """
    q = (query or "").strip()
    if not q:
        return ""
    k_eff = k or int(os.getenv("EMBED_TOP_K", "5"))

    # Try runtime embeddings first
    hits = []
    try:
        hits = await retrieve_top_k(q, k=k_eff, max_time_ms=max_time_ms)
    except Exception:
        hits = []

    parts: List[str] = []
    if hits:
        # Deduplicate identical previews and content by hash; keep chronological order
        seen: set[str] = set()
        seen_hash: set[str] = set()
        import time as _t
        def _ts(h):
            try:
                return float((h[2].get("meta", {}).get("ts") or 0.0))
            except Exception:
                return 0.0
        for score, src, obj in sorted(hits, key=_ts):
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or "").strip()
            if not pv or pv in seen or is_noise_text(pv):
                continue
            csha = (meta.get("content_sha256") or "").strip()
            if csha:
                if csha in seen_hash:
                    continue
                seen_hash.add(csha)
            seen.add(pv)
            parts.append(pv)
            if sum(len(p) for p in parts) > max_chars:
                break
    # Fallback to compact.md selection when empty
    if not parts:
        try:
            raw = (await read_compact()) or ""
        except Exception:
            raw = ""
        if raw:
            # Take tail-biased non-empty lines, dedup, within budget
            lines = [ln.strip() for ln in raw.splitlines() if (ln or "").strip()]
            seen: set[str] = set()
            for ln in lines[-200:]:
                if ln in seen or is_noise_text(ln):
                    continue
                seen.add(ln)
                parts.append(ln)
                if sum(len(p) for p in parts) > max_chars:
                    break
    if not parts:
        return ""
    body = "\n".join(parts)
    return f"<embeddings_memory>\n{body}\n</embeddings_memory>"


__all__ = ["build_memory_context_for"]
