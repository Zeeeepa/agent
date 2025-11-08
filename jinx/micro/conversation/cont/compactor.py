from __future__ import annotations

import os
from typing import List, Dict, Any

from jinx.async_utils.fs import read_text, write_text
from jinx.micro.embeddings.pipeline import embed_text

_CACHE_PATH = os.path.join(".jinx", "tmp", "continuity.json")


async def _load_cache() -> Dict[str, Any]:
    try:
        s = await read_text(_CACHE_PATH)
        return {} if not s else __import__("json").loads(s)
    except Exception:
        return {}


async def _save_cache(obj: Dict[str, Any]) -> None:
    try:
        body = __import__("json").dumps(obj, ensure_ascii=False)
        await write_text(_CACHE_PATH, body)
    except Exception:
        return


def _recent_state_previews(k: int) -> List[str]:
    out: List[str] = []
    try:
        # Lazy import to avoid circular imports at module import time
        try:
            from jinx.micro.embeddings.pipeline import iter_recent_items as _iter_recent
            it = list(_iter_recent())
        except Exception:
            it = []
        for obj in it:
            meta = obj.get("meta", {})
            if (meta.get("source") or "").strip().lower() != "state":
                continue
            kind = (meta.get("kind") or "").strip().lower()
            # Prefer frames; concept will be synthesized here
            if kind not in ("frame", "concept"):
                continue
            pv = (meta.get("text_preview") or "").strip()
            if not pv:
                continue
            out.append(pv)
            if len(out) >= k:
                break
    except Exception:
        pass
    return out


async def maybe_compact_state_frames() -> None:
    """Compact recent StateFrames into a single ConceptFrame and embed it.

    Controlled by env:
      - JINX_STATEFRAME_COMPACT_N: run compaction every N frames (default 6)
      - JINX_STATEFRAME_RECENT_K: how many recent frames to compact (default 8)
    """
    try:
        import os
        n = int(os.getenv("JINX_STATEFRAME_COMPACT_N", "6"))
        k = int(os.getenv("JINX_STATEFRAME_RECENT_K", "8"))
    except Exception:
        n, k = 6, 8

    if n <= 0 or k <= 1:
        return

    cache = await _load_cache()
    cnt = int(cache.get("frame_n", 0))
    cnt += 1
    cache["frame_n"] = cnt

    # Not the time to compact yet
    if (cnt % n) != 0:
        await _save_cache(cache)
        return

    previews = _recent_state_previews(k)
    if not previews:
        await _save_cache(cache)
        return

    # Build a compact "concept" text using previews (unique, preserving order)
    uniq: List[str] = []
    seen = set()
    for p in previews:
        if p and p not in seen:
            uniq.append(p)
            seen.add(p)
        if len(uniq) >= k:
            break
    if not uniq:
        await _save_cache(cache)
        return

    body_lines = ["[ConceptFrame v1]"]
    for p in uniq:
        body_lines.append(f"• {p}")
    concept_text = "\n".join(body_lines)[:900]
    try:
        await embed_text(concept_text, source="state", kind="concept")
    except Exception:
        pass

    await _save_cache(cache)
