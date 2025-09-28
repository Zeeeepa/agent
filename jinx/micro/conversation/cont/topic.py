from __future__ import annotations

from jinx.micro.embeddings.retrieval import retrieve_top_k


async def detect_topic_shift(query: str, *, k: int = 6, max_time_ms: int = 120) -> bool:
    """Return True if the current query appears to shift topic (no 'state' hits among top-k).

    Tuned for speed: small k, tight time budget. Env toggle JINX_TOPIC_SHIFT_CHECK governs usage.
    """
    try:
        q = (query or "").strip()
        if not q:
            return False
        hits = await retrieve_top_k(q, k=k, max_time_ms=max_time_ms)
        if not hits:
            return False
        # consider a shift if zero hits originate from 'state'
        has_state = False
        for score, src, obj in hits:
            meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
            src_meta = (meta.get("source") or "").strip().lower()
            src_i = (src or "").strip().lower()
            if src_meta == "state" or src_i == "state":
                has_state = True
                break
        return not has_state
    except Exception:
        return False
