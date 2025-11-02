from __future__ import annotations

import os
import json
import asyncio
import re
from typing import Any, Dict, List, Tuple

from .embed_cache import embed_text_cached  # reuse shared cache/TTL
from jinx.micro.memory.storage import read_compact, read_evergreen
from jinx.micro.brain.attention import get_attention_weights as _atten_get

# Simple memory retrieval over a JSONL index of items with optional stored embeddings.
# Each line: {"id": str, "text": str, "embedding": [float,...] (optional), "meta": {...}}
# Env:
#  - EMBED_AGG_MEM_INDEX: path to JSONL (default: mem/index.jsonl)
#  - EMBED_MEM_MAX_ITEMS: max items to scan (default: 200)
#  - EMBED_MEM_SCORE_MIN: minimum score to keep (default: 0.0)


def _mem_index_path() -> str:
    p = os.getenv("EMBED_AGG_MEM_INDEX", "mem/index.jsonl").strip() or "mem/index.jsonl"
    return p


async def _read_jsonl_tail(path: str, max_items: int) -> List[Dict[str, Any]]:
    try:
        def _read() -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for ln in lines[-max_items:]:
                ln = (ln or "").strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict) and (obj.get("text") or obj.get("content")):
                        out.append(obj)
                except Exception:
                    continue
            return out
        return await asyncio.to_thread(_read)
    except FileNotFoundError:
        return []
    except Exception:
        return []


async def _fallback_items(max_items: int) -> List[Dict[str, Any]]:
    """Build memory items from compact/evergreen text as a fallback when no JSONL index.

    - Compact: take last non-empty lines.
    - Evergreen: include channel-like lines as separate items.
    """
    items: List[Dict[str, Any]] = []
    try:
        comp = await read_compact()
    except Exception:
        comp = ""
    try:
        ever = await read_evergreen()
    except Exception:
        ever = ""
    # Recent compact lines
    if comp:
        lines = [ln.strip() for ln in comp.splitlines() if ln.strip()]
        for i, ln in enumerate(lines[-max_items:]):
            items.append({"id": f"compact:{i}", "text": ln})
    # Evergreen facts (prefix-based)
    if ever:
        lines = [ln.strip() for ln in ever.splitlines() if ln.strip()]
        for i, ln in enumerate(lines[-max_items:]):
            low = ln.lower()
            if low.startswith(("path:", "symbol:", "pref:", "decision:")):
                items.append({"id": f"ever:{i}", "text": ln})
    return items[:max_items]


def _dot(a: List[float] | None, b: List[float] | None) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        try:
            s += float(a[i] or 0.0) * float(b[i] or 0.0)
        except Exception:
            continue
    return s


async def retrieve_memory_top_k(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    q = (query or "").strip()
    if not q:
        return []
    try:
        max_items = max(10, int(os.getenv("EMBED_MEM_MAX_ITEMS", "200")))
    except Exception:
        max_items = 200
    try:
        score_min = float(os.getenv("EMBED_MEM_SCORE_MIN", "0.0"))
    except Exception:
        score_min = 0.0

    # Load memory index tail (most recent entries)
    items = await _read_jsonl_tail(_mem_index_path(), max_items)
    if not items:
        # Fallback to compact/evergreen memory
        items = await _fallback_items(max_items)
        if not items:
            return []

    # Compute query embedding
    try:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    except Exception:
        model = "text-embedding-3-small"
    try:
        q_emb = await embed_text_cached(q, model=model)
    except Exception:
        q_emb = []

    # Score items by dot product; compute missing embeddings on-the-fly (bounded)
    # Also integrate attention weights on item terms
    try:
        atten = _atten_get()
    except Exception:
        atten = {}
    _tok_re = re.compile(r"(?u)[\w\.]{3,}")
    try:
        gamma = float(os.getenv("EMBED_MEM_ATTEN_GAMMA", "0.25"))
    except Exception:
        gamma = 0.25
    results: List[Tuple[float, str, Dict[str, Any]]] = []
    for idx, it in enumerate(items):
        try:
            mid = str(it.get("id") or f"{idx}")
            text = str(it.get("text") or it.get("content") or "").strip()
            if not text:
                continue
            memb = it.get("embedding")
            if not memb:
                try:
                    memb = await embed_text_cached(text, model=model)
                except Exception:
                    memb = []
            sc = _dot(q_emb, memb)
            if atten:
                try:
                    att_sum = 0.0
                    for m in _tok_re.finditer(text.lower()):
                        t = (m.group(0) or "").strip().lower()
                        if t and len(t) >= 3:
                            try:
                                att_sum += float(atten.get(f"term: {t}", 0.0))
                            except Exception:
                                continue
                    if att_sum != 0.0:
                        sc += gamma * att_sum
                except Exception:
                    pass
            if sc < score_min:
                continue
            meta = {
                "memory_id": mid,
                "text_preview": text[:256],
            }
            results.append((float(sc or 0.0), f"memory://{mid}", {"meta": meta}))
        except Exception:
            continue

    # Sort by score desc and return top-k
    results.sort(key=lambda h: float(h[0] or 0.0), reverse=True)
    return results[: max(1, int(k or 1))]
