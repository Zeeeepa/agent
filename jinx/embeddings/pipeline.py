from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from typing import Optional, Dict, Any, Deque, Iterable
import re

from jinx.net import get_openai_client
from jinx.config import ALL_TAGS

EMBED_ROOT = os.path.join("log", "embeddings")
INDEX_DIR = os.path.join(EMBED_ROOT, "index")
_RECENT_MAX = 200
_recent: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_MAX)

_RE_CODEY = re.compile(r"^\s*(print\s*\(|return\b|def\b|class\b|import\b|from\b)", re.I)
_RE_NUMBERY = re.compile(r"^[\s\d+\-*/().]+$")

def _is_noise_text(pv: str) -> bool:
    pv = (pv or "").strip()
    if len(pv) < 4:
        return True
    if _RE_CODEY.match(pv):
        return True
    if _RE_NUMBERY.match(pv) and any(ch.isdigit() for ch in pv):
        return True
    return False


# Remove known wrapper tags like <machine_123>, </machine_123>, <python_...>
_TAG_OPEN_RE = re.compile(r"<([a-zA-Z_]+)(?:_\d+)?\s*>")
_TAG_CLOSE_RE = re.compile(r"</([a-zA-Z_]+)(?:_\d+)?\s*>")


def _strip_known_tags(text: str) -> str:
    if not text:
        return text
    def repl_open(m: re.Match) -> str:
        base = m.group(1).lower()
        return "" if base in ALL_TAGS else m.group(0)
    def repl_close(m: re.Match) -> str:
        base = m.group(1).lower()
        return "" if base in ALL_TAGS else m.group(0)
    cleaned = _TAG_OPEN_RE.sub(repl_open, text)
    cleaned = _TAG_CLOSE_RE.sub(repl_close, cleaned)
    # Also drop lines that are only leftover angle brackets or whitespace
    cleaned_lines = []
    for ln in cleaned.splitlines():
        s = ln.strip()
        if not s:
            continue
        # drop a line if it became just <> or similar
        if s in {"<>", "</>", "<>:"}:
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines)


def _ensure_dirs() -> None:
    os.makedirs(EMBED_ROOT, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _now_ts() -> float:
    return time.time()


async def embed_text(text: str, *, source: str, kind: str = "text") -> Dict[str, Any]:
    """Create an embedding for text and persist versioned artifact.

    Storage layout:
    - log/embeddings/{source}/{hash}.json  -> embedding item
    - log/embeddings/index/{source}.jsonl  -> append-only index
    """
    _ensure_dirs()
    raw = (text or "").strip()
    if not raw:
        return {"skipped": True, "reason": "empty"}
    cleaned = _strip_known_tags(raw)
    text = cleaned.strip()
    if not text:
        return {"skipped": True, "reason": "empty_after_tags"}
    if _is_noise_text(text):
        return {"skipped": True, "reason": "empty"}

    content_id = _sha256(text)
    source_dir = os.path.join(EMBED_ROOT, source)
    os.makedirs(source_dir, exist_ok=True)

    item_path = os.path.join(source_dir, f"{content_id}.json")
    if os.path.exists(item_path):
        # Already embedded; still record a touch in index
        meta = json.loads(open(item_path, "r", encoding="utf-8").read())
        await _append_index(source, {
            "ts": _now_ts(),
            "source": source,
            "kind": kind,
            "content_id": content_id,
            "dedup": True,
        })
        # Also surface to recent cache for real-time retrieval
        try:
            _recent.appendleft(meta)
        except Exception:
            pass
        return {"cached": True, **meta}

    # Call OpenAI embeddings through existing client; use to_thread for sync SDK
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    async def _call() -> Any:
        return await asyncio.to_thread(
            get_openai_client().embeddings.create,
            model=model,
            input=text,
        )

    resp = await _call()
    vec = resp.data[0].embedding if getattr(resp, "data", None) else None

    meta: Dict[str, Any] = {
        "ts": _now_ts(),
        "model": model,
        "source": source,
        "kind": kind,
        "content_sha256": content_id,
        "dims": len(vec) if vec is not None else 0,
        "text_preview": text[:256],
    }

    payload = {
        "meta": meta,
        "embedding": vec,
    }

    with open(item_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    await _append_index(source, {
        "ts": meta["ts"],
        "source": source,
        "kind": kind,
        "content_id": content_id,
    })

    # Push to in-memory recent cache for real-time use
    try:
        _recent.appendleft(payload)
    except Exception:
        pass

    return payload


async def _append_index(source: str, row: Dict[str, Any]) -> None:
    # Some sources may include path separators (e.g., "sandbox/<file>").
    # Sanitize to a flat filename for the index while keeping a readable hint.
    safe_source = source.replace(os.sep, "__").replace("/", "__")
    path = os.path.join(INDEX_DIR, f"{safe_source}.jsonl")
    # Append-only, tolerate concurrent writers via simple retry
    line = json.dumps(row, ensure_ascii=False)
    for _ in range(3):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return
        except Exception:
            await asyncio.sleep(0.05)
    # Best-effort; drop on persistent failure


def iter_recent_items() -> Iterable[Dict[str, Any]]:
    """Return a snapshot iterator over recent embedded payloads (most recent first)."""
    return list(_recent)
