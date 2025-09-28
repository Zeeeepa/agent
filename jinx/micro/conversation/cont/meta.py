from __future__ import annotations

import json
import time
from typing import Dict, Optional

from jinx.async_utils.fs import read_text, write_text
from .cache import CACHE_FILE, _ensure_tmp_dir


def load_cache_meta_sync() -> Dict[str, Optional[str]]:
    """Synchronous helper to load minimal continuity cache metadata.

    Returns {"ts": str|None, "frame_sha": str|None}.
    """
    try:
        import json as _json
        from pathlib import Path
        p = Path(CACHE_FILE)
        if not p.exists():
            return {}
        s = p.read_text(encoding="utf-8").strip()
        if not s:
            return {}
        obj = _json.loads(s)
        out: Dict[str, Optional[str]] = {}
        if "ts" in obj:
            out["ts"] = str(obj.get("ts"))
        if "frame_sha" in obj:
            out["frame_sha"] = str(obj.get("frame_sha"))
        return out
    except Exception:
        return {}


async def load_cache_meta() -> Dict[str, Optional[str]]:
    try:
        s = await read_text(CACHE_FILE)
        if not s:
            return {}
        obj = json.loads(s)
        out: Dict[str, Optional[str]] = {}
        if "ts" in obj:
            out["ts"] = str(obj.get("ts"))
        if "frame_sha" in obj:
            out["frame_sha"] = str(obj.get("frame_sha"))
        return out
    except Exception:
        return {}


async def save_last_context_with_meta(proj_ctx: str, anchors: Dict | None = None, *, frame_sha: str | None = None) -> None:
    try:
        _ensure_tmp_dir()
        payload = {"proj_ctx": (proj_ctx or "")[:20000], "ts": int(time.time() * 1000)}
        if anchors:
            payload["anchors"] = anchors
        if frame_sha:
            payload["frame_sha"] = frame_sha
        body = json.dumps(payload, ensure_ascii=False)
        await write_text(CACHE_FILE, body)
    except Exception:
        return
