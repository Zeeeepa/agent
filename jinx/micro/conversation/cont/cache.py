from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from jinx.async_utils.fs import read_text, write_text
from .util import _is_short_reply

TMP_DIR = os.path.join(".jinx", "tmp")
CACHE_FILE = os.path.join(TMP_DIR, "continuity.json")


def _ensure_tmp_dir() -> None:
    try:
        os.makedirs(TMP_DIR, exist_ok=True)
    except Exception:
        pass


async def load_last_context() -> str:
    try:
        _ensure_tmp_dir()
        s = await read_text(CACHE_FILE)
        if not s:
            return ""
        obj = json.loads(s)
        # TTL guard
        try:
            ttl_ms = int(os.getenv("JINX_CONTINUITY_CACHE_TTL_MS", "600000"))  # 10 minutes default
        except Exception:
            ttl_ms = 600000
        ts = float(obj.get("ts", 0.0))
        if ttl_ms > 0 and ts > 0.0:
            age_ms = max(0.0, (time.time() * 1000.0) - ts)
            if age_ms > ttl_ms:
                return ""
        return str(obj.get("proj_ctx") or "")
    except Exception:
        return ""


async def save_last_context(proj_ctx: str, anchors: Optional[Dict[str, List[str]]] = None) -> None:
    try:
        _ensure_tmp_dir()
        payload = {"proj_ctx": (proj_ctx or "")[:20000], "ts": int(time.time() * 1000)}
        if anchors:
            payload["anchors"] = anchors
        body = json.dumps(payload, ensure_ascii=False)
        await write_text(CACHE_FILE, body)
    except Exception:
        return


async def maybe_reuse_last_context(x: str, proj_ctx: str, synth: str) -> str:
    """If current proj_ctx is empty and input is short reply, reuse last cached proj_ctx."""
    if proj_ctx and proj_ctx.strip():
        return ""
    if not _is_short_reply(x or ""):
        return ""
    return await load_last_context()


async def load_last_anchors() -> Dict[str, List[str]]:
    try:
        s = await read_text(CACHE_FILE)
        if not s:
            return {}
        obj = json.loads(s)
        # TTL guard aligns with context TTL
        try:
            ttl_ms = int(os.getenv("JINX_CONTINUITY_CACHE_TTL_MS", "600000"))
        except Exception:
            ttl_ms = 600000
        ts = float(obj.get("ts", 0.0))
        if ttl_ms > 0 and ts > 0.0:
            age_ms = max(0.0, (time.time() * 1000.0) - ts)
            if age_ms > ttl_ms:
                return {}
        anc = obj.get("anchors") or {}
        if isinstance(anc, dict):
            return {k: list(v) for k, v in anc.items() if isinstance(v, list)}
        return {}
    except Exception:
        return {}
