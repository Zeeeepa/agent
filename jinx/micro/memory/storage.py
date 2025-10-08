from __future__ import annotations

import os
import time

from jinx.async_utils.fs import read_text_raw, write_text
from jinx.state import shard_lock
from jinx.log_paths import INK_SMEARED_DIARY, EVERGREEN_MEMORY
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT


def _memory_dir() -> str:
    # Primary memory directory under the project root (default: .jinx/memory)
    try:
        sub = os.getenv("JINX_MEMORY_DIR", os.path.join(".jinx", "memory"))
    except Exception:
        sub = os.path.join(".jinx", "memory")
    root = PROJECT_ROOT or os.getcwd()
    return os.path.join(root, sub)


# Canonical memory paths
_MEM_DIR = _memory_dir()
_COMPACT_PATH = os.path.join(_MEM_DIR, "compact.md")
_EVERGREEN_PATH = os.path.join(_MEM_DIR, "evergreen.md")
_HIST_DIR = os.path.join(_MEM_DIR, "history")
_CH_PATHS = os.path.join(_MEM_DIR, "paths.md")
_CH_SYMBOLS = os.path.join(_MEM_DIR, "symbols.md")
_CH_PREFS = os.path.join(_MEM_DIR, "prefs.md")
_CH_DECS = os.path.join(_MEM_DIR, "decisions.md")
_TOPICS_DIR = os.path.join(_MEM_DIR, "topics")
_TOKEN_HINT = os.path.join(_MEM_DIR, ".last_prompt_tokens")
try:
    _HIST_KEEP = max(0, int(os.getenv("JINX_MEM_HISTORY_KEEP", "50")))
except Exception:
    _HIST_KEEP = 50


def _ensure_dirs() -> None:
    try:
        os.makedirs(_MEM_DIR, exist_ok=True)
        os.makedirs(_HIST_DIR, exist_ok=True)
        os.makedirs(_TOPICS_DIR, exist_ok=True)
    except Exception:
        pass


async def write_token_hint(tokens: int) -> None:
    """Write last prompt token estimate to memory dir."""
    async with shard_lock:
        try:
            await write_text(_TOKEN_HINT, str(int(tokens)))
        except Exception:
            pass


async def read_token_hint() -> int:
    """Read last prompt token estimate; 0 if missing."""
    async with shard_lock:
        try:
            if os.path.exists(_TOKEN_HINT):
                txt = await read_text_raw(_TOKEN_HINT)
                return int((txt or "0").strip() or "0")
        except Exception:
            return 0
    return 0


def memory_dir() -> str:
    """Return absolute path to the canonical memory directory (.jinx/memory by default)."""
    return _MEM_DIR


async def read_evergreen() -> str:
    """Read evergreen memory, preferring .jinx/memory/evergreen.md, fallback to log/evergreen_memory.txt."""
    # Prefer new location
    async with shard_lock:
        try:
            if os.path.exists(_EVERGREEN_PATH):
                txt = await read_text_raw(_EVERGREEN_PATH)
                if txt != "":
                    return txt
        except Exception:
            pass
        evergreen = await read_text_raw(EVERGREEN_MEMORY)
    return evergreen or ""


def get_memory_mtimes() -> tuple[int, int]:
    """Return (compact_mtime_ms, evergreen_mtime_ms). Missing files -> 0.

    Note: best-effort synchronous stats for caching layers.
    """
    try:
        c = int(os.stat(_COMPACT_PATH).st_mtime * 1000) if os.path.exists(_COMPACT_PATH) else 0
    except Exception:
        c = 0
    try:
        e = int(os.stat(_EVERGREEN_PATH).st_mtime * 1000) if os.path.exists(_EVERGREEN_PATH) else 0
    except Exception:
        e = 0
    return (c, e)


async def read_compact() -> str:
    """Read compact memory, preferring .jinx/memory/compact.md, fallback to log/ink_smeared_diary.txt."""
    async with shard_lock:
        try:
            if os.path.exists(_COMPACT_PATH):
                txt = await read_text_raw(_COMPACT_PATH)
                if txt != "":
                    return txt
        except Exception:
            pass
        diary = await read_text_raw(INK_SMEARED_DIARY)
    return diary or ""


def _parse_channels(durable_text: str) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "paths": [],
        "symbols": [],
        "prefs": [],
        "decisions": [],
    }
    for raw in (durable_text or "").splitlines():
        line = (raw or "").strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("path: "):
            buckets["paths"].append(line)
        elif low.startswith("symbol: "):
            buckets["symbols"].append(line)
        elif low.startswith("pref: "):
            buckets["prefs"].append(line)
        elif low.startswith("decision: "):
            buckets["decisions"].append(line)
    return buckets


async def read_channel(kind: str) -> str:
    """Read a specific memory channel file if present (paths/symbols/prefs/decisions)."""
    k = (kind or "").strip().lower()
    path = {
        "paths": _CH_PATHS,
        "symbols": _CH_SYMBOLS,
        "prefs": _CH_PREFS,
        "decisions": _CH_DECS,
    }.get(k)
    if not path:
        return ""


async def read_topic(name: str) -> str:
    """Read a topic file from .jinx/memory/topics/<name>.md if present."""
    topic = (name or "").strip()
    if not topic:
        return ""
    path = os.path.join(_TOPICS_DIR, f"{topic}.md")
    async with shard_lock:
        try:
            if os.path.exists(path):
                txt = await read_text_raw(path)
                if txt != "":
                    return txt
        except Exception:
            return ""
    return ""


def ensure_nl(s: str) -> str:
    return s + ("\n" if s and not s.endswith("\n") else "")


async def write_state(compact: str, durable: str | None) -> None:
    """Persist current memory state to both the canonical .jinx/memory/ store and legacy log files.

    - Writes compact context to compact.md and legacy INK_SMEARED_DIARY.
    - Writes evergreen durable memory to evergreen.md and legacy EVERGREEN_MEMORY when provided.
    - Appends a snapshot into .jinx/memory/history/ for auditing and longitudinal learning.
    """
    _ensure_dirs()
    compact_out = ensure_nl(compact)
    ts = int(time.time() * 1000)
    hist_name = f"{ts}_state.md"
    hist_path = os.path.join(_HIST_DIR, hist_name)

    async with shard_lock:
        # Canonical store
        try:
            await write_text(_COMPACT_PATH, compact_out)
        except Exception:
            pass
        if durable is not None:
            try:
                durable_out = ensure_nl(durable)
                await write_text(_EVERGREEN_PATH, durable_out)
            except Exception:
                pass
            # Derive channel files from durable content (best-effort)
            try:
                buckets = _parse_channels(durable_out)
                if buckets.get("paths") is not None:
                    await write_text(_CH_PATHS, ensure_nl("\n".join(buckets.get("paths") or [])))
                if buckets.get("symbols") is not None:
                    await write_text(_CH_SYMBOLS, ensure_nl("\n".join(buckets.get("symbols") or [])))
                if buckets.get("prefs") is not None:
                    await write_text(_CH_PREFS, ensure_nl("\n".join(buckets.get("prefs") or [])))
                if buckets.get("decisions") is not None:
                    await write_text(_CH_DECS, ensure_nl("\n".join(buckets.get("decisions") or [])))
            except Exception:
                pass
        # Legacy files for backward compatibility with existing readers
        try:
            await write_text(INK_SMEARED_DIARY, compact_out)
        except Exception:
            pass
        if durable is not None:
            try:
                durable_out = ensure_nl(durable)
                await write_text(EVERGREEN_MEMORY, durable_out)
            except Exception:
                pass
        # History snapshot (best-effort)
        try:
            snap_lines = ["---", f"ts_ms: {ts}", "---\n", "# compact\n", compact_out, "\n# evergreen\n", ensure_nl(durable or "")]
            await write_text(hist_path, "\n".join(snap_lines))
        except Exception:
            pass

    # Rotate history to keep bounded number of files
    try:
        if _HIST_KEEP > 0 and os.path.isdir(_HIST_DIR):
            files = [os.path.join(_HIST_DIR, f) for f in os.listdir(_HIST_DIR) if f.endswith("_state.md")]
            if len(files) > _HIST_KEEP:
                # Sort by filename (ts prefix) and remove oldest
                for p in sorted(files)[: max(0, len(files) - _HIST_KEEP)]:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    except Exception:
        pass
