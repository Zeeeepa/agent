from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Any

from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT


def _mem_dir() -> str:
    root = PROJECT_ROOT or os.getcwd()
    d = os.path.join(root, ".jinx", "memory")
    os.makedirs(d, exist_ok=True)
    return d


_BOARD_PATH = os.path.join(_mem_dir(), "board.json")
_FEN_PATH = os.path.join(_mem_dir(), "board_fen.txt")


@dataclass
class BoardState:
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    # Session/group state
    session: str = "main"
    active_turns: int = 0
    turns_total: int = 0
    errors_total: int = 0
    last_error: str = ""
    last_query: str = ""
    # Capabilities/skills acquired
    skills: Set[str] = field(default_factory=set)
    # Patch outcomes
    patches_ok: int = 0
    patches_fail: int = 0
    last_patch_msg: str = ""
    # API intents and artifacts (heuristic)
    api_intents: int = 0
    api_endpoints_seen: Set[str] = field(default_factory=set)
    # Self-update/health (basic counters)
    selfupdate_success: int = 0
    selfupdate_fail: int = 0
    # High-level compact cognition
    goals: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    capability_gaps: List[str] = field(default_factory=list)
    next_action: str = ""
    mem_pointers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert sets to sorted lists for JSON
        d["skills"] = sorted(self.skills)
        d["api_endpoints_seen"] = sorted(self.api_endpoints_seen)
        return d

    def fen(self) -> str:
        # Compact snapshot string (JIN-FEN): key=value; pairs, minimal spacing
        parts = [
            f"t={self.ts_ms}",
            f"sess={self.session}",
            f"act={self.active_turns}",
            f"turns={self.turns_total}",
            f"err={self.errors_total}",
            f"okp={self.patches_ok}",
            f"fpp={self.patches_fail}",
            f"sucsu={self.selfupdate_success}",
            f"faisu={self.selfupdate_fail}",
            f"skills={len(self.skills)}",
            f"apii={self.api_intents}",
            f"apis={len(self.api_endpoints_seen)}",
        ]
        return ";".join(parts)


# In-process lock to serialize writes
_lock = asyncio.Lock()
_embed_lock = asyncio.Lock()
_last_embed_ms: int = 0
_EMBED_DEBOUNCE_MS = 2500


async def _read_board() -> BoardState:
    try:
        if not os.path.exists(_BOARD_PATH):
            return BoardState()
        def _load() -> Dict[str, Any]:
            with open(_BOARD_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        data = await asyncio.to_thread(_load)
        st = BoardState()
        st.ts_ms = int(data.get("ts_ms", st.ts_ms))
        st.session = str(data.get("session", st.session))
        st.active_turns = int(data.get("active_turns", 0))
        st.turns_total = int(data.get("turns_total", 0))
        st.errors_total = int(data.get("errors_total", 0))
        st.last_error = str(data.get("last_error", ""))
        st.last_query = str(data.get("last_query", ""))
        st.skills = set(data.get("skills", []) or [])
        st.patches_ok = int(data.get("patches_ok", 0))
        st.patches_fail = int(data.get("patches_fail", 0))
        st.last_patch_msg = str(data.get("last_patch_msg", ""))
        st.api_intents = int(data.get("api_intents", 0))
        st.api_endpoints_seen = set(data.get("api_endpoints_seen", []) or [])
        st.selfupdate_success = int(data.get("selfupdate_success", 0))
        st.selfupdate_fail = int(data.get("selfupdate_fail", 0))
        st.goals = list(data.get("goals", []) or [])
        st.plan = list(data.get("plan", []) or [])
        st.capability_gaps = list(data.get("capability_gaps", []) or [])
        st.next_action = str(data.get("next_action", ""))
        st.mem_pointers = list(data.get("mem_pointers", []) or [])
        return st
    except Exception:
        return BoardState()


async def _write_board(st: BoardState) -> None:
    d = st.to_dict()
    def _dump() -> None:
        with open(_BOARD_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    def _dump_fen() -> None:
        with open(_FEN_PATH, "w", encoding="utf-8") as f:
            f.write(st.fen())
    await asyncio.to_thread(_dump)
    await asyncio.to_thread(_dump_fen)


async def maybe_embed_board(tag: str | None = None) -> None:
    """Debounced embedding of board snapshots (FEN and JSON) into embeddings.
    Stored under source='state' with kind 'board_fen'/'board_json'.
    """
    global _last_embed_ms
    now = int(time.time() * 1000)
    if (now - _last_embed_ms) < _EMBED_DEBOUNCE_MS:
        return
    async with _embed_lock:
        now2 = int(time.time() * 1000)
        if (now2 - _last_embed_ms) < _EMBED_DEBOUNCE_MS:
            return
        _last_embed_ms = now2
        try:
            st = await _read_board()
            # Build JSON and FEN
            import json as _json
            body_json = _json.dumps(st, ensure_ascii=False)
            from jinx.micro.memory.board_state import BoardState as _BS
            bs = _BS()
            bs.session = str(st.get("session") or "main")
            bs.active_turns = int(st.get("active_turns") or 0)
            bs.turns_total = int(st.get("turns_total") or 0)
            bs.errors_total = int(st.get("errors_total") or 0)
            bs.patches_ok = int(st.get("patches_ok") or 0)
            bs.patches_fail = int(st.get("patches_fail") or 0)
            bs.selfupdate_success = int(st.get("selfupdate_success") or 0)
            bs.selfupdate_fail = int(st.get("selfupdate_fail") or 0)
            bs.skills = set(st.get("skills") or [])
            bs.api_intents = int(st.get("api_intents") or 0)
            bs.api_endpoints_seen = set(st.get("api_endpoints_seen") or [])
            body_fen = bs.fen()
        except Exception:
            return
        # Fire-and-forget embedding
        async def _emit() -> None:
            try:
                from jinx.micro.embeddings.pipeline import embed_text as _embed
                await _embed(body_fen, source="state", kind="board_fen")
                await _embed(body_json, source="state", kind="board_json")
            except Exception:
                pass
        asyncio.create_task(_emit())


async def touch_board(**kwargs: Any) -> None:
    """Update the board with given fields atomically and persist JSON + FEN."""
    async with _lock:
        st = await _read_board()
        st.ts_ms = int(time.time() * 1000)
        # Apply updates
        if "session" in kwargs and kwargs["session"]:
            st.session = str(kwargs["session"])
        if "active_turns" in kwargs:
            try:
                st.active_turns = max(0, int(kwargs["active_turns"]))
            except Exception:
                pass
        if "turns_inc" in kwargs:
            st.turns_total += int(kwargs["turns_inc"]) or 0
        if "errors_inc" in kwargs:
            st.errors_total += int(kwargs["errors_inc"]) or 0
        if "last_error" in kwargs:
            st.last_error = str(kwargs["last_error"] or "")
        if "last_query" in kwargs:
            st.last_query = str(kwargs["last_query"] or "")
        if "skill_add" in kwargs and kwargs["skill_add"]:
            st.skills.add(str(kwargs["skill_add"]))
        if "patch_ok" in kwargs and kwargs["patch_ok"]:
            st.patches_ok += 1
            st.last_patch_msg = str(kwargs.get("patch_msg") or "")
        if "patch_fail" in kwargs and kwargs["patch_fail"]:
            st.patches_fail += 1
            st.last_patch_msg = str(kwargs.get("patch_msg") or "")
        if "api_intent" in kwargs and kwargs["api_intent"]:
            st.api_intents += 1
        if "api_endpoint" in kwargs and kwargs["api_endpoint"]:
            st.api_endpoints_seen.add(str(kwargs["api_endpoint"]))
        if "selfupdate_ok" in kwargs and kwargs["selfupdate_ok"]:
            st.selfupdate_success += 1
        if "selfupdate_fail" in kwargs and kwargs["selfupdate_fail"]:
            st.selfupdate_fail += 1
        await _write_board(st)


async def read_board() -> Dict[str, Any]:
    st = await _read_board()
    return st.to_dict()
