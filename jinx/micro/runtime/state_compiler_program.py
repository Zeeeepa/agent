from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

from jinx.micro.runtime.program import MicroProgram


class StateCompilerProgram(MicroProgram):
    """Compile a compact "board-state" cognition from runtime + memory.

    Periodically produces:
      - goals: list[str]
      - plan: list[str]
      - capability_gaps: list[str]
      - next_action: str
      - mem_pointers: list[str]

    Writes into board via touch_board(), with strict RT budgets.
    """

    def __init__(self) -> None:
        super().__init__(name="StateCompilerProgram")
        try:
            self._period_sec = float(os.getenv("JINX_STATE_COMPILER_SEC", "2.5"))
        except Exception:
            self._period_sec = 2.5
        try:
            self._budget_ms = int(os.getenv("JINX_STATE_COMPILER_BUDGET_MS", "500"))
        except Exception:
            self._budget_ms = 500
        self._last_ts = 0.0
        # LLM gating (default OFF to avoid idle API churn)
        try:
            self._llm_on = str(os.getenv("JINX_STATE_COMPILER_LLM", "0")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            self._llm_on = False
        self._prev_sig: str = ""

    async def run(self) -> None:
        await self.log("state-compiler online")
        while True:
            try:
                await self._tick()
            except Exception:
                pass
            await asyncio.sleep(self._period_sec)

    async def _tick(self) -> None:
        t0 = time.perf_counter()
        # Do not overrun budget; quick exit when close
        def time_left_ms() -> int:
            return int(max(1.0, self._budget_ms - (time.perf_counter() - t0) * 1000.0))

        # Load inputs
        try:
            from jinx.micro.memory.board_state import read_board as _read_board
            board = await _read_board()
        except Exception:
            board = {}
        try:
            from jinx.micro.memory.storage import read_evergreen as _read_ev, read_summary_heads as _sum_heads
            evergreen = await _read_ev()
            heads = await _sum_heads(max_n=2)
        except Exception:
            evergreen = ""; heads = []
        memo_lines = [h[1] for h in (heads or []) if h and h[1]]
        last_query = str(board.get("last_query") or "")

        # Assemble minimal context
        ctx = {
            "board": board,
            "last_query": last_query,
            "memory": ("\n".join(memo_lines) if memo_lines else "")[:1200],
            "evergreen": (evergreen or "")[:2000],
        }

        # Build a compact signature to detect meaningful changes
        try:
            sig = "|".join([
                (last_query or "").strip(),
                str(int(board.get("turns_total") or 0)),
                str(int(board.get("errors_total") or 0)),
                str(len(board.get("skills") or [])),
                str(int(board.get("api_intents") or 0)),
            ])
        except Exception:
            sig = ""

        # Try LLM JSON compile under tight timeout, only if enabled and meaningful changes present
        out_obj: Dict[str, Any] = {}
        try:
            should_llm = (
                self._llm_on and (time_left_ms() > 180) and (
                    bool(last_query.strip()) or sig != (self._prev_sig or "")
                )
            )
            if should_llm:
                prompt = (
                    "You are a systems planner. Given compact runtime board, last_query, and memory, "
                    "produce STRICT JSON ONLY with keys: goals (array of short strings), plan (array of 3-6 short steps), "
                    "capability_gaps (array), next_action (string), mem_pointers (array of short pointers).\n"
                    "Rules: minimal, actionable, avoid redundancy, ASCII only, no code fences.\n\n"
                    f"BOARD_JSON:\n{json.dumps(ctx['board'], ensure_ascii=False)}\n\n"
                    f"LAST_QUERY:\n{ctx['last_query']}\n\n"
                    f"MEMORY_SNIPPETS:\n{ctx['memory']}\n\n"
                    f"EVERGREEN:\n{ctx['evergreen']}\n"
                )
                from jinx.micro.llm.service import spark_openai as _spark
                raw, _ = await asyncio.wait_for(_spark(prompt), timeout=max(0.2, time_left_ms()) / 1000.0)
                if raw:
                    try:
                        m = json.loads(_extract_json(raw))
                        if isinstance(m, dict):
                            out_obj = m
                    except Exception:
                        out_obj = {}
                # Update signature only after a successful LLM cycle
                self._prev_sig = sig
        except Exception:
            out_obj = {}

        # Fallback heuristics if LLM failed or incomplete
        goals = out_obj.get("goals") if isinstance(out_obj.get("goals"), list) else []
        plan = out_obj.get("plan") if isinstance(out_obj.get("plan"), list) else []
        gaps = out_obj.get("capability_gaps") if isinstance(out_obj.get("capability_gaps"), list) else []
        next_action = out_obj.get("next_action") if isinstance(out_obj.get("next_action"), str) else ""
        mem_ptrs = out_obj.get("mem_pointers") if isinstance(out_obj.get("mem_pointers"), list) else []

        if not goals:
            goals = ["answer_user", "improve_capabilities", "stabilize_runtime"]
        if not plan:
            plan = [
                "resolve_query_or_route_task",
                "retrieve_project+memory_context",
                "apply_patch_or_call_skill",
                "verify_results",
            ]
        if not next_action:
            next_action = "resolve_query_or_route_task"
        if not mem_ptrs and memo_lines:
            mem_ptrs = ["summary:recent"]
        # Simple gap inference
        try:
            if int(board.get("patches_fail") or 0) > int(board.get("patches_ok") or 0):
                gaps.append("patch_reliability")
        except Exception:
            pass
        try:
            if int(board.get("errors_total") or 0) > 0 and "error_handling" not in gaps:
                gaps.append("error_handling")
        except Exception:
            pass
        # Deduplicate and clamp sizes
        def _dedup(xs):
            seen = set(); out = []
            for x in xs:
                s = (str(x)[:80]).strip()
                if s and s not in seen:
                    seen.add(s); out.append(s)
            return out[:8]
        goals = _dedup(goals)
        plan = _dedup(plan)
        gaps = _dedup(gaps)
        mem_ptrs = _dedup(mem_ptrs)
        next_action = (next_action or "")[:100]

        # Update board
        try:
            from jinx.micro.memory.board_state import touch_board as _touch, maybe_embed_board as _embed
            await _touch(goals=goals, plan=plan, capability_gaps=gaps, next_action=next_action, mem_pointers=mem_ptrs)
            await _embed("state_compiler")
        except Exception:
            pass


def _extract_json(s: str) -> str:
    try:
        import re
        m = re.search(r"\{[\s\S]*\}", s)
        return m.group(0) if m else s.strip()
    except Exception:
        return s or "{}"


async def spawn_state_compiler() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(StateCompilerProgram())
