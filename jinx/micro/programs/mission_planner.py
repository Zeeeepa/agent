from __future__ import annotations

import asyncio
import os
import json
from typing import Any, Dict, Optional, Tuple

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.common.log import log_info, log_warn
from jinx.micro.rt.activity import (
    set_activity as _set_act,
    set_activity_detail as _set_det,
    clear_activity as _clear_act,
    clear_activity_detail as _clear_det,
)


class MissionPlannerProgram(MicroProgram):
    """Autonomous mission planner.

    - No user input required.
    - Periodically inspects repository and runtime signals.
    - Decides next high-level goals and dispatches concrete tasks to the runtime.
    - Initial focus: API architecture synthesis if missing.
    """

    def __init__(self) -> None:
        super().__init__(name="MissionPlanner")
        self._tick_sec = float(os.getenv("JINX_MISSION_TICK_SEC", "4.0") or 4.0)
        # Prevent duplicate submissions within session
        self._arch_submitted = False

    async def run(self) -> None:
        await self.log("mission planner online")
        while True:
            try:
                await self._tick()
            except Exception:
                pass
            await asyncio.sleep(self._tick_sec)

    async def _tick(self) -> None:
        # If shutdown soon or throttle, skip heavy planning
        try:
            import jinx.state as jx_state
            if jx_state.shutdown_event.is_set():
                return
        except Exception:
            pass

        # 1) API architecture goal: if missing app, generate it
        if await self._needs_api_architecture() and not self._arch_submitted:
            _set_act("planning: api_architecture")
            _set_det({"mission": {"goal": "api_architecture"}})
            ok = await self._submit_api_architecture()
            self._arch_submitted = ok
            _clear_det(); _clear_act()

    async def _needs_api_architecture(self) -> bool:
        root = os.getcwd()
        ap = os.path.join(root, "api", "app.py")
        try:
            if os.path.isfile(ap):
                return False
        except Exception:
            pass
        # Also require: avoid repeated runs when a marker exists
        marker = os.path.join(root, ".jinx", "plan", "arch.done")
        try:
            if os.path.exists(marker):
                return False
        except Exception:
            pass
        return True

    async def _submit_api_architecture(self) -> bool:
        # Build context and synthesize spec automatically
        spec = await self._auto_spec()
        # Submit task regardless (program will default if spec None)
        try:
            from jinx.micro.runtime.api import submit_task as _submit
        except Exception:
            return False
        try:
            framework = (os.getenv("JINX_API_ARCH_FRAMEWORK", "fastapi") or "fastapi").strip().lower()
            budget_ms = int(os.getenv("JINX_API_ARCH_BUDGET_MS", "1600"))
            await _submit("architect.api", spec=spec, framework=framework, budget_ms=budget_ms)
            # Record marker asynchronously
            try:
                await asyncio.to_thread(self._write_marker)
            except Exception:
                pass
            await self.log("api architecture task submitted")
            log_info("mission.arch.submit", framework=framework)
            return True
        except Exception:
            log_warn("mission.arch.submit_failed")
            return False

    def _write_marker(self) -> None:
        root = os.getcwd()
        d = os.path.join(root, ".jinx", "plan")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "arch.done"), "w", encoding="utf-8") as f:
            f.write("ok")

    async def _auto_spec(self) -> Optional[Dict[str, Any]]:
        """Build a compact API spec without user prompts.

        Strategy:
        - Derive project name from cwd or README.md
        - Extract candidate resources from directory/file names
        - Use LLM to refine into strict JSON when available; fallback to heuristic default
        """
        name = self._infer_project_name()
        resources = self._infer_resources()
        # Try LLM refinement
        spec = await self._try_llm_spec(name, resources)
        if spec:
            return spec
        # Fallback minimal
        return {
            "name": name or "api",
            "resources": resources or [
                {"name": "users", "fields": {"id": "int", "email": "str", "name": "str"}, "endpoints": ["list", "get", "create", "update", "delete"]}
            ]
        }

    def _infer_project_name(self) -> str:
        try:
            root = os.getcwd()
            base = os.path.basename(root).strip().lower().replace(" ", "_")
            # Prefer README title if present
            rd = os.path.join(root, "README.md")
            if os.path.isfile(rd):
                try:
                    with open(rd, "r", encoding="utf-8", errors="ignore") as f:
                        head = "\n".join([next(f, "") for _ in range(5)])
                    for line in head.splitlines():
                        s = line.strip("# ")
                        if 2 <= len(s) <= 64:
                            return s.strip().lower().replace(" ", "_")
                except Exception:
                    pass
            return base or "api"
        except Exception:
            return "api"

    def _infer_resources(self) -> list[dict[str, Any]]:
        root = os.getcwd()
        candidates = set()
        try:
            for d in os.listdir(root):
                dn = d.strip().lower()
                if dn in {"api", ".git", ".jinx", "jinx", "venv", ".venv", "node_modules", "__pycache__"}:
                    continue
                if os.path.isdir(os.path.join(root, d)) and 3 <= len(dn) <= 24:
                    candidates.add(dn)
        except Exception:
            pass
        # Map to resource dicts with minimal fields
        res = []
        for c in list(candidates)[:4]:
            res.append({
                "name": c,
                "fields": {"id": "int", f"{c}_name": "str"},
                "endpoints": ["list", "get", "create", "update", "delete"],
            })
        return res

    async def _try_llm_spec(self, name: str, resources: list[dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            from jinx.micro.llm.service import spark_openai as _spark
        except Exception:
            return None
        prompt = (
            "You are an autonomous system planner. Produce ONLY a JSON object for a REST API spec.\n"
            "Shape: {\"name\": str, \"resources\": [{\"name\": str, \"fields\": {k: type}, \"endpoints\": [list,get,create,update,delete]}]}\n"
            "Use ascii only, no code fences. Max 4 resources, max 6 fields each.\n\n"
            f"Project name: {name}\n"
            f"Candidate resources: {json.dumps([r['name'] for r in resources])}\n"
        )
        try:
            out, _ = await asyncio.wait_for(_spark(prompt), timeout=1.2)
            if not out or not isinstance(out, str):
                return None
            import re as _re
            m = _re.search(r"\{[\s\S]*\}", out)
            s = m.group(0) if m else out.strip()
            obj = json.loads(s)
            if isinstance(obj, dict) and obj.get("resources"):
                return obj
        except Exception:
            return None
        return None


async def spawn_mission_planner() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(MissionPlannerProgram())
