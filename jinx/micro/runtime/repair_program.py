from __future__ import annotations

import asyncio
import os
import importlib
from typing import Any, Dict

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST
from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.micro.common.log import log_info, log_warn
from jinx.micro.rt.activity import (
    set_activity as _set_act,
    set_activity_detail as _set_det,
    clear_activity as _clear_act,
    clear_activity_detail as _clear_det,
)


class RepairProgram(MicroProgram):
    """Repairs missing/broken jinx.* modules by creating safe stub modules then (optionally) asking LLM to synthesize better code.

    Handles TASK_REQUEST "repair.import_missing" with kwargs: { module: str }
    """

    def __init__(self) -> None:
        super().__init__(name="RepairProgram")

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        await self.log("repair online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            name = str(payload.get("name") or "")
            tid = str(payload.get("id") or "")
            if name != "repair.import_missing" or not tid:
                return
            kw = payload.get("kwargs") or {}
            mod = str(kw.get("module") or "").strip()
            if not mod or not mod.startswith("jinx."):
                return
            try:
                _set_act(f"repair: {mod}")
                _set_det({"repair": {"module": mod, "phase": "start"}})
            except Exception:
                pass
            await self._repair_module(mod)
        except Exception:
            pass
        finally:
            # Clear spinner activity/detail to avoid stale status
            try:
                _clear_det()
                _clear_act()
            except Exception:
                pass

    async def _repair_module(self, mod: str) -> None:
        root = _resolve_root()
        parts = mod.split(".")
        rel_path = os.path.join(*parts)
        # Prefer module.py; if directory chosen, ensure __init__.py
        target_py = os.path.join(root, rel_path + ".py")
        target_pkg_init = os.path.join(root, rel_path, "__init__.py")
        # If module exists and imports fine, skip
        try:
            importlib.import_module(mod)
            return
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(target_py), exist_ok=True)
        except Exception:
            return
        # Create stub module if not present
        if not os.path.exists(target_py) and not os.path.isdir(os.path.dirname(target_py)):
            # parent presumably not a package yet
            pass
        # Decide whether to write module.py or __init__.py
        path = target_py
        try:
            # If the path is a directory or module import suggests a package path, use __init__.py
            if os.path.isdir(os.path.join(root, rel_path)):
                os.makedirs(os.path.join(root, rel_path), exist_ok=True)
                path = target_pkg_init
        except Exception:
            path = target_py
        try:
            code = self._stub_code(mod)
            # Write atomically
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(code)
            os.replace(tmp, path)
            log_info("repair.stub_written", module=mod, path=path)
            try:
                _set_det({"repair": {"module": mod, "phase": "stub", "path": path}})
            except Exception:
                pass
        except Exception:
            return
        # Try importing after write
        try:
            importlib.invalidate_caches()
            importlib.import_module(mod)
        except Exception:
            pass
        # Optional: attempt LLM synthesis to improve stub (best-effort)
        await self._maybe_llm_improve(mod, path)

    def _stub_code(self, mod: str) -> str:
        return (
            "from __future__ import annotations\n\n"
            "# Auto-generated resilient stub for missing module.\n"
            f"__module_name__ = {mod!r}\n"
            "__all__ = []\n"
            "class _Dummy:\n"
            "    def __getattr__(self, _): return self\n"
            "    def __call__(self, *a, **k): return None\n"
            "    def __iter__(self): return iter(())\n"
            "    def __await__(self):\n"
            "        async def _noop(): return None\n"
            "        return _noop().__await__()\n"
            "_dummy = _Dummy()\n"
            "def __getattr__(name: str):\n"
            "    return _dummy\n"
        )

    async def _maybe_llm_improve(self, mod: str, path: str) -> None:
        # Best-effort, gated by env (default ON)
        try:
            import os
            on = str(os.getenv("JINX_REPAIR_LLM", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            on = True
        if not on:
            return
        try:
            from jinx.micro.llm.service import spark_openai as _spark
        except Exception:
            return
        try:
            from jinx.prompts import get_prompt as _get_prompt
            _tmpl = _get_prompt("repair_stub")
            prompt = _tmpl.format(module=mod)
        except Exception:
            prompt = f"Generate a minimal, SAFE Python module implementation for '{mod}'."
        try:
            out, _ = await _spark(prompt)
            # naive extract code block: keep whole output as code
            code = out.strip()
            if code:
                tmp = path + ".llm.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(code)
                os.replace(tmp, path)
                log_warn("repair.llm_applied", module=mod, path=path)
                try:
                    _set_det({"repair": {"module": mod, "phase": "llm_applied", "path": path}})
                except Exception:
                    pass
        except Exception:
            pass


async def spawn_repair() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(RepairProgram())
