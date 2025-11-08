from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST


def _slugify(text: str, max_len: int = 32) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", (text or "").strip())
    s = s.strip("_") or "skill"
    return s[:max_len]


def _allowlist_path(path: str) -> str:
    # Force under jinx/skills
    base = os.path.join(os.getcwd(), "jinx", "skills")
    os.makedirs(base, exist_ok=True)
    name = os.path.basename(path or "auto_skill.py")
    if not name.endswith(".py"):
        name += ".py"
    return os.path.join(base, name)


class SkillAcquirerProgram(MicroProgram):
    """Autonomous skill acquisition program.

    Handles TASK_REQUEST "skill.acquire" with kwargs:
        { query: str }

    Strategy:
      - Ask LLM for JSON { path, code }
      - Sanitize path to jinx/skills allowlist
      - Write code, compile to validate, import to smoke-test
      - Publish event and report result
    """

    def __init__(self) -> None:
        super().__init__(name="SkillAcquirerProgram")
        try:
            self._budget_ms = int(os.getenv("JINX_SKILL_BUDGET_MS", "1600"))
        except Exception:
            self._budget_ms = 1600

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        await self.log("skill-acquirer online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        name = str(payload.get("name") or "").strip()
        tid = str(payload.get("id") or "").strip()
        if name != "skill.acquire" or not tid:
            return
        kw = payload.get("kwargs") or {}
        q = str(kw.get("query") or "").strip()
        await self._acquire_skill(tid, q)

    async def _acquire_skill(self, tid: str, query: str) -> None:
        from jinx.micro.runtime.api import report_progress as _progress, report_result as _result
        await _progress(tid, 2.0, "skill acquire start")
        target_name = f"auto_{_slugify(query)}.py"
        allow_path = _allowlist_path(target_name)
        spec = await self._llm_spec(query, target_name)
        path = _allowlist_path(spec.get("path") or target_name)
        code = str(spec.get("code") or "").strip()
        if not code:
            # Fallback minimal stub
            code = (
                "from __future__ import annotations\n\n"
                "async def handle(query: str) -> str:\n"
                "    return f'Not yet implemented for: {query}'\n"
            )
        # Write file
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            await _result(tid, False, error=f"write_error: {e}")
            return
        # Compile validation
        try:
            import ast
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            ast.parse(src)
        except Exception as e:
            await _result(tid, False, error=f"compile_error: {e}")
            return
        # Import smoke
        try:
            import importlib.util, importlib
            modname = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(path))[0])
            spec_i = importlib.util.spec_from_file_location(modname, path)
            if spec_i and spec_i.loader:
                mod = importlib.util.module_from_spec(spec_i)
                spec_i.loader.exec_module(mod)  # type: ignore[attr-defined]
        except Exception as e:
            await _result(tid, False, error=f"import_error: {e}")
            return
        # Publish
        try:
            from jinx.micro.runtime.plugins import publish_event as _pub
            _pub("auto.skill_acquired", {"path": path, "query": query})
        except Exception:
            pass
        await _result(tid, True, result={"path": path})

    async def _llm_spec(self, query: str, target_name: str) -> Dict[str, Any]:
        """Ask LLM for JSON {path, code}. Returns dict."""
        prompt = (
            "You are a senior Jinx systems engineer. Create a minimal Python skill module to help answer the user query.\n"
            "Return STRICT JSON ONLY with keys: path (string), code (string). No code fences. ASCII only.\n"
            "Constraints:\n"
            "- Micro-modular, async-friendly, no blocking IO in top-level.\n"
            "- Prefer a function 'async def handle(query: str) -> str' that returns a textual result.\n"
            "- Keep external deps to standard library only.\n"
            "- Keep code small and safe.\n"
            f"User query: {query}\n"
            f"Suggested path (under jinx/skills): jinx/skills/{target_name}\n"
        )
        try:
            from jinx.micro.llm.service import spark_openai as _spark
        except Exception:
            return {"path": f"jinx/skills/{target_name}", "code": ""}
        try:
            out, _ = await asyncio.wait_for(_spark(prompt), timeout=max(0.5, self._budget_ms) / 1000.0)
        except Exception:
            return {"path": f"jinx/skills/{target_name}", "code": ""}
        if not out:
            return {"path": f"jinx/skills/{target_name}", "code": ""}
        # Extract JSON
        try:
            m = re.search(r"\{[\s\S]*\}", out)
            s = m.group(0) if m else out.strip()
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {"path": f"jinx/skills/{target_name}", "code": ""}
        return {"path": f"jinx/skills/{target_name}", "code": ""}


async def spawn_skill_acquirer() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(SkillAcquirerProgram())
