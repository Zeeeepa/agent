from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST
from jinx.micro.common.log import log_info, log_warn
from jinx.micro.rt.activity import (
    set_activity as _set_act,
    set_activity_detail as _set_det,
    clear_activity as _clear_act,
    clear_activity_detail as _clear_det,
)


class APIArchitectProgram(MicroProgram):
    """Autonomous API architect.

    Listens for TASK_REQUEST "architect.api" and generates a production-grade API
    skeleton under ./api/ with routers, schemas, services, repositories.

    Goals:
    - Async-first, micro-modular, RT-conscious
    - Uses FastAPI if available; falls back to a minimal ASGI stub otherwise
    - Non-destructive: creates new files only; does not overwrite existing ones
    - Emits compact outcome and triggers verification
    """

    def __init__(self) -> None:
        super().__init__(name="APIArchitect")

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on  # lazy import
        await _on(TASK_REQUEST, self._on_task)
        await self.log("api_architect online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            name = str(payload.get("name") or "")
            tid = str(payload.get("id") or "")
            if name != "architect.api" or not tid:
                return
            kw = payload.get("kwargs") or {}
            spec = kw.get("spec")
            framework = (kw.get("framework") or "fastapi").strip().lower()
            try:
                budget_ms = max(300, int(kw.get("budget_ms") or int(os.getenv("JINX_API_ARCH_BUDGET_MS", "1600"))))
            except Exception:
                budget_ms = 1600
            _set_act("architecting API")
            _set_det({"api": {"phase": "parse", "framework": framework}})
            plan = self._parse_spec(spec)
            _set_det({"api": {"phase": "generate", "resources": len(plan.get('resources') or [])}})
            ok, files, msg = await self._generate(plan, framework=framework, budget_ms=budget_ms)
            _set_det({"api": {"phase": "verify", "files": len(files)}})
            try:
                await self._verify("api_architecture", files, diff="")
            except Exception:
                pass
            if ok:
                log_info("api.arch.generated", files=len(files), framework=framework)
                await self.log(f"generated {len(files)} files: {', '.join(files[:6])}{'...' if len(files)>6 else ''}")
            else:
                log_warn("api.arch.failed", reason=msg)
                await self.log(f"generation failed: {msg}", level="warn")
        except Exception:
            pass
        finally:
            try:
                _clear_det(); _clear_act()
            except Exception:
                pass

    def _parse_spec(self, spec: Any) -> Dict[str, Any]:
        # Accept dict or JSON string; fallback to a default sample
        if isinstance(spec, dict):
            return self._normalize_spec(spec)
        if isinstance(spec, str) and spec.strip():
            try:
                obj = json.loads(spec)
                if isinstance(obj, dict):
                    return self._normalize_spec(obj)
            except Exception:
                pass
        # Default example
        return {
            "name": "example_api",
            "resources": [
                {
                    "name": "users",
                    "fields": {"id": "int", "email": "str", "name": "str"},
                    "endpoints": ["list", "get", "create", "update", "delete"]
                },
                {
                    "name": "projects",
                    "fields": {"id": "int", "title": "str", "owner_id": "int"},
                    "endpoints": ["list", "get", "create", "update", "delete"]
                }
            ]
        }

    def _normalize_spec(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        name = str(obj.get("name") or "api").strip().lower().replace(" ", "_")
        res = []
        for r in obj.get("resources", []) or []:
            try:
                rn = str(r.get("name") or "res").strip().lower().replace(" ", "_")
                fields = dict(r.get("fields") or {})
                eps = list(r.get("endpoints") or ["list", "get", "create", "update", "delete"])
                res.append({"name": rn, "fields": fields, "endpoints": eps})
            except Exception:
                continue
        return {"name": name or "api", "resources": res}

    async def _generate(self, plan: Dict[str, Any], *, framework: str, budget_ms: int) -> tuple[bool, List[str], str]:
        start = asyncio.get_running_loop().time()
        root = os.getcwd()
        out_dir = os.path.join(root, "api")
        files: List[str] = []

        async def _ensure_dir(p: str) -> None:
            await asyncio.to_thread(os.makedirs, p, True)

        async def _write_if_absent(path: str, content: str) -> bool:
            def _write() -> bool:
                if os.path.exists(path):
                    return False
                d = os.path.dirname(path)
                if d and not os.path.isdir(d):
                    os.makedirs(d, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return True
            created = await asyncio.to_thread(_write)
            if created:
                files.append(os.path.relpath(path, root).replace("\\", "/"))
            return created

        # Base layout
        await _ensure_dir(out_dir)
        await _ensure_dir(os.path.join(out_dir, "routers"))
        await _ensure_dir(os.path.join(out_dir, "schemas"))
        await _ensure_dir(os.path.join(out_dir, "services"))
        await _ensure_dir(os.path.join(out_dir, "repositories"))

        # app
        if framework == "fastapi":
            app_code = self._tpl_app_fastapi(plan)
        else:
            app_code = self._tpl_app_asgi(plan)
        await _write_if_absent(os.path.join(out_dir, "app.py"), app_code)
        await _write_if_absent(os.path.join(out_dir, "__init__.py"), "__all__ = ['app']\n")

        # resources
        for r in plan.get("resources", []):
            # Budget check
            if (asyncio.get_running_loop().time() - start) * 1000 > budget_ms:
                return False, files, "time_budget_exceeded"
            rn = r["name"]
            fields = r.get("fields") or {}
            # schemas
            sch_path = os.path.join(out_dir, "schemas", f"{rn}.py")
            await _write_if_absent(sch_path, self._tpl_schema(rn, fields))
            # services
            srv_path = os.path.join(out_dir, "services", f"{rn}_service.py")
            await _write_if_absent(srv_path, self._tpl_service(rn))
            # repositories (in-memory stub)
            repo_path = os.path.join(out_dir, "repositories", f"{rn}_repo.py")
            await _write_if_absent(repo_path, self._tpl_repo(rn))
            # routers
            router_path = os.path.join(out_dir, "routers", f"{rn}.py")
            await _write_if_absent(router_path, self._tpl_router(framework, rn))

        return True, files, "ok"

    def _tpl_app_fastapi(self, plan: Dict[str, Any]) -> str:
        imports = [f"from .routers.{r['name']} import router as {r['name']}_router" for r in plan.get("resources", [])]
        includes = [f"app.include_router({r['name']}_router, prefix='/{r['name']}', tags=['{r['name']}'])" for r in plan.get("resources", [])]
        return (
            "from __future__ import annotations\n\n"
            "from fastapi import FastAPI\n\n"
            f"app = FastAPI(title='{plan.get('name','api')}', version='1.0.0')\n\n"
            + ("\n".join(imports) + ("\n\n" if imports else ""))
            + ("\n".join(includes) + ("\n" if includes else ""))
        )

    def _tpl_app_asgi(self, plan: Dict[str, Any]) -> str:
        # Minimal ASGI stub if FastAPI unavailable/undesired
        return (
            "from __future__ import annotations\n\n"
            "import json\n\n"
            "async def app(scope, receive, send):\n"
            "    assert scope['type'] == 'http'\n"
            "    body = json.dumps({'name': '%s', 'status': 'ok'}).encode('utf-8')\n" % plan.get('name','api') +
            "    await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'application/json')]})\n"
            "    await send({'type': 'http.response.body', 'body': body})\n"
        )

    def _tpl_schema(self, name: str, fields: Dict[str, str]) -> str:
        # FastAPI/Pydantic schema; if pydantic missing, remains import but user can install
        anns = []
        for k, v in fields.items():
            t = {
                'int': 'int', 'str': 'str', 'float': 'float', 'bool': 'bool'
            }.get(str(v).lower(), 'str')
            default = ' | None = None' if k != 'id' else ''
            anns.append(f"    {k}: {t}{default}")
        body = "\n".join(anns) or "    id: int | None = None"
        return (
            "from __future__ import annotations\n\n"
            "from pydantic import BaseModel\n\n"
            f"class {name.capitalize()}(BaseModel):\n"
            f"{body}\n"
        )

    def _tpl_service(self, name: str) -> str:
        return (
            "from __future__ import annotations\n\n"
            f"from ..repositories.{name}_repo import {name.capitalize()}Repository\n"
            f"from ..schemas.{name} import {name.capitalize()}\n\n"
            f"class {name.capitalize()}Service:\n"
            f"    def __init__(self):\n"
            f"        self.repo = {name.capitalize()}Repository()\n\n"
            f"    async def list(self) -> list[{name.capitalize()}]:\n"
            f"        return await self.repo.list()\n\n"
            f"    async def get(self, id: int) -> {name.capitalize()} | None:\n"
            f"        return await self.repo.get(id)\n\n"
            f"    async def create(self, obj: {name.capitalize()}) -> {name.capitalize()}:\n"
            f"        return await self.repo.create(obj)\n\n"
            f"    async def update(self, id: int, obj: {name.capitalize()}) -> {name.capitalize()} | None:\n"
            f"        return await self.repo.update(id, obj)\n\n"
            f"    async def delete(self, id: int) -> bool:\n"
            f"        return await self.repo.delete(id)\n"
        )

    def _tpl_repo(self, name: str) -> str:
        return (
            "from __future__ import annotations\n\n"
            "import asyncio\n"
            f"from ..schemas.{name} import {name.capitalize()}\n\n"
            f"class {name.capitalize()}Repository:\n"
            f"    def __init__(self):\n"
            f"        self._data: dict[int, {name.capitalize()}] = {{}}\n"
            f"        self._next_id = 1\n"
            f"        self._lock = asyncio.Lock()\n\n"
            f"    async def list(self) -> list[{name.capitalize()}]:\n"
            f"        async with self._lock:\n"
            f"            return list(self._data.values())\n\n"
            f"    async def get(self, id: int) -> {name.capitalize()} | None:\n"
            f"        async with self._lock:\n"
            f"            return self._data.get(id)\n\n"
            f"    async def create(self, obj: {name.capitalize()}) -> {name.capitalize()}:\n"
            f"        async with self._lock:\n"
            f"            oid = self._next_id; self._next_id += 1\n"
            f"            obj.id = oid\n"
            f"            self._data[oid] = obj\n"
            f"            return obj\n\n"
            f"    async def update(self, id: int, obj: {name.capitalize()}) -> {name.capitalize()} | None:\n"
            f"        async with self._lock:\n"
            f"            if id not in self._data:\n"
            f"                return None\n"
            f"            obj.id = id\n"
            f"            self._data[id] = obj\n"
            f"            return obj\n\n"
            f"    async def delete(self, id: int) -> bool:\n"
            f"        async with self._lock:\n"
            f"            return self._data.pop(id, None) is not None\n"
        )

    def _tpl_router(self, framework: str, name: str) -> str:
        if framework == "fastapi":
            cls = name.capitalize()
            return (
                "from __future__ import annotations\n\n"
                "from fastapi import APIRouter, HTTPException\n"
                f"from ..schemas.{name} import {cls}\n"
                f"from ..services.{name}_service import {cls}Service\n\n"
                f"router = APIRouter()\n"
                f"svc = {cls}Service()\n\n"
                f"@router.get('/')\n"
                f"async def list_{name}() -> list[{cls}]:\n"
                f"    return await svc.list()\n\n"
                f"@router.get('/{{id}}')\n"
                f"async def get_{name}(id: int) -> {cls}:\n"
                f"    obj = await svc.get(id)\n"
                f"    if not obj:\n"
                f"        raise HTTPException(status_code=404, detail='{cls} not found')\n"
                f"    return obj\n\n"
                f"@router.post('/')\n"
                f"async def create_{name}(payload: {cls}) -> {cls}:\n"
                f"    return await svc.create(payload)\n\n"
                f"@router.put('/{{id}}')\n"
                f"async def update_{name}(id: int, payload: {cls}) -> {cls}:\n"
                f"    obj = await svc.update(id, payload)\n"
                f"    if not obj:\n"
                f"        raise HTTPException(status_code=404, detail='{cls} not found')\n"
                f"    return obj\n\n"
                f"@router.delete('/{{id}}')\n"
                f"async def delete_{name}(id: int) -> dict:\n"
                f"    ok = await svc.delete(id)\n"
                f"    if not ok:\n"
                f"        raise HTTPException(status_code=404, detail='{cls} not found')\n"
                f"    return {{'ok': True}}\n"
            )
        # ASGI fallback: provide a stub module
        return (
            "from __future__ import annotations\n\n"
            "# Router not available without FastAPI; install fastapi/uvicorn to enable.\n"
        )

    async def _verify(self, goal: str, files: List[str], diff: str) -> None:
        try:
            from jinx.micro.runtime.verify_integration import maybe_verify
        except Exception:
            return
        await maybe_verify(goal, files, diff)


async def spawn_api_architect() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(APIArchitectProgram())
