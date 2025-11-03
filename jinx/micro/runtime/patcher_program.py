from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, List, Dict
from .program import MicroProgram
from .api import on
from .contracts import TASK_REQUEST
from jinx.micro.runtime.patch import AutoPatchArgs
from jinx.micro.runtime.verify_integration import maybe_verify as _maybe_verify
from jinx.micro.runtime.patcher_handlers import (
    handle_write as _h_write,
    handle_line_patch as _h_line,
    handle_symbol_patch as _h_symbol,
    handle_anchor_patch as _h_anchor,
    handle_regex_patch as _h_regex,
    handle_auto_patch as _h_auto,
    handle_batch_patch as _h_batch,
    handle_dump_symbol as _h_dump_symbol,
    handle_dump_by_query as _h_dump_query,
    handle_dump_by_query_global as _h_dump_query_global,
    handle_refactor_move_symbol as _h_refactor_move,
    handle_refactor_split_file as _h_refactor_split,
)
from jinx.micro.runtime.handlers.rename_handler import handle_refactor_rename as _h_refactor_rename

# removed stray placeholder

class AutoPatchProgram(MicroProgram):
    """Background patcher that applies file writes and line-range patches.

    Supported task names (via TASK_REQUEST bus):
    - "patch.write": {id, name, args: [path, text], kwargs: {}}
    - "patch.line": {id, name, args: [path, line_start, line_end, replacement], kwargs: {}}
    - "patch.symbol": {id, name, args: [path, symbol, replacement], kwargs: {}}
    - "patch.anchor": {id, name, args: [path, anchor, replacement], kwargs: {}}
    - "patch.auto": {id, name, args: [], kwargs: {path?, code?, line_start?, line_end?, symbol?, anchor?}}
    - "patch.batch": {id, name, args: [], kwargs: {ops: List[dict], force?: bool}}

    Dumping (source extraction to a file):
    - "dump.symbol": {id, name, args: [], kwargs: {src_path: str, symbol: str, out_path: str, include_decorators?: bool, include_docstring?: bool}}
    - "dump.query": {id, name, args: [], kwargs: {src_path: str, query: str, out_path: str, include_decorators?: bool, include_docstring?: bool}}
    - "dump.query_global": {id, name, args: [], kwargs: {query: str, out_path: str, topk?: int, include_decorators?: bool, include_docstring?: bool}}

    Refactor (module reorg without breaking code):
    - "refactor.move": {id, name, args: [], kwargs: {src_path: str, symbol: str, dst_path: str, create_init?: bool, insert_shim?: bool, force?: bool}}
    - "refactor.split": {id, name, args: [], kwargs: {src_path: str, out_dir: str, create_init?: bool, insert_shim?: bool, force?: bool}}
    
    Semantics:
    - Best-effort, async, non-blocking. All I/O uses aiofiles.
    - Adds progress + final result to the bus for observability.
    """

    def __init__(self) -> None:
        super().__init__(name="AutoPatchProgram")
        # Lightweight export surface for macros: {{export:last_patch_*}}
        self.exports: Dict[str, str] = {}

    def get_export(self, key: str) -> str:
        try:
            k = str(key or "").strip().lower()
            val = self.exports.get(k)
            if val is None:
                return ""
            # Guard overly large expansions in prompts
            try:
                cap = max(512, int(os.getenv("JINX_PATCH_EXPORT_MAXCHARS", "6000")))
            except Exception:
                cap = 6000
            if len(val) > cap:
                return val[:cap] + "\n...<truncated>"
            return val
        except Exception:
            return ""

    async def _verify_cb(self, goal: Optional[str], files: List[str], diff: str) -> None:
        try:
            await _maybe_verify(goal, files, diff)
        except Exception:
            pass

    async def run(self) -> None:
        # Subscribe to task requests once; event bus dispatches asynchronously
        await on(TASK_REQUEST, self._on_task)
        await self.log("patcher online")
        # Idle loop to keep program alive until stopped
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: dict) -> None:
        try:
            name = str(payload.get("name") or "")
            tid = str(payload.get("id") or "")
            if not name or not tid:
                return
            if name == "patch.write":
                args = payload.get("args") or []
                if len(args) < 2:
                    return
                path = str(args[0])
                text = str(args[1])
                await self._handle_write(tid, path, text)
            elif name == "patch.line":
                args = payload.get("args") or []
                if len(args) < 4:
                    return
                path = str(args[0])
                ls = int(args[1])
                le = int(args[2])
                replacement = str(args[3])
                await self._handle_line_patch(tid, path, ls, le, replacement)
            elif name == "patch.symbol":
                args = payload.get("args") or []
                if len(args) < 3:
                    return
                path = str(args[0])
                symbol = str(args[1])
                replacement = str(args[2])
                await self._handle_symbol_patch(tid, path, symbol, replacement)
            elif name == "patch.anchor":
                args = payload.get("args") or []
                if len(args) < 3:
                    return
                path = str(args[0])
                anchor = str(args[1])
                replacement = str(args[2])
                await self._handle_anchor_patch(tid, path, anchor, replacement)
            elif name == "patch.regex":
                # Supports both positional args and optional flags kwarg
                args = payload.get("args") or []
                if len(args) < 3:
                    return
                path = str(args[0])
                pattern = str(args[1])
                replacement = str(args[2])
                kw = payload.get("kwargs") or {}
                flags = kw.get("flags")
                if flags is None and len(args) >= 4:
                    flags = str(args[3])
                await self._handle_regex_patch(tid, path, pattern, replacement, flags=(str(flags) if flags is not None else None))
            elif name == "patch.auto":
                kw = payload.get("kwargs") or {}
                await self._handle_auto_patch(
                    tid,
                    AutoPatchArgs(
                        path=str(kw.get("path") or "") or None,
                        code=str(kw.get("code") or "") if (kw.get("code") is not None) else None,
                        line_start=int(kw.get("line_start")) if kw.get("line_start") is not None else None,
                        line_end=int(kw.get("line_end")) if kw.get("line_end") is not None else None,
                        symbol=str(kw.get("symbol") or "") or None,
                        anchor=str(kw.get("anchor") or "") or None,
                        query=str(kw.get("query") or "") or None,
                        preview=bool(kw.get("preview") or False),
                        max_span=int(kw.get("max_span")) if kw.get("max_span") is not None else None,
                        force=bool(kw.get("force") or False),
                        context_before=str(kw.get("context_before") or "") or None,
                        context_tolerance=float(kw.get("context_tolerance")) if kw.get("context_tolerance") is not None else None,
                    ),
                )
            elif name == "patch.batch":
                kw = payload.get("kwargs") or {}
                ops = kw.get("ops") or []
                await self._handle_batch_patch(tid, ops, bool(kw.get("force") or False))
            elif name == "dump.symbol":
                kw = payload.get("kwargs") or {}
                src_path = str(kw.get("src_path") or "")
                symbol = str(kw.get("symbol") or "")
                out_path = str(kw.get("out_path") or "")
                include_decorators = kw.get("include_decorators")
                include_docstring = kw.get("include_docstring")
                await self._handle_dump_symbol(
                    tid,
                    src_path,
                    symbol,
                    out_path,
                    include_decorators=bool(include_decorators) if include_decorators is not None else None,
                    include_docstring=bool(include_docstring) if include_docstring is not None else None,
                )
            elif name == "dump.query":
                kw = payload.get("kwargs") or {}
                src_path = str(kw.get("src_path") or "")
                query = str(kw.get("query") or "")
                out_path = str(kw.get("out_path") or "")
                include_decorators = kw.get("include_decorators")
                include_docstring = kw.get("include_docstring")
                await self._handle_dump_by_query(
                    tid,
                    src_path,
                    query,
                    out_path,
                    include_decorators=bool(include_decorators) if include_decorators is not None else None,
                    include_docstring=bool(include_docstring) if include_docstring is not None else None,
                )
            elif name == "dump.query_global":
                kw = payload.get("kwargs") or {}
                query = str(kw.get("query") or "")
                out_path = str(kw.get("out_path") or "")
                topk = kw.get("topk")
                include_decorators = kw.get("include_decorators")
                include_docstring = kw.get("include_docstring")
                await self._handle_dump_by_query_global(
                    tid,
                    query,
                    out_path,
                    topk=int(topk) if topk is not None else None,
                    include_decorators=bool(include_decorators) if include_decorators is not None else None,
                    include_docstring=bool(include_docstring) if include_docstring is not None else None,
                )
            elif name == "refactor.move":
                kw = payload.get("kwargs") or {}
                src_path = str(kw.get("src_path") or "")
                symbol = str(kw.get("symbol") or "")
                dst_path = str(kw.get("dst_path") or "")
                create_init = kw.get("create_init")
                insert_shim = kw.get("insert_shim")
                force = kw.get("force")
                await self._handle_refactor_move(
                    tid,
                    src_path,
                    symbol,
                    dst_path,
                    create_init=bool(create_init) if create_init is not None else None,
                    insert_shim=bool(insert_shim) if insert_shim is not None else None,
                    force=bool(force) if force is not None else None,
                )
            elif name == "refactor.split":
                kw = payload.get("kwargs") or {}
                src_path = str(kw.get("src_path") or "")
                out_dir = str(kw.get("out_dir") or "")
                create_init = kw.get("create_init")
                insert_shim = kw.get("insert_shim")
                force = kw.get("force")
                await self._handle_refactor_split(
                    tid,
                    src_path,
                    out_dir,
                    create_init=bool(create_init) if create_init is not None else None,
                    insert_shim=bool(insert_shim) if insert_shim is not None else None,
                    force=bool(force) if force is not None else None,
                )
            elif name == "refactor.rename":
                kw = payload.get("kwargs") or {}
                symbol = str(kw.get("symbol") or "").strip()
                new_name = str(kw.get("new_name") or "").strip()
                if not symbol or not new_name:
                    await self.log("rename: missing symbol or new_name")
                    return
                try:
                    await _h_refactor_rename(tid, symbol, new_name, verify_cb=self._verify_cb, exports=self.exports)
                except Exception:
                    await self.log("rename handler failed")
        except Exception:
            # Do not let handler raise; bus must never fail
            pass

    async def _handle_write(self, tid: str, path: str, text: str) -> None:
        await _h_write(tid, path, text, verify_cb=self._verify_cb, exports=self.exports)
    async def _handle_line_patch(self, tid: str, path: str, ls: int, le: int, replacement: str) -> None:
        await _h_line(tid, path, ls, le, replacement, verify_cb=self._verify_cb, exports=self.exports)

    async def _handle_symbol_patch(self, tid: str, path: str, symbol: str, replacement: str) -> None:
        await _h_symbol(tid, path, symbol, replacement, verify_cb=self._verify_cb, exports=self.exports)

    async def _handle_anchor_patch(self, tid: str, path: str, anchor: str, replacement: str) -> None:
        await _h_anchor(tid, path, anchor, replacement, verify_cb=self._verify_cb, exports=self.exports)

    async def _handle_regex_patch(self, tid: str, path: str, pattern: str, replacement: str, flags: str | None) -> None:
        await _h_regex(tid, path, pattern, replacement, flags=flags, verify_cb=self._verify_cb, exports=self.exports)

    async def _handle_auto_patch(self, tid: str, a: AutoPatchArgs) -> None:
        await _h_auto(tid, a, verify_cb=self._verify_cb, exports=self.exports)

    async def _handle_batch_patch(self, tid: str, ops: List[Dict[str, Any]], force: bool) -> None:
        await _h_batch(tid, ops, force, verify_cb=self._verify_cb, exports=self.exports)

    async def _handle_dump_symbol(
        self,
        tid: str,
        src_path: str,
        symbol: str,
        out_path: str,
        *,
        include_decorators: bool | None,
        include_docstring: bool | None,
    ) -> None:
        await _h_dump_symbol(
            tid,
            src_path,
            symbol,
            out_path,
            include_decorators=include_decorators,
            include_docstring=include_docstring,
            verify_cb=self._verify_cb,
            exports=self.exports,
        )

    async def _handle_dump_by_query(
        self,
        tid: str,
        src_path: str,
        query: str,
        out_path: str,
        *,
        include_decorators: bool | None,
        include_docstring: bool | None,
    ) -> None:
        await _h_dump_query(
            tid,
            src_path,
            query,
            out_path,
            include_decorators=include_decorators,
            include_docstring=include_docstring,
            verify_cb=self._verify_cb,
            exports=self.exports,
        )

    async def _handle_dump_by_query_global(
        self,
        tid: str,
        query: str,
        out_path: str,
        *,
        topk: int | None,
        include_decorators: bool | None,
        include_docstring: bool | None,
    ) -> None:
        await _h_dump_query_global(
            tid,
            query,
            out_path,
            topk=topk,
            include_decorators=include_decorators,
            include_docstring=include_docstring,
            verify_cb=self._verify_cb,
            exports=self.exports,
        )

    async def _handle_refactor_move(
        self,
        tid: str,
        src_path: str,
        symbol: str,
        dst_path: str,
        *,
        create_init: bool | None,
        insert_shim: bool | None,
        force: bool | None,
    ) -> None:
        await _h_refactor_move(
            tid,
            src_path,
            symbol,
            dst_path,
            verify_cb=self._verify_cb,
            exports=self.exports,
            create_init=create_init,
            insert_shim=insert_shim,
            force=force,
        )

    async def _handle_refactor_split(
        self,
        tid: str,
        src_path: str,
        out_dir: str,
        *,
        create_init: bool | None,
        insert_shim: bool | None,
        force: bool | None,
    ) -> None:
        await _h_refactor_split(
            tid,
            src_path,
            out_dir,
            verify_cb=self._verify_cb,
            exports=self.exports,
            create_init=create_init,
            insert_shim=insert_shim,
            force=force,
        )
