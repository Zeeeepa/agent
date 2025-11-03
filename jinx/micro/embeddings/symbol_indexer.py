from __future__ import annotations

import os
import asyncio
import contextlib
from typing import Dict, List

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.embeddings.symbol_index import update_symbol_index
from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
import jinx.state as jx_state

_EXCLUDE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", "emb", "build", "dist"}


def _is_excluded(path: str) -> bool:
    parts = (path or "").replace("\\", "/").split("/")
    return any(p in _EXCLUDE_DIRS for p in parts)


async def _list_py_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            ap = os.path.join(dirpath, fn)
            rel = os.path.relpath(ap, root).replace("\\", "/")
            if _is_excluded(rel):
                continue
            out.append(rel)
    return out


class SymbolIndexProgram(MicroProgram):
    """Background program that maintains a project-wide Python symbol index.

    - Builds an initial index on start (batched, cooperative).
    - Periodically scans for modified files via mtime and refreshes batches.
    - Env toggles:
      JINX_SYMBOL_INDEX_ENABLE (default 1)
      JINX_SYMBOL_INDEX_INTERVAL_SEC (default 45)
      JINX_SYMBOL_INDEX_BATCH (default 60)
    """

    def __init__(self) -> None:
        super().__init__(name="SymbolIndexProgram")
        self._mtimes: Dict[str, float] = {}

    async def run(self) -> None:
        root = _resolve_root()
        try:
            interval = int(os.getenv("JINX_SYMBOL_INDEX_INTERVAL_SEC", "45"))
        except Exception:
            interval = 45
        try:
            batch_n = int(os.getenv("JINX_SYMBOL_INDEX_BATCH", "60"))
        except Exception:
            batch_n = 60
        # Initial build
        files = await _list_py_files(root)
        await self.log(f"symbol-index: initial {len(files)} files")
        pos = 0
        while pos < len(files):
            chunk = files[pos: pos + batch_n]
            pos += batch_n
            await update_symbol_index(chunk)
            # update mtimes
            for rel in chunk:
                ap = os.path.join(root, rel)
                try:
                    self._mtimes[rel] = os.path.getmtime(ap)
                except Exception:
                    self._mtimes[rel] = 0.0
            # Cooperative yield and fast-exit on shutdown
            await asyncio.sleep(0)
            if jx_state.shutdown_event.is_set():
                return
        # Periodic refresh
        while not jx_state.shutdown_event.is_set():
            # Sleep with cancel-on-shutdown
            sleep_task = asyncio.create_task(asyncio.sleep(interval))
            stop_task = asyncio.create_task(jx_state.shutdown_event.wait())
            done, pending = await asyncio.wait({sleep_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            if stop_task in done:
                break
            try:
                cur_files = await _list_py_files(root)
            except Exception:
                cur_files = []
            dirty: List[str] = []
            for rel in cur_files:
                ap = os.path.join(root, rel)
                try:
                    mt = os.path.getmtime(ap)
                except Exception:
                    mt = 0.0
                if rel not in self._mtimes or mt > (self._mtimes.get(rel) or 0.0):
                    dirty.append(rel)
                    self._mtimes[rel] = mt
            if not dirty:
                continue
            # batch refresh dirty
            pos = 0
            while pos < len(dirty):
                chunk = dirty[pos: pos + batch_n]
                pos += batch_n
                await update_symbol_index(chunk)
                await asyncio.sleep(0)
                if jx_state.shutdown_event.is_set():
                    return


def start_symbol_indexer_task() -> asyncio.Task:
    async def _run() -> None:
        try:
            from jinx.micro.runtime.api import spawn as _spawn
            await _spawn(SymbolIndexProgram())
        except Exception:
            pass
    return asyncio.create_task(_run())
