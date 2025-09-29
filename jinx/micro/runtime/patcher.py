from __future__ import annotations

from typing import Any, List, Dict

from .api import spawn, list_programs
from .patcher_program import AutoPatchProgram as _AutoPatchProgramNew
from .patcher_submit import (
    submit_write_file as _submit_write_file,
    submit_line_patch as _submit_line_patch,
    submit_symbol_patch as _submit_symbol_patch,
    submit_anchor_insert_after as _submit_anchor_insert_after,
    submit_autopatch as _submit_autopatch,
    submit_autopatch_ex as _submit_autopatch_ex,
    submit_batch_patch as _submit_batch_patch,
    submit_dump_symbol as _submit_dump_symbol,
    submit_dump_by_query as _submit_dump_by_query,
    submit_dump_by_query_global as _submit_dump_by_query_global,
    submit_refactor_move_symbol as _submit_refactor_move_symbol,
    submit_refactor_split_file as _submit_refactor_split_file,
)

# Backward-compat: expose AutoPatchProgram from this facade
AutoPatchProgram = _AutoPatchProgramNew

_PATCHER_PID: str | None = None
_PATCHER_STARTED: bool = False


# Helper APIs for callers (model-generated code)
async def spawn_patcher() -> str:
    """Spawn the AutoPatchProgram and return its id."""
    global _PATCHER_PID, _PATCHER_STARTED
    # Spawn the micro-module implementation of the patcher program
    pid = await spawn(_AutoPatchProgramNew())
    _PATCHER_PID = pid
    _PATCHER_STARTED = True
    return pid


async def ensure_patcher_running() -> str | None:
    """Ensure there is exactly one AutoPatchProgram running in this session.

    Best-effort guard: caches the spawned pid and also checks live registry.
    Returns the pid if running, else None on failure.
    """
    global _PATCHER_PID, _PATCHER_STARTED
    try:
        if _PATCHER_STARTED and _PATCHER_PID:
            # quick path
            return _PATCHER_PID
        # Validate cache against live registry
        pids = await list_programs()
        if _PATCHER_PID and (_PATCHER_PID in (pids or [])):
            _PATCHER_STARTED = True
            return _PATCHER_PID
        # Spawn fresh instance
        pid = await spawn_patcher()
        return pid
    except Exception:
        return None


async def submit_write_file(path: str, text: str) -> str:
    """Submit a simple write task. Returns task id."""
    return await _submit_write_file(path, text)


async def submit_line_patch(path: str, line_start: int, line_end: int, replacement: str) -> str:
    """Submit a line-range patch task. Returns task id."""
    return await _submit_line_patch(path, int(line_start), int(line_end), replacement)


async def submit_symbol_patch(path: str, symbol: str, replacement: str) -> str:
    """Submit a symbol-level patch task for Python files."""
    return await _submit_symbol_patch(path, symbol, replacement)


async def submit_anchor_insert_after(path: str, anchor: str, replacement: str) -> str:
    """Submit an anchor-based insert-after patch task."""
    return await _submit_anchor_insert_after(path, anchor, replacement)


async def submit_autopatch(*, path: str | None = None, code: str | None = None, line_start: int | None = None, line_end: int | None = None, symbol: str | None = None, anchor: str | None = None) -> str:
    """Submit an intelligent autopatch task that selects the best strategy."""
    return await _submit_autopatch(path=path, code=code, line_start=line_start, line_end=line_end, symbol=symbol, anchor=anchor)


async def submit_autopatch_ex(*, path: str | None = None, code: str | None = None, line_start: int | None = None, line_end: int | None = None, symbol: str | None = None, anchor: str | None = None, query: str | None = None, preview: bool | None = None, max_span: int | None = None, force: bool | None = None, context_before: str | None = None, context_tolerance: float | None = None) -> str:
    """Advanced autopatch submit with query/preview/max_span/force controls."""
    return await _submit_autopatch_ex(
        path=path,
        code=code,
        line_start=line_start,
        line_end=line_end,
        symbol=symbol,
        anchor=anchor,
        query=query,
        preview=preview,
        max_span=max_span,
        force=force,
        context_before=context_before,
        context_tolerance=context_tolerance,
    )


async def submit_batch_patch(ops: List[Dict[str, Any]], *, force: bool | None = None) -> str:
    """Submit a batch of patch operations with preview/commit gating."""
    return await _submit_batch_patch(ops, force=force)


async def submit_dump_symbol(src_path: str, symbol: str, out_path: str, *, include_decorators: bool | None = None, include_docstring: bool | None = None) -> str:
    """Dump a function/class by name from src_path into out_path using the patch pipeline."""
    return await _submit_dump_symbol(src_path, symbol, out_path, include_decorators=include_decorators, include_docstring=include_docstring)


async def submit_dump_by_query(src_path: str, query: str, out_path: str, *, include_decorators: bool | None = None, include_docstring: bool | None = None) -> str:
    """Dump the enclosing function/class around a query in src_path into out_path."""
    return await _submit_dump_by_query(src_path, query, out_path, include_decorators=include_decorators, include_docstring=include_docstring)


async def submit_dump_by_query_global(query: str, out_path: str, *, topk: int | None = None, include_decorators: bool | None = None, include_docstring: bool | None = None) -> str:
    """Find a symbol by query via embeddings and dump it into out_path."""
    return await _submit_dump_by_query_global(query, out_path, topk=topk, include_decorators=include_decorators, include_docstring=include_docstring)


async def submit_refactor_move_symbol(src_path: str, symbol: str, dst_path: str, *, create_init: bool | None = None, insert_shim: bool | None = None, force: bool | None = None) -> str:
    """Move a top-level function/class to another module with safe shims and package exports."""
    return await _submit_refactor_move_symbol(src_path, symbol, dst_path, create_init=create_init, insert_shim=insert_shim, force=force)


async def submit_refactor_split_file(src_path: str, out_dir: str, *, create_init: bool | None = None, insert_shim: bool | None = None, force: bool | None = None) -> str:
    """Split a Python file into per-symbol modules in out_dir and convert the source into a shim importing moved symbols."""
    return await _submit_refactor_split_file(src_path, out_dir, create_init=create_init, insert_shim=insert_shim, force=force)
