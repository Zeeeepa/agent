from __future__ import annotations

from typing import Any, List, Dict

from .api import submit_task


async def submit_write_file(path: str, text: str) -> str:
    """Submit a simple write task. Returns task id."""
    return await submit_task("patch.write", path, text)


async def submit_line_patch(path: str, line_start: int, line_end: int, replacement: str) -> str:
    """Submit a line-range patch task. Returns task id."""
    return await submit_task("patch.line", path, int(line_start), int(line_end), replacement)


async def submit_symbol_patch(path: str, symbol: str, replacement: str) -> str:
    """Submit a symbol-level patch task for Python files."""
    return await submit_task("patch.symbol", path, symbol, replacement)


async def submit_anchor_insert_after(path: str, anchor: str, replacement: str) -> str:
    """Submit an anchor-based insert-after patch task."""
    return await submit_task("patch.anchor", path, anchor, replacement)


async def submit_regex_patch(path: str, pattern: str, replacement: str, *, flags: str | None = None) -> str:
    """Submit a regex-based patch task (search/replace) with optional flags (i,m,s,x).

    Args:
        path: file to patch
        pattern: regex pattern (Python syntax)
        replacement: replacement string
        flags: optional regex flags string, e.g. "im" for IGNORECASE and MULTILINE
    """
    if flags is None or flags == "":
        return await submit_task("patch.regex", path, pattern, replacement)
    return await submit_task("patch.regex", path, pattern, replacement, flags=str(flags))


async def submit_autopatch(*, path: str | None = None, code: str | None = None, line_start: int | None = None, line_end: int | None = None, symbol: str | None = None, anchor: str | None = None) -> str:
    """Submit an intelligent autopatch task that selects the best strategy."""
    return await submit_task(
        "patch.auto",
        path=path,
        code=code,
        line_start=line_start,
        line_end=line_end,
        symbol=symbol,
        anchor=anchor,
    )


async def submit_autopatch_ex(*, path: str | None = None, code: str | None = None, line_start: int | None = None, line_end: int | None = None, symbol: str | None = None, anchor: str | None = None, query: str | None = None, preview: bool | None = None, max_span: int | None = None, force: bool | None = None, context_before: str | None = None, context_tolerance: float | None = None) -> str:
    """Advanced autopatch submit with query/preview/max_span/force controls."""
    return await submit_task(
        "patch.auto",
        path=path,
        code=code,
        line_start=line_start,
        line_end=line_end,
        symbol=symbol,
        anchor=anchor,
        query=query,
        preview=bool(preview) if (preview is not None) else False,
        max_span=int(max_span) if (max_span is not None) else None,
        force=bool(force) if (force is not None) else False,
        context_before=context_before,
        context_tolerance=context_tolerance,
    )


async def submit_batch_patch(ops: List[Dict[str, Any]], *, force: bool | None = None) -> str:
    """Submit a batch of patch operations with preview/commit gating."""
    if force is None:
        return await submit_task("patch.batch", ops=ops)
    return await submit_task("patch.batch", ops=ops, force=bool(force))


async def submit_dump_symbol(src_path: str, symbol: str, out_path: str, *, include_decorators: bool | None = None, include_docstring: bool | None = None) -> str:
    """Dump a function/class by name from src_path into out_path using the patch pipeline."""
    return await submit_task(
        "dump.symbol",
        src_path=src_path,
        symbol=symbol,
        out_path=out_path,
        include_decorators=include_decorators if include_decorators is not None else None,
        include_docstring=include_docstring if include_docstring is not None else None,
    )


async def submit_dump_by_query(src_path: str, query: str, out_path: str, *, include_decorators: bool | None = None, include_docstring: bool | None = None) -> str:
    """Dump the enclosing function/class around a query in src_path into out_path."""
    return await submit_task(
        "dump.query",
        src_path=src_path,
        query=query,
        out_path=out_path,
        include_decorators=include_decorators if include_decorators is not None else None,
        include_docstring=include_docstring if include_docstring is not None else None,
    )


async def submit_dump_by_query_global(query: str, out_path: str, *, topk: int | None = None, include_decorators: bool | None = None, include_docstring: bool | None = None) -> str:
    """Find a symbol by query via embeddings and dump it into out_path."""
    return await submit_task(
        "dump.query_global",
        query=query,
        out_path=out_path,
        topk=int(topk) if topk is not None else None,
        include_decorators=include_decorators if include_decorators is not None else None,
        include_docstring=include_docstring if include_docstring is not None else None,
    )


async def submit_refactor_move_symbol(src_path: str, symbol: str, dst_path: str, *, create_init: bool | None = None, insert_shim: bool | None = None, force: bool | None = None) -> str:
    """Move a top-level function/class to another module with safe shims and package exports."""
    return await submit_task(
        "refactor.move",
        src_path=src_path,
        symbol=symbol,
        dst_path=dst_path,
        create_init=create_init if create_init is not None else None,
        insert_shim=insert_shim if insert_shim is not None else None,
        force=force if force is not None else None,
    )


async def submit_refactor_split_file(src_path: str, out_dir: str, *, create_init: bool | None = None, insert_shim: bool | None = None, force: bool | None = None) -> str:
    """Split a Python file into per-symbol modules in out_dir and convert the source into a shim importing moved symbols."""
    return await submit_task(
        "refactor.split",
        src_path=src_path,
        out_dir=out_dir,
        create_init=create_init if create_init is not None else None,
        insert_shim=insert_shim if insert_shim is not None else None,
        force=force if force is not None else None,
    )
