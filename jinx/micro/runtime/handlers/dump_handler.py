from __future__ import annotations

import os
from typing import Callable, Awaitable, List, Dict, Optional

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_write as _patch_write,
    should_autocommit as _should_autocommit,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize
from jinx.micro.runtime.source_extract import (
    extract_symbol_source,
    find_enclosing_symbol,
)
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT
from jinx.micro.embeddings.project_search_api import search_project

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


def _truthy(name: str, default: str = "1") -> bool:
    try:
        return str(os.getenv(name, default)).strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def _abs_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    base = PROJECT_ROOT or os.getcwd()
    return os.path.normpath(os.path.join(base, p))


async def _write_with_gate(tid: str, out_path: str, code: str, *, verify_cb: VerifyCB, exports: Dict[str, str], strategy: str) -> None:
    await report_progress(tid, 35.0, f"preview write {out_path}")
    ok_prev, diff = await _patch_write(out_path, code, preview=True)
    if not ok_prev:
        await report_result(tid, False, error=diff)
        return
    exports["last_patch_preview"] = diff or ""
    okc, reason = _should_autocommit(strategy, diff)
    # Dump operations can be forced via env to maximize task success
    force_dump = False
    try:
        if strategy.startswith("dump"):
            force_dump = _truthy("JINX_DUMP_FORCE", "1")
    except Exception:
        force_dump = False
    if not okc and not force_dump:
        exports["last_patch_reason"] = f"needs_confirmation: {reason}"
        exports["last_patch_strategy"] = strategy
        await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": out_path, "diff": diff})
        return
    await report_progress(tid, 65.0, f"commit write {out_path}")
    ok_commit, diff2 = await _patch_write(out_path, code, preview=False)
    if ok_commit:
        warn = await maybe_warn_filesize(out_path)
        exports["last_patch_commit"] = diff2 or ""
        exports["last_patch_strategy"] = strategy
        if warn:
            exports["last_watchdog_warn"] = warn
        await report_result(
            tid,
            True,
            {"path": out_path, "bytes": len(code), "diff": diff2, **({"watchdog": warn} if warn else {})},
        )
        await verify_cb(goal=None, files=[out_path], diff=diff2)
    else:
        await report_result(tid, False, error=diff2)


async def handle_dump_symbol(
    tid: str,
    src_path: str,
    symbol: str,
    out_path: str,
    *,
    include_decorators: Optional[bool],
    include_docstring: Optional[bool],
    verify_cb: VerifyCB,
    exports: Dict[str, str],
) -> None:
    try:
        await report_progress(tid, 10.0, "extracting symbol source")
        inc_deco = include_decorators if include_decorators is not None else _truthy("JINX_DUMP_INCLUDE_DECORATORS", "1")
        inc_doc = include_docstring if include_docstring is not None else _truthy("JINX_DUMP_INCLUDE_DOCSTRING", "1")
        ok, code, meta = await extract_symbol_source(src_path, symbol, include_decorators=inc_deco, include_docstring=inc_doc)
        if not ok:
            await report_result(tid, False, error=str(meta.get("error") or "extract failed"))
            return
        apath = _abs_path(out_path)
        await _write_with_gate(tid, apath, code, verify_cb=verify_cb, exports=exports, strategy="dump_symbol")
    except Exception as e:
        await report_result(tid, False, error=f"dump_symbol failed: {e}")


async def handle_dump_by_query(
    tid: str,
    src_path: str,
    query: str,
    out_path: str,
    *,
    include_decorators: Optional[bool],
    include_docstring: Optional[bool],
    verify_cb: VerifyCB,
    exports: Dict[str, str],
) -> None:
    try:
        await report_progress(tid, 12.0, "locating symbol by query")
        ok_loc, data = await find_enclosing_symbol(src_path, query)
        if not ok_loc:
            await report_result(tid, False, error=str(data.get("error") or "symbol not found by query"))
            return
        symbol = str(data.get("symbol") or "")
        if not symbol:
            await report_result(tid, False, error="empty symbol")
            return
        await handle_dump_symbol(
            tid,
            src_path,
            symbol,
            out_path,
            include_decorators=include_decorators,
            include_docstring=include_docstring,
            verify_cb=verify_cb,
            exports=exports,
        )
    except Exception as e:
        await report_result(tid, False, error=f"dump_by_query failed: {e}")


async def handle_dump_by_query_global(
    tid: str,
    query: str,
    out_path: str,
    *,
    topk: Optional[int],
    include_decorators: Optional[bool],
    include_docstring: Optional[bool],
    verify_cb: VerifyCB,
    exports: Dict[str, str],
) -> None:
    try:
        await report_progress(tid, 10.0, "searching project for query")
        k = int(topk) if (topk is not None) else 3
        hits = await search_project(query, k=max(1, k), max_time_ms=300)
        if not hits:
            await report_result(tid, False, error="no hits for query")
            return
        # iterate hits to find a symbol to dump
        for h in hits:
            rel = str(h.get("file") or "")
            if not rel:
                continue
            src_path = _abs_path(rel)
            ok_loc, data = await find_enclosing_symbol(src_path, query)
            if not ok_loc:
                continue
            symbol = str(data.get("symbol") or "")
            if not symbol:
                continue
            await handle_dump_symbol(
                tid,
                src_path,
                symbol,
                out_path,
                include_decorators=include_decorators,
                include_docstring=include_docstring,
                verify_cb=verify_cb,
                exports=exports,
            )
            return
        await report_result(tid, False, error="no symbol found in top hits")
    except Exception as e:
        await report_result(tid, False, error=f"dump_by_query_global failed: {e}")
