from __future__ import annotations

import os
from typing import Any, Dict, List, Callable, Awaitable

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    AutoPatchArgs,
    autopatch as _autopatch,
    patch_symbol_python as _patch_symbol,
    patch_anchor_insert_after as _patch_anchor,
    patch_line_range as _patch_line,
    patch_write as _patch_write,
    should_autocommit as _should_autocommit,
    diff_stats as _diff_stats,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_batch_patch(tid: str, ops: List[Dict[str, Any]], force: bool, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        if not isinstance(ops, list) or not ops:
            await report_result(tid, False, error="ops required (list)")
            return
        await report_progress(tid, 10.0, f"batch preview {len(ops)} ops")
        # detect refactor intent from ops meta (if provided by upstream handlers)
        try:
            is_refactor = any(bool((op.get("meta") or {}).get("refactor")) for op in ops)
        except Exception:
            is_refactor = False
        previews: List[Dict[str, Any]] = []
        combined_diff_parts: List[str] = []
        for i, op in enumerate(ops):
            typ = str(op.get("type") or "auto").strip().lower()
            path = str(op.get("path") or "")
            code = str(op.get("code") or "")
            if typ == "write":
                ok, diff = await _patch_write(path, code, preview=True)
                previews.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path})
                combined_diff_parts.append(diff)
            elif typ == "line":
                ls = int(op.get("line_start") or 0); le = int(op.get("line_end") or 0)
                try:
                    max_span = int(os.getenv("JINX_PATCH_MAX_SPAN", "80"))
                except Exception:
                    max_span = 80
                ok, diff = await _patch_line(path, ls, le, code, preview=True, max_span=max_span)
                previews.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "ls": ls, "le": le})
                combined_diff_parts.append(diff)
            elif typ == "symbol":
                sym = str(op.get("symbol") or "")
                ok, diff = await _patch_symbol(path, sym, code, preview=True)
                previews.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "symbol": sym})
                combined_diff_parts.append(diff)
            elif typ == "anchor":
                anc = str(op.get("anchor") or "")
                ok, diff = await _patch_anchor(path, anc, code, preview=True)
                previews.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "anchor": anc})
                combined_diff_parts.append(diff)
            else:
                a = AutoPatchArgs(
                    path=path or None,
                    code=code or None,
                    line_start=int(op.get("line_start")) if op.get("line_start") is not None else None,
                    line_end=int(op.get("line_end")) if op.get("line_end") is not None else None,
                    symbol=str(op.get("symbol") or "") or None,
                    anchor=str(op.get("anchor") or "") or None,
                    query=str(op.get("query") or "") or None,
                    preview=True,
                    max_span=int(op.get("max_span")) if op.get("max_span") is not None else None,
                )
                ok, strat, diff = await _autopatch(a)
                previews.append({"i": i, "type": f"auto:{strat}", "ok": ok, "diff": diff, "path": path})
                combined_diff_parts.append(diff)
        combined_diff = "\n".join([d for d in combined_diff_parts if d])
        add, rem = _diff_stats(combined_diff)
        # export preview for macros/prompts
        exports["last_patch_preview"] = combined_diff or ""
        exports["last_patch_strategy"] = "batch:refactor" if is_refactor else "batch"
        okc, reason = _should_autocommit("batch", combined_diff)
        if not okc and not force:
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"previews": previews, "diff_add": add, "diff_rem": rem})
            return
        await report_progress(tid, 55.0, "batch commit")
        results: List[Dict[str, Any]] = []
        changed_files: List[str] = []
        for i, op in enumerate(ops):
            typ = str(op.get("type") or "auto").strip().lower()
            path = str(op.get("path") or "")
            code = str(op.get("code") or "")
            if typ == "write":
                ok, diff = await _patch_write(path, code, preview=False)
                results.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path})
                if ok and path:
                    changed_files.append(path)
            elif typ == "line":
                ls = int(op.get("line_start") or 0); le = int(op.get("line_end") or 0)
                try:
                    max_span = int(os.getenv("JINX_PATCH_MAX_SPAN", "80"))
                except Exception:
                    max_span = 80
                ok, diff = await _patch_line(path, ls, le, code, preview=False, max_span=max_span)
                results.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "ls": ls, "le": le})
                if ok and path:
                    changed_files.append(path)
            elif typ == "symbol":
                sym = str(op.get("symbol") or "")
                ok, diff = await _patch_symbol(path, sym, code, preview=False)
                results.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "symbol": sym})
                if ok and path:
                    changed_files.append(path)
            elif typ == "anchor":
                anc = str(op.get("anchor") or "")
                ok, diff = await _patch_anchor(path, anc, code, preview=False)
                results.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "anchor": anc})
                if ok and path:
                    changed_files.append(path)
            else:
                a = AutoPatchArgs(
                    path=path or None,
                    code=code or None,
                    line_start=int(op.get("line_start")) if op.get("line_start") is not None else None,
                    line_end=int(op.get("line_end")) if op.get("line_end") is not None else None,
                    symbol=str(op.get("symbol") or "") or None,
                    anchor=str(op.get("anchor") or "") or None,
                    query=str(op.get("query") or "") or None,
                    preview=False,
                    max_span=int(op.get("max_span")) if op.get("max_span") is not None else None,
                )
                ok, strat, diff = await _autopatch(a)
                results.append({"i": i, "type": f"auto:{strat}", "ok": ok, "diff": diff, "path": path})
        # Compute watchdog warnings for changed files
        warnings: List[str] = []
        for p in list({p for p in changed_files if p}):
            try:
                warn = await maybe_warn_filesize(p)
                if warn:
                    warnings.append(warn)
            except Exception:
                pass
        if warnings:
            exports["last_watchdog_warn"] = warnings[-1]
        # build combined commit diff once
        combined_commit = "\n".join([str(r.get("diff") or "") for r in results if r.get("ok")])
        # export combined commit for macros/prompts
        exports["last_patch_commit"] = combined_commit or ""
        await report_result(tid, True, {"results": results, "diff_add": add, "diff_rem": rem, **({"watchdog": warnings} if warnings else {})})
        try:
            await verify_cb(goal=None, files=list({p.replace('\\','/') for p in changed_files}), diff=combined_commit)
        except Exception:
            pass
    except Exception as e:
        await report_result(tid, False, error=f"batch patch failed: {e}")
