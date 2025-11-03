from __future__ import annotations

import os
from typing import Any, Dict, List, Callable, Awaitable
import re

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
from jinx.micro.embeddings.symbol_index import update_symbol_index as _symindex_update
from jinx.async_utils.fs import read_text_raw
from jinx.micro.runtime.patch.validators_integration import validate_text
from jinx.micro.core.edit_core import finalize_commit
from jinx.micro.common.log import log_info, log_error

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
        # Snapshot originals for potential revert on validation failures
        originals: Dict[str, str] = {}
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
            elif typ == "regex":
                pat = str(op.get("pattern") or "")
                flags = str(op.get("flags") or "")
                fl = 0
                if "i" in flags.lower(): fl |= re.IGNORECASE
                if "m" in flags.lower(): fl |= re.MULTILINE
                if "s" in flags.lower(): fl |= re.DOTALL
                if "x" in flags.lower(): fl |= re.VERBOSE
                try:
                    rx = re.compile(pat, fl)
                except re.error as e:
                    previews.append({"i": i, "type": typ, "ok": False, "diff": f"invalid regex: {e}", "path": path})
                    combined_diff_parts.append("")
                    continue
                try:
                    original = await read_text_raw(path)
                except Exception:
                    previews.append({"i": i, "type": typ, "ok": False, "diff": f"cannot read: {path}", "path": path})
                    combined_diff_parts.append("")
                    continue
                new_text, count = rx.subn(code, original)
                if count <= 0 or new_text == original:
                    previews.append({"i": i, "type": typ, "ok": False, "diff": "no match or no change", "path": path, "matches": count})
                    combined_diff_parts.append("")
                    continue
                ok, diff = await _patch_write(path, new_text, preview=True)
                previews.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "matches": count})
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
            try:
                log_error("batch.preview.needs_confirmation", ops=len(ops), add=add, rem=rem, reason=str(reason))
            except Exception:
                pass
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"previews": previews, "diff_add": add, "diff_rem": rem})
            return
        try:
            log_info("batch.preview.ok", ops=len(ops), add=add, rem=rem, is_refactor=bool(is_refactor))
        except Exception:
            pass
        await report_progress(tid, 55.0, "batch commit")
        results: List[Dict[str, Any]] = []
        changed_files: List[str] = []
        for i, op in enumerate(ops):
            typ = str(op.get("type") or "auto").strip().lower()
            path = str(op.get("path") or "")
            code = str(op.get("code") or "")
            if typ == "write":
                # Snapshot
                try:
                    if path and path not in originals:
                        originals[path] = await read_text_raw(path)
                except Exception:
                    originals[path] = ""
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
                # Snapshot
                try:
                    if path and path not in originals:
                        originals[path] = await read_text_raw(path)
                except Exception:
                    originals[path] = ""
                ok, diff = await _patch_line(path, ls, le, code, preview=False, max_span=max_span)
                results.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "ls": ls, "le": le})
                if ok and path:
                    changed_files.append(path)
            elif typ == "regex":
                pat = str(op.get("pattern") or "")
                flags = str(op.get("flags") or "")
                fl = 0
                if "i" in flags.lower(): fl |= re.IGNORECASE
                if "m" in flags.lower(): fl |= re.MULTILINE
                if "s" in flags.lower(): fl |= re.DOTALL
                if "x" in flags.lower(): fl |= re.VERBOSE
                # Snapshot
                try:
                    if path and path not in originals:
                        originals[path] = await read_text_raw(path)
                except Exception:
                    originals[path] = ""
                try:
                    rx = re.compile(pat, fl)
                except re.error as e:
                    results.append({"i": i, "type": typ, "ok": False, "diff": f"invalid regex: {e}", "path": path})
                    continue
                try:
                    original = originals.get(path) or await read_text_raw(path)
                except Exception:
                    results.append({"i": i, "type": typ, "ok": False, "diff": f"cannot read: {path}", "path": path})
                    continue
                new_text, count = rx.subn(code, original)
                if count <= 0 or new_text == original:
                    results.append({"i": i, "type": typ, "ok": False, "diff": "no match or no change", "path": path, "matches": count})
                    continue
                ok, diff = await _patch_write(path, new_text, preview=False)
                results.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "matches": count})
                if ok and path:
                    changed_files.append(path)
            elif typ == "symbol":
                sym = str(op.get("symbol") or "")
                try:
                    if path and path not in originals:
                        originals[path] = await read_text_raw(path)
                except Exception:
                    originals[path] = ""
                ok, diff = await _patch_symbol(path, sym, code, preview=False)
                results.append({"i": i, "type": typ, "ok": ok, "diff": diff, "path": path, "symbol": sym})
                if ok and path:
                    changed_files.append(path)
            elif typ == "anchor":
                anc = str(op.get("anchor") or "")
                try:
                    if path and path not in originals:
                        originals[path] = await read_text_raw(path)
                except Exception:
                    originals[path] = ""
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
        # build combined commit diff once
        combined_commit = "\n".join([str(r.get("diff") or "") for r in results if r.get("ok")])
        # export combined commit for macros/prompts
        exports["last_patch_commit"] = combined_commit or ""
        # Unified finalize commit across all changed files
        files_norm = list({p.replace('\\','/') for p in changed_files if p})
        core = await finalize_commit(files_norm, combined_commit or "", snapshots=originals, strategy=("batch:refactor" if is_refactor else "batch"))
        if not core.ok:
            exports["last_patch_reason"] = "validation_failed_batch"
            exports["last_patch_strategy"] = "batch"
            try:
                log_error("batch.commit.revert", files=len(files_norm), errs=len(core.errors))
            except Exception:
                pass
            await report_result(tid, False, error=(core.errors[0] if core.errors else "validation failed"), result={"errors": [{"path": e.split(':',1)[0], "error": e} for e in core.errors], "reverted_files": files_norm})
            return
        if core.warnings:
            exports["last_watchdog_warn"] = core.warnings[-1]
        try:
            log_info("batch.commit.ok", files=len(files_norm), warns=len(core.warnings or []))
        except Exception:
            pass
        await report_result(tid, True, {"results": results, "diff_add": add, "diff_rem": rem, **({"watchdog": core.warnings} if core.warnings else {})})
    except Exception as e:
        try:
            from jinx.micro.common.repair_utils import maybe_schedule_repairs_from_error as _rep
            await _rep(e)
        except Exception:
            pass
        try:
            log_error("batch.error", msg=str(e))
        except Exception:
            pass
        await report_result(tid, False, error=f"batch patch failed: {e}")
