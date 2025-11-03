from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result, submit_task
from jinx.micro.runtime.patch import (
    patch_write as _patch_write,
    should_autocommit as _should_autocommit,
    diff_stats as _diff_stats,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize
from jinx.async_utils.fs import read_text_raw
from jinx.micro.runtime.patch.validators_integration import validate_text
from jinx.micro.core.edit_core import finalize_commit
from jinx.micro.common.log import log_info, log_error
from jinx.micro.embeddings.symbol_index import update_symbol_index as _symindex_update
import ast

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_write(tid: str, path: str, text: str, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        await report_progress(tid, 15.0, f"preview write {path}")
        ok_prev, diff = await _patch_write(path, text, preview=True)
        if not ok_prev:
            try:
                log_error("patch.preview.fail", strategy="write", path=path)
            except Exception:
                pass
            await report_result(tid, False, error=diff)
            return
        try:
            add_p, rem_p = _diff_stats(diff)
            log_info("patch.preview.ok", strategy="write", path=path, add=add_p, rem=rem_p)
        except Exception:
            pass
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit("write", diff)
        if not okc:
            add, rem = _diff_stats(diff)
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = "write"
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": path, "diff": diff, "diff_add": add, "diff_rem": rem})
            return
        await report_progress(tid, 45.0, f"commit write {path}")
        # Snapshot current contents for potential revert on validation failure
        try:
            cur_before = await read_text_raw(path)
        except Exception:
            cur_before = ""
        # Pre-commit AST check for Python files to avoid committing syntax errors
        try:
            if path.lower().endswith(".py"):
                ast.parse(text)
        except SyntaxError as se:
            await report_result(tid, False, error=f"syntax error: {se}")
            return
        ok_commit, diff2 = await _patch_write(path, text, preview=False)
        if ok_commit:
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = "write"
            # Unified finalize pipeline
            snapshots = {path: cur_before or ""}
            core = await finalize_commit([path], diff2 or "", snapshots=snapshots, strategy="write")
            if not core.ok:
                try:
                    log_error("patch.commit.revert", strategy="write", path=path, errs=len(core.errors))
                except Exception:
                    pass
                # Background autofix attempt (env-gated by program)
                try:
                    await submit_task("autofix.retry", path=path, code=text, symbol=None, query=None)
                except Exception:
                    pass
                await report_result(tid, False, error=(core.errors[0] if core.errors else "validation failed"), result={"path": path, "reverted": True})
                return
            add2, rem2 = _diff_stats(diff2)
            # Attach last watchdog warning if any
            if core.warnings:
                exports["last_watchdog_warn"] = core.warnings[-1]
            try:
                log_info("patch.commit.ok", strategy="write", path=path, add=add2, rem=rem2, warns=len(core.warnings or []))
            except Exception:
                pass
            # Update symbol index (best-effort)
            try:
                await _symindex_update([path])
            except Exception:
                pass
            await report_result(tid, True, {"path": path, "bytes": len(text), "diff": diff2, "diff_add": add2, "diff_rem": rem2, **({"watchdog": core.warnings} if core.warnings else {})})
        else:
            try:
                log_error("patch.commit.fail", strategy="write", path=path)
            except Exception:
                pass
            await report_result(tid, False, error=diff2)
    except Exception as e:
        try:
            from jinx.micro.common.repair_utils import maybe_schedule_repairs_from_error as _rep
            await _rep(e)
        except Exception:
            pass
        await report_result(tid, False, error=f"write failed: {e}")
