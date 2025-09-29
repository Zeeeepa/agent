from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_write as _patch_write,
    should_autocommit as _should_autocommit,
    diff_stats as _diff_stats,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_write(tid: str, path: str, text: str, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        await report_progress(tid, 15.0, f"preview write {path}")
        ok_prev, diff = await _patch_write(path, text, preview=True)
        if not ok_prev:
            await report_result(tid, False, error=diff)
            return
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit("write", diff)
        if not okc:
            add, rem = _diff_stats(diff)
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = "write"
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": path, "diff": diff, "diff_add": add, "diff_rem": rem})
            return
        await report_progress(tid, 45.0, f"commit write {path}")
        ok_commit, diff2 = await _patch_write(path, text, preview=False)
        if ok_commit:
            warn = await maybe_warn_filesize(path)
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = "write"
            if warn:
                exports["last_watchdog_warn"] = warn
            add2, rem2 = _diff_stats(diff2)
            await report_result(tid, True, {"path": path, "bytes": len(text), "diff": diff2, "diff_add": add2, "diff_rem": rem2, **({"watchdog": warn} if warn else {})})
            await verify_cb(goal=None, files=[path], diff=diff2)
        else:
            await report_result(tid, False, error=diff2)
    except Exception as e:
        await report_result(tid, False, error=f"write failed: {e}")
