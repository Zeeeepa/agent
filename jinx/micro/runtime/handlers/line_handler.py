from __future__ import annotations

import os
from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_line_range as _patch_line,
    should_autocommit as _should_autocommit,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_line_patch(tid: str, path: str, ls: int, le: int, replacement: str, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        if ls <= 0 or le <= 0 or le < ls:
            await report_result(tid, False, error="invalid line range")
            return
        try:
            max_span = int(os.getenv("JINX_PATCH_MAX_SPAN", "80"))
        except Exception:
            max_span = 80
        await report_progress(tid, 15.0, f"preview patch {path}:{ls}-{le}")
        ok_prev, diff = await _patch_line(path, ls, le, replacement, preview=True, max_span=max_span)
        if not ok_prev:
            await report_result(tid, False, error=diff)
            return
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit("line", diff)
        if not okc:
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = "line"
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": path, "lines": [ls, le], "diff": diff})
            return
        await report_progress(tid, 55.0, f"commit patch {path}:{ls}-{le}")
        ok_commit, diff2 = await _patch_line(path, ls, le, replacement, preview=False, max_span=max_span)
        if ok_commit:
            warn = await maybe_warn_filesize(path)
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = "line"
            if warn:
                exports["last_watchdog_warn"] = warn
            await report_result(tid, True, {"path": path, "lines": [ls, le], "diff": diff2, **({"watchdog": warn} if warn else {})})
            await verify_cb(goal=None, files=[path], diff=diff2)
        else:
            await report_result(tid, False, error=diff2)
    except Exception as e:
        await report_result(tid, False, error=f"line patch failed: {e}")
