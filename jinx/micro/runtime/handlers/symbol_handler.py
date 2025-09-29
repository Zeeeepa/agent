from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.patch import (
    patch_symbol_python as _patch_symbol,
    should_autocommit as _should_autocommit,
)
from jinx.micro.runtime.watchdog import maybe_warn_filesize

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_symbol_patch(tid: str, path: str, symbol: str, replacement: str, *, verify_cb: VerifyCB, exports: Dict[str, str]) -> None:
    try:
        await report_progress(tid, 15.0, f"preview symbol {symbol} in {path}")
        ok_prev, diff = await _patch_symbol(path, symbol, replacement, preview=True)
        if not ok_prev:
            await report_result(tid, False, error=diff)
            return
        exports["last_patch_preview"] = diff or ""
        okc, reason = _should_autocommit("symbol", diff)
        if not okc:
            exports["last_patch_reason"] = f"needs_confirmation: {reason}"
            exports["last_patch_strategy"] = "symbol"
            await report_result(tid, False, error=f"needs_confirmation: {reason}", result={"path": path, "symbol": symbol, "diff": diff})
            return
        await report_progress(tid, 55.0, f"commit symbol {symbol} in {path}")
        ok_commit, diff2 = await _patch_symbol(path, symbol, replacement, preview=False)
        if ok_commit:
            warn = await maybe_warn_filesize(path)
            exports["last_patch_commit"] = diff2 or ""
            exports["last_patch_strategy"] = "symbol"
            if warn:
                exports["last_watchdog_warn"] = warn
            await report_result(tid, True, {"path": path, "symbol": symbol, "diff": diff2, **({"watchdog": warn} if warn else {})})
            await verify_cb(goal=None, files=[path], diff=diff2)
        else:
            await report_result(tid, False, error=diff2)
    except Exception as e:
        await report_result(tid, False, error=f"symbol patch failed: {e}")
