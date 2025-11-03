from __future__ import annotations

from typing import Callable, Awaitable, List, Dict

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.handlers.batch_handler import handle_batch_patch as _h_batch
from jinx.micro.runtime.patch.rename_symbol import build_rename_ops as _build_rename_ops
from jinx.micro.common.log import log_info, log_error

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_refactor_rename(
    tid: str,
    symbol: str,
    new_name: str,
    *,
    verify_cb: VerifyCB,
    exports: Dict[str, str],
) -> None:
    """Graph-guided rename across defs and callsites using the symbol index.

    Conservative: renames top-level def/class names, bare calls, and single-line import entries.
    Delegates commit to batch handler for unified preview/gate/commit/verify.
    """
    try:
        await report_progress(tid, 8.0, "build rename ops")
        ops = await _build_rename_ops(symbol, new_name)
        if not ops:
            await report_result(tid, False, error="no rename edits produced (index empty or symbol not found)")
            return
        try:
            log_info("rename.preview.ok", symbol=symbol, new=new_name, ops=len(ops))
        except Exception:
            pass
        await report_progress(tid, 22.0, "preview rename (batch)")
        # Force to reduce gating friction (still validated by finalize pipeline)
        await _h_batch(tid, ops, True, verify_cb=verify_cb, exports=exports)
    except Exception as e:
        try:
            from jinx.micro.common.repair_utils import maybe_schedule_repairs_from_error as _rep
            await _rep(e)
        except Exception:
            pass
        try:
            log_error("rename.error", symbol=symbol, new=new_name, msg=str(e))
        except Exception:
            pass
        await report_result(tid, False, error=f"refactor rename failed: {e}")
