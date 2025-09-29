from __future__ import annotations

import os
from typing import Callable, Awaitable, List, Dict, Optional, Tuple

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.handlers.batch_handler import handle_batch_patch as _h_batch
from jinx.micro.runtime.source_extract import extract_symbol_source
from .refactor_utils import (
    _abs_path,
    _truthy,
    _module_name_from_path,
    _ensure_newline,
    _read,
    _import_insertion_index,
)
from .refactor_imports import scan_and_rewrite_imports


VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def _build_move_plan(
    src_path: str,
    symbol: str,
    dst_path: str,
    *,
    create_init: bool,
    insert_shim: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    """Construct a batch ops plan to move a symbol to dst_path, optionally inserting shim import in source
    and updating __init__ in destination package.

    Returns (ops, meta) where ops is a list of batch operations and meta has diagnostic info.
    """
    ap_src = _abs_path(src_path)
    ap_dst = _abs_path(dst_path)
    ok, code, meta = await extract_symbol_source(ap_src, symbol, include_decorators=True, include_docstring=True)
    if not ok:
        raise RuntimeError(f"extract failed: {meta.get('error')}")
    src_text = await _read(ap_src)
    try:
        dst_text = await _read(ap_dst)
    except Exception:
        dst_text = ""

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(ap_dst), exist_ok=True)

    # Prepare destination content: append code with spacing
    dst_new = dst_text or ""
    if dst_new and not dst_new.endswith("\n\n"):
        if not dst_new.endswith("\n"):
            dst_new += "\n"
        dst_new += "\n"
    dst_new += _ensure_newline(code)

    # Prepare source shim content: remove symbol block and add import binding
    s = int(meta.get("start") or 0)
    e = int(meta.get("end") or 0)
    src_lines = src_text.splitlines()
    # bounds safety
    s = max(1, min(s, len(src_lines) if src_lines else 1))
    e = max(s, min(e, len(src_lines) if src_lines else s))
    kept = src_lines[: s - 1] + src_lines[e:]

    if insert_shim:
        # Compose import binding: from <dst_mod> import <symbol>
        dst_mod = _module_name_from_path(ap_dst)
        import_line = f"from {dst_mod} import {symbol}"
        idx = _import_insertion_index("\n".join(kept))
        # insert after idx
        kept = kept[:idx] + [import_line] + kept[idx:]

    src_new = _ensure_newline("\n".join(kept))

    ops: List[Dict[str, object]] = []
    # 1) write destination file
    ops.append({"type": "write", "path": ap_dst, "code": dst_new, "meta": {"refactor": "move", "role": "dst", "symbol": symbol}})
    # 2) write source file with shim
    ops.append({"type": "write", "path": ap_src, "code": src_new, "meta": {"refactor": "move", "role": "src", "symbol": symbol}})

    # 3) optional: ensure destination package __init__.py exports the symbol
    if create_init:
        dst_dir = os.path.dirname(ap_dst)
        init_path = os.path.join(dst_dir, "__init__.py")
        try:
            init_text = await _read(init_path)
        except Exception:
            init_text = ""
        if init_text == "":
            # file may not exist -> create base
            init_text = "\n"
        rel_mod = os.path.splitext(os.path.basename(ap_dst))[0]
        export_line = f"from .{rel_mod} import {symbol}"
        lines = init_text.splitlines()
        if not any(ln.strip() == export_line for ln in lines):
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(export_line)
        init_new = _ensure_newline("\n".join(lines))
        ops.append({"type": "write", "path": init_path, "code": init_new, "meta": {"refactor": "move", "role": "dst_init", "symbol": symbol}})

    # Optional: project-wide conservative import rewrite
    if _truthy("JINX_REFACTOR_REWRITE_IMPORTS", "0"):
        old_mod = _module_name_from_path(ap_src)
        new_mod = _module_name_from_path(ap_dst)
        grouped = _truthy("JINX_REFACTOR_REWRITE_GROUPED_IMPORTS", "1")
        more_ops = await scan_and_rewrite_imports(old_mod, new_mod, symbol, grouped_enabled=grouped)
        ops.extend(more_ops)

    return ops, {"dst_module": _module_name_from_path(ap_dst)}


async def handle_refactor_move_symbol(
    tid: str,
    src_path: str,
    symbol: str,
    dst_path: str,
    *,
    verify_cb: VerifyCB,
    exports: Dict[str, str],
    create_init: Optional[bool] = None,
    insert_shim: Optional[bool] = None,
    force: Optional[bool] = None,
) -> None:
    """Move a top-level function/class to another module with safe shims and package exports.

    - Writes symbol code to dst_path (appending, creates file if needed)
    - Removes symbol from src_path and inserts `from <dst_module> import <symbol>` at the top
    - Optionally updates __init__.py in destination package to re-export symbol
    - Runs through batch handler to preserve preview/gate/commit/watchdog/verify
    """
    try:
        await report_progress(tid, 9.0, "build move plan")
        ci = create_init if create_init is not None else _truthy("JINX_REFACTOR_CREATE_INIT", "1")
        sh = insert_shim if insert_shim is not None else _truthy("JINX_REFACTOR_INSERT_SHIM", "1")
        ops, _ = await _build_move_plan(src_path, symbol, dst_path, create_init=ci, insert_shim=sh)
        await report_progress(tid, 22.0, "preview refactor (batch)")
        # Delegate to batch handler (with force to reduce gating friction)
        f = bool(force) if force is not None else _truthy("JINX_REFACTOR_FORCE", "1")
        await _h_batch(tid, ops, f, verify_cb=verify_cb, exports=exports)
    except Exception as e:
        await report_result(tid, False, error=f"refactor move failed: {e}")
