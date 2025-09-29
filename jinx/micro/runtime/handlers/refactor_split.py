from __future__ import annotations

import os
import asyncio
import ast
from typing import Callable, Awaitable, List, Dict, Tuple, Optional

from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.handlers.batch_handler import handle_batch_patch as _h_batch
from .refactor_utils import (
    _abs_path,
    _truthy,
    _module_name_from_path,
    _import_insertion_index,
    _read,
)


VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


async def handle_refactor_split_file(
    tid: str,
    src_path: str,
    out_dir: str,
    *,
    verify_cb: VerifyCB,
    exports: Dict[str, str],
    create_init: Optional[bool] = None,
    insert_shim: Optional[bool] = None,
    force: Optional[bool] = None,
) -> None:
    """Split a Python module into per-symbol modules in out_dir, and convert the source into a shim importing moved symbols.

    Strategy: list top-level defs/classes, move each to its own file, aggregate into a single batch.
    """
    try:
        await report_progress(tid, 8.0, "parse symbols")
        ap_src = _abs_path(src_path)
        text = await _read(ap_src)
        try:
            tree = await asyncio.to_thread(ast.parse, text)
        except Exception as e:
            await report_result(tid, False, error=f"ast parse failed: {e}")
            return
        symbols: List[Tuple[str, int, int, str]] = []  # (name, lineno, end_lineno, kind)
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = getattr(node, "name", "")
                if name and not name.startswith("__"):
                    symbols.append((name, int(getattr(node, "lineno", 1)), int(getattr(node, "end_lineno", 1)), node.__class__.__name__))
        if not symbols:
            await report_result(tid, False, error="no top-level symbols to split")
            return
        ci = create_init if create_init is not None else _truthy("JINX_REFACTOR_CREATE_INIT", "1")
        sh = insert_shim if insert_shim is not None else _truthy("JINX_REFACTOR_INSERT_SHIM", "1")
        ap_out = _abs_path(out_dir)
        os.makedirs(ap_out, exist_ok=True)
        ops_all: List[Dict[str, object]] = []
        # Prepare cumulative shim imports for src
        shim_imports: List[str] = []
        # Build write for src after removing all symbols
        lines = text.splitlines()
        mask = [False] * len(lines)
        for _, s, e, _ in symbols:
            s1 = max(1, min(s, len(lines)))
            e1 = max(s1, min(e, len(lines)))
            for i in range(s1 - 1, e1):
                mask[i] = True
        kept = [ln for i, ln in enumerate(lines) if not mask[i]]
        if sh:
            for name, _, _, _ in symbols:
                dst_file = os.path.join(ap_out, f"{name}.py")
                dst_module = _module_name_from_path(dst_file)
                shim_imports.append(f"from {dst_module} import {name}")
            # Insert shim imports into kept at insertion index
            idx = _import_insertion_index("\n".join(kept))
            kept = kept[:idx] + shim_imports + kept[idx:]
        src_new = ("\n".join(kept)).rstrip("\n") + "\n"
        # Build per-symbol destination writes and package __init__
        pkg_init_path = os.path.join(ap_out, "__init__.py")
        try:
            pkg_init_text = await _read(pkg_init_path)
        except Exception:
            pkg_init_text = ""
        if pkg_init_text == "":
            pkg_init_text = "\n"
        init_lines = pkg_init_text.splitlines()
        for name, s, e, _ in symbols:
            # extract code slice
            s1 = max(1, min(s, len(lines)))
            e1 = max(s1, min(e, len(lines)))
            code = "\n".join(lines[s1 - 1 : e1])
            dst_file = os.path.join(ap_out, f"{name}.py")
            try:
                dst_text = await _read(dst_file)
            except Exception:
                dst_text = ""
            dst_new = dst_text or ""
            if dst_new and not dst_new.endswith("\n\n"):
                if not dst_new.endswith("\n"):
                    dst_new += "\n"
                dst_new += "\n"
            dst_new += (code.rstrip("\n") + "\n")
            ops_all.append({"type": "write", "path": dst_file, "code": dst_new, "meta": {"refactor": "split", "role": "dst", "symbol": name}})
            if ci:
                export_line = f"from .{name} import {name}"
                if not any(ln.strip() == export_line for ln in init_lines):
                    init_lines.append(export_line)
        if ci:
            init_new = ("\n".join(init_lines)).rstrip("\n") + "\n"
            ops_all.append({"type": "write", "path": pkg_init_path, "code": init_new, "meta": {"refactor": "split", "role": "dst_init"}})
        # Finally, write source shim
        ops_all.append({"type": "write", "path": ap_src, "code": src_new, "meta": {"refactor": "split", "role": "src"}})
        await report_progress(tid, 22.0, "preview refactor split (batch)")
        f = bool(force) if force is not None else _truthy("JINX_REFACTOR_FORCE", "1")
        await _h_batch(tid, ops_all, f, verify_cb=verify_cb, exports=exports)
    except Exception as e:
        await report_result(tid, False, error=f"refactor split failed: {e}")
