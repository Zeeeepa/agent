from __future__ import annotations

# DEPRECATED: This module has been split into micro-modules:
#  - refactor_utils.py
#  - refactor_imports.py
#  - refactor_move.py
#  - refactor_split.py
# It is kept for historical reference; the handlers package re-exports from the micro-modules.

import os
import asyncio
import ast
from typing import Callable, Awaitable, List, Dict, Tuple, Optional

from jinx.async_utils.fs import read_text_raw
from jinx.micro.runtime.api import report_progress, report_result
from jinx.micro.runtime.handlers.batch_handler import handle_batch_patch as _h_batch
from jinx.micro.runtime.source_extract import extract_symbol_source
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT, EXCLUDE_DIRS

VerifyCB = Callable[[str | None, List[str], str], Awaitable[None]]


def _abs_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    base = PROJECT_ROOT or os.getcwd()
    return os.path.normpath(os.path.join(base, p))


def _truthy(name: str, default: str = "1") -> bool:
    try:
        return str(os.getenv(name, default)).strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


def _module_name_from_path(path: str) -> str:
    ap = _abs_path(path)
    root = os.path.normpath(PROJECT_ROOT or os.getcwd())
    if ap.startswith(root):
        rel = os.path.relpath(ap, root)
    else:
        rel = ap
    if rel.endswith("__init__.py"):
        rel = os.path.dirname(rel)
    else:
        rel = rel[:-3] if rel.lower().endswith(".py") else rel
    parts = []
    for seg in rel.replace("\\", "/").split("/"):
        if seg and seg != ".":
            parts.append(seg)
    return ".".join(parts)


def _ensure_newline(s: str) -> str:
    return s if (not s or s.endswith("\n")) else (s + "\n")


async def _read(path: str) -> str:
    return await read_text_raw(path)


def _import_insertion_index(text: str) -> int:
    """Find a safe index (line number 0-based) to insert import lines: after shebang/encoding/docstring/future imports."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    # shebang
    if i < n and lines[i].startswith("#!"):
        i += 1
    # encoding cookie
    if i < n and ("coding:" in lines[i] or "coding=" in lines[i]):
        i += 1
    # docstring (triple-quoted on first non-empty line)
    while i < n and not lines[i].strip():
        i += 1
    if i < n and (lines[i].lstrip().startswith("\"\"\"") or lines[i].lstrip().startswith("'''")):
        quote = '"""' if lines[i].lstrip().startswith('"""') else "'''"
        # advance to closing
        j = i
        while j < n:
            if quote in lines[j] and (j != i or lines[j].count(quote) >= 2):
                i = j + 1
                break
            j += 1
    # future imports block
    k = i
    while k < n and lines[k].lstrip().startswith("from __future__ import"):
        k += 1
    return k


def _append_unique_line(lines: List[str], line: str) -> List[str]:
    if any(ln.strip() == line.strip() for ln in lines):
        return lines
    return lines[:1] + [line] + lines[1:] if lines else [line]


async def _build_move_plan(src_path: str, symbol: str, dst_path: str, *, create_init: bool, insert_shim: bool) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    """Construct a batch ops plan to move a symbol to dst_path, optionally inserting shim import in source and updating __init__ in destination package.

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
        more_ops = await _scan_and_rewrite_imports(old_mod, new_mod, symbol)
        ops.extend(more_ops)

    return ops, {"dst_module": _module_name_from_path(ap_dst)}


async def _scan_and_rewrite_imports(old_mod: str, new_mod: str, symbol: str) -> List[Dict[str, object]]:
    """Conservatively rewrite 'from old_mod import symbol' to new_mod across project.

    Only rewrites lines where the imported list is a single name (optionally 'as alias'), to avoid
    changing imports of other symbols from old_mod. Preserves alias.
    """
    root = os.path.normpath(PROJECT_ROOT or os.getcwd())
    ops: List[Dict[str, object]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in set(EXCLUDE_DIRS)]
        for fn in filenames:
            if not fn.lower().endswith(".py"):
                continue
            ap = os.path.join(dirpath, fn)
            try:
                text = await _read(ap)
            except Exception:
                continue
            if text == "":
                continue
            lines = text.splitlines()
            changed = False
            for i, ln in enumerate(lines):
                # match: from old_mod import <names>[#comment]
                if not ln.lstrip().startswith("from "):
                    continue
                # quick contains check
                if f"from {old_mod} import" not in ln:
                    continue
                # Extract the import list preserving alias
                try:
                    before, _, after = ln.partition(f"from {old_mod} import ")
                    if not after:
                        continue
                    # strip trailing comment but keep content
                    code_part, *_ = after.split("#", 1)
                    names_raw = code_part.strip()
                    # Only rewrite if exactly one imported entry
                    parts = [p.strip() for p in names_raw.split(",") if p.strip()]
                    # Case A: exactly the symbol (with optional alias)
                    if len(parts) == 1:
                        entry = parts[0]
                        base = entry.split(" as ")[0].strip()
                        if base != symbol:
                            continue
                        new_line = before + f"from {new_mod} import " + entry
                        if new_line != ln:
                            lines[i] = new_line
                            changed = True
                    # Case B: grouped import containing our symbol -> split when enabled
                    elif symbol in [p.split(" as ")[0].strip() for p in parts]:
                        if not _truthy("JINX_REFACTOR_REWRITE_GROUPED_IMPORTS", "1"):
                            continue
                        # keep others in old_mod, move our entry to new_mod on a new line
                        keep_parts = []
                        move_entry = None
                        alias = None
                        for p in parts:
                            base = p.split(" as ")[0].strip()
                            if base == symbol and move_entry is None:
                                # preserve alias text
                                move_entry = p
                                if " as " in p:
                                    alias = p.split(" as ", 1)[1].strip()
                            else:
                                keep_parts.append(p)
                        if move_entry is None:
                            continue
                        indent = before  # includes leading whitespace
                        new_lines_block = []
                        # from new_mod import symbol[ as alias]
                        new_lines_block.append(indent + f"from {new_mod} import {move_entry}")
                        # from old_mod import other1, other2 (if any left)
                        if keep_parts:
                            new_lines_block.append(indent + f"from {old_mod} import {', '.join(keep_parts)}")
                        # replace the single line with 1-2 lines
                        lines[i:i+1] = new_lines_block
                        changed = True
                except Exception:
                    continue
            if changed:
                new_text = ("\n".join(lines)).rstrip("\n") + "\n"
                ops.append({"type": "write", "path": ap, "code": new_text})
    return ops


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
        ops, meta = await _build_move_plan(src_path, symbol, dst_path, create_init=ci, insert_shim=sh)
        await report_progress(tid, 22.0, "preview refactor (batch)")
        # Delegate to batch handler (with force to reduce gating friction)
        f = bool(force) if force is not None else _truthy("JINX_REFACTOR_FORCE", "1")
        await _h_batch(tid, ops, f, verify_cb=verify_cb, exports=exports)
    except Exception as e:
        await report_result(tid, False, error=f"refactor move failed: {e}")


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

    Strategy: list top-level defs/classes, move each via the same plan as handle_refactor_move_symbol, but aggregate into a single batch.
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
        for node in tree.body:
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
