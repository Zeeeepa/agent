from __future__ import annotations

import ast
import os
from typing import List, Set, Tuple

from jinx.async_utils.fs import read_text_raw
from jinx.micro.runtime.patch.write_patch import patch_write as _patch_write
from jinx.micro.embeddings.symbol_index import query_symbol_index as _sym_query
from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root


def _module_docstring_lines(m: ast.Module, src: str) -> Tuple[int, int]:
    """Return (start_line, end_line) of module docstring if present, else (0,0)."""
    try:
        if m.body and isinstance(m.body[0], ast.Expr) and isinstance(getattr(m.body[0], "value", None), ast.Constant) and isinstance(m.body[0].value.value, str):
            # Extract raw text lines to find extent; ast gives lineno but not end_lineno on 3.10-
            ds = m.body[0].value.value
            # naive: search the docstring content in source to measure lines; fallback to ast.lineno
            start = getattr(m.body[0], "lineno", 1)
            end = getattr(m.body[0], "end_lineno", start)
            return int(start), int(end)
    except Exception:
        pass
    return 0, 0


def _imports_defined(m: ast.Module) -> Tuple[Set[str], Set[str]]:
    """Return (imported_names, imported_modules)."""
    names: Set[str] = set()
    mods: Set[str] = set()
    for n in ast.walk(m):
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.asname:
                    names.add(alias.asname)
                else:
                    # 'import pkg.sub' binds 'pkg'
                    root = (alias.name or "").split(".", 1)[0]
                    if root:
                        names.add(root)
                    mods.add(alias.name or "")
        elif isinstance(n, ast.ImportFrom):
            for alias in n.names:
                nm = alias.asname or alias.name
                if nm:
                    names.add(nm)
    return names, mods


def _top_defs(m: ast.Module) -> Set[str]:
    out: Set[str] = set()
    for n in getattr(m, "body", []) or []:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = getattr(n, "name", None)
            if nm:
                out.add(nm)
        elif isinstance(n, ast.Assign):
            for t in getattr(n, "targets", []) or []:
                if isinstance(t, ast.Name) and t.id:
                    out.add(t.id)
    return out


def _used_names(m: ast.Module) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and isinstance(getattr(n, "ctx", None), ast.Load):
            if n.id:
                out.add(n.id)
    return out


def _builtins_set() -> Set[str]:
    try:
        return set(dir(__builtins__))  # type: ignore[name-defined]
    except Exception:
        return set()


async def _propose_imports_for(path: str, text: str) -> List[str]:
    try:
        mod = ast.parse(text or "")
    except Exception:
        return []
    imported_names, _imported_mods = _imports_defined(mod)
    defined = _top_defs(mod)
    used = _used_names(mod)
    missing = sorted(list(used - imported_names - defined - _builtins_set()))
    if not missing:
        return []
    proposals: List[str] = []
    for name in missing:
        try:
            idx = await _sym_query(name)
        except Exception:
            idx = {"defs": [], "calls": []}
        defs = idx.get("defs") or []
        if len(defs) != 1:
            continue
        rel, _ln = defs[0]
        # Skip if provider is same file
        try:
            root = _resolve_root()
            ap_src = os.path.abspath(path if os.path.isabs(path) else os.path.join(root, path))
            ap_def = os.path.abspath(os.path.join(root, rel))
            if ap_src == ap_def:
                continue
        except Exception:
            pass
        # from rel_module import name
        mod_name = rel[:-3].replace("\\", "/")
        if mod_name.endswith("/__init__"):
            mod_name = mod_name[:-9]
        mod_name = ".".join([p for p in mod_name.split("/") if p])
        if not mod_name:
            continue
        proposals.append(f"from {mod_name} import {name}")
    return proposals


def _insert_imports(text: str, imports: List[str]) -> str:
    if not imports:
        return text
    lines = (text or "").splitlines()
    # Find insertion point after module docstring and existing imports
    try:
        mod = ast.parse(text or "")
    except Exception:
        mod = None
    insert_at = 0
    if mod is not None:
        s, e = _module_docstring_lines(mod, text or "")
        insert_at = max(insert_at, e)
        # scan existing imports
        for n in getattr(mod, "body", []) or []:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                insert_at = max(insert_at, int(getattr(n, "end_lineno", getattr(n, "lineno", 0)) or 0))
            else:
                break
    # Build block with a dividing blank line if needed
    block = []
    if insert_at > 0 and insert_at <= len(lines) and (lines[insert_at-1] or "").strip():
        block.append("")
    block.extend(imports)
    block.append("")
    new_lines = lines[:insert_at] + block + lines[insert_at:]
    return "\n".join(new_lines)


async def synthesize_and_patch_imports(path: str) -> Tuple[bool, str]:
    """Analyze a Python file and insert import lines for unique missing symbols.

    Returns (ok, diff) using the patch_write pipeline for atomicity and diff reporting.
    """
    cur = await read_text_raw(path)
    if cur == "":
        return False, "file read error or empty"
    props = await _propose_imports_for(path, cur)
    if not props:
        return True, ""  # nothing to do
    new = _insert_imports(cur, props)
    if new == cur:
        return True, ""
    ok, diff = await _patch_write(path, new, preview=False)
    return ok, diff
