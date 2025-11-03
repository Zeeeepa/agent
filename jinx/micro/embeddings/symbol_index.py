from __future__ import annotations

import os
import json
import ast
import asyncio
from typing import Any, Dict, List, Tuple, Optional

from jinx.micro.embeddings.project_paths import PROJECT_STATE_DIR, ensure_project_dirs
from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.async_utils.fs import read_text_abs_thread, write_text

_INDEX_PATH = os.path.join(PROJECT_STATE_DIR, "symbol_index.json")
_EXCLUDE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", "emb", "build", "dist"}


def _is_excluded(rel_path: str) -> bool:
    parts = (rel_path or "").split(os.sep)
    return any(p in _EXCLUDE_DIRS for p in parts)


async def _read_index() -> Dict[str, Any]:
    try:
        ensure_project_dirs()
    except Exception:
        pass
    try:
        txt = await read_text_abs_thread(_INDEX_PATH)
    except Exception:
        txt = ""
    if not txt:
        return {"files": {}, "defs": {}, "calls": {}}
    try:
        return json.loads(txt)
    except Exception:
        return {"files": {}, "defs": {}, "calls": {}}


async def _write_index(data: Dict[str, Any]) -> None:
    try:
        ensure_project_dirs()
    except Exception:
        pass
    try:
        await write_text(_INDEX_PATH, json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


def _relpath(abs_path: str, root: str) -> str:
    try:
        return os.path.relpath(abs_path, root).replace("\\", "/")
    except Exception:
        return abs_path.replace("\\", "/")


def _call_name(node: ast.AST) -> Optional[str]:
    # Extract a human symbol name from ast.Call
    try:
        if isinstance(getattr(node, "func", None), ast.Name):
            return node.func.id
        if isinstance(getattr(node, "func", None), ast.Attribute):
            # use attribute tail, e.g., obj.method -> method
            return node.func.attr
    except Exception:
        return None
    return None


def _collect_for(text: str) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int]]]:
    """Return (defs, calls)
    defs: list of (name, line, kind={"func"|"class"})
    calls: list of (name, line)
    """
    defs: List[Tuple[str, int, str]] = []
    calls: List[Tuple[str, int]] = []
    try:
        m = ast.parse(text or "")
    except Exception:
        return defs, calls
    for n in ast.walk(m):
        if isinstance(n, ast.FunctionDef):
            defs.append((n.name, int(getattr(n, "lineno", 1) or 1), "func"))
        elif isinstance(n, ast.AsyncFunctionDef):
            defs.append((n.name, int(getattr(n, "lineno", 1) or 1), "func"))
        elif isinstance(n, ast.ClassDef):
            defs.append((n.name, int(getattr(n, "lineno", 1) or 1), "class"))
        elif isinstance(n, ast.Call):
            nm = _call_name(n)
            if nm:
                calls.append((nm, int(getattr(n, "lineno", 1) or 1)))
    return defs, calls


async def update_symbol_index(files: List[str]) -> None:
    """Incrementally update the symbol index for given files (project-relative or absolute).
    Ignores excluded dirs. Safe no-op on errors.
    """
    root = _resolve_root()
    idx = await _read_index()
    files_map: Dict[str, Any] = idx.get("files", {})
    defs_ix: Dict[str, List[Tuple[str, int]]] = idx.get("defs", {})
    calls_ix: Dict[str, List[Tuple[str, int]]] = idx.get("calls", {})

    # remove stale entries for incoming files
    def _drop_entries_for(rel: str) -> None:
        nonlocal defs_ix, calls_ix
        for sym, lst in list(defs_ix.items()):
            defs_ix[sym] = [pair for pair in lst if pair[0] != rel]
            if not defs_ix[sym]:
                defs_ix.pop(sym, None)
        for sym, lst in list(calls_ix.items()):
            calls_ix[sym] = [pair for pair in lst if pair[0] != rel]
            if not calls_ix[sym]:
                calls_ix.pop(sym, None)

    async def _process_one(p: str) -> None:
        # Determine absolute and relative path
        ap = p
        if not os.path.isabs(ap):
            ap = os.path.join(root, p)
        rel = _relpath(ap, root)
        if _is_excluded(rel) or not rel.endswith(".py"):
            return
        try:
            text = await read_text_abs_thread(ap)
        except Exception:
            return
        d_list, c_list = _collect_for(text or "")
        files_map[rel] = {"defs": d_list, "calls": c_list}
        _drop_entries_for(rel)
        for nm, ln, kind in d_list:
            defs_ix.setdefault(nm, []).append((rel, ln))
        for nm, ln in c_list:
            calls_ix.setdefault(nm, []).append((rel, ln))

    # Process sequentially to avoid heavy CPU; parsing per file is cheap
    for fp in (files or []):
        try:
            await _process_one(fp)
        except Exception:
            continue

    idx["files"] = files_map
    idx["defs"] = defs_ix
    idx["calls"] = calls_ix
    await _write_index(idx)


async def query_symbol_index(token: str) -> Dict[str, List[Tuple[str, int]]]:
    """Return mapping: {"defs": [(rel, line)], "calls": [(rel, line)]} for exact token.
    Safe empty on error.
    """
    try:
        idx = await _read_index()
        out: Dict[str, List[Tuple[str, int]]] = {"defs": [], "calls": []}
        if not token:
            return out
        out["defs"] = list(idx.get("defs", {}).get(token, []) or [])
        out["calls"] = list(idx.get("calls", {}).get(token, []) or [])
        return out
    except Exception:
        return {"defs": [], "calls": []}
