from __future__ import annotations

import os
import sys
import ast
import importlib.util
from typing import List, Tuple, Dict

from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.async_utils.fs import read_text_abs_thread, read_text_raw, write_text


def _top_module(name: str) -> str:
    return (name or "").split(".", 1)[0]


def _is_std_or_builtin(mod: str) -> bool:
    # Conservative check: builtin or stdlib set (3.11+), fallback to attempting find_spec
    try:
        if mod in sys.builtin_module_names:  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    try:
        if hasattr(sys, "stdlib_module_names") and mod in getattr(sys, "stdlib_module_names"):
            return True
    except Exception:
        pass
    return False


def _project_has_module(root: str, mod: str) -> bool:
    # Check if module path exists in project tree
    parts = mod.split(".")
    p1 = os.path.join(root, *parts) + ".py"
    p2 = os.path.join(root, *parts, "__init__.py")
    return os.path.isfile(p1) or os.path.isfile(p2)


def _missing_for_text(root: str, text: str) -> List[str]:
    missing: List[str] = []
    try:
        m = ast.parse(text or "")
    except Exception:
        # Syntax errors are handled elsewhere; skip dep analysis
        return []
    mods: List[str] = []
    for n in ast.walk(m):
        if isinstance(n, ast.Import):
            for alias in n.names:
                mods.append(_top_module(alias.name))
        elif isinstance(n, ast.ImportFrom):
            if n.level and n.level > 0:
                # relative import — assume project-local
                continue
            if n.module:
                mods.append(_top_module(n.module))
    seen: set[str] = set()
    for mod in mods:
        if not mod or mod in seen:
            continue
        seen.add(mod)
        if _is_std_or_builtin(mod):
            continue
        if _project_has_module(root, mod):
            continue
        try:
            if importlib.util.find_spec(mod) is not None:
                continue
        except Exception:
            pass
        missing.append(mod)
    return missing


async def discover_missing_deps(files: List[str]) -> List[str]:
    root = _resolve_root()
    acc: set[str] = set()
    for p in (files or []):
        try:
            if not str(p).endswith(".py"):
                continue
            # Attempt absolute read; fallback to project-relative
            text = await read_text_raw(p)
            if text == "":
                # maybe project-relative
                ap = os.path.join(root, p)
                text = await read_text_abs_thread(ap)
            if not text:
                continue
            missing = _missing_for_text(root, text)
            for m in missing:
                acc.add(m)
        except Exception:
            continue
    return sorted(acc)


async def ensure_requirements_updated(missing: List[str]) -> Tuple[bool, str]:
    """Append missing packages to requirements.txt if enabled.

    Controlled by JINX_REQS_AUTOWRITE (default: off).
    Returns (updated, message).
    """
    import os as _os
    on = str(_os.getenv("JINX_REQS_AUTOWRITE", "0")).lower() not in ("", "0", "false", "off", "no")
    if not on:
        return False, "autowrite disabled"
    root = _resolve_root()
    req_path = os.path.join(root, "requirements.txt")
    try:
        cur = await read_text_abs_thread(req_path)
    except Exception:
        cur = ""
    already = {ln.strip().split("==")[0] for ln in (cur or "").splitlines() if ln.strip() and not ln.strip().startswith("#")}
    add = [m for m in (missing or []) if m not in already]
    if not add:
        return False, "no new packages"
    new = (cur or "").rstrip() + ("\n" if cur and not cur.endswith("\n") else "") + "\n".join(add) + "\n"
    try:
        await write_text(req_path, new)
    except Exception as e:
        return False, f"failed to update requirements.txt: {e}"
    return True, f"requirements.txt updated: {', '.join(add)}"
