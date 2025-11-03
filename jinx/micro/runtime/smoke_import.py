from __future__ import annotations

import os
import sys
import importlib
import traceback
from typing import List, Tuple

from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root


def _to_module(rel_path: str) -> str | None:
    try:
        if not rel_path.endswith(".py"):
            return None
        rel = rel_path[:-3].replace("\\", "/")
        if rel.endswith("/__init__"):
            rel = rel[:-9]
        parts = [p for p in rel.split("/") if p]
        if not parts:
            return None
        return ".".join(parts)
    except Exception:
        return None


def _rel_under_root(p: str, root: str) -> str | None:
    try:
        ap = p if os.path.isabs(p) else os.path.join(root, p)
        ap = os.path.abspath(ap)
        root_abs = os.path.abspath(root)
        if not ap.startswith(root_abs):
            return None
        return os.path.relpath(ap, root_abs)
    except Exception:
        return None


def smoke_import_paths(files: List[str]) -> List[str]:
    """Attempt to import Python modules for given file paths; return error lines.

    - Converts project-relative paths to module names.
    - Temporarily prepends project root to sys.path.
    - Returns list of error strings per failed import.
    """
    root = _resolve_root()
    errs: List[str] = []
    # Prepare sys.path guard
    added = False
    if root and root not in sys.path:
        sys.path.insert(0, root)
        added = True
    try:
        for p in (files or []):
            try:
                rel = _rel_under_root(p, root)
                if not rel:
                    continue
                mod = _to_module(rel)
                if not mod:
                    continue
                try:
                    importlib.invalidate_caches()
                    importlib.import_module(mod)
                except BaseException as e:
                    tb = "".join(traceback.format_exception_only(type(e), e)).strip()
                    errs.append(f"import failed: {mod} ({rel}): {tb}")
            except Exception:
                continue
    finally:
        if added:
            try:
                sys.path.remove(root)
            except ValueError:
                pass
    return errs
