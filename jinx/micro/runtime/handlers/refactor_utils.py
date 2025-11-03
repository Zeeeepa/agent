from __future__ import annotations

import os
from typing import List

from jinx.async_utils.fs import read_text_raw
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT
from jinx.micro.common.env import truthy as _truthy


def _abs_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    base = PROJECT_ROOT or os.getcwd()
    return os.path.normpath(os.path.join(base, p))


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
