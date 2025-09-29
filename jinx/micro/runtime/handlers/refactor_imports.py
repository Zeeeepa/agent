from __future__ import annotations

import os
from typing import List, Dict

from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT, EXCLUDE_DIRS
from .refactor_utils import _read


async def scan_and_rewrite_imports(old_mod: str, new_mod: str, symbol: str, *, grouped_enabled: bool = True) -> List[Dict[str, object]]:
    """Conservatively rewrite 'from old_mod import symbol' to new_mod across project.

    Only rewrites lines where the imported list is a single name (optionally 'as alias'), to avoid
    changing imports of other symbols from old_mod. Preserves alias. When grouped imports are enabled
    via env in the calling layer, a single line may be split into two lines to move our symbol.
    """
    root = os.path.normpath(PROJECT_ROOT or os.getcwd())
    ops: List[Dict[str, object]] = []
    exclude = set(EXCLUDE_DIRS)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in exclude]
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
            i = 0
            while i < len(lines):
                ln = lines[i]
                if not ln.lstrip().startswith("from "):
                    i += 1
                    continue
                if f"from {old_mod} import" not in ln:
                    i += 1
                    continue
                try:
                    before, _, after = ln.partition(f"from {old_mod} import ")
                    if not after:
                        i += 1
                        continue
                    code_part, *_ = after.split("#", 1)
                    names_raw = code_part.strip()
                    parts = [p.strip() for p in names_raw.split(",") if p.strip()]
                    if len(parts) == 1:
                        entry = parts[0]
                        base = entry.split(" as ")[0].strip()
                        if base != symbol:
                            i += 1
                            continue
                        new_line = before + f"from {new_mod} import " + entry
                        if new_line != ln:
                            lines[i] = new_line
                            changed = True
                        i += 1
                        continue
                    # grouped import: leave decision to caller (they enabled this mode)
                    if not grouped_enabled:
                        i += 1
                        continue
                    # keep others in old_mod, move our entry to new_mod on a new line
                    found_idx = -1
                    for j, p in enumerate(parts):
                        if p.split(" as ")[0].strip() == symbol:
                            found_idx = j
                            break
                    if found_idx < 0:
                        i += 1
                        continue
                    move_entry = parts[found_idx]
                    keep_parts = [p for j, p in enumerate(parts) if j != found_idx]
                    indent = before
                    new_lines_block = [indent + f"from {new_mod} import {move_entry}"]
                    if keep_parts:
                        new_lines_block.append(indent + f"from {old_mod} import {', '.join(keep_parts)}")
                    # replace one line with up to two lines
                    lines[i:i+1] = new_lines_block
                    changed = True
                    i += len(new_lines_block)
                    continue
                except Exception:
                    i += 1
                    continue
            if changed:
                new_text = ("\n".join(lines)).rstrip("\n") + "\n"
                ops.append({"type": "write", "path": ap, "code": new_text})
    return ops
