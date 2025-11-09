from __future__ import annotations

import asyncio
from typing import Optional


def preview_rename_text(src: str, old_name: str, new_name: str) -> str:
    try:
        import libcst as cst  # type: ignore
    except Exception:
        # If libcst is not present, return original to avoid breaking
        return src

    class Renamer(cst.CSTTransformer):
        def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:  # type: ignore[override]
            if original_node.value == old_name:
                return updated_node.with_changes(value=new_name)
            return updated_node

        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:  # type: ignore[override]
            try:
                if original_node.name.value == old_name:
                    return updated_node.with_changes(name=cst.Name(new_name))
            except Exception:
                pass
            return updated_node

        def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:  # type: ignore[override]
            try:
                if original_node.name.value == old_name:
                    return updated_node.with_changes(name=cst.Name(new_name))
            except Exception:
                pass
            return updated_node

    try:
        mod = cst.parse_module(src)
        out = mod.visit(Renamer())
        return out.code
    except Exception:
        return src


async def rename_symbol_file(path: str, *, old_name: str, new_name: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return False
    dst = await asyncio.to_thread(preview_rename_text, src, old_name, new_name)
    if dst == src:
        # Treat as success if nothing changed but target may not exist in file
        return True
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(dst)
        return True
    except Exception:
        return False
