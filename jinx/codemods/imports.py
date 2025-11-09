from __future__ import annotations

import asyncio
from typing import Optional


def _add_import_code(src: str, module: str, name: Optional[str] = None, alias: Optional[str] = None) -> str:
    try:
        import libcst as cst  # type: ignore
        from libcst import RemovalSentinel as _RS  # type: ignore
    except Exception:
        return src

    class Inserter(cst.CSTTransformer):
        def __init__(self) -> None:
            self.inserted = False

        def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.CSTNode:  # type: ignore[override]
            if self.inserted:
                return updated_node
            if name:
                alias_obj = cst.AsName(name=cst.Name(alias)) if alias else None
                imp = cst.SimpleStatementLine(
                    body=[
                        cst.ImportFrom(
                            module=cst.Name(module),
                            names=[cst.ImportAlias(name=cst.Name(name), asname=alias_obj)],
                        )
                    ]
                )
            else:
                alias_obj = cst.AsName(name=cst.Name(alias)) if alias else None
                imp = cst.SimpleStatementLine(
                    body=[
                        cst.Import(names=[cst.ImportAlias(cst.Name(module), asname=alias_obj)])
                    ]
                )
            self.inserted = True
            return updated_node.with_changes(body=[imp] + list(updated_node.body))

    try:
        mod = cst.parse_module(src)
        out = mod.visit(Inserter())
        return out.code
    except Exception:
        return src


def _replace_import_code(src: str, old_module: str, new_module: str) -> str:
    try:
        import libcst as cst  # type: ignore
    except Exception:
        return src

    class Replacer(cst.CSTTransformer):
        def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.CSTNode:  # type: ignore[override]
            names = []
            for a in updated_node.names:
                nm = getattr(a, "name", None)
                if isinstance(nm, cst.Name) and nm.value == old_module:
                    names.append(a.with_changes(name=cst.Name(new_module)))
                else:
                    names.append(a)
            return updated_node.with_changes(names=names)

        def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.CSTNode:  # type: ignore[override]
            mod = getattr(updated_node, "module", None)
            if isinstance(mod, cst.Name) and mod.value == old_module:
                return updated_node.with_changes(module=cst.Name(new_module))
            return updated_node

    try:
        mod = cst.parse_module(src)
        out = mod.visit(Replacer())
        return out.code
    except Exception:
        return src


async def add_import_to_file(path: str, module: str, *, name: Optional[str] = None, alias: Optional[str] = None) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return False
    dst = await asyncio.to_thread(_add_import_code, src, module, name, alias)
    if dst == src:
        return True
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(dst)
        return True
    except Exception:
        return False


async def replace_import_in_file(path: str, old_module: str, new_module: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return False
    dst = await asyncio.to_thread(_replace_import_code, src, old_module, new_module)
    if dst == src:
        return True
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(dst)
        return True
    except Exception:
        return False
