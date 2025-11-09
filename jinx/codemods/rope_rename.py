from __future__ import annotations

"""
Optional project-wide rename via rope. If rope is unavailable, returns False.

Usage:
  await project_rename_symbol(root, module_rel, old_name, new_name)
"""

import asyncio
from typing import Optional
import os


async def project_rename_symbol(root: str, module_rel: str, *, old_name: str, new_name: str) -> bool:
    try:
        from rope.base.project import Project  # type: ignore
        from rope.base.libutils import path_to_resource  # type: ignore
        from rope.refactor.rename import Rename  # type: ignore
    except Exception:
        return False
    try:
        proj = Project(root)
        try:
            abs_path = os.path.join(root, module_rel)
            res = path_to_resource(proj, abs_path)
            renamer = Rename(proj, res, offset=None)
            # Run in thread to avoid blocking loop
            def _do() -> bool:
                try:
                    changes = renamer.get_changes(new_name)
                    proj.do(changes)
                    return True
                except Exception:
                    return False
            ok = await asyncio.to_thread(_do)
            proj.close()
            return ok
        except Exception:
            try:
                proj.close()
            except Exception:
                pass
            return False
    except Exception:
        return False
