from __future__ import annotations

from typing import List, Iterable, Optional
import re

# Utilities to parse missing internal module errors and schedule repairs.

_PAT_NO_MODULE = re.compile(r"No module named '([^']+)'", re.IGNORECASE)
_PAT_CANNOT_IMPORT = re.compile(r"cannot import name .* from '([^']+)'", re.IGNORECASE)


def parse_missing_jinx_modules(text: str) -> List[str]:
    mods: List[str] = []
    s = (text or "").strip()
    if not s:
        return mods
    for m in _PAT_NO_MODULE.finditer(s):
        name = (m.group(1) or '').strip()
        if name.startswith("jinx.") and name not in mods:
            mods.append(name)
    for m in _PAT_CANNOT_IMPORT.finditer(s):
        name = (m.group(1) or '').strip()
        if name.startswith("jinx.") and name not in mods:
            mods.append(name)
    return mods


async def schedule_repairs_for(mods: Iterable[str]) -> None:
    items = [m for m in (mods or []) if isinstance(m, str) and m.startswith("jinx.")]
    if not items:
        return
    try:
        from jinx.micro.runtime.api import submit_task as _submit
    except Exception:
        return
    for m in items:
        try:
            await _submit("repair.import_missing", module=m)
        except Exception:
            pass


async def maybe_schedule_repairs_from_error(err: object) -> None:
    try:
        s = str(err)
    except Exception:
        s = ""
    mods = parse_missing_jinx_modules(s)
    if mods:
        await schedule_repairs_for(mods)
