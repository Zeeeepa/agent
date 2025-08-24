from __future__ import annotations

import io
import contextlib
import traceback
from typing import Any
from jinx.text_service import slice_fuse


def blast_zone(x: str, stack: dict[str, Any], shrap: "multiprocessing.managers.DictProxy") -> None:
    """Execute code in a clean globals dict and capture output/errors.

    Pure function suitable as a multiprocessing target.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(x, stack)
            shrap["error"] = None
        except Exception:
            shrap["error"] = traceback.format_exc()
    shrap["output"] = slice_fuse(buf.getvalue())
