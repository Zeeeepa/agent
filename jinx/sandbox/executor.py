from __future__ import annotations

import io
import os
import contextlib
import traceback
from typing import Any, Optional
from jinx.text_service import slice_fuse


def blast_zone(
    x: str,
    stack: dict[str, Any],
    shrap: "multiprocessing.managers.DictProxy",
    log_path: Optional[str] = None,
) -> None:
    """Execute code in a clean globals dict and capture output/errors.

    If ``log_path`` is provided, stream stdout/stderr directly to the file so
    long-running programs record progress incrementally. A short slice of the
    final output is still returned via ``shrap['output']`` for summary.
    """
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8", buffering=1) as f:
            buf = io.StringIO()
            # Tee output to both file (for full history) and buffer (for summary)
            class Tee(io.TextIOBase):
                def write(self, s: str) -> int:  # type: ignore[override]
                    f.write(s)
                    return buf.write(s)

                def flush(self) -> None:  # type: ignore[override]
                    f.flush()
                    buf.flush()

            tee = Tee()
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                try:
                    exec(x, stack)
                    shrap["error"] = None
                except Exception:
                    shrap["error"] = traceback.format_exc()
            shrap["output"] = slice_fuse(buf.getvalue())
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(x, stack)
                shrap["error"] = None
            except Exception:
                shrap["error"] = traceback.format_exc()
        shrap["output"] = slice_fuse(buf.getvalue())
