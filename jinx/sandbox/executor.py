from __future__ import annotations

import io
import os
import contextlib
import traceback
import multiprocessing  # needed for type annotations resolution
from multiprocessing.managers import DictProxy as MPDictProxy
from typing import Any, Optional, TextIO, cast
from jinx.text_service import slice_fuse
try:
    # Best-effort: apply per-run resource limits (POSIX only)
    from jinx.sandbox.policy import select_limits, apply_limits  # type: ignore
except Exception:  # pragma: no cover - import-safe
    select_limits = None  # type: ignore
    apply_limits = None  # type: ignore


def blast_zone(
    x: str,
    stack: dict[str, Any],
    shrap: MPDictProxy,
    log_path: Optional[str] = None,
) -> None:
    """Execute code in a clean globals dict and capture output/errors.

    If ``log_path`` is provided, stream stdout/stderr directly to the file so
    long-running programs record progress incrementally. A short slice of the
    final output is still returned via ``shrap['output']`` for summary.
    """
    # Apply resource limits at the beginning of the child process (no-op on Windows)
    try:
        if apply_limits and select_limits:
            # Kind can be controlled via env JINX_SANDBOX_POLICY
            lim = select_limits(os.getenv("JINX_SANDBOX_POLICY", None))  # type: ignore[call-arg]
            apply_limits(lim)  # type: ignore[misc]
    except Exception:
        pass

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8", buffering=1) as f:
            buf = io.StringIO()
            # Tee output to both file (for full history) and buffer (for summary)
            class Tee(io.TextIOBase):
                def __init__(self, file_obj: io.TextIOBase, buffer: io.StringIO, encoding: str = "utf-8") -> None:
                    self._file = file_obj
                    self._buf = buffer
                    self._enc = encoding

                # IO[str] protocol methods
                def write(self, s: str) -> int:  # type: ignore[override]
                    self._file.write(s)
                    return self._buf.write(s)

                def flush(self) -> None:  # type: ignore[override]
                    self._file.flush()
                    self._buf.flush()

                @property
                def encoding(self) -> str:  # type: ignore[override]
                    return self._enc

                def readable(self) -> bool:  # type: ignore[override]
                    return False

                def writable(self) -> bool:  # type: ignore[override]
                    return True

                def seekable(self) -> bool:  # type: ignore[override]
                    return False

                def close(self) -> None:  # type: ignore[override]
                    try:
                        self._file.flush()
                        self._buf.flush()
                    except Exception:
                        pass

            tee: TextIO = cast(TextIO, Tee(f, buf))
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
