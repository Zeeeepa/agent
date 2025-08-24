from __future__ import annotations

from typing import Iterable


def append_and_trim(path: str, text: str, keep_lines: int = 500) -> None:
    """Append text to transcript and trim file to last ``keep_lines`` lines.

    Pure, no locking; caller is responsible for synchronization.
    """
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            lines = []
        lines = lines + ["", text]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[-keep_lines:]) + "\n")
    except Exception:
        # Best-effort; swallow I/O errors to mirror existing semantics
        pass
