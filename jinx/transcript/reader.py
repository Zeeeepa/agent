from __future__ import annotations


def read_transcript(path: str) -> str:
    """Return the contents of the transcript file or empty string on error.

    Pure function: no locking here. Caller is responsible for synchronization.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""
