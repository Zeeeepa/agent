from __future__ import annotations

import os
import datetime
from typing import Optional

from jinx.async_utils.fs import write_text
import aiofiles


async def write_openai_request_dump(
    target_dir: str,
    kind: str,
    instructions: str,
    input_text: str,
    model: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Write a single OpenAI request dump to a uniquely named file.

    Parameters
    ----------
    target_dir : str
        Directory to store the dump files (created if missing).
    kind : str
        Label for the request type, e.g. "GENERAL" or "MEMORY".
    instructions : str
        The instructions/prompt header sent to the API.
    input_text : str
        The input/payload sent to the API.
    model : Optional[str]
        Optional model name to include in the dump header.
    suffix : Optional[str]
        Optional suffix added to the filename (before extension).

    Returns
    -------
    str
        The full path to the written dump file.
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
        base = ts if not suffix else f"{ts}_{suffix}"
        path = os.path.join(target_dir, f"{base}.txt")
        header = [
            f"===== {kind} REQUEST START =====\n",
            f"time_utc: {ts}\n",
        ]
        if model:
            header.append(f"model: {model}\n")
        content = "".join(
            header
            + [
                "[instructions]\n",
                instructions,
                "\n[input]\n",
                input_text,
                f"\n===== {kind} REQUEST END =====\n",
            ]
        )
        await write_text(path, content)
        return path
    except Exception:
        # Best-effort; swallow I/O errors
        return ""


async def _append_text(path: str, text: str) -> None:
    """Append arbitrary text to a file, creating parent directories.

    Best-effort semantics: swallow I/O errors to avoid surfacing logging failures.
    """
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(text or "")
    except Exception:
        pass


async def write_openai_response_append(path: str, kind: str, output_text: str) -> bool:
    """Append a response section to an existing OpenAI request dump file.

    Parameters
    ----------
    path : str
        Path returned by `write_openai_request_dump`.
    kind : str
        Label corresponding to the request (e.g., "GENERAL" or "MEMORY").
    output_text : str
        The model output to append.

    Returns
    -------
    bool
        True on best-effort attempt (file may be empty path), False if skipped.
    """
    if not path:
        return False
    try:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
        content = (
            f"\n===== {kind} RESPONSE START =====\n"
            f"time_utc: {ts}\n"
            f"[output]\n"
            f"{output_text or ''}\n"
            f"===== {kind} RESPONSE END =====\n"
        )
        await _append_text(path, content)
        return True
    except Exception:
        return False
