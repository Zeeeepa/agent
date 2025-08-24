"""Configuration and tagging utilities."""

from __future__ import annotations

import platform, getpass, time

# Tags used by parser_service and openai_service outputs
CODE_TAGS = {"python", "python_question"}
ALL_TAGS = {"machine", *CODE_TAGS}


# Active prompt selection (None -> let prompts.get_prompt() resolve via env/default)
PROMPT_NAME: str | None = "burning_logic"


def set_prompt(name: str | None) -> None:
    """Set active prompt name (e.g., "burning_logic", "chaos_bloom").

    Pass None or empty string to defer to environment/default resolution.
    """
    global PROMPT_NAME
    PROMPT_NAME = (name or "").strip().lower() or None


def neon_stat() -> dict[str, str]:
    """Return a snapshot of host identity for instruction headers."""
    return dict(
        os=platform.system() + " " + platform.release(),
        arch=platform.machine(),
        host=platform.node(),
        user=getpass.getuser(),
    )


def jinx_tag() -> tuple[str, dict[str, dict[str, str]]]:
    """Return a unique fuse id and corresponding start/end tags for all blocks."""
    fuse = str(int(time.time()))
    flames = {b: dict(start=f"<{b}_{fuse}>\n", end=f"</{b}_{fuse}>") for b in ALL_TAGS}
    return fuse, flames
