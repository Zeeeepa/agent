"""Banner service.

Responsible for rendering the startup banner. Kept separate for clarity and
potential future customization (branding, version info, etc.).
"""

from __future__ import annotations

from jinx.bootstrap import ensure_optional

art = ensure_optional(["art"])["art"]  # type: ignore


def show_banner() -> None:
    """Render the startup banner."""
    art.tprint("Jinx", "random")
