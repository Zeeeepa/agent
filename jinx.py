"""Agent entrypoint.

This module serves as the CLI bootstrap for the Jinx agent runtime. It
delegates orchestration to ``jinx.orchestrator.main()`` and provides a thin
exception boundary suitable for production usage.

Design goals
------------
* Keep the entrypoint minimal and dependency-light.
* Provide predictable behavior on interrupts.
* Ensure non-zero exit codes on unexpected exceptions.
"""

from __future__ import annotations

import sys
from jinx.orchestrator import main as jinx_main


def _run() -> int:
    """Execute the agent runtime.

    Returns
    -------
    int
        Process exit code. ``0`` on success, non-zero on handled errors.
    """
    try:
        jinx_main()
        return 0
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        return 130  # Conventional exit code for SIGINT
    except Exception as exc:  # pragma: no cover - safety net
        # Last-resort guard to avoid silent crashes.
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_run())
