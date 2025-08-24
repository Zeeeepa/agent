"""Package entrypoint for `python -m jinx`. Delegates to `orchestrator.main`."""

from __future__ import annotations

from .orchestrator import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
