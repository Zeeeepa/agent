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


def _run() -> int:
    """Execute the agent runtime.

    Returns
    -------
    int
        Process exit code. ``0`` on success, non-zero on handled errors.
    """
    try:
        # Install resilient import hook before importing orchestrator/runtime
        try:
            from jinx.micro.runtime.resilience import install_resilience as _install
            _install()
        except Exception:
            pass
        # Avoid package shadowing when this script name matches the package name
        # Inject a lightweight package stub into sys.modules that points to the real package directory
        try:
            import os as _os, types as _types
            pkg_dir = _os.path.join(_os.path.dirname(__file__), "jinx")
            # Blue-green override: allow alternate package dir via env
            ov = (_os.getenv("JINX_PACKAGE_DIR") or "").strip()
            if ov and _os.path.isdir(ov):
                pkg_dir = ov
            if _os.path.isdir(pkg_dir):
                import sys as _sys
                if "jinx" not in _sys.modules:
                    _pkg = _types.ModuleType("jinx")
                    setattr(_pkg, "__path__", [pkg_dir])  # mark as package
                    _sys.modules["jinx"] = _pkg
        except Exception:
            pass
        # Kernel boot (resilience, env, auto-config, prewarm)
        try:
            from jinx.kernel import boot as _kernel_boot
            _kernel_boot()
        except Exception:
            pass
        # Lazy import orchestrator after resilience and anti-shadowing are in place
        from jinx.orchestrator import main as jinx_main
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
