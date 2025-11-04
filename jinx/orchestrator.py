"""Top-level orchestrator.

This module exposes the synchronous ``main()`` function that boots the
asynchronous runtime loop via ``jinx.runtime_service.pulse_core``. Keeping this
adapter minimal ensures clean separation between synchronous CLI entrypoints
and the async runtime core.
"""

from __future__ import annotations

import asyncio
from jinx.bootstrap import load_env, ensure_optional


def main() -> None:
    """Run the async runtime loop and block until completion.

    This function is intentionally synchronous so it can be used directly from
    standard CLI entrypoints without requiring the caller to manage an event
    loop.
    """
    # Install crash diagnostics FIRST
    try:
        from jinx.micro.runtime.crash_diagnostics import install_crash_diagnostics, record_operation
        install_crash_diagnostics()
        record_operation("startup", details={'stage': 'orchestrator'}, success=True)
    except Exception:
        pass
    
    # Install shutdown event monitor
    try:
        import jinx.state as jx_state
        import traceback as tb
        from jinx.micro.logger.debug_logger import debug_log_sync
        
        original_set = jx_state.shutdown_event.set
        
        def monitored_set():
            try:
                debug_log_sync("="*70, "SHUTDOWN")
                debug_log_sync("shutdown_event.set() called!", "SHUTDOWN")
                debug_log_sync("="*70, "SHUTDOWN")
                debug_log_sync("Call stack:", "SHUTDOWN")
                for line in tb.format_stack()[:-1]:
                    debug_log_sync(line.strip(), "SHUTDOWN")
                debug_log_sync("="*70, "SHUTDOWN")
            except Exception:
                pass
            return original_set()
        
        jx_state.shutdown_event.set = monitored_set
    except Exception:
        pass
    
    # Ensure environment variables (e.g., OPENAI_API_KEY) are loaded from .env
    load_env()
    # Ensure runtime optional deps are present before importing runtime_service
    ensure_optional([
        "aiofiles",      # async file IO used by runtime
        "regex",         # fuzzy regex stage
        "rapidfuzz",     # fuzzy line matching
        "jedi",          # Python identifier references
        "libcst",        # CST structural patterns
        "astunparse",    # pretty-printing annotations (optional)
    ])

    # Defer import until after dependencies are ensured to avoid early import errors
    from jinx.runtime_service import pulse_core

    try:
        record_operation("runtime_start", success=True)
        asyncio.run(pulse_core())
        record_operation("runtime_end", success=True)
        
        # Mark as normal shutdown
        from jinx.micro.runtime.crash_diagnostics import mark_normal_shutdown
        mark_normal_shutdown("normal_completion")
    except KeyboardInterrupt:
        record_operation("runtime_interrupted", details={'reason': 'KeyboardInterrupt'}, success=True)
        from jinx.micro.runtime.crash_diagnostics import mark_normal_shutdown
        mark_normal_shutdown("keyboard_interrupt")
        raise
    except Exception as e:
        record_operation("runtime_error", details={'error': str(e)}, success=False, error=str(e))
        raise
