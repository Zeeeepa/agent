from __future__ import annotations

from typing import Any, Dict
import inspect

try:
    from jinx.logging_service import bomb_log as _bomb
except Exception:  # pragma: no cover
    _bomb = None


def _fmt(msg: str, fields: Dict[str, Any] | None) -> str:
    if not fields:
        return msg
    kv = []
    for k, v in fields.items():
        try:
            kv.append(f"{k}={v}")
        except Exception:
            kv.append(f"{k}=<err>")
    return f"{msg} | " + ", ".join(kv)


def log_info(msg: str, **fields: Any) -> None:
    s = _fmt(msg, fields)
    if _bomb:
        try:
            import jinx.log_paths as _lp
            # If bomb_log is async, schedule it without awaiting to avoid warnings
            if inspect.iscoroutinefunction(_bomb):  # type: ignore[arg-type]
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    loop.create_task(_bomb(s, _lp.BLUE_WHISPERS))  # type: ignore[misc]
                    return
                except RuntimeError:
                    # No running loop: run in a background daemon thread
                    import threading, asyncio
                    def _runner() -> None:
                        try:
                            asyncio.run(_bomb(s, _lp.BLUE_WHISPERS))  # type: ignore[misc]
                        except Exception:
                            pass
                    threading.Thread(target=_runner, daemon=True).start()
                    return
            else:
                # Synchronous implementation
                _bomb(s, _lp.BLUE_WHISPERS)  # type: ignore[misc]
                return
        except Exception:
            pass
    print(s)


def log_warn(msg: str, **fields: Any) -> None:
    log_info("WARN: " + msg, **fields)


def log_error(msg: str, **fields: Any) -> None:
    log_info("ERROR: " + msg, **fields)
