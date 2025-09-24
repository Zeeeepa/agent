from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple, Any


@dataclass
class WatchHandle:
    observer: Any

    def stop(self) -> None:
        try:
            # type: ignore[attr-defined]
            self.observer.stop()
            # type: ignore[attr-defined]
            self.observer.join(timeout=1.0)
        except Exception:
            pass


def try_start_watch(root: str, loop: asyncio.AbstractEventLoop, changes_q: "asyncio.Queue[tuple[str, str]]") -> Optional[WatchHandle]:
    """Try to start a watchdog-based file watcher.

    Returns a WatchHandle if started successfully, otherwise None.
    """
    try:
        importlib.import_module("watchdog")
        from watchdog.observers import Observer  # type: ignore
        from watchdog.events import FileSystemEventHandler  # type: ignore
    except Exception:
        return None

    class H(FileSystemEventHandler):  # type: ignore
        def on_modified(self, event):
            if not getattr(event, "is_directory", False):
                try:
                    loop.call_soon_threadsafe(lambda: changes_q.put_nowait(("modified", event.src_path)))
                except Exception:
                    pass

        def on_created(self, event):
            if not getattr(event, "is_directory", False):
                try:
                    loop.call_soon_threadsafe(lambda: changes_q.put_nowait(("created", event.src_path)))
                except Exception:
                    pass

        def on_moved(self, event):
            if not getattr(event, "is_directory", False):
                try:
                    loop.call_soon_threadsafe(lambda: changes_q.put_nowait(("deleted", event.src_path)))
                    loop.call_soon_threadsafe(lambda: changes_q.put_nowait(("created", event.dest_path)))
                except Exception:
                    pass

        def on_deleted(self, event):
            if not getattr(event, "is_directory", False):
                try:
                    loop.call_soon_threadsafe(lambda: changes_q.put_nowait(("deleted", event.src_path)))
                except Exception:
                    pass

    handler = H()
    observer = Observer()
    observer.schedule(handler, root, recursive=True)
    observer.daemon = True
    observer.start()
    return WatchHandle(observer=observer)


def drain_queue(q: "asyncio.Queue[tuple[str, str]]") -> list[tuple[str, str]]:
    drained: list[tuple[str, str]] = []
    while True:
        try:
            drained.append(q.get_nowait())
        except Exception:
            break
    return drained
