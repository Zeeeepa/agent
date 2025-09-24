from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List, Tuple

from .project_paths import ensure_project_dirs, PROJECT_INDEX_DIR, safe_rel_path
from .project_hashdb import load_hash_db, save_hash_db, get_record
from .project_config import (
    ENABLE,
    ROOT,
    SCAN_INTERVAL_MS,
    MAX_CONCURRENCY,
    USE_WATCHDOG,
    MAX_FILE_BYTES,
    RECONCILE_SEC,
    INCLUDE_EXTS,
    EXCLUDE_DIRS,
)
from .project_iter import iter_candidate_files
from .project_prune import prune_deleted, prune_single
from .project_watch import try_start_watch, drain_queue, WatchHandle
from .project_tasks import embed_if_changed
from .project_util import file_should_include




class ProjectEmbeddingsService:
    def __init__(self, *, root: str | None = None) -> None:
        self.root = os.path.abspath(root or ROOT)
        self._task: asyncio.Task | None = None
        self._use_watchdog = USE_WATCHDOG
        self._watch: WatchHandle | None = None

    async def run(self) -> None:
        if not ENABLE:
            return
        ensure_project_dirs()
        db = load_hash_db()
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        # Initial delay to let app start
        await asyncio.sleep(0.2)

        # One initial full scan to build baseline
        mutated = False
        tasks0: List[asyncio.Task] = []
        for abs_p, rel_p in iter_candidate_files(
            self.root,
            include_exts=INCLUDE_EXTS,
            exclude_dirs=EXCLUDE_DIRS,
            max_file_bytes=MAX_FILE_BYTES,
        ):
            # Prefilter by mtime to avoid unnecessary tasks
            try:
                st = os.stat(abs_p)
                mtime = float(st.st_mtime)
            except FileNotFoundError:
                continue
            rec = get_record(db, rel_p) or {}
            prev_m = float(rec.get("mtime", 0.0) or 0.0)
            # Also schedule if artifacts missing
            safe = safe_rel_path(rel_p)
            index_path = os.path.join(PROJECT_INDEX_DIR, f"{safe}.json")
            artifacts_ok = os.path.exists(index_path)
            if prev_m >= mtime and rec.get("sha") and artifacts_ok:
                continue
            tasks0.append(asyncio.create_task(embed_if_changed(db, abs_p, rel_p, sem=sem)))
        if tasks0:
            for t in asyncio.as_completed(tasks0):
                try:
                    if await t:
                        mutated = True
                except Exception:
                    pass
        if prune_deleted(self.root, db):
            mutated = True
        if mutated:
            save_hash_db(db)

        # Try to start watchdog-based watcher
        watcher_ok = False
        changes_q: asyncio.Queue[Tuple[str, str]] = asyncio.Queue(maxsize=1000)  # (event, abs_path)
        if self._use_watchdog:
            try:
                loop = asyncio.get_running_loop()
                handle = try_start_watch(self.root, loop, changes_q)
                if handle is not None:
                    watcher_ok = True
                    self._watch = handle
            except Exception:
                watcher_ok = False

        if watcher_ok:
            # Event-driven loop
            try:
                last_reconcile = time.time()
                while True:
                    # Batch events for a short period to coalesce bursts
                    await asyncio.sleep(0.15)
                    drained = drain_queue(changes_q)
                    mutated = False
                    # Periodic reconcile even if queue is empty
                    now = time.time()
                    need_reconcile = RECONCILE_SEC > 0 and (now - last_reconcile) >= RECONCILE_SEC
                    if not drained and not need_reconcile:
                        continue
                    mutated = False
                    seen_paths: set[str] = set()
                    for ev, abs_path in drained:
                        if not abs_path:
                            continue
                        # Normalize and filter include/exclude
                        abs_p = os.path.abspath(abs_path)
                        try:
                            rel_p = os.path.relpath(abs_p, start=self.root)
                        except Exception:
                            continue
                        if rel_p in seen_paths:
                            continue
                        seen_paths.add(rel_p)
                        if ev == "deleted":
                            if prune_single(self.root, db, rel_p):
                                mutated = True
                            continue
                        # Re-apply filters for events (extension and exclude dirs), then quick size check
                        if not file_should_include(
                            abs_p,
                            include_exts=INCLUDE_EXTS,
                            exclude_dirs=EXCLUDE_DIRS,
                        ):
                            continue
                        try:
                            if os.path.getsize(abs_p) > MAX_FILE_BYTES:
                                continue
                        except Exception:
                            continue
                        try:
                            if await embed_if_changed(db, abs_p, rel_p, sem=sem):
                                mutated = True
                        except Exception:
                            pass
                    if mutated:
                        save_hash_db(db)
                        last_reconcile = time.time()

                    # Reconcile pass (scan mtime & missing artifacts) on schedule
                    if need_reconcile:
                        recon_tasks: List[asyncio.Task] = []
                        for abs_p, rel_p in iter_candidate_files(
                            self.root,
                            include_exts=INCLUDE_EXTS,
                            exclude_dirs=EXCLUDE_DIRS,
                            max_file_bytes=MAX_FILE_BYTES,
                        ):
                            try:
                                st = os.stat(abs_p)
                                mtime = float(st.st_mtime)
                            except FileNotFoundError:
                                continue
                            rec = get_record(db, rel_p) or {}
                            prev_m = float(rec.get("mtime", 0.0) or 0.0)
                            safe = safe_rel_path(rel_p)
                            index_path = os.path.join(PROJECT_INDEX_DIR, f"{safe}.json")
                            artifacts_ok = os.path.exists(index_path)
                            if prev_m >= mtime and rec.get("sha") and artifacts_ok:
                                continue
                            recon_tasks.append(asyncio.create_task(embed_if_changed(db, abs_p, rel_p, sem=sem)))
                        if recon_tasks:
                            for t in asyncio.as_completed(recon_tasks):
                                try:
                                    if await t:
                                        mutated = True
                                except Exception:
                                    pass
                        if prune_deleted(self.root, db):
                            mutated = True
                        if mutated:
                            save_hash_db(db)
                        last_reconcile = time.time()
            finally:
                # Stop watchdog observer on exit
                if self._watch is not None:
                    self._watch.stop()
        else:
            # Fallback to periodic scanning loop
            while True:
                t0 = time.perf_counter()
                mutated = False
                scan_tasks: List[asyncio.Task] = []
                for abs_p, rel_p in iter_candidate_files(
                    self.root,
                    include_exts=INCLUDE_EXTS,
                    exclude_dirs=EXCLUDE_DIRS,
                    max_file_bytes=MAX_FILE_BYTES,
                ):
                    try:
                        st = os.stat(abs_p)
                        mtime = float(st.st_mtime)
                    except FileNotFoundError:
                        continue
                    rec = get_record(db, rel_p) or {}
                    prev_m = float(rec.get("mtime", 0.0) or 0.0)
                    safe = safe_rel_path(rel_p)
                    index_path = os.path.join(PROJECT_INDEX_DIR, f"{safe}.json")
                    artifacts_ok = os.path.exists(index_path)
                    if prev_m >= mtime and rec.get("sha") and artifacts_ok:
                        continue
                    scan_tasks.append(asyncio.create_task(embed_if_changed(db, abs_p, rel_p, sem=sem)))
                if scan_tasks:
                    for t in asyncio.as_completed(scan_tasks):
                        try:
                            if await t:
                                mutated = True
                        except Exception:
                            # best-effort
                            pass
                if prune_deleted(self.root, db):
                    mutated = True
                if mutated:
                    save_hash_db(db)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                wait_ms = max(50.0, SCAN_INTERVAL_MS - elapsed_ms)
                await asyncio.sleep(wait_ms / 1000.0)


def start_project_embeddings_task(root: str | None = None) -> asyncio.Task[None]:
    svc = ProjectEmbeddingsService(root=root)
    return asyncio.create_task(svc.run(), name="project-embeddings-service")
