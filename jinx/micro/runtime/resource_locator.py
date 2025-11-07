"""Resource Locator - Production file auto-discovery and resolution.

Features:
- Fast project-wide file index (cached to disk)
- Fuzzy name matching (handles typos: e.g., "фала" -> "файла" ignored)
- Extension-aware ranking (e.g., .py priority if mentioned)
- Optional semantic content signal (uses existing project chunk embeddings)
- Async, non-blocking; respects tight time budgets (hard RT)
- Micro-modular; no tight coupling to task classifier

Outputs stable, deterministic candidates suitable for automated flows.
"""

from __future__ import annotations

import asyncio
import os
import time
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque, defaultdict

# Optional fuzzy matching
try:
    from rapidfuzz import fuzz
    _HAS_FUZZ = True
except Exception:
    import difflib
    _HAS_FUZZ = False

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class FileEntry:
    path: str            # absolute path
    rel: str             # project-relative path
    name: str            # filename
    stem: str            # name without extension
    ext: str             # extension with dot
    size: int
    mtime: float


@dataclass
class ResolvedResource:
    path: str            # absolute path
    rel: str             # project-relative path
    score: float         # 0..1
    reason: str          # explanation of ranking


# -----------------------------------------------------------------------------
# Resource Locator
# -----------------------------------------------------------------------------

class ResourceLocator:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._initialized = False
        self._root: str = os.getcwd()
        self._index: List[FileEntry] = []
        self._name_map: Dict[str, List[int]] = defaultdict(list)  # stem -> indices
        self._cache_path: str = os.path.join('.jinx', 'project_index.json')
        self._max_index_files: int = 20000
        self._last_refresh: float = 0.0
        self._refresh_interval: float = 60.0  # seconds
        self._ignore_dirs = {'.git', '.hg', '.svn', '__pycache__', 'node_modules', '.venv', 'venv', '.jinx', '.idea', '.vscode'}
        self._include_exts = {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.kt', '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.sh', '.yml', '.yaml', '.toml', '.json', '.sql', '.md'}

    async def initialize(self, project_root: Optional[str] = None):
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            try:
                root = (project_root or os.getenv('JINX_PROJECT_ROOT') or os.getcwd()).strip()
                self._root = os.path.abspath(root)
                await self._load_index()
                if not self._index:
                    await self._build_index(time_budget_ms=500)
                self._initialized = True
            except Exception:
                # Ensure graceful fallback
                self._initialized = True

    # ------------------------ Index Build/Load -------------------------

    async def _load_index(self):
        try:
            if not os.path.exists(self._cache_path):
                return
            def _read():
                with open(self._cache_path, 'r', encoding='utf-8') as r:
                    return json.load(r)
            data = await asyncio.to_thread(_read)
            entries = []
            for item in data.get('files', []):
                try:
                    fe = FileEntry(**item)
                    entries.append(fe)
                except Exception:
                    continue
            self._index = entries
            self._rebuild_name_map()
        except Exception:
            self._index = []
            self._name_map.clear()

    async def _save_index(self):
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            payload = {'root': self._root, 'files': [asdict(e) for e in self._index]}
            def _write():
                with open(self._cache_path, 'w', encoding='utf-8') as w:
                    json.dump(payload, w, ensure_ascii=False)
            await asyncio.to_thread(_write)
        except Exception:
            pass

    def _rebuild_name_map(self):
        self._name_map.clear()
        for i, fe in enumerate(self._index):
            self._name_map[fe.stem].append(i)
            self._name_map[fe.name].append(i)

    async def _build_index(self, *, time_budget_ms: int = 500):
        """Scan project for files, respecting time budget and limits."""
        start = time.perf_counter()
        entries: List[FileEntry] = []
        count = 0
        for root, dirs, files in os.walk(self._root):
            # Prune ignored dirs in-place
            dirs[:] = [d for d in dirs if d not in self._ignore_dirs]
            for fn in files:
                if len(entries) >= self._max_index_files:
                    break
                ext = os.path.splitext(fn)[1].lower()
                if ext and (ext in self._include_exts or len(entries) < 2000):
                    abspath = os.path.join(root, fn)
                    try:
                        st = os.stat(abspath)
                        rel = os.path.relpath(abspath, self._root)
                        entries.append(FileEntry(
                            path=abspath,
                            rel=rel.replace('\\', '/'),
                            name=fn,
                            stem=os.path.splitext(fn)[0],
                            ext=ext,
                            size=st.st_size,
                            mtime=st.st_mtime,
                        ))
                    except Exception:
                        pass
                count += 1
                # Budget check
                if (time.perf_counter() - start) * 1000 > time_budget_ms:
                    break
            if (time.perf_counter() - start) * 1000 > time_budget_ms:
                break
        # Merge with existing (prefer newer mtime)
        if entries:
            existing: Dict[str, FileEntry] = {e.rel: e for e in self._index}
            for e in entries:
                prev = existing.get(e.rel)
                if not prev or e.mtime >= prev.mtime:
                    existing[e.rel] = e
            self._index = list(existing.values())
            self._rebuild_name_map()
            await self._save_index()
        self._last_refresh = time.time()

    async def refresh_index_if_needed(self):
        if time.time() - self._last_refresh > self._refresh_interval:
            await self._build_index(time_budget_ms=300)

    # ------------------------ Matching Logic ---------------------------

    def _extract_file_hints(self, text: str) -> List[str]:
        # Very light extraction of tokens resembling file names; language-agnostic
        # Accept tokens with dots or common separators
        hints: List[str] = []
        text_norm = (text or '').strip()
        # Split on whitespace and punctuation, keep tokens with a dot or with letters/digits
        tokens = re.split(r"[^\w\.\-/]+", text_norm)
        for tok in tokens:
            if not tok:
                continue
            if '.' in tok:
                # e.g., crawler.py, app.module.ts, path/to/file.py
                hints.append(tok)
            elif len(tok) >= 5:  # long stems e.g., crawler
                hints.append(tok)
        return hints[:10]

    def _fuzzy_score(self, a: str, b: str) -> float:
        a = a.lower(); b = b.lower()
        if a == b:
            return 1.0
        if _HAS_FUZZ:
            try:
                return float(fuzz.partial_ratio(a, b)) / 100.0
            except Exception:
                pass
        # Fallback to difflib similarity
        try:
            import difflib
            return difflib.SequenceMatcher(a=a, b=b).ratio()
        except Exception:
            return 0.0

    async def _semantic_signals(self, query: str, k: int = 20, max_time_ms: int = 200) -> Dict[str, float]:
        """Use existing chunk embeddings to get file-level semantic signals."""
        try:
            from jinx.micro.embeddings.vector_stage_semantic import semantic_search
            results = await semantic_search(query, k=k, max_time_ms=max_time_ms)
            # Aggregate best score per file
            per_file: Dict[str, float] = {}
            for score, file_rel, obj in results:
                if score <= 0:
                    continue
                if file_rel not in per_file or score > per_file[file_rel]:
                    per_file[file_rel] = float(score)
            return per_file
        except Exception:
            return {}

    async def locate(self, query: str, *, prefer_ext: Optional[str] = None, k: int = 5, budget_ms: int = 120) -> List[ResolvedResource]:
        """Resolve resources mentioned in a free-form query.

        Combines fuzzy filename matching with optional semantic content signals.
        Time-bounded for real-time responsiveness.
        """
        start = time.perf_counter()
        if not self._initialized:
            await self.initialize()
        await self.refresh_index_if_needed()

        # Extract hints
        hints = self._extract_file_hints(query)
        prefer_ext = (prefer_ext or '').lower().strip()

        # Precompute semantic map under tight budget (run concurrently)
        sem_task = asyncio.create_task(self._semantic_signals(query, k=30, max_time_ms=min(200, max(50, budget_ms // 2))))

        # Name-based candidates
        scores: Dict[int, float] = defaultdict(float)
        reasons: Dict[int, str] = {}

        for hint in hints:
            hint_norm = os.path.basename(hint).lower()
            hint_stem, hint_ext = os.path.splitext(hint_norm)
            for idx, fe in enumerate(self._index):
                # Quick filter: filename contains subset tokens
                base = fe.name.lower()
                if hint_stem and hint_stem in base:
                    s = self._fuzzy_score(hint_norm, base)
                    # Extension boost
                    if hint_ext and fe.ext == hint_ext:
                        s += 0.2
                    elif prefer_ext and fe.ext == prefer_ext:
                        s += 0.1
                    # Clamp
                    s = min(1.0, s)
                    if s > scores[idx]:
                        scores[idx] = s
                        reasons[idx] = f"name:{hint_norm}"
                # Also compare stem-to-stem
                if hint_stem:
                    s2 = self._fuzzy_score(hint_stem, fe.stem)
                    if prefer_ext and fe.ext == prefer_ext:
                        s2 += 0.1
                    s2 = min(1.0, s2)
                    if s2 > scores[idx]:
                        scores[idx] = s2
                        reasons[idx] = f"stem:{hint_stem}"
            # Budget check
            if (time.perf_counter() - start) * 1000 > budget_ms:
                break

        # Merge semantic signals
        try:
            sem_map = await sem_task
        except Exception:
            sem_map = {}

        if sem_map:
            # Map rel->idx
            rel_to_idx: Dict[str, int] = {fe.rel: i for i, fe in enumerate(self._index)}
            for rel, sem_score in sem_map.items():
                idx = rel_to_idx.get(rel)
                if idx is None:
                    continue
                # Blend: 0.7 name, 0.3 semantic (if name present), else use semantic alone
                name_score = scores.get(idx, 0.0)
                blended = max(name_score, 0.7 * name_score + 0.3 * sem_score)
                if blended > scores.get(idx, 0.0):
                    scores[idx] = blended
                    prev = reasons.get(idx, '')
                    reasons[idx] = (prev + ('+' if prev else '') + f"sem:{sem_score:.2f}")

        # Build result list
        items: List[Tuple[float, int]] = sorted(((sc, idx) for idx, sc in scores.items()), key=lambda x: x[0], reverse=True)
        results: List[ResolvedResource] = []
        for sc, idx in items[:k]:
            fe = self._index[idx]
            results.append(ResolvedResource(path=fe.path, rel=fe.rel, score=float(min(1.0, sc)), reason=reasons.get(idx, '')))

        # If no hints at all, fall back to pure semantic top files (if any)
        if not results and sem_map:
            sorted_sem = sorted(sem_map.items(), key=lambda x: x[1], reverse=True)[:k]
            rel_to_idx: Dict[str, int] = {fe.rel: i for i, fe in enumerate(self._index)}
            for rel, sem_score in sorted_sem:
                idx = rel_to_idx.get(rel)
                if idx is None:
                    continue
                fe = self._index[idx]
                results.append(ResolvedResource(path=fe.path, rel=fe.rel, score=float(min(1.0, sem_score)), reason=f"sem:{sem_score:.2f}"))

        return results


# Singleton
_locator: Optional[ResourceLocator] = None
_locator_lock = asyncio.Lock()


async def get_resource_locator() -> ResourceLocator:
    global _locator
    if _locator is None:
        async with _locator_lock:
            if _locator is None:
                _locator = ResourceLocator()
                await _locator.initialize()
    return _locator


__all__ = [
    "ResourceLocator",
    "ResolvedResource",
    "get_resource_locator",
]
