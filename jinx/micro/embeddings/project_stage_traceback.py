from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Tuple, Pattern
from functools import lru_cache
from dataclasses import dataclass

from .project_config import ROOT


@dataclass(frozen=True)
class TracebackPattern:
    """Configurable traceback pattern with metadata."""
    name: str
    pattern: str
    priority: int = 1  # Higher = more reliable


class TracebackParser:
    """Advanced traceback parsing with extensible patterns."""
    
    def __init__(self):
        self._patterns: List[Tuple[int, Pattern[str]]] = []
        self._setup_default_patterns()
    
    def _setup_default_patterns(self) -> None:
        """Initialize default traceback patterns."""
        default_patterns = [
            # CPython standard traceback format (highest priority)
            TracebackPattern(
                "cpython_std",
                r'File\s+"(?P<path>[^"]+)"\s*,\s*line\s+(?P<line>\d+)',
                priority=10
            ),
            # Pytest/pytest-cov style
            TracebackPattern(
                "pytest",
                r'(?P<path>[^\s:<>"\']+\.py):(?P<line>\d+):',
                priority=9
            ),
            # filename.py:123 style (common in linters/editors)
            TracebackPattern(
                "file_line",
                r'(?P<path>[^\s:<>"\']+\.py)[:\(](?P<line>\d+)\)?',
                priority=7
            ),
            # Stack trace with 'at' keyword
            TracebackPattern(
                "at_style",
                r'at\s+(?P<path>[^\s:<>"\']+\.py):(?P<line>\d+)',
                priority=8
            ),
            # VSCode/IDE format
            TracebackPattern(
                "vscode",
                r'(?P<path>[a-zA-Z]:[^:]+\.py):(?P<line>\d+):',
                priority=6
            ),
        ]
        
        for pattern_def in default_patterns:
            self._add_pattern(pattern_def)
        
        # Sort by priority (descending)
        self._patterns.sort(key=lambda x: x[0], reverse=True)
    
    def _add_pattern(self, pattern_def: TracebackPattern) -> None:
        """Add a traceback pattern with compilation."""
        try:
            compiled = re.compile(pattern_def.pattern, re.IGNORECASE)
            self._patterns.append((pattern_def.priority, compiled))
        except re.error:
            # Skip invalid patterns
            pass
    
    def extract_frames(self, text: str) -> List[Tuple[str, int]]:
        """Extract file/line frames from traceback text."""
        frames: List[Tuple[str, int]] = []
        if not text:
            return frames
        
        # Try patterns in priority order
        for priority, pattern in self._patterns:
            for match in pattern.finditer(text):
                try:
                    path = _norm_path(match.group("path") or "")
                    line = int(match.group("line"))
                    if path and line > 0:
                        frames.append((path, line))
                except (ValueError, KeyError, IndexError):
                    continue
        
        # Deduplicate preserving order
        seen: set[Tuple[str, int]] = set()
        unique_frames: List[Tuple[str, int]] = []
        for frame in frames:
            if frame not in seen:
                seen.add(frame)
                unique_frames.append(frame)
        
        return unique_frames[:4]


# Global parser instance
_parser = TracebackParser()

# Legacy compatibility
_TB_FILE_PATTERNS = [pat for _, pat in _parser._patterns]


@lru_cache(maxsize=2048)
def _norm_path(p: str) -> str:
    """Normalize and relativize path with caching."""
    p = (p or "").strip().strip("\u200b\ufeff")
    if not p:
        return ""
    
    # Normalize separators
    p = p.replace("\\", os.sep).replace("/", os.sep)
    
    # Remove quotes that might have leaked through
    p = p.strip('"\'')
    
    # If absolute under ROOT, make it relative
    try:
        ap = os.path.abspath(p)
        ar = os.path.abspath(ROOT)
        if ap.startswith(ar + os.sep) or ap.startswith(ar + "/"):
            return os.path.relpath(ap, ar)
    except Exception:
        pass
    
    # If already relative from ROOT
    return p


def _extract_frames(q: str) -> List[Tuple[str, int]]:
    """Extract file/line frames from traceback text (legacy wrapper)."""
    return _parser.extract_frames(q)


def stage_traceback_hits(query: str, k: int, *, max_time_ms: int | None = 100) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: parse Python traceback-like text and return precise file windows.

    Extremely precise when users paste error logs with file/line info.
    """
    q = (query or "").strip()
    if not q:
        return []
    frames = _extract_frames(q)
    if not frames:
        return []

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    for rel_p, ln in frames:
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            break
        abs_p = os.path.join(ROOT, rel_p)
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if not text:
            continue
        lines = text.splitlines()
        a = max(1, ln - 12)
        b = min(len(lines), ln + 12)
        preview = "\n".join(lines[a - 1 : b]).strip()
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": preview,
                "line_start": a,
                "line_end": b,
            },
        }
        # Highest precision among heuristics
        hits.append((0.996, rel_p, obj))
        if len(hits) >= k:
            break
    return hits


__all__ = ["stage_traceback_hits"]
