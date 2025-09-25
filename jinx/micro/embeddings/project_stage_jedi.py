from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Tuple

try:
    import jedi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    jedi = None  # type: ignore[assignment]

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files


def _extract_identifiers(q: str) -> List[str]:
    q = (q or "").strip()
    if not q:
        return []
    names: List[str] = []
    # function calls like foo(
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", q):
        name = m.group(1)
        if name and name not in names:
            names.append(name)
    # plain identifiers (variables, function names without parens)
    for m in re.finditer(r"\b([A-Za-z_]\w*)\b", q):
        name = m.group(1)
        if name and name not in names:
            names.append(name)
    return names


def stage_jedi_hits(query: str, k: int, *, max_time_ms: int | None = 220) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage -1.5: Jedi-based identifier/reference discovery (optional).

    If `jedi` is available, use it to scan Python files and find references of
    identifiers present in the query. Returns (score, file_rel, obj).
    """
    if jedi is None:
        return []

    t0 = time.perf_counter()
    cand = _extract_identifiers(query)
    if not cand:
        return []

    hits: List[Tuple[float, str, Dict[str, Any]]] = []
    seen: set[str] = set()

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    for abs_p, rel_p in iter_candidate_files(
        ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES
    ):
        if time_up():
            break
        if rel_p in seen:
            continue
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
        except Exception:
            continue
        if not src:
            continue
        try:
            script = jedi.Script(src, path=abs_p)  # type: ignore[arg-type]
            # Collect names seen in the file including references
            names = script.get_names(all_scopes=True, definitions=False, references=True)  # type: ignore[attr-defined]
        except Exception:
            continue
        # Gather candidate line numbers
        lines: List[int] = []
        try:
            for nm in names:
                nm_name = getattr(nm, "name", None)
                ln = getattr(nm, "line", None)
                if isinstance(nm_name, str) and isinstance(ln, int) and nm_name in cand:
                    lines.append(ln)
        except Exception:
            lines = []
        if not lines:
            continue
        seen.add(rel_p)
        # Window around the first occurrence
        lno = max(1, lines[0])
        all_lines = src.splitlines()
        a = max(1, lno - 12)
        b = min(len(all_lines), lno + 12)
        preview = "\n".join(all_lines[a - 1 : b]).strip()
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": preview,
                "line_start": a,
                "line_end": b,
            },
        }
        # Jedi is conservative; set score slightly below AST but above phrase hits
        hits.append((0.992, rel_p, obj))
        if len(hits) >= k:
            break
    return hits


__all__ = ["stage_jedi_hits"]
