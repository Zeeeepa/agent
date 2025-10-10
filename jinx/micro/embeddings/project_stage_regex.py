from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

try:
    import regex as _rx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _rx = None  # type: ignore[assignment]

from .project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_query_core import extract_code_core
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from .project_iter import iter_candidate_files


def _make_fuzzy_pattern(q: str):
    if _rx is None:
        return None
    s = (q or "").strip()
    if len(s) < 3:
        return None
    parts = [p for p in s.split() if p]
    if not parts:
        return None
    # Join parts with optional whitespace to tolerate spaces
    base = r"\s*".join(_rx.escape(p) for p in parts)
    # Allow a small number of edits proportional to length
    err = max(1, min(4, len(s) // 10))
    try:
        return _rx.compile(f"({base}){{e<={err}}}", _rx.BESTMATCH | _rx.IGNORECASE | _rx.DOTALL)
    except Exception:
        return None


def stage_regex_hits(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: fuzzy regex phrase matching across candidate files.

    Uses the optional 'regex' package if available. Returns [(score, file_rel, obj)].
    """
    if _rx is None:
        return []
    q = (query or "").strip()
    if not q:
        return []
    # Prefer code-core for building the fuzzy pattern to improve robustness on code fragments
    q_eff = extract_code_core(q) or q
    pat = _make_fuzzy_pattern(q_eff)
    if pat is None:
        return []

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    codey = _is_code_like(q)
    include_exts = ["py"] if codey else INCLUDE_EXTS
    for abs_p, rel_p in iter_candidate_files(
        ROOT,
        include_exts=include_exts,
        exclude_dirs=EXCLUDE_DIRS,
        max_file_bytes=MAX_FILE_BYTES,
    ):
        if time_up():
            break
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if not text:
            continue
        try:
            m = pat.search(text)
        except Exception:
            m = None
        if not m:
            continue
        pos0, pos1 = m.start(), m.end()
        pre = text[:pos0]
        ls = pre.count("\n") + 1
        le = ls + max(1, text[pos0:pos1].count("\n"))
        lines_all = text.splitlines()
        a = max(1, ls - 12)
        b = min(len(lines_all), le + 12)
        snip = "\n".join(lines_all[a - 1 : b]).strip()
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": snip or text[:300].strip(),
                "line_start": a,
                "line_end": b,
            },
        }
        # Slightly below AST but above Jedi and phrase hits
        hits.append((0.993, rel_p, obj))
        if len(hits) >= k:
            break
    return hits


__all__ = ["stage_regex_hits"]
