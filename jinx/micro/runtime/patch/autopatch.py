from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Awaitable, Dict
import time

from jinx.micro.embeddings.search_cache import search_project_cached
from .write_patch import patch_write
from .line_patch import patch_line_range
from .anchor_patch import patch_anchor_insert_after
from .symbol_patch import patch_symbol_python
from .symbol_body_patch import patch_symbol_body_python
from .context_patch import patch_context_replace
from .semantic_patch import patch_semantic_in_file
from .utils import diff_stats as _diff_stats, should_autocommit as _should_autocommit


@dataclass
class AutoPatchArgs:
    path: Optional[str] = None
    code: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    symbol: Optional[str] = None
    anchor: Optional[str] = None
    query: Optional[str] = None
    preview: bool = False
    max_span: Optional[int] = None
    force: bool = False
    context_before: Optional[str] = None
    context_tolerance: Optional[float] = None


def _truthy(name: str, default: str = "1") -> bool:
    try:
        return str(os.getenv(name, default)).strip().lower() not in ("", "0", "false", "off", "no")
    except Exception:
        return True


async def _evaluate_candidate(name: str, coro) -> Tuple[str, bool, str, int]:
    """Run candidate in preview mode, return (name, ok, diff_or_detail, total_changes)."""
    ok, detail = await coro
    # detail is usually a diff for preview=True; compute size as risk proxy
    add, rem = _diff_stats(detail or "")
    total = add + rem
    return name, ok, detail, total


async def autopatch(args: AutoPatchArgs) -> Tuple[bool, str, str]:
    """Choose best patch strategy using a smart, candidate-based selector.

    - Builds an ordered list of viable strategies from the provided args.
    - Evaluates candidates in preview mode with timeboxing; scores by autocommit suitability and diff size.
    - Commits the best candidate (or returns its preview when args.preview is True).

    Returns (ok, strategy, detail_or_diff).
    """
    path = args.path or ""
    code = args.code or ""
    start_ts = time.monotonic()
    try:
        max_ms = int(os.getenv("JINX_AUTOPATCH_MAX_MS", "800"))
    except Exception:
        max_ms = 800
    # Exhaustive mode: disable timeboxing and search caps if enabled
    no_budgets = _truthy("JINX_AUTOPATCH_NO_BUDGETS", "1")
    max_ms_local = None if no_budgets else max_ms
    try:
        search_k = int(os.getenv("JINX_AUTOPATCH_SEARCH_TOPK", "3"))
    except Exception:
        search_k = 3

    # Gather candidates (as tuples of (name, preview_coro_factory, commit_coro_factory))
    candidates: List[Tuple[str, Callable[[], Awaitable[Tuple[bool, str]]], Callable[[], Awaitable[Tuple[bool, str]]]]] = []

    # 1) explicit line range
    if (args.line_start or 0) > 0 and (args.line_end or 0) > 0 and path:
        ls = int(args.line_start)
        le = int(args.line_end)
        candidates.append((
            "line",
            lambda: patch_line_range(path, ls, le, code, preview=True, max_span=args.max_span),
            lambda: patch_line_range(path, ls, le, code, preview=False, max_span=args.max_span),
        ))

    # 2) python symbol (requires path)
    if (args.symbol or "") and (str(path).endswith(".py") if path else False):
        code_l = code.lstrip()
        if code_l.startswith(("def ", "class ", "async def ")):
            candidates.append((
                "symbol",
                lambda: patch_symbol_python(path, args.symbol or "", code, preview=True),
                lambda: patch_symbol_python(path, args.symbol or "", code, preview=False),
            ))
        else:
            candidates.append((
                "symbol_body",
                lambda: patch_symbol_body_python(path, args.symbol or "", code, preview=True),
                lambda: patch_symbol_body_python(path, args.symbol or "", code, preview=False),
            ))

    # 3) anchor insert
    if (args.anchor or "") and path:
        candidates.append((
            "anchor",
            lambda: patch_anchor_insert_after(path, args.anchor or "", code, preview=True),
            lambda: patch_anchor_insert_after(path, args.anchor or "", code, preview=False),
        ))

    # 3b) context replace
    if (args.context_before or "") and path:
        try:
            tol = float(args.context_tolerance) if (args.context_tolerance is not None) else float(os.getenv("JINX_PATCH_CONTEXT_TOL", "0.72"))
        except Exception:
            tol = 0.72
        candidates.append((
            "context",
            lambda: patch_context_replace(path, args.context_before or "", code, preview=True, tolerance=tol),
            lambda: patch_context_replace(path, args.context_before or "", code, preview=False, tolerance=tol),
        ))

    # 4) semantic in-file when we know the path and have a query
    if path and (args.query or ""):
        candidates.append((
            "semantic",
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=True),
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=False),
        ))

    # 5) write new file or overwrite
    if path:
        candidates.append((
            "write",
            lambda: patch_write(path, code, preview=True),
            lambda: patch_write(path, code, preview=False),
        ))

    # 6) search-based if query provided (multi-hit)
    if not path and (args.query or ""):
        try:
            limit_ms = None if no_budgets else min(max_ms, 600)
            hits = await search_project_cached(args.query or "", k=max(1, search_k), max_time_ms=limit_ms)
        except Exception:
            hits = []
        for h in (hits or []):
            f = str(h.get("file") or "")
            if not f:
                continue
            fpath = os.path.join(os.getenv("EMBED_PROJECT_ROOT", os.getcwd()), f)
            ls_h = int(h.get("line_start") or 1)
            le_h = int(h.get("line_end") or 1)
            # Prefer semantic first for each hit, then fallback to line
            candidates.append((
                "search_semantic",
                (lambda fpath=fpath: patch_semantic_in_file(fpath, args.query or "", code, preview=True)),
                (lambda fpath=fpath: patch_semantic_in_file(fpath, args.query or "", code, preview=False)),
            ))
            candidates.append((
                "search_line",
                (lambda fpath=fpath, ls_h=ls_h, le_h=le_h: patch_line_range(fpath, ls_h, le_h, code, preview=True, max_span=args.max_span)),
                (lambda fpath=fpath, ls_h=ls_h, le_h=le_h: patch_line_range(fpath, ls_h, le_h, code, preview=False, max_span=args.max_span)),
            ))

    # Timeboxed evaluation and selection
    best: Dict[str, object] | None = None
    for name, prev_factory, commit_factory in candidates:
        # timebox unless disabled
        if (max_ms_local is not None) and ((time.monotonic() - start_ts) * 1000.0 > max_ms_local):
            break
        cname, ok, diff, total = await _evaluate_candidate(name, prev_factory())
        if not ok:
            continue
        # score: prefer autocommit suitability first, then smaller diff
        okc, reason = _should_autocommit(name.replace("search_", "").replace("symbol_body", "symbol"), diff)
        score = (1 if okc else 0, -total)
        if best is None or (score > best["score"]):
            best = {"name": cname, "diff": diff, "score": score, "commit": commit_factory}

    # If nothing succeeded during preview, attempt last resort paths (old behavior fallbacks)
    if best is None:
        # Try the original simple flow as a final fallback
        if path and code:
            ok, detail = await patch_write(path, code, preview=bool(args.preview))
            return ok, "write", detail
        if args.query:
            limit_ms2 = None if no_budgets else min(max_ms, 300)
            hits = await search_project_cached(args.query, k=1, max_time_ms=limit_ms2)
            if hits:
                h = hits[0]
                fpath = os.path.join(os.getenv("EMBED_PROJECT_ROOT", os.getcwd()), h.get("file") or "")
                if fpath:
                    ok, detail = await patch_semantic_in_file(fpath, args.query or "", code, preview=bool(args.preview))
                    if ok:
                        return True, "search_semantic", detail
                    ok2, detail2 = await patch_line_range(fpath, int(h.get("line_start") or 1), int(h.get("line_end") or 1), code, preview=bool(args.preview), max_span=args.max_span)
                    return ok2, "search_line", detail2
            return False, "search", "no hits"
        return False, "auto", "insufficient arguments for autopatch"

    # We have a best candidate selected by preview. If preview requested, return its diff.
    if args.preview:
        return True, str(best["name"]), str(best["diff"])

    # Commit the chosen candidate
    okc, detailc = await best["commit"]()
    return okc, str(best["name"]), detailc
