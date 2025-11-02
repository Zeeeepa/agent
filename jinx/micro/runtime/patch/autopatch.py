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
from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.micro.common.internal_paths import is_restricted_path
from jinx.micro.brain.concepts import activate_concepts as _brain_activate
from jinx.micro.brain.attention import record_attention as _att_rec
import re as _re
from jinx.micro.memory.graph import apply_feedback as _kg_feedback
from .strategy_bandit import bandit_order_for_context as _bandit_order, bandit_update as _bandit_update
from .symbol_patch_generic import patch_symbol_generic as _patch_symbol_generic
from jinx.micro.brain.concepts import record_reinforcement as _brain_reinf


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
        max_ms = int(os.getenv("JINX_AUTOPATCH_MAX_MS", "900"))
    except Exception:
        max_ms = 900
    # Exhaustive mode: disable timeboxing and search caps if enabled
    no_budgets = _truthy("JINX_AUTOPATCH_NO_BUDGETS", "1")
    max_ms_local = None if no_budgets else max_ms
    try:
        search_k = int(os.getenv("JINX_AUTOPATCH_SEARCH_TOPK", "4"))
    except Exception:
        search_k = 4

    # Resolve project root once
    ROOT = _resolve_root()

    # Brain-driven query expansion (future-grade retrieval): expand terms/symbols
    exp_query = (args.query or "").strip()
    brain_pairs: List[Tuple[str, float]] = []
    if exp_query:
        try:
            if _truthy("EMBED_BRAIN_ENABLE", "1"):
                btop = max(4, int(os.getenv("EMBED_BRAIN_TOP_K", "10")))
                bmax = max(2, int(os.getenv("EMBED_BRAIN_EXPAND_MAX_TOKENS", "6")))
                brain_pairs = await _brain_activate(exp_query, top_k=btop)
                seen_bt: set[str] = set()
                btoks: List[str] = []
                for key, _sc in brain_pairs:
                    low = (key or "").lower()
                    tok = ""
                    if low.startswith("term: "):
                        tok = low.split(": ", 1)[1]
                    elif low.startswith("symbol: "):
                        tok = low.split(": ", 1)[1]
                    if tok and tok not in seen_bt:
                        btoks.append(tok)
                        seen_bt.add(tok)
                    if len(btoks) >= bmax:
                        break
                if btoks:
                    exp_query = (exp_query + " " + " ".join(btoks)).strip()
        except Exception:
            brain_pairs = []

    # Gather candidates (as tuples of (name, preview_coro_factory, commit_coro_factory))
    candidates: List[Tuple[str, Callable[[], Awaitable[Tuple[bool, str]]], Callable[[], Awaitable[Tuple[bool, str]]]]] = []

    # Guard: skip restricted paths
    if path and is_restricted_path(path):
        return False, "restricted", f"path is restricted: {path}"

    # 1) explicit line range
    if (args.line_start or 0) > 0 and (args.line_end or 0) > 0 and path:
        ls = int(args.line_start)
        le = int(args.line_end)
        candidates.append((
            "line",
            lambda: patch_line_range(path, ls, le, code, preview=True, max_span=args.max_span),
            lambda: patch_line_range(path, ls, le, code, preview=False, max_span=args.max_span),
        ))

    # 2) python symbol (requires path) with smarter header detection
    if (args.symbol or "") and (str(path).endswith(".py") if path else False):
        def _looks_like_header(snippet: str) -> bool:
            sn = (snippet or "").lstrip()
            if sn.startswith(("def ", "class ", "async def ")):
                return True
            # Try AST parse to see if snippet declares a def/class at top-level
            try:
                import ast as _ast
                m = _ast.parse(snippet or "")
                for n in getattr(m, "body", []) or []:
                    if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                        return True
            except Exception:
                pass
            return False

        if _looks_like_header(code):
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

    # 3b) context replace (multi-variant tolerances)
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
        # add laxer variants to increase hit probability
        for tol_v in (0.64, 0.55):
            candidates.append((
                f"context_{tol_v}",
                (lambda tol_v=tol_v: patch_context_replace(path, args.context_before or "", code, preview=True, tolerance=tol_v)),
                (lambda tol_v=tol_v: patch_context_replace(path, args.context_before or "", code, preview=False, tolerance=tol_v)),
            ))

    # 4) semantic in-file when we know the path and have a query (add wide variant)
    if path and (args.query or ""):
        candidates.append((
            "semantic",
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=True),
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=False),
        ))
        # wider window variant
        candidates.append((
            "semantic_wide",
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=True, topk=8, margin=10, tol=0.5),
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=False, topk=8, margin=10, tol=0.5),
        ))
    # 4b) Node (TS/JS) direct symbol patch if we know the path and symbol
    if path and (args.symbol or ""):
        try:
            ext = os.path.splitext(path)[1].lower()
        except Exception:
            ext = ""
        if ext in (".js", ".jsx"):
            candidates.append((
                "node_symbol",
                lambda: _patch_symbol_generic(path, "js", args.symbol or "", code, preview=True),
                lambda: _patch_symbol_generic(path, "js", args.symbol or "", code, preview=False),
            ))

    # 5) write new file or overwrite
    if path:
        candidates.append((
            "write",
            lambda: patch_write(path, code, preview=True),
            lambda: patch_write(path, code, preview=False),
        ))

    # 6) search-based if query provided (multi-hit across expanded query; add wide semantic variants and Node symbol patchers)
    if not path and (args.query or ""):
        try:
            limit_ms = None if no_budgets else min(max_ms, 600)
            hits_base = await search_project_cached(args.query or "", k=max(1, search_k), max_time_ms=limit_ms)
        except Exception:
            hits_base = []
        hits_exp: List[Dict] = []
        if exp_query and exp_query != (args.query or ""):
            try:
                limit_ms2 = None if no_budgets else min(max_ms, 600)
                hits_exp = await search_project_cached(exp_query, k=max(1, search_k), max_time_ms=limit_ms2)
            except Exception:
                hits_exp = []
        # Merge by file and range
        seen_keys: set[Tuple[str, int, int]] = set()
        merged: List[Dict] = []
        for lst in (hits_base or []), (hits_exp or []):
            for h in (lst or []):
                f = str(h.get("file") or "").strip()
                if not f:
                    continue
                ls_h = int(h.get("line_start") or 1)
                le_h = int(h.get("line_end") or 1)
                kx = (f, ls_h, le_h)
                if kx in seen_keys:
                    continue
                seen_keys.add(kx)
                merged.append(h)
        for h in merged:
            f = str(h.get("file") or "").strip()
            if not f:
                continue
            fpath = os.path.join(ROOT, f)
            if is_restricted_path(fpath):
                continue
            ls_h = int(h.get("line_start") or 1)
            le_h = int(h.get("line_end") or 1)
            # If symbol provided and Node file, add generic symbol patch candidates
            if (args.symbol or ""):
                try:
                    ext = os.path.splitext(fpath)[1].lower()
                except Exception:
                    ext = ""
                if ext in (".js", ".jsx"):
                    candidates.append((
                        "node_symbol",
                        (lambda fpath=fpath: _patch_symbol_generic(fpath, "js", args.symbol or "", code, preview=True)),
                        (lambda fpath=fpath: _patch_symbol_generic(fpath, "js", args.symbol or "", code, preview=False)),
                    ))
            # Prefer semantic first for each hit, then fallback to line
            candidates.append((
                "search_semantic",
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=True)),
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=False)),
            ))
            candidates.append((
                "search_semantic_wide",
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=True, topk=8, margin=10, tol=0.5)),
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=False, topk=8, margin=10, tol=0.5)),
            ))
            candidates.append((
                "search_line",
                (lambda fpath=fpath, ls_h=ls_h, le_h=le_h: patch_line_range(fpath, ls_h, le_h, code, preview=True, max_span=args.max_span)),
                (lambda fpath=fpath, ls_h=ls_h, le_h=le_h: patch_line_range(fpath, ls_h, le_h, code, preview=False, max_span=args.max_span)),
            ))

    # Reorder candidates by bandit per context (language, symbol/anchor/query flags)
    try:
        ext = os.path.splitext(path)[1].lower() if path else ""
    except Exception:
        ext = ""
    ctx = "|".join([
        (ext or ""),
        ("sym" if (args.symbol or "") else ""),
        ("anc" if (args.anchor or "") else ""),
        ("qry" if (args.query or "") else ""),
    ])
    cand_names = [name for name, _p, _c in candidates]
    try:
        order = _bandit_order(ctx, cand_names)
    except Exception:
        order = cand_names
    rank: Dict[str, int] = {nm: i for i, nm in enumerate(order)}
    candidates.sort(key=lambda t: rank.get(t[0], 1_000_000))

    # Timeboxed concurrent evaluation and selection (batched)
    best: Dict[str, object] | None = None
    try:
        PREV_CONC = max(1, int(os.getenv("JINX_AUTOPATCH_PREVIEW_CONC", "4")))
    except Exception:
        PREV_CONC = 4
    # Helper scorer
    def _score(name: str, diff: str, total: int) -> Tuple[int, int, int]:
        okc, _reason = _should_autocommit(name.replace("search_", "").replace("symbol_body", "symbol"), diff)
        pref = {
            "symbol": 9, "symbol_body": 8, "semantic": 7, "search_semantic": 6,
            "context": 5, "anchor": 4, "line": 3, "search_line": 2, "write": 1,
        }
        base = name
        for key in ("_wide", "_0.64", "_0.55"):
            base = base.replace(key, "")
        return (1 if okc else 0, pref.get(base, 0), -total)

    # Evaluate in windows to respect timebox while still exploring
    pos = 0
    n = len(candidates)
    while pos < n:
        # timebox check
        if (max_ms_local is not None) and ((time.monotonic() - start_ts) * 1000.0 > max_ms_local):
            break
        batch = candidates[pos: pos + PREV_CONC]
        pos += PREV_CONC
        # launch previews concurrently
        tasks = []
        for name, prev_factory, commit_factory in batch:
            tasks.append((name, commit_factory, asyncio.create_task(_evaluate_candidate(name, prev_factory()))))
        # gather
        done_list = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
        for (name, commit_factory, _), res in zip(tasks, done_list):
            if isinstance(res, Exception):
                continue
            cname, ok, diff, total = res
            if not ok:
                continue
            sc = _score(name, diff, total)
            if best is None or (sc > best["score"]):
                best = {"name": name, "diff": diff, "score": sc, "commit": commit_factory}

    # Strategy preference ordering (higher is better) for risk-aware selection
    pref_weight = {
        "symbol": 9,
        "symbol_body": 8,
        "semantic": 7,
        "search_semantic": 6,
        "context": 5,
        "anchor": 4,
        "line": 3,
        "search_line": 2,
        "write": 1,
    }
    for name, prev_factory, commit_factory in candidates:
        # timebox unless disabled
        if (max_ms_local is not None) and ((time.monotonic() - start_ts) * 1000.0 > max_ms_local):
            break
        cname, ok, diff, total = await _evaluate_candidate(name, prev_factory())
        if not ok:
            continue
        # score: prefer autocommit suitability, then strategy preference, then smaller diff
        okc, reason = _should_autocommit(name.replace("search_", "").replace("symbol_body", "symbol"), diff)
        score = (1 if okc else 0, pref_weight.get(name, 0), -total)
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
            hits = await search_project_cached(exp_query or (args.query or ""), k=1, max_time_ms=limit_ms2)
            if hits:
                h = hits[0]
                fpath = os.path.join(ROOT, h.get("file") or "")
                if fpath:
                    ok, detail = await patch_semantic_in_file(fpath, exp_query or (args.query or ""), code, preview=bool(args.preview))
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
    # Bandit update
    try:
        _bandit_update(ctx, str(best["name"]), bool(okc))
    except Exception:
        pass
    # Record attention on success to reinforce short-term working memory
    try:
        if okc:
            keys: List[str] = []
            # Path attention (relative)
            if path:
                try:
                    relp = os.path.relpath(path, start=ROOT)
                except Exception:
                    relp = path
                keys.append(f"path: {relp}")
            # Symbol attention
            if args.symbol:
                keys.append(f"symbol: {args.symbol}")
            # Query term attention
            qtxt = exp_query or (args.query or "")
            for m in _re.finditer(r"(?u)[\w\.]{3,}", qtxt):
                t = (m.group(0) or "").strip().lower()
                if t and len(t) >= 3:
                    keys.append(f"term: {t}")
            await _att_rec(keys, weight=1.0)
            # Apply KG feedback: nodes and edges among co-activated concepts
            try:
                fb_nodes = [(k, 0.6) for k in keys]
                # build sparse pairwise edges (cap to first 6 to limit work)
                cap = min(6, len(keys))
                fb_edges: List[tuple[str, str, float]] = []
                for i in range(cap):
                    for j in range(i + 1, cap):
                        fb_edges.append((keys[i], keys[j], 0.4))
                _kg_feedback(fb_nodes, fb_edges)
            except Exception:
                pass
            # Persist reinforcement for brain concepts (with decay applied later)
            try:
                await _brain_reinf(keys, weight=1.0)
            except Exception:
                pass
    except Exception:
        pass
    return okc, str(best["name"]), detailc
