from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple
import re as _re

from .project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_line_window import find_line_window
from .project_scan_store import iter_project_chunks
from .project_query_tokens import expand_strong_tokens, codeish_tokens


def _expand_tokens(q: str, max_items: int = 32) -> List[str]:
    strong = expand_strong_tokens(q, max_items=max_items)
    simple = codeish_tokens(q)
    # Deduplicate preserving order, prefer strong first
    out: List[str] = []
    seen: set[str] = set()
    for t in strong + simple:
        tl = (t or "").lower()
        if not tl or tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out[:max_items]


def stage_textscan_hits(query: str, k: int, *, max_time_ms: int | None = 250) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage -1: direct text scan over project files (no embeddings).

    - First tries a flexible phrase match (whitespace-insensitive, tolerant around ()=, ,)
    - Then falls back to token scanning using code-like tokens.

    Returns a list of (score, file_rel, obj) sorted by score desc.
    """
    q = (query or "").strip()
    if not q:
        return []
    toks = _expand_tokens(q)

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []
    seen_rel: set[str] = set()

    # Precompile a flexible phrase pattern to match the raw query with optional whitespace
    def _flex_phrase_pattern(s: str):
        s = (s or "").strip()
        if len(s) < 3:
            return None
        # Try to extract a code-core from the query (e.g., "a = func(x)" inside a NL sentence)
        def _extract_code_core(src: str) -> str | None:
            try:
                import re as __re, ast as __ast
                # Candidates consisting mostly of code characters
                cands = list(__re.finditer(r'[A-Za-z0-9_\./:\-+*<>=!"\'\[\]\(\)\{\),\s]+', src))
                def _can_parse(sn: str) -> bool:
                    s = sn.strip()
                    if len(s) < 3:
                        return False
                    try:
                        __ast.parse(s, mode='exec')
                        return True
                    except Exception:
                        pass
                    try:
                        __ast.parse(s, mode='eval')
                        return True
                    except Exception:
                        pass
                    try:
                        __ast.parse(f'({s})', mode='eval')
                        return True
                    except Exception:
                        return False
                best = ""
                best_len = 0
                for m in cands:
                    frag = (m.group(0) or "").strip()
                    if len(frag) < 6:
                        continue
                    if _can_parse(frag) and len(frag) > best_len:
                        best = frag
                        best_len = len(frag)
                if best:
                    return best
                # Fallback: choose the longest candidate if nothing parses
                if cands:
                    frag2 = max((m.group(0) or '').strip() for m in cands if (m.group(0) or '').strip())
                    return frag2 or None
                return None
            except Exception:
                return None
        s_eff = _extract_code_core(s) or s
        parts = [p for p in s_eff.split() if p]
        if not parts:
            return None
        # Build fuzzy phrase using 'regex' library (auto-installed at startup)
        try:
            import regex as _rx  # type: ignore
            needle = " ".join(parts)
            err = max(1, min(6, len(needle) // 10))
            return _rx.compile(f"({_rx.escape(needle)}){{e<={err}}}", _rx.BESTMATCH | _rx.IGNORECASE | _rx.DOTALL)
        except Exception:
            # If 'regex' is unavailable for any reason, let later stages handle matching
            return None

    phrase_pat = _flex_phrase_pattern(q)

    # Pass 1: only files already present in embeddings store (fast and highly relevant)
    rel_files: List[str] = []
    try:
        seen_f: set[str] = set()
        for fr, obj in iter_project_chunks():
            rel = fr or str((obj.get("meta") or {}).get("file_rel") or "")
            if rel and rel not in seen_f:
                seen_f.add(rel)
                rel_files.append(rel)
    except Exception:
        rel_files = []

    def _scan_abs_rel(abs_p: str, rel_p: str) -> bool:
        # Basic time budget check
        if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
            return True  # signal to stop
        # Avoid scanning the same rel path twice (shouldn't happen, but be safe)
        if rel_p in seen_rel:
            return False
        seen_rel.add(rel_p)
        try:
            with open(abs_p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            text = ""
        if not text:
            return False
        # Try flexible phrase match first
        ls_meta = 0
        le_meta = 0
        snip = ""
        is_phrase_hit = False
        if phrase_pat is not None:
            # Try original text, then normalized variants while preserving line counts
            texts_to_try: List[Tuple[str, str]] = [("orig", text)]
            try:
                gen_strip = _re.sub(r"\[[^\]\n]*\]", "", text)
                texts_to_try.append(("gen_strip", gen_strip))
            except Exception:
                pass
            try:
                gen_T = _re.sub(r"\[[^\]\n]*\]", "T", text)
                texts_to_try.append(("gen_T", gen_T))
            except Exception:
                pass
            # Remove underscores to match 'QueueT' vs 'Queue_T'
            try:
                no_unders = text.replace("_", " ")
                texts_to_try.append(("no_unders", no_unders))
            except Exception:
                pass
            # Remove most punctuation (keep newlines) for typo-tolerant matching in comments/docstrings
            try:
                no_punct = _re.sub(r"[^\w\s]", " ", text)
                texts_to_try.append(("no_punct", no_punct))
            except Exception:
                pass
            for _kind, src in texts_to_try:
                m = phrase_pat.search(src)
                if m:
                    pos0, pos1 = m.start(), m.end()
                    pre = src[:pos0]
                    ls = pre.count("\n") + 1
                    le = ls + max(1, src[pos0:pos1].count("\n"))
                    lines_all = text.splitlines()
                    a = max(1, ls - 12)
                    b = min(len(lines_all), le + 12)
                    snip = "\n".join(lines_all[a-1:b]).strip()
                    ls_meta, le_meta = a, b
                    is_phrase_hit = True
                    break
        # Fallback: token-based match, then approximate line-level match
        if not (ls_meta and le_meta):
            low = text.lower()
            if any(t.lower() in low for t in toks):
                a, b, snip2 = find_line_window(text, toks, around=12)
                if a or b:
                    snip = snip or snip2 or text[:300].strip()
                    ls_meta = int(a or 0)
                    le_meta = int(b or 0)
            # Approximate match (typo-tolerant) if still not found
            if not (ls_meta and le_meta):
                try:
                    import difflib as _df
                except Exception:
                    _df = None  # type: ignore
                if _df is not None:
                    # Use top few longest tokens as anchors
                    anchors = sorted({t for t in toks if t and len(t) >= 5}, key=len, reverse=True)[:3]
                    if anchors:
                        lines_all = text.splitlines()
                        # Bound work: check up to N lines and bail if time budget exceeds
                        max_lines = min(len(lines_all), 1200)
                        best = (0.0, 0, 0)
                        for idx in range(max_lines):
                            if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
                                break
                            ln = lines_all[idx]
                            ln_low = ln.lower()
                            for a_tok in anchors:
                                r = _df.SequenceMatcher(None, a_tok.lower(), ln_low).ratio()
                                if r >= 0.86 and r > best[0]:
                                    a = max(1, idx + 1 - 12)
                                    b = min(len(lines_all), idx + 1 + 12)
                                    best = (r, a, b)
                        if best[0] >= 0.86:
                            _, a, b = best
                            snip = snip or "\n".join(lines_all[a-1:b]).strip()
                            ls_meta = a
                            le_meta = b
        # Final fallback: fuzzy full-query vs individual lines (very tolerant)
        if not (ls_meta and le_meta):
            try:
                from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
            except Exception:  # pragma: no cover - optional
                _rf_fuzz = None  # type: ignore
            try:
                import difflib as _df2
            except Exception:
                _df2 = None  # type: ignore
            def _norm(s: str) -> str:
                s2 = (s or "").lower()
                try:
                    s2 = _re.sub(r"[^\w\s]", " ", s2)
                except Exception:
                    pass
                s2 = s2.replace("_", " ")
                s2 = " ".join(s2.split())
                return s2
            # Prefer fuzzy on the code core if present
            def _extract_code_core2(src: str) -> str | None:
                try:
                    m = _re.search(r"([A-Za-z_][A-Za-z0-9_\.]*\s*[=]\s*.+)$", src)
                    if m:
                        return (m.group(1) or "").strip()
                except Exception:
                    pass
                try:
                    # Fallback: look for a call-like pattern
                    m2 = _re.search(r"([A-Za-z_][A-Za-z0-9_\.]*)\s*\(.*\)", src)
                    if m2:
                        return (m2.group(0) or "").strip()
                except Exception:
                    pass
                return None
            core_q = _extract_code_core2(q) or q
            nq = _norm(core_q)
            if nq and len(nq) >= 3:
                lines_all = text.splitlines()
                max_lines = min(len(lines_all), 1500)
                best_score = 0.0
                best_idx = -1
                for idx in range(max_lines):
                    if max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms:
                        break
                    ln = _norm(lines_all[idx])
                    score = 0.0
                    if _rf_fuzz is not None:
                        try:
                            score = float(_rf_fuzz.token_set_ratio(nq, ln)) / 100.0
                        except Exception:
                            score = 0.0
                    elif _df2 is not None:
                        try:
                            score = _df2.SequenceMatcher(None, nq, ln).ratio()
                        except Exception:
                            score = 0.0
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx >= 0 and best_score >= 0.78:
                    a = max(1, best_idx + 1 - 12)
                    b = min(len(lines_all), best_idx + 1 + 12)
                    snip = snip or "\n".join(lines_all[a-1:b]).strip()
                    ls_meta = a
                    le_meta = b
        obj = {
            "embedding": [],
            "meta": {
                "file_rel": rel_p,
                "text_preview": snip or text[:300].strip(),
                "line_start": ls_meta,
                "line_end": le_meta,
            },
        }
        # Slightly prioritize phrase hits over token hits
        score = 0.99 if is_phrase_hit else 0.98
        hits.append((score, rel_p, obj))
        if len(hits) >= k:
            return True
        return False

    # Scan embeddings-known files first
    for rel_p in rel_files:
        abs_p = os.path.join(ROOT, rel_p)
        if _scan_abs_rel(abs_p, rel_p):
            return hits[:k]

    # Pass 2: fallback to general project walk (slower)
    for abs_p, rel_p in iter_candidate_files(
        ROOT,
        include_exts=INCLUDE_EXTS,
        exclude_dirs=EXCLUDE_DIRS,
        max_file_bytes=MAX_FILE_BYTES,
    ):
        if _scan_abs_rel(abs_p, rel_p):
            return hits[:k]

    # Keep original order; all scores equal
    return hits[:k]
