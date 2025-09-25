from __future__ import annotations

import ast
import time
from typing import Any, Dict, List, Tuple

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files
from .project_query_tokens import expand_strong_tokens, codeish_tokens


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _query_tokens(q: str, max_items: int = 24) -> List[str]:
    toks: List[str] = []
    # Prefer strong tokens, then simpler ones
    for t in expand_strong_tokens(q, max_items=max_items) + codeish_tokens(q):
        tl = (t or "").strip().lower()
        if tl and tl not in toks:
            toks.append(tl)
    return toks[:max_items]


def stage_pydoc_hits(query: str, k: int, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: scan Python docstrings and match against the query (tokens or fuzzy).

    Returns a list of (score, file_rel, obj) where obj.meta contains file_rel, line_start, line_end, preview.
    """
    q = (query or "").strip()
    if not q:
        return []
    qn = _norm(q)
    if len(qn) < 3:
        return []
    toks = _query_tokens(q)

    t0 = time.perf_counter()
    hits: List[Tuple[float, str, Dict[str, Any]]] = []

    def time_up() -> bool:
        return max_time_ms is not None and (time.perf_counter() - t0) * 1000.0 > max_time_ms

    for abs_p, rel_p in iter_candidate_files(
        ROOT, include_exts=["py"], exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES
    ):
        if time_up():
            break
        try:
            with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
        except Exception:
            continue
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        lines = src.splitlines()

        def _consider(node: ast.AST) -> None:
            if time_up():
                return
            try:
                doc = ast.get_docstring(node, clean=False)  # type: ignore[arg-type]
            except Exception:
                doc = None
            if not doc:
                return
            doc_lno = None
            doc_end = None
            try:
                if getattr(node, "body", None):  # type: ignore[attr-defined]
                    first = node.body[0]  # type: ignore[index]
                    if isinstance(first, ast.Expr):
                        val = first.value  # type: ignore[attr-defined]
                        # py3.8+: Constant; py3.7: Str
                        if hasattr(val, "lineno"):
                            doc_lno = int(getattr(val, "lineno", 0) or 0)
                        if hasattr(val, "end_lineno"):
                            doc_end = int(getattr(val, "end_lineno", 0) or 0)
            except Exception:
                pass
            dl = int(doc_lno or 0)
            de = int(doc_end or (dl + 1))
            # Quick token match
            dn = _norm(doc)
            match = False
            if toks and any(t in dn for t in toks):
                match = True
            else:
                # Light fuzzy via difflib on normalized strings
                try:
                    import difflib as _df
                except Exception:
                    _df = None
                if _df is not None:
                    try:
                        r = _df.SequenceMatcher(None, qn, dn).ratio()
                        match = r >= 0.78
                    except Exception:
                        match = False
            if not match:
                return
            a = max(1, dl - 12)
            b = min(len(lines), de + 12)
            preview = "\n".join(lines[a - 1 : b]).strip()
            obj = {
                "embedding": [],
                "meta": {
                    "file_rel": rel_p,
                    "text_preview": preview or (doc[:300].strip() if isinstance(doc, str) else ""),
                    "line_start": a,
                    "line_end": b,
                },
            }
            hits.append((0.991, rel_p, obj))

        class V(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[name-defined]
                _consider(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[name-defined]
                _consider(node)
                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[name-defined]
                _consider(node)
                self.generic_visit(node)

        V().visit(tree)
        if len(hits) >= k:
            break
    return hits[:k]


__all__ = ["stage_pydoc_hits"]
