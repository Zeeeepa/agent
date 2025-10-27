from __future__ import annotations

import os
from typing import List, Tuple, Any

from .project_chunk_types import Chunk
from .project_chunk_token import chunk_text_token
from .project_chunk_char import chunk_text_char
from .snippet_segments import build_multi_segment_python

try:
    import libcst as _cst  # type: ignore
    from libcst.metadata import PositionProvider as _PositionProvider  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _cst = None  # type: ignore
    _PositionProvider = None  # type: ignore


# Semantic chunker parameters (query-agnostic; robust and non-primitive)
CHARS_PER_CHUNK = int(os.getenv("EMBED_PROJECT_SEM_CHARS", "1400"))
MIN_CHUNK_CHARS = int(os.getenv("EMBED_PROJECT_SEM_MIN_CHARS", "200"))
MAX_CHUNKS_PER_FILE = int(os.getenv("EMBED_PROJECT_SEM_MAX_CHUNKS", "220"))
HEAD_LINES = int(os.getenv("EMBED_PROJECT_SEM_HEAD_LINES", "10"))
TAIL_LINES = int(os.getenv("EMBED_PROJECT_SEM_TAIL_LINES", "10"))
MID_WINDOWS = int(os.getenv("EMBED_PROJECT_SEM_MID_WINDOWS", "3"))
MID_AROUND = int(os.getenv("EMBED_PROJECT_SEM_MID_AROUND", "20"))


def _py_module_spans(text: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, str]]]:
    """Return (module_doc_spans, def_spans) using libcst if available.

    def_spans entries are (start_line, end_line, kind) with kind in {"func", "method", "class"}.
    """
    if _cst is None or _PositionProvider is None or not text:
        return ([], [])
    try:
        mod = _cst.parse_module(text)  # type: ignore[attr-defined]
        wrapper = _cst.metadata.MetadataWrapper(mod)  # type: ignore[attr-defined]
        pos = wrapper.resolve(_PositionProvider)
    except Exception:
        return ([], [])

    lines = text.splitlines()
    n = len(lines)
    module_doc: List[Tuple[int, int]] = []
    defs: List[Tuple[int, int, str]] = []

    # Module docstring: first statement simple string
    try:
        body = getattr(mod, "body", [])
        if body:
            first = body[0]
            if hasattr(first, "body") and getattr(first, "body", None) is None:
                pass  # not a simple statement wrapper
    except Exception:
        pass
    # Safer: detect by source slice of first statement
    try:
        first_stmt = body[0] if body else None  # type: ignore[name-defined]
        if first_stmt is not None:
            p = pos.get(first_stmt)  # type: ignore[attr-defined]
            a = int(getattr(getattr(p, "start", None), "line", 1) or 1)
            b = int(getattr(getattr(p, "end", None), "line", a) or a)
            # Heuristic: if first non-empty, non-comment line starts with a quote, treat as docstring
            head_block = "\n".join(lines[a - 1 : min(n, b + 2)])
            if head_block.lstrip().startswith(("'", '"')):
                module_doc.append((a, min(n, b)))
    except Exception:
        pass

    # Collect defs/classes and class methods with their spans
    C = _cst  # type: ignore[assignment]

    class _SpanVisitor(getattr(C, "CSTVisitor", object)):  # type: ignore[misc]
        def __init__(self) -> None:
            self.defs: List[Tuple[int, int, str]] = []

        def visit_FunctionDef(self, node: Any) -> None:  # type: ignore[override]
            try:
                p = pos.get(node)  # type: ignore[attr-defined]
                a = int(getattr(getattr(p, "start", None), "line", 1) or 1)
                b = int(getattr(getattr(p, "end", None), "line", a) or a)
                self.defs.append((max(1, a), max(a, b), "func"))
            except Exception:
                return

        def visit_ClassDef(self, node: Any) -> None:  # type: ignore[override]
            try:
                p = pos.get(node)  # type: ignore[attr-defined]
                a = int(getattr(getattr(p, "start", None), "line", 1) or 1)
                b = int(getattr(getattr(p, "end", None), "line", a) or a)
                self.defs.append((max(1, a), max(a, b), "class"))
            except Exception:
                pass

    try:
        v = _SpanVisitor()
        wrapper.visit(v)  # type: ignore[attr-defined]
        defs = v.defs
    except Exception:
        defs = []
    return (module_doc, defs)


def chunk_text_semantic(text: str) -> List[Chunk]:
    """Semantic chunker that prefers CST-driven Python segmentation with multi-segment synthesis.

    Falls back to token-based chunker, then char-based chunker. Non-primitive; reuses existing
    snippet construction logic for robust segments.
    """
    if not text:
        return []
    # Try Python CST path first
    mod_spans, def_spans = _py_module_spans(text)
    lines = text.splitlines()
    chunks: List[Chunk] = []

    def _add_chunk(a: int, b: int, body_text: str) -> None:
        if not body_text:
            return
        if len(body_text) < MIN_CHUNK_CHARS and not (len(chunks) == 0 and len(text) < MIN_CHUNK_CHARS):
            return
        chunks.append({"text": body_text, "line_start": int(a), "line_end": int(b)})

    if def_spans:
        # Optional module header/doc chunk
        for a, b in (mod_spans or [])[:1]:
            try:
                head = "\n".join(lines[a - 1 : b]).strip()[: CHARS_PER_CHUNK]
                _add_chunk(a, b, head)
            except Exception:
                pass

        # Build function/class chunks using multi-segment synthesis
        for a, b, kind in def_spans:
            if len(chunks) >= MAX_CHUNKS_PER_FILE:
                break
            try:
                snippet = build_multi_segment_python(
                    file_lines=lines,
                    scope_start=int(a),
                    scope_end=int(b),
                    query="",  # query-agnostic; falls back to control-flow anchors
                    per_hit_chars=CHARS_PER_CHUNK,
                    head_lines=HEAD_LINES,
                    tail_lines=TAIL_LINES,
                    mid_windows=MID_WINDOWS,
                    mid_around=MID_AROUND,
                    strip_comments=True,
                    extra_centers=None,
                )
            except Exception:
                # Fallback: take raw scope trimmed to budget
                snippet = "\n".join(lines[a - 1 : b]).strip()[: CHARS_PER_CHUNK]
            _add_chunk(a, b, snippet)

        if chunks:
            return chunks[:MAX_CHUNKS_PER_FILE]

    # Fallback to token chunker (more semantic than char-only)
    try:
        toks = chunk_text_token(text)
    except Exception:
        toks = []
    if toks:
        return toks[:MAX_CHUNKS_PER_FILE]

    # Final fallback to char chunker
    return chunk_text_char(text)
