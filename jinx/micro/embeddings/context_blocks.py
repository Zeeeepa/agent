from __future__ import annotations

from typing import List, Tuple
import os


def build_code_block(parts: List[str]) -> str:
    body = "\n".join(parts or [])
    return f"<embeddings_code>\n{body}\n</embeddings_code>" if body else ""


def build_brain_block(brain_pairs: List[Tuple[str, float]] | None, *, topn: int | None = None) -> str:
    if not brain_pairs:
        return ""
    try:
        _topn = max(1, int(topn or int(os.getenv("EMBED_BRAIN_BLOCK_TOP", "8"))))
    except Exception:
        _topn = 8
    lines: List[str] = []
    for key, sc in (brain_pairs or [])[:_topn]:
        try:
            lines.append(f"- {key} ({sc:.2f})")
        except Exception:
            lines.append(f"- {key}")
    if not lines:
        return ""
    return f"<embeddings_brain>\n{os.linesep.join(lines)}\n</embeddings_brain>"


def build_refs_block(refs_parts: List[str] | None, *, policy: str | None = None, refs_min: int | None = None, refs_max_chars: int | None = None, codey: bool = False) -> str:
    if not refs_parts:
        return ""
    pol = (policy or os.getenv("JINX_REFS_POLICY", "always")).strip().lower()
    try:
        rmin = max(1, int(refs_min if refs_min is not None else os.getenv("JINX_REFS_AUTO_MIN", "2")))
    except Exception:
        rmin = 2
    try:
        rmax = max(200, int(refs_max_chars if refs_max_chars is not None else os.getenv("JINX_REFS_MAX_CHARS", "1600")))
    except Exception:
        rmax = 1600
    def _allow() -> bool:
        if pol in ("never", "0", "off", "false", ""):
            return False
        if pol in ("always", "1", "on", "true"):
            return True
        return bool(codey) or (len(refs_parts or []) >= rmin)
    if not _allow():
        return ""
    acc: List[str] = []
    total = 0
    for p in (refs_parts or []):
        plen = len(p) + 1
        if total + plen > rmax:
            break
        acc.append(p)
        total += plen
    if not acc:
        return ""
    rbody = "\n".join(acc)
    return f"<embeddings_refs>\n{rbody}\n</embeddings_refs>"


def build_graph_block(graph_parts: List[str] | None) -> str:
    if not graph_parts:
        return ""
    gbody = "\n".join(graph_parts)
    return f"<embeddings_graph>\n{gbody}\n</embeddings_graph>"


def build_memory_block(mem_parts: List[str] | None) -> str:
    if not mem_parts:
        return ""
    mbody = "\n".join(mem_parts)
    return f"<embeddings_memory>\n{mbody}\n</embeddings_memory>"


def join_blocks(blocks: List[str]) -> str:
    return "\n\n".join([b for b in (blocks or []) if (b or "").strip()])


__all__ = [
    "build_code_block",
    "build_brain_block",
    "build_refs_block",
    "build_graph_block",
    "build_memory_block",
    "join_blocks",
]
