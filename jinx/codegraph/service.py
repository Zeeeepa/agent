from __future__ import annotations

import os
import re
from typing import List, Tuple

from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.micro.embeddings.project_config import INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES
from jinx.micro.embeddings.project_iter import iter_candidate_files
try:
    from jinx.codegraph.ts_backend import defs_for_token as _ts_defs  # type: ignore
    _TS_OK = True
except Exception:
    _TS_OK = False
from jinx.micro.embeddings.symbol_index import query_symbol_index as _sym_query


def _lang_for_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".cs": "csharp",
    }.get(ext, "")


def _read_text(abs_path: str) -> str:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _window(text: str, line: int, around: int = 8) -> Tuple[int, int, str]:
    lines = (text or "").splitlines()
    if not lines:
        return 0, 0, ""
    a = max(1, line - around)
    b = min(len(lines), line + around)
    snippet = "\n".join(lines[a - 1 : b])
    return a, b, snippet


def _tokenize(text: str, max_tokens: int = 8) -> List[str]:
    toks = []
    seen = set()
    for m in re.finditer(r"(?u)[A-Za-z_][A-Za-z0-9_]{2,}", text or ""):
        t = (m.group(0) or "").strip()
        if t and t not in seen:
            seen.add(t)
            toks.append(t)
        if len(toks) >= max_tokens:
            break
    return toks


async def snippets_for_tokens(tokens: List[str], *, max_per_token: int = 1, around: int = 8) -> List[Tuple[str, str]]:
    root = _resolve_root()
    out: List[Tuple[str, str]] = []
    for tok in tokens:
        try:
            q = await _sym_query(tok)
        except Exception:
            q = {"defs": [], "calls": []}
        defs = q.get("defs") or []
        for rel, line in defs[: max(1, max_per_token)]:
            abs_path = os.path.join(root, rel)
            txt = _read_text(abs_path)
            if not txt:
                continue
            a, b, snip = _window(txt, int(line), around=around)
            lang = _lang_for_file(abs_path)
            hdr = f"[DEF {tok}] [{rel}:{a}-{b}]"
            block = f"```{lang}\n{snip}\n```" if lang else f"```\n{snip}\n```"
            out.append((hdr, block))
            if len(out) >= 6:
                return out
    return out


async def snippets_for_text(text: str, *, max_tokens: int = 8, max_snippets: int = 4) -> List[Tuple[str, str]]:
    toks = _tokenize(text or "", max_tokens=max_tokens)
    # Python-first via symbol index
    pairs = await snippets_for_tokens(toks, max_per_token=1)
    # If still not enough, try tree-sitter across non-Python files (best-effort)
    if len(pairs) < max_snippets and _TS_OK:
        extra = await _ts_pairs_for_tokens(toks, want=max_snippets - len(pairs))
        pairs.extend(extra)
    return pairs[:max_snippets]


async def _ts_pairs_for_tokens(tokens: List[str], *, want: int) -> List[Tuple[str, str]]:
    if not tokens or not _TS_OK or want <= 0:
        return []
    root = _resolve_root()
    out: List[Tuple[str, str]] = []
    seen: set[Tuple[str,int,int]] = set()
    # Iterate candidate files with reasonable limits
    for abs_p, rel_p in iter_candidate_files(root, include_exts=INCLUDE_EXTS, exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
        if rel_p.endswith('.py'):
            continue
        for tok in tokens:
            try:
                spans = await __import__('asyncio').to_thread(_ts_defs, abs_p, tok, max_items=1)  # type: ignore
            except Exception:
                spans = []
            if not spans:
                continue
            ls, le = spans[0]
            key = (rel_p, ls, le)
            if key in seen:
                continue
            seen.add(key)
            text = _read_text(abs_p)
            a, b, snip = _window(text, int((ls+le)//2), around=8)
            lang = _lang_for_file(abs_p)
            hdr = f"[DEF {tok}] [{rel_p}:{a}-{b}]"
            block = f"```{lang}\n{snip}\n```" if lang else f"```\n{snip}\n```"
            out.append((hdr, block))
            if len(out) >= want:
                return out
    return out
