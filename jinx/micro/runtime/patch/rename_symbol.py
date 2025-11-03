from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.async_utils.fs import read_text_abs_thread
from jinx.micro.embeddings.symbol_index import query_symbol_index as _sym_query
from jinx.micro.common.env import truthy

_DEF_PAT_CACHE: Dict[str, re.Pattern[str]] = {}
_CALL_PAT_CACHE: Dict[str, re.Pattern[str]] = {}
_IMPORT_ONE_PAT_CACHE: Dict[str, re.Pattern[str]] = {}
_ATTR_CALL_PAT_CACHE: Dict[str, re.Pattern[str]] = {}


def _pat_def(sym: str) -> re.Pattern[str]:
    p = _DEF_PAT_CACHE.get(sym)
    if p is None:
        # Top-level def/class header begins lines
        p = re.compile(rf"(?m)^(\s*)(def|class)\s+{re.escape(sym)}\b")
        _DEF_PAT_CACHE[sym] = p
    return p


def _pat_call(sym: str) -> re.Pattern[str]:
    p = _CALL_PAT_CACHE.get(sym)
    if p is None:
        # Standalone call 'sym(' not preceded by '.' and as a bare identifier
        p = re.compile(rf"(?<!\.)\b{re.escape(sym)}\s*\(")
        _CALL_PAT_CACHE[sym] = p
    return p


def _pat_import_one(sym: str) -> re.Pattern[str]:
    p = _IMPORT_ONE_PAT_CACHE.get(sym)
    if p is None:
        # from X import sym [as alias]
        p = re.compile(rf"(?m)^(\s*from\s+[^\n]+\s+import\s+)(.*\b){re.escape(sym)}\b(.*)$")
        _IMPORT_ONE_PAT_CACHE[sym] = p
    return p


def _pat_attr_call(sym: str) -> re.Pattern[str]:
    p = _ATTR_CALL_PAT_CACHE.get(sym)
    if p is None:
        # Attribute call like .sym( — ensure a leading dot and '('; conservative
        p = re.compile(rf"\.{re.escape(sym)}\s*\(")
        _ATTR_CALL_PAT_CACHE[sym] = p
    return p


def _rewrite_text(text: str, sym: str, new_name: str, *, rename_attr: bool = False) -> str:
    s = text or ""
    if not s:
        return s
    # 1) def/class headers
    s = _pat_def(sym).sub(lambda m: f"{m.group(1)}{m.group(2)} {new_name}", s)
    # 2) bare calls
    s = _pat_call(sym).sub(lambda m: f"{new_name}(", s)
    # 3) attribute calls (optional, conservative)
    if rename_attr:
        s = _pat_attr_call(sym).sub(lambda m: f".{new_name}(", s)
    # 4) simple import entries (single-line)
    def _imp_sub(m: re.Match[str]) -> str:
        head = m.group(1)
        before_sym = m.group(2) or ""
        after = m.group(3) or ""
        # Replace last occurrence of sym in the import list portion
        combined = before_sym + sym + after
        # Replace only the symbol token
        combined2 = re.sub(rf"\b{re.escape(sym)}\b", new_name, combined, count=1)
        return head + combined2
    s = _pat_import_one(sym).sub(_imp_sub, s)
    return s


async def build_rename_ops(symbol: str, new_name: str) -> List[Dict[str, object]]:
    """Build batch write ops to rename symbol across defs and calls based on symbol index.

    Conservative:
      - Renames top-level def/class names and bare function calls 'sym(...)'.
      - Renames entries in 'from X import sym'.
      - Does not rename attribute calls 'obj.sym(...)'.
    """
    root = _resolve_root()
    idx = await _sym_query(symbol)
    defs: List[Tuple[str, int]] = list(idx.get("defs") or [])  # (rel, line)
    calls: List[Tuple[str, int]] = list(idx.get("calls") or [])
    files_rel: List[str] = []
    for rel, _ in defs:
        if rel and rel not in files_rel:
            files_rel.append(rel)
    for rel, _ in calls:
        if rel and rel not in files_rel:
            files_rel.append(rel)
    # Enable attribute-call rename only when safe: unique def and env gate
    rename_attr = truthy("JINX_RENAME_ATTR_CONF", "0") and (len(defs) == 1)
    ops: List[Dict[str, object]] = []
    seen: set[str] = set()
    for rel in files_rel:
        ap = os.path.join(root, rel)
        if ap in seen:
            continue
        seen.add(ap)
        try:
            text = await read_text_abs_thread(ap)
        except Exception:
            continue
        if not text:
            continue
        new = _rewrite_text(text, symbol, new_name, rename_attr=rename_attr)
        if new != text:
            ops.append({"type": "write", "path": ap, "code": new, "meta": {"refactor": "rename", "symbol": symbol, "new_name": new_name}})
    return ops
