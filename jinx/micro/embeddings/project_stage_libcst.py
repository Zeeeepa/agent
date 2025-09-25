from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, cast

try:
    import libcst as cst  # type: ignore
    from libcst import matchers as m  # type: ignore
    from libcst.metadata import PositionProvider  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cst = None  # type: ignore
    m = None  # type: ignore
    PositionProvider = None  # type: ignore

from .project_config import ROOT, EXCLUDE_DIRS, MAX_FILE_BYTES
from .project_iter import iter_candidate_files


def _ltxt(s: str) -> str:
    return (s or "").strip().lower()


def _tokset(q: str, limit: int = 24) -> List[str]:
    import re
    toks: List[str] = []
    for mobj in re.finditer(r"[A-Za-z_][\w\.]{1,}", q or ""):
        t = _ltxt(mobj.group(0))
        if t and t not in toks:
            toks.append(t)
    return toks[:limit]


def _node_text_names(n: Any) -> List[str]:
    names: List[str] = []
    C = cast(Any, cst)
    try:
        if isinstance(n, C.Name):
            names.append(_ltxt(n.value))
        elif isinstance(n, C.Attribute):
            parts: List[str] = []
            cur = n
            while isinstance(cur, C.Attribute):
                parts.append(_ltxt(cur.attr.value))
                cur = cur.value
            if isinstance(cur, C.Name):
                parts.append(_ltxt(cur.value))
            parts.reverse()
            if parts:
                names.append(".".join(parts))
                names.append(parts[-1])
                names.append(parts[0])
    except Exception:
        pass
    # dedupe
    out: List[str] = []
    seen: set[str] = set()
    for t in names:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _match_interest(node: Any, toks: List[str]) -> float:
    # Return a score > 0 if node matches known patterns.
    if not toks:
        return 0.0
    tset = set(toks)
    sc = 0.0
    C = cast(Any, cst)

    # Return of a Call: return foo.bar(arg)
    if isinstance(node, C.Return):
        val = node.value
        if isinstance(val, C.Call):
            fn_names = []
            try:
                fn_names = _node_text_names(val.func)
            except Exception:
                fn_names = []
            if any(n in tset for n in fn_names):
                sc += 1.2
            # args names or strings
            try:
                for a in val.args:
                    nms: List[str] = []
                    if isinstance(a.value, C.Name) or isinstance(a.value, C.Attribute):
                        nms = _node_text_names(a.value)
                    elif isinstance(a.value, C.SimpleString):
                        try:
                            txt = a.value.evaluated_value  # may be str|bytes
                            nms = [_ltxt(txt.decode() if isinstance(txt, (bytes, bytearray)) else str(txt))]
                        except Exception:
                            nms = []
                    if any(n in tset for n in nms):
                        sc += 0.6
            except Exception:
                pass
            return sc

    # Assign target = Call(...)
    if isinstance(node, C.Assign) and isinstance(node.value, C.Call):
        fn_names = _node_text_names(node.value.func)
        if any(n in tset for n in fn_names):
            sc += 1.0
        try:
            for tgt in node.targets:
                tn = tgt.target
                nms = _node_text_names(tn) if isinstance(tn, (C.Name, C.Attribute)) else []
                if any(n in tset for n in nms):
                    sc += 0.5
        except Exception:
            pass
        return sc

    # Await Call(...)
    if isinstance(node, C.Await) and isinstance(node.expression, C.Call):
        fn_names = _node_text_names(node.expression.func)
        if any(n in tset for n in fn_names):
            sc += 1.0
        return sc

    # Raise Exception or Raise Call(Exception(...))
    if isinstance(node, C.Raise):
        exc = node.exc
        if isinstance(exc, C.Call):
            fn_names = _node_text_names(exc.func)
            if any(n in tset for n in fn_names):
                sc += 1.0
        elif isinstance(exc, (C.Name, C.Attribute)):
            fn_names = _node_text_names(exc)
            if any(n in tset for n in fn_names):
                sc += 0.8
        return sc

    # With context manager calls
    if isinstance(node, C.With):
        try:
            for it in node.items:
                expr = it.context_expr
                if isinstance(expr, C.Call):
                    fn_names = _node_text_names(expr.func)
                    if any(n in tset for n in fn_names):
                        sc += 0.8
        except Exception:
            pass
        return sc

    # Decorators on function/class
    if isinstance(node, (cast(Any, cst).FunctionDef, cast(Any, cst).ClassDef)):
        try:
            decs = node.decorators  # type: ignore[attr-defined]
            for d in decs:
                name_nodes = _node_text_names(d.decorator)
                if any(n in tset for n in name_nodes):
                    sc += 0.7
        except Exception:
            pass
        # Name tokens inside definition name
        try:
            nm = _ltxt(getattr(node, "name").value)  # type: ignore[attr-defined]
            if nm in tset:
                sc += 0.6
        except Exception:
            pass
        return sc

    # Imports
    if isinstance(node, C.Import):
        try:
            for n in node.names:
                nm = _ltxt(getattr(n.name, "value", ""))
                if nm in tset or any(p in tset for p in nm.split(".")):
                    sc += 0.7
        except Exception:
            pass
        return sc
    if isinstance(node, C.ImportFrom):
        try:
            mod = _ltxt(getattr(node.module, "value", "") if node.module else "")
            if mod and (mod in tset or any(p in tset for p in mod.split("."))):
                sc += 0.7
            for n in node.names or []:
                nm = _ltxt(getattr(getattr(n, "name", None), "value", ""))
                if nm in tset:
                    sc += 0.6
        except Exception:
            pass
        return sc

    # Type annotations (rough): look inside Annotation nodes
    if isinstance(node, C.Annotation):
        try:
            txt = _ltxt(node.annotation.code)
            for t in tset:
                if t in txt:
                    sc += 0.5
                    break
        except Exception:
            pass
        return sc

    # Logging calls: *.info/debug/warning/error/exception/critical
    if isinstance(node, C.Call):
        fn_names = _node_text_names(node.func)
        if any(n in ("info","debug","warning","error","exception","critical") for n in fn_names):
            sc += 0.6
        if any("log" in n for n in fn_names):
            sc += 0.4
        if any(n in tset for n in fn_names):
            sc += 0.5
        return sc

    return sc


def stage_libcst_hits(query: str, k: int, *, max_time_ms: int | None = 220) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Stage: Python CST-based structural matching for many common query patterns.

    Requires libcst; otherwise returns [].
    """
    if cst is None or m is None or PositionProvider is None:
        return []
    q = (query or "").strip()
    if not q:
        return []
    toks = _tokset(q)
    if not toks:
        return []

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
            mod = cast(Any, cst).parse_module(src)
            wrapper = cast(Any, cst).metadata.MetadataWrapper(mod)
            PP = PositionProvider  # not None due to earlier guard
            pos = wrapper.resolve(PP)
        except Exception:
            continue
        lines = src.splitlines()

        best_for_file: Tuple[float, int, int] | None = None

        BaseVisitor = cast(Any, getattr(cst, "CSTVisitor", object))
        class V(BaseVisitor):
            def on_visit(self, node: Any) -> bool:  # type: ignore[override]
                nonlocal best_for_file
                if time_up():
                    return False
                try:
                    score = _match_interest(node, toks)
                except Exception:
                    score = 0.0
                if score > 0.0:
                    try:
                        p = pos.get(node)  # type: ignore[attr-defined]
                        ps = getattr(p, "start", None)
                        pe = getattr(p, "end", None)
                        sl = int(getattr(ps, "line", 1))
                        el = int(getattr(pe, "line", sl + 24))
                        a = max(1, sl)
                        b = min(len(lines), el)
                    except Exception:
                        a = 1
                        b = min(len(lines), a + 24)
                    cur = (score, a, b)
                    if best_for_file is None or cur[0] > best_for_file[0]:
                        best_for_file = cur
                return True

        try:
            wrapper.visit(V())
        except Exception:
            pass
        if best_for_file is not None:
            sc, a, b = best_for_file
            preview = "\n".join(lines[a - 1 : min(len(lines), b + 12)]).strip()
            obj = {
                "embedding": [],
                "meta": {
                    "file_rel": rel_p,
                    "text_preview": preview,
                    "line_start": a,
                    "line_end": min(len(lines), b + 12),
                },
            }
            # Priority: below AST call-sites and PyFlow, above Jedi/regex
            hits.append((0.9933, rel_p, obj))
            if len(hits) >= k:
                break

    return hits


__all__ = ["stage_libcst_hits"]
