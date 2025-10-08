from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Tuple

from jinx.micro.memory.storage import memory_dir
from jinx.micro.embeddings.project_identifiers import extract_identifiers

_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\r\n]+|(?:\.|\.{1,2})?/(?:[^\s/]+/)*[^\s/]+\.[A-Za-z0-9]{1,6})")
_WORD_RE = re.compile(r"(?u)[\w\.]{4,}")
_DIALOG_PREFIX_RE = re.compile(r"^(?:User|Jinx|Error|State|Note):\s*", re.IGNORECASE)
_STOP_TERMS = {"user", "jinx", "error", "state", "note"}


def _graph_path() -> str:
    return os.path.join(memory_dir(), "graph.json")


def _stamp_path() -> str:
    return os.path.join(memory_dir(), ".graph_last_run")


def _load_graph() -> Dict[str, Any]:
    try:
        with open(_graph_path(), "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj  # type: ignore[return-value]
    except Exception:
        pass
    return {"nodes": {}, "edges": {}, "updated_ts": 0}


def read_graph_edges() -> Dict[str, float]:
    """Return edges mapping {'a||b'->weight} from graph.json (best-effort)."""
    try:
        g = _load_graph()
        out: Dict[str, float] = {}
        for ek, ed in (g.get("edges") or {}).items():
            try:
                out[str(ek)] = float((ed or {}).get("w", 0.0))
            except Exception:
                continue
        return out
    except Exception:
        return {}


def read_graph_nodes() -> Dict[str, float]:
    """Return nodes mapping {key->weight} from graph.json (best-effort)."""
    try:
        g = _load_graph()
        out: Dict[str, float] = {}
        for k, nd in (g.get("nodes") or {}).items():
            try:
                out[str(k)] = float(nd.get("w", 0.0))
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _save_graph(g: Dict[str, Any]) -> None:
    try:
        os.makedirs(memory_dir(), exist_ok=True)
        with open(_graph_path(), "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False)
    except Exception:
        pass


def _add_node(g: Dict[str, Any], key: str, ntype: str, w: float) -> None:
    nodes = g.setdefault("nodes", {})
    nd = nodes.get(key) or {"t": ntype, "w": 0.0}
    nd["t"] = ntype
    try:
        nd["w"] = float(nd.get("w", 0.0)) + float(w)
    except Exception:
        nd["w"] = float(w)
    nodes[key] = nd


def _edge_key(a: str, b: str) -> str:
    return a + "||" + b if a <= b else b + "||" + a


def _add_edge(g: Dict[str, Any], a: str, b: str, w: float) -> None:
    if not a or not b or a == b:
        return
    ek = _edge_key(a, b)
    edges = g.setdefault("edges", {})
    ed = edges.get(ek) or {"w": 0.0}
    try:
        ed["w"] = float(ed.get("w", 0.0)) + float(w)
    except Exception:
        ed["w"] = float(w)
    edges[ek] = ed


def _tokens_from_line(line: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (paths, symbols, terms)."""
    # Strip dialogue label prefixes to avoid polluting term graph
    if line:
        line = _DIALOG_PREFIX_RE.sub("", line.strip())
    paths: List[str] = []
    for m in _PATH_RE.finditer(line or ""):
        p = (m.group(0) or "").strip()
        if p:
            paths.append(p.lower())
    syms: List[str] = []
    try:
        for tok in extract_identifiers(line or "", max_items=128):
            t = (tok or "").strip()
            if t:
                syms.append(t.lower())
    except Exception:
        pass
    terms: List[str] = []
    for m in _WORD_RE.finditer(line or ""):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 4:
            if t not in _STOP_TERMS:
                terms.append(t)
    return paths, syms, terms


async def update_graph(compact: str | None, evergreen: str | None) -> None:
    """Update knowledge graph from compact and evergreen memory.

    - Applies decay based on last update time.
    - Adds nodes and edges by co-occurrence in compact lines.
    - Adds light weight to evergreen facts' nodes.
    - Throttled by JINX_MEM_GRAPH_MIN_INTERVAL_MS (default 45000 ms).
    """
    try:
        min_interval = int(os.getenv("JINX_MEM_GRAPH_MIN_INTERVAL_MS", "45000"))
    except Exception:
        min_interval = 45000
    stamp = _stamp_path()
    try:
        st = os.stat(stamp)
        last = int(st.st_mtime * 1000)
    except Exception:
        last = 0
    now = int(time.time() * 1000)
    if min_interval > 0 and (now - last) < min_interval:
        return

    g = _load_graph()

    # decay
    try:
        half_days = float(os.getenv("JINX_MEM_GRAPH_HALF_LIFE_DAYS", "7"))
    except Exception:
        half_days = 7.0
    last_ts = int(g.get("updated_ts") or 0)
    if last_ts > 0 and half_days > 0:
        decay = 0.5 ** (max(0.0, (now - last_ts) / 1000.0) / (half_days * 86400.0))
        # nodes
        nodes = g.get("nodes") or {}
        new_nodes: Dict[str, Any] = {}
        for k, nd in nodes.items():
            try:
                w = float(nd.get("w", 0.0)) * float(decay)
            except Exception:
                w = 0.0
            if w >= 0.4:
                nd["w"] = w
                new_nodes[k] = nd
        g["nodes"] = new_nodes
        # edges
        edges = g.get("edges") or {}
        new_edges: Dict[str, Any] = {}
        for ek, ed in edges.items():
            try:
                w = float(ed.get("w", 0.0)) * float(decay)
            except Exception:
                w = 0.0
            # keep edges by absolute weight to preserve inhibitory links too
            if abs(w) >= 0.3:
                ed["w"] = w
                new_edges[ek] = ed
        g["edges"] = new_edges

    # add evergreen nodes (light weight)
    if evergreen:
        for raw in (evergreen or "").splitlines():
            line = (raw or "").strip()
            low = line.lower()
            if low.startswith("path: "):
                _add_node(g, line, "path", 0.5)
            elif low.startswith("symbol: "):
                _add_node(g, line, "symbol", 0.5)
            elif low.startswith("pref: "):
                _add_node(g, line, "pref", 0.5)
            elif low.startswith("decision: "):
                _add_node(g, line, "decision", 0.5)

    # helper: inhibitory edges from corrections/renames/deprecations
    def _emit_inhib_from_line(line: str) -> List[tuple[str, str, float]]:
        res: List[tuple[str, str, float]] = []
        s = (line or "").strip()
        low = s.lower()
        # pattern: corrected X -> Y (or →)
        if "corrected" in low and ("->" in s or "→" in s):
            try:
                parts = re.split(r"corrected", s, flags=re.IGNORECASE)
                tail = parts[-1]
                if "->" in tail:
                    lpart, rpart = tail.split("->", 1)
                else:
                    lpart, rpart = tail.split("→", 1)
                ltok = (lpart or "").strip()
                rtok = (rpart or "").strip()
                if ltok and rtok:
                    res.append((f"term: {ltok}", f"term: {rtok}", -1.0))
            except Exception:
                pass
        # deprecated/rename/remove lines — add mild inhibitory among extracted symbols/paths
        if any(k in low for k in ("deprecated", "rename", "renamed", "remove", "removed")):
            paths, syms, terms = _tokens_from_line(s)
            toks = [f"path: {p}" for p in paths] + [f"symbol: {x}" for x in syms] + [f"term: {t}" for t in terms]
            for i in range(len(toks)):
                for j in range(i + 1, len(toks)):
                    res.append((toks[i], toks[j], -0.5))
        return res

    # co-occurrence from compact lines
    lines = [(ln or "").strip() for ln in (compact or "").splitlines() if (ln or "").strip()]
    for ln in lines[-800:]:  # bound work
        paths, syms, terms = _tokens_from_line(ln)
        nodes: List[Tuple[str, str]] = []
        for p in paths:
            nodes.append((f"path: {p}", "path"))
        for s in syms[:64]:
            nodes.append((f"symbol: {s}", "symbol"))
        for t in terms[:64]:
            nodes.append((f"term: {t}", "term"))
        # update
        keys = [k for k, _t in nodes]
        for k, t in nodes:
            _add_node(g, k, t, 1.0)
        # connect pairs
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                _add_edge(g, keys[i], keys[j], 1.0)
        # inhibitory edges from this line, if any
        for a, b, w in _emit_inhib_from_line(ln):
            _add_node(g, a, (a.split(":",1)[0]).strip(), 0.2)
            _add_node(g, b, (b.split(":",1)[0]).strip(), 0.2)
            _add_edge(g, a, b, w)

    g["updated_ts"] = now
    # Prune graph size for RT
    try:
        max_nodes = int(os.getenv("JINX_MEM_GRAPH_MAX_NODES", "0"))
    except Exception:
        max_nodes = 0
    try:
        max_edges = int(os.getenv("JINX_MEM_GRAPH_MAX_EDGES", "0"))
    except Exception:
        max_edges = 0
    try:
        if max_nodes > 0:
            items = sorted((g.get("nodes") or {}).items(), key=lambda kv: -abs(float((kv[1] or {}).get("w", 0.0))))
            if len(items) > max_nodes:
                g["nodes"] = {k: v for k, v in items[:max_nodes]}
        if max_edges > 0:
            eitems = sorted((g.get("edges") or {}).items(), key=lambda kv: -abs(float((kv[1] or {}).get("w", 0.0))))
            if len(eitems) > max_edges:
                g["edges"] = {k: v for k, v in eitems[:max_edges]}
    except Exception:
        pass
    _save_graph(g)
    try:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception:
        pass


async def query_graph(term: str, k: int = 8) -> List[str]:
    """Query neighbors for a node matching term (substring match on keys)."""
    q = (term or "").strip().lower()
    if not q:
        return []
    g = _load_graph()
    nodes = g.get("nodes") or {}
    edges = g.get("edges") or {}
    # find matching node keys
    cand: List[str] = [nk for nk in nodes.keys() if q in nk.lower()]
    if not cand:
        return []
    # aggregate neighbor scores
    scores: Dict[str, float] = {}
    for a in cand:
        for ek, ed in edges.items():
            if a in ek:
                b = ek.split("||")
                other = b[1] if b[0] == a else (b[0] if b[1] == a else None)
                if not other:
                    continue
                w = float(ed.get("w", 0.0)) * float((nodes.get(other) or {}).get("w", 1.0))
                scores[other] = scores.get(other, 0.0) + w
    ranked = sorted(scores.items(), key=lambda x: -x[1])[: max(1, k)]
    out: List[str] = []
    for key, sc in ranked:
        out.append(f"{key} ({sc:.2f})")
    return out
