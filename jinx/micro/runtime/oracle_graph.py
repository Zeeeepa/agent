from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple, Iterable
import re as _re

from jinx.micro.embeddings.project_config import ROOT

_PATH = os.path.join(ROOT, ".jinx", "memory", "oracle_graph.json")
_TOK = _re.compile(r"(?u)[\w\.]{3,}")

Graph = Dict[str, Dict[str, float]]  # adjacency: node -> {neighbor: weight}


def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass


def load_graph(max_nodes: int | None = None) -> Graph:
    try:
        with open(_PATH, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
            adj: Graph = obj.get("adj", {})  # type: ignore[assignment]
    except Exception:
        adj = {}
    # light prune on load
    if max_nodes is not None and len(adj) > max_nodes:
        # keep by degree
        items = sorted(adj.items(), key=lambda kv: -sum(float(v) for v in kv[1].values()))[:max_nodes]
        adj = {k: dict(v) for k, v in items}
    return adj


def save_graph(adj: Graph) -> None:
    try:
        _ensure_dir(_PATH)
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"adj": adj, "ts": time.time()}, f, ensure_ascii=False)
        os.replace(tmp, _PATH)
    except Exception:
        pass


def _tokify(text: str, max_n: int = 48) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for m in _TOK.finditer(text or ""):
        t = (m.group(0) or "").strip().lower()
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def _nkey(tok: str) -> str:
    return f"term: {tok}"


def update_from_text(adj: Graph, text: str, *, window: int = 3, w: float = 1.0) -> None:
    toks = _tokify(text)
    nodes = [_nkey(t) for t in toks]
    n = len(nodes)
    for i in range(n):
        a = nodes[i]
        ai = adj.setdefault(a, {})
        for j in range(i + 1, min(i + 1 + window, n)):
            b = nodes[j]
            if b == a:
                continue
            # undirected weight increments (symmetric)
            ai[b] = float(ai.get(b, 0.0)) + w
            bj = adj.setdefault(b, {})
            bj[a] = float(bj.get(a, 0.0)) + w


def prune(adj: Graph, *, max_nodes: int = 8000, max_deg: int = 64) -> None:
    if len(adj) <= max_nodes:
        # still cap degrees
        for k, nbrs in list(adj.items()):
            if len(nbrs) > max_deg:
                items = sorted(nbrs.items(), key=lambda kv: -float(kv[1]))[:max_deg]
                adj[k] = {kk: vv for kk, vv in items}
        return
    # keep by degree sum
    deg: List[Tuple[str, float]] = [(k, sum(float(v) for v in nbrs.values())) for k, nbrs in adj.items()]
    deg.sort(key=lambda x: -x[1])
    keep = {k for k, _ in deg[:max_nodes]}
    for k in list(adj.keys()):
        if k not in keep:
            adj.pop(k, None)
    # cap degrees after prune
    for k, nbrs in list(adj.items()):
        if len(nbrs) > max_deg:
            items = sorted(nbrs.items(), key=lambda kv: -float(kv[1]))[:max_deg]
            adj[k] = {kk: vv for kk, vv in items}


def _norm(nbrs: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in nbrs.values())
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in nbrs.items()}


def predict_ppr(adj: Graph, seeds: Iterable[str], *, alpha: float = 0.2, iters: int = 16, top_k: int = 6) -> List[str]:
    # Build seed vector over existing nodes (use term: namespace)
    seed_nodes: List[str] = []
    seen: set[str] = set()
    for s in seeds or []:
        key = s if s.startswith("term:") else _nkey(s)
        if key in adj and key not in seen:
            seen.add(key)
            seed_nodes.append(key)
    if not seed_nodes:
        return []
    # Prepare transition (row-normalized)
    T: Dict[str, Dict[str, float]] = {u: _norm(nbrs) for u, nbrs in adj.items() if nbrs}
    # Initialize r0 uniformly over seeds
    r: Dict[str, float] = {u: 0.0 for u in T.keys()}
    for u in seed_nodes:
        if u in r:
            r[u] = r.get(u, 0.0) + (1.0 / len(seed_nodes))
    # Power iteration with restart
    for _ in range(max(1, iters)):
        nr: Dict[str, float] = {u: 0.0 for u in r.keys()}
        for u, nbrs in T.items():
            wu = (1.0 - alpha) * r.get(u, 0.0)
            if wu == 0.0:
                continue
            for v, p in nbrs.items():
                if v in nr:
                    nr[v] += wu * p
        # restart to seeds
        for u in seed_nodes:
            if u in nr:
                nr[u] += alpha * (1.0 / len(seed_nodes))
        r = nr
    # Top ranks excluding seeds
    items = sorted(((u, sc) for u, sc in r.items() if u not in seed_nodes), key=lambda x: -x[1])
    out: List[str] = []
    for u, _ in items:
        if u.startswith("term:"):
            tok = u.split(": ", 1)[1]
            out.append(tok)
            if len(out) >= top_k:
                break
    return out


__all__ = [
    "load_graph",
    "save_graph",
    "update_from_text",
    "prune",
    "predict_ppr",
]
