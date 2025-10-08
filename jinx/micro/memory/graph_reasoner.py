from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple

from jinx.micro.memory.graph import read_graph_nodes, read_graph_edges
from jinx.micro.memory.graph_fast import read_fast_edges, update_fast

_WORD_RE = re.compile(r"(?u)[\w\.]{3,}")


def _tokens(s: str) -> List[str]:
    out: List[str] = []
    for m in _WORD_RE.finditer(s or ""):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 3:
            out.append(t)
    return out


def _node_terms(key: str) -> List[str]:
    # key like 'symbol: foo.bar' -> terms ['foo', 'bar'] ; 'path: a/b.py' -> ['a','b','py'] ; 'term: token'
    s = (key or "").strip().lower()
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    # split by separators
    parts = re.split(r"[\s/\\\.:_\-]+", s)
    return [p for p in parts if p]


def _node_type_weight(key: str) -> float:
    k = (key or "").strip().lower()
    try:
        w_path = float(os.getenv("JINX_MEM_GRAPH_NODEW_PATH", "1.0"))
    except Exception:
        w_path = 1.0
    try:
        w_sym = float(os.getenv("JINX_MEM_GRAPH_NODEW_SYMBOL", "1.2"))
    except Exception:
        w_sym = 1.2
    try:
        w_pref = float(os.getenv("JINX_MEM_GRAPH_NODEW_PREF", "1.1"))
    except Exception:
        w_pref = 1.1
    try:
        w_dec = float(os.getenv("JINX_MEM_GRAPH_NODEW_DECISION", "1.4"))
    except Exception:
        w_dec = 1.4
    try:
        w_term = float(os.getenv("JINX_MEM_GRAPH_NODEW_TERM", "1.0"))
    except Exception:
        w_term = 1.0
    if k.startswith("path: "):
        return w_path
    if k.startswith("symbol: "):
        return w_sym
    if k.startswith("pref: "):
        return w_pref
    if k.startswith("decision: "):
        return w_dec
    if k.startswith("term: "):
        return w_term
    return 1.0


def _topic_for_key(key: str) -> str:
    k = (key or "").strip().lower()
    if k.startswith("path: "):
        p = k[6:].strip()
        segs = re.split(r"[\\/]+", p)
        segs = [s for s in segs if s]
        if not segs:
            return "misc"
        if "jinx" in segs and len(segs) > segs.index("jinx") + 1:
            return segs[segs.index("jinx") + 1]
        return segs[0]
    if k.startswith("symbol: "):
        s = k[8:].strip()
        return s.split(".", 1)[0] if "." in s else "symbols"
    if k.startswith("pref: "):
        return "prefs"
    if k.startswith("decision: "):
        return "decisions"
    if k.startswith("term: "):
        t = k[6:].strip()
        return t.split(".", 1)[0] if "." in t else "terms"
    return "misc"


def _match_seed_nodes(qtoks: List[str], nodes: Dict[str, float], limit: int) -> List[Tuple[str, float]]:
    cand: List[Tuple[int, float, str]] = []
    for k, w in nodes.items():
        terms = _node_terms(k)
        hit = 0
        for t in qtoks:
            if t in terms or t in k.lower():
                hit += 1
        if hit > 0:
            cand.append((hit, float(w), k))
    cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
    # amplitude = hit * log(1 + node_weight)
    import math
    out: List[Tuple[str, float]] = []
    for h, w, k in cand[:limit]:
        amp = float(h) * math.log1p(max(0.0, float(w))) if w > 0 else float(h)
        out.append((k, max(0.5, amp)))
    return out


async def activate(query: str, k: int = 12, steps: int = 2) -> List[Tuple[str, float]]:
    """BDH-inspired local activation on the memory graph.

    - Seeds: nodes whose terms match query tokens (best-effort).
    - Dynamics: few steps of symmetric propagation with damping.
    - Returns top-k nodes (key, score).
    Controls (env):
      JINX_MEM_GRAPH_ALPHA (damping, default 0.85)
      JINX_MEM_GRAPH_SEED_TOP (default 24)
      JINX_MEM_GRAPH_MAX_EDGES (default 40000)
      JINX_MEM_GRAPH_MAX_MS (default 20)
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        alpha = float(os.getenv("JINX_MEM_GRAPH_ALPHA", "0.85"))
    except Exception:
        alpha = 0.85
    try:
        seed_top = int(os.getenv("JINX_MEM_GRAPH_SEED_TOP", "24"))
    except Exception:
        seed_top = 24
    try:
        max_edges = int(os.getenv("JINX_MEM_GRAPH_MAX_EDGES", "40000"))
    except Exception:
        max_edges = 40000
    try:
        max_ms = float(os.getenv("JINX_MEM_GRAPH_MAX_MS", "20"))
    except Exception:
        max_ms = 20.0

    nodes = read_graph_nodes() or {}
    edges = read_graph_edges() or {}
    # Merge fast overlay edges
    try:
        f_edges = read_fast_edges() or {}
        for ek, w in f_edges.items():
            edges[ek] = float(edges.get(ek, 0.0)) + float(w)
    except Exception:
        pass
    if not nodes or not edges:
        return []
    qtoks = _tokens(q)
    seeds = _match_seed_nodes(qtoks, nodes, seed_top)
    if not seeds:
        return []

    act: Dict[str, float] = {}
    for n, amp in seeds:
        act[n] = float(amp) * _node_type_weight(n)
    t0 = time.perf_counter()
    # Prepare small edge list (bounded)
    e_items = list(edges.items())
    if max_edges > 0 and len(e_items) > max_edges:
        e_items = e_items[-max_edges:]

    # Topic constraint
    try:
        use_topic = str(os.getenv("JINX_MEM_GRAPH_TOPIC_CONSTRAINT", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        use_topic = True
    allowed_topics = set(_topic_for_key(s) for s, _a in seeds) if use_topic else set()

    # Edge damping by sign
    try:
        u_exc = float(os.getenv("JINX_MEM_GRAPH_U_EXC", "1.0"))
    except Exception:
        u_exc = 1.0
    try:
        u_inh = float(os.getenv("JINX_MEM_GRAPH_U_INH", "0.75"))
    except Exception:
        u_inh = 0.75

    for _ in range(max(1, steps)):
        new_act: Dict[str, float] = {}
        for ek, w in e_items:
            if (time.perf_counter() - t0) * 1000.0 > max_ms:
                break
            try:
                a, b = ek.split("||", 1)
            except ValueError:
                continue
            if use_topic:
                ta = _topic_for_key(a)
                tb = _topic_for_key(b)
                if ta not in allowed_topics and tb not in allowed_topics:
                    continue
            # apply damping based on sign
            w_adj = (u_exc * w) if w >= 0.0 else (u_inh * w)
            wa = act.get(a, 0.0)
            wb = act.get(b, 0.0)
            if wa > 0.0:
                new_act[b] = new_act.get(b, 0.0) + wa * w_adj
            if wb > 0.0:
                new_act[a] = new_act.get(a, 0.0) + wb * w_adj
        # damping + reseed
        for k0 in list(new_act.keys())[:]:
            new_act[k0] *= alpha
        for s in seeds:
            new_act[s] = new_act.get(s, 0.0) + (1.0 - alpha)
        # Optional clamp: keep Top-K activations per step, clip negatives
        try:
            step_topk = int(os.getenv("JINX_MEM_GRAPH_STEP_TOPK", "256"))
        except Exception:
            step_topk = 256
        # non-negative clamp to avoid runaway inhibitory accumulation
        items = sorted(((max(0.0, v), k) for k, v in new_act.items()), key=lambda x: x[0], reverse=True)
        if step_topk > 0:
            items = items[:step_topk]
        act = {k: v for v, k in items}
        if (time.perf_counter() - t0) * 1000.0 > max_ms:
            break

    ranked = sorted(act.items(), key=lambda x: -x[1])[: max(1, k)]
    # Update fast overlay with positive reinforcement seeds->winners
    try:
        amount = float(os.getenv("JINX_MEM_GRAPH_FAST_AMOUNT", "0.5"))
    except Exception:
        amount = 0.5
    try:
        update_fast([s for s, _a in seeds], [key for key, _ in ranked], amount=amount)
    except Exception:
        pass
    return ranked
