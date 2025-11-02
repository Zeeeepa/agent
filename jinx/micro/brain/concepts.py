from __future__ import annotations

import os
import json
import asyncio
import time
from typing import Any, Dict, List, Tuple

from .paths import ensure_brain_dirs, CONCEPTS_PATH
from jinx.micro.embeddings.project_paths import PROJECT_INDEX_DIR
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT
from jinx.log_paths import TRIGGER_ECHOES, BLUE_WHISPERS
from jinx.micro.memory.storage import read_compact, read_evergreen
from jinx.micro.memory.graph import read_graph_edges, read_graph_nodes
from jinx.micro.brain.attention import get_attention_weights as _atten_get
from jinx.micro.brain.scanners import (
    scan_import_graph as _scan_import_graph_mod,
    scan_error_classes as _scan_error_classes_mod,
    scan_framework_markers as _scan_framework_markers_mod,
)
from .paths import BRAIN_ROOT
from jinx.micro.brain.scanners.project import scan_project_concepts as _scan_project_mod
from jinx.micro.brain.scanners.memory import scan_memory_concepts as _scan_memory_mod


def _now_ms() -> int:
    try:
        return int(time.time() * 1000)
    except Exception:
        return 0


async def _read_json(path: str) -> Dict[str, Any]:
    try:
        def _load() -> Dict[str, Any]:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
        return await asyncio.to_thread(_load)
    except Exception:
        return {}


async def _write_json(path: str, obj: Dict[str, Any]) -> None:
    try:
        def _save() -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
        await asyncio.to_thread(_save)
    except Exception:
        pass


def _add(node_map: Dict[str, float], key: str, w: float) -> None:
    try:
        node_map[key] = float(node_map.get(key, 0.0)) + float(w)
    except Exception:
        node_map[key] = float(w)


async def _scan_project_concepts() -> Dict[str, float]:
    """Delegate to micro-module project scanner (RT-bounded internally)."""
    try:
        return await _scan_project_mod()
    except Exception:
        return {}


async def _scan_memory_concepts() -> Dict[str, float]:
    """Delegate to micro-module memory scanner (RT-bounded internally)."""
    try:
        return await _scan_memory_mod()
    except Exception:
        return {}


async def _scan_import_graph() -> Dict[str, float]:
    """Lightweight import graph scan from indexed JSON files.

    Reads a bounded number of project index json files (PROJECT_INDEX_DIR) and extracts
    module import names (if present under key 'imports'). Adds concepts as 'import: <module>'.
    Bounded to respect hard RT constraints.
    """
    nodes: Dict[str, float] = {}
    try:
        entries = [os.path.join(PROJECT_INDEX_DIR, f) for f in os.listdir(PROJECT_INDEX_DIR) if f.endswith('.json')]
    except Exception:
        entries = []
    cnt = 0
    for p in entries:
        cnt += 1
        if cnt > 400:
            break
        try:
            obj = await _read_json(p)
            imports = obj.get('imports') or []
            if isinstance(imports, dict):
                imports = list(imports.keys())
            imp_list = list(imports)[:64]
            for imp in imp_list:
                try:
                    mod = str(imp or '').strip().split('.')[0].lower()
                    if mod:
                        _add(nodes, f"import: {mod}", 0.5)
                except Exception:
                    continue
        except Exception:
            continue
    return nodes


async def _scan_error_classes() -> Dict[str, float]:
    """Scan recent log tails for error class names and add as 'error: <Class>'."""
    nodes: Dict[str, float] = {}
    paths = [TRIGGER_ECHOES, BLUE_WHISPERS]
    for fp in paths:
        try:
            if not fp or not os.path.exists(fp):
                continue
            def _tail() -> str:
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read()
                        # keep tail to bound work
                        return txt[-40000:]
                except Exception:
                    return ''
            tail = await asyncio.to_thread(_tail)
            if not tail:
                continue
            import re as _re
            for m in _re.finditer(r"(?m)\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b", tail):
                name = (m.group(1) or '').strip()
                if name:
                    _add(nodes, f"error: {name}", 1.5)
        except Exception:
            continue
    return nodes


async def _scan_framework_markers() -> Dict[str, float]:
    """Detect frameworks/tools by combining multiple bounded signals (deps, scripts, configs, indexes, tools).

    Adds 'framework: <name>' and 'lang: <name>' concepts with cumulative weights.
    All operations are RT-bounded and rely on file existence checks or tiny reads.
    """
    nodes: Dict[str, float] = {}
    import time as _tm
    t0 = _tm.monotonic()
    def _timeup(ms: int = 240) -> bool:
        try:
            return (_tm.monotonic() - t0) * 1000.0 > ms
        except Exception:
            return False

    def _read_text_safe(path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ''

    # -----------------------------
    # NodeJS signals (package.json)
    # -----------------------------
    try:
        pkg = os.path.join(PROJECT_ROOT, 'package.json')
        if os.path.exists(pkg):
            def _read_pkg() -> dict:
                try:
                    import json as _json
                    with open(pkg, 'r', encoding='utf-8') as f:
                        return _json.load(f) or {}
                except Exception:
                    return {}
            obj = await asyncio.to_thread(_read_pkg)
            deps = {}
            deps.update(obj.get('dependencies') or {})
            deps.update(obj.get('devDependencies') or {})
            scripts = obj.get('scripts') or {}
            keys = {str(k).strip().lower() for k in deps.keys()}
            # Dependency-based detection
            dep_map = {
                'react': 2.0,
                'next': 2.5,
                'vue': 2.0,
                'nuxt': 2.5,
                'svelte': 2.0,
                'angular': 2.0,
                'express': 1.8,
                'koa': 1.6,
                'nestjs': 2.2,
                'vite': 1.6,
                'webpack': 1.4,
                'astro': 1.8,
                'gatsby': 1.8,
                'storybook': 1.4,
            }
            for k in list(keys)[:2000]:
                if k in dep_map:
                    _add(nodes, f"framework: {k}", dep_map[k])
                    _add(nodes, "lang: node", 1.0)
            # Script-based detection (more confident)
            s_low = {str(a).lower(): str(b).lower() for a, b in (scripts or {}).items()}
            script_map = {
                'next': ('next dev', 'next build', 'next start'),
                'react': ('react-scripts',),
                'vue': ('vue-cli-service',),
                'nuxt': ('nuxt',),
                'svelte': ('svelte-kit', 'vite dev'),
                'angular': ('ng ', 'angular-cli'),
                'nestjs': ('nest ',),
                'vite': ('vite ',),
                'webpack': ('webpack ',),
                'storybook': ('storybook ',),
            }
            for fw, patt in script_map.items():
                try:
                    for name, cmd in s_low.items():
                        if any(p in cmd for p in patt):
                            _add(nodes, f"framework: {fw}", 1.2)
                            _add(nodes, "lang: node", 0.6)
                            break
                except Exception:
                    continue
            # Config file presence
            cfg_files = {
                'next': ('next.config.js', 'next.config.mjs', 'next.config.ts'),
                'nuxt': ('nuxt.config.js', 'nuxt.config.ts'),
                'svelte': ('svelte.config.js', 'svelte.config.ts'),
                'vite': ('vite.config.js', 'vite.config.ts'),
                'webpack': ('webpack.config.js',),
                'angular': ('angular.json',),
                'nestjs': ('nest-cli.json',),
                'storybook': ('.storybook',),
            }
            for fw, files in cfg_files.items():
                for fname in files:
                    if _timeup():
                        break
                    try:
                        path = os.path.join(PROJECT_ROOT, fname)
                        if os.path.exists(path):
                            _add(nodes, f"framework: {fw}", 0.8)
                            _add(nodes, "lang: node", 0.4)
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    if _timeup():
        return nodes

    # ----------------------------------------
    # Python signals (pyproject/requirements + index imports)
    # ----------------------------------------
    try:
        pyproj = os.path.join(PROJECT_ROOT, 'pyproject.toml')
        reqs = os.path.join(PROJECT_ROOT, 'requirements.txt')
        text = ''
        if os.path.exists(pyproj):
            text += (await asyncio.to_thread(_read_text_safe, pyproj))[-60000:]
        if os.path.exists(reqs):
            text += '\n' + (await asyncio.to_thread(_read_text_safe, reqs))[-60000:]
        if text:
            low = text.lower()
            for lib, fw in (
                ("django", "django"), ("fastapi", "fastapi"), ("flask", "flask"), ("starlette", "starlette"),
                ("pydantic", "pydantic"), ("celery", "celery"), ("sqlalchemy", "sqlalchemy"),
            ):
                if lib in low:
                    _add(nodes, f"framework: {fw}", 2.0)
                    _add(nodes, "lang: python", 1.0)
    except Exception:
        pass

    # Imports from project index (quick pass)
    try:
        entries = [os.path.join(PROJECT_INDEX_DIR, f) for f in os.listdir(PROJECT_INDEX_DIR) if f.endswith('.json')]
    except Exception:
        entries = []
    seen_fw: set[str] = set()
    for p in entries[:120]:
        if _timeup():
            break
        try:
            obj = await _read_json(p)
            imports = obj.get('imports') or []
            if isinstance(imports, dict):
                imports = list(imports.keys())
            s = {str(x).lower() for x in imports[:128]}
            for lib, fw in (('django','django'),('fastapi','fastapi'),('flask','flask'),('starlette','starlette'),('celery','celery'),('sqlalchemy','sqlalchemy')):
                if fw in seen_fw:
                    continue
                if lib in s:
                    _add(nodes, f"framework: {fw}", 1.2)
                    _add(nodes, "lang: python", 0.6)
                    seen_fw.add(fw)
        except Exception:
            continue

    if _timeup():
        return nodes

    # -----------------
    # Tooling markers
    # -----------------
    try:
        # Poetry/Pipenv/Conda
        if os.path.exists(os.path.join(PROJECT_ROOT, 'Pipfile')):
            _add(nodes, 'framework: pipenv', 0.8)
            _add(nodes, 'lang: python', 0.4)
        # Poetry: pyproject already read above; check marker string quickly (avoid re-read if possible)
        try:
            if 'text' in locals() and ('[tool.poetry]' in text or 'tool.poetry' in (text or '')):
                _add(nodes, 'framework: poetry', 0.8)
                _add(nodes, 'lang: python', 0.4)
        except Exception:
            pass
        if os.path.exists(os.path.join(PROJECT_ROOT, 'environment.yml')) or os.path.exists(os.path.join(PROJECT_ROOT, 'environment.yaml')):
            _add(nodes, 'framework: conda', 0.8)
            _add(nodes, 'lang: python', 0.4)
        # Docker / Compose
        if os.path.exists(os.path.join(PROJECT_ROOT, 'Dockerfile')):
            _add(nodes, 'framework: docker', 1.2)
        if os.path.exists(os.path.join(PROJECT_ROOT, 'docker-compose.yml')) or os.path.exists(os.path.join(PROJECT_ROOT, 'docker-compose.yaml')):
            _add(nodes, 'framework: docker-compose', 1.0)
        # Kubernetes (shallow)
        for fn in ('k8s', 'deployment.yaml', 'deployment.yml'):
            try:
                if os.path.exists(os.path.join(PROJECT_ROOT, fn)):
                    _add(nodes, 'framework: kubernetes', 1.0)
                    break
            except Exception:
                continue
    except Exception:
        pass

    return nodes


async def ensure_concepts_fresh(max_age_sec: int = 120, *, max_time_ms: int | None = None) -> Dict[str, Any]:
    """Ensure concepts snapshot exists and is fresh; return the snapshot object.

    Snapshot format: {"ts": ms, "nodes": {concept_key->weight}}
    """
    ensure_brain_dirs()
    try:
        st = os.stat(CONCEPTS_PATH)
        age_sec = max(0.0, (time.time() - st.st_mtime))
    except Exception:
        age_sec = 1e9
    if age_sec <= max_age_sec:
        snap = await _read_json(CONCEPTS_PATH)
        if snap:
            return snap
    # Rebuild snapshot (micro-modular scanners + local ones), RT-bounded
    tasks = [
        asyncio.create_task(_scan_project_concepts()),
        asyncio.create_task(_scan_memory_concepts()),
        asyncio.create_task(_scan_import_graph_mod()),
        asyncio.create_task(_scan_error_classes_mod()),
        asyncio.create_task(_scan_framework_markers_mod(PROJECT_ROOT)),
    ]
    results: List[Dict[str, float]] = []
    if max_time_ms is not None and max_time_ms > 0:
        try:
            rr = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=max_time_ms / 1000.0)
        except Exception:
            rr = []
            for t in tasks:
                if t.done() and (not t.cancelled()):
                    try:
                        r = t.result()
                    except Exception:
                        r = {}
                    rr.append(r if isinstance(r, dict) else {})
        for r in rr:
            results.append(r if isinstance(r, dict) else {})
    else:
        rr = await asyncio.gather(*tasks, return_exceptions=True)
        for r in rr:
            results.append(r if isinstance(r, dict) else {})
    proj_nodes, mem_nodes, imp_nodes, err_nodes, fw_nodes = (results + [{}] * 5)[:5]
    nodes: Dict[str, float] = {}
    for d in (proj_nodes, mem_nodes, imp_nodes, err_nodes, fw_nodes):
        for k, v in d.items():
            _add(nodes, k, float(v or 0.0))
    # Apply reinforcement layer (optional, with decay)
    try:
        reinf = await _read_reinforce()
        if reinf:
            _apply_reinforce(nodes, reinf)
    except Exception:
        pass
    snap = {"ts": _now_ms(), "nodes": nodes}
    await _write_json(CONCEPTS_PATH, snap)
    return snap


def _edge_key(a: str, b: str) -> str:
    return a + "||" + b if a <= b else b + "||" + a


def _query_terms(q: str) -> List[str]:
    import re as _re
    out: List[str] = []
    for m in _re.finditer(r"(?u)[\w\.]{3,}", (q or "")):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 3:
            out.append(t)
    # dedupe
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _type_weight(key: str) -> float:
    low = (key or "").lower()
    if low.startswith("symbol: "):
        return 1.3
    if low.startswith("path: "):
        return 1.15
    if low.startswith("error: "):
        return 1.2
    if low.startswith("framework: "):
        return 1.1
    if low.startswith("import: "):
        return 1.05
    if low.startswith("lang: "):
        return 0.9
    if low.startswith("pref: "):
        return 1.1
    if low.startswith("decision: "):
        return 1.05
    return 1.0


async def _idf_for_terms(q_terms: List[str]) -> Dict[str, float]:
    import math as _m
    idf: Dict[str, float] = {t: 1.0 for t in q_terms}
    if not q_terms:
        return idf


# --- Reinforcement state (optional) ---------------------------------------------------------
async def _read_reinforce() -> Dict[str, Any]:
    path = os.path.join(BRAIN_ROOT, "reinforce.json")
    try:
        return await _read_json(path)
    except Exception:
        return {}


def _apply_reinforce(nodes: Dict[str, float], reinforce: Dict[str, Any]) -> None:
    try:
        rn = (reinforce.get("nodes") or {}) if isinstance(reinforce, dict) else {}
        ts = float(reinforce.get("ts") or 0.0) if isinstance(reinforce, dict) else 0.0
        # Recency decay for reinforcement
        age_ms = max(0.0, float(_now_ms() - ts)) if ts else 0.0
        try:
            half_sec = max(10.0, float(os.getenv("JINX_BRAIN_REINF_HALF_SEC", "300")))
        except Exception:
            half_sec = 300.0
        decay = 0.5 ** (age_ms / 1000.0 / half_sec) if half_sec > 0 else 1.0
        for k, v in rn.items():
            try:
                nodes[k] = float(nodes.get(k, 0.0)) + float(v or 0.0) * float(decay)
            except Exception:
                continue
    except Exception:
        pass


# --- Semantic co-occurrence (bounded) -------------------------------------------------------
async def _semantic_cooccur_map(q_terms: List[str], *, time_budget_ms: int | None = 80) -> Dict[str, float]:
    """Compute a tiny PMI-like co-occurrence score for query terms using indexed chunks.

    Returns map term->score. Bounded by file and time budgets for RT.
    """
    import time as _tm
    t0 = _tm.monotonic()
    def _timeup() -> bool:
        if time_budget_ms is None or time_budget_ms <= 0:
            return False
        return (_tm.monotonic() - t0) * 1000.0 > time_budget_ms
    out: Dict[str, float] = {t: 0.0 for t in q_terms}
    if not q_terms:
        return out
    try:
        entries = [os.path.join(PROJECT_INDEX_DIR, f) for f in os.listdir(PROJECT_INDEX_DIR) if f.endswith('.json')]
    except Exception:
        entries = []
    import math as _m
    N = 0
    df: Dict[str, int] = {t: 0 for t in q_terms}
    co: Dict[tuple, int] = {}
    for p in entries[:200]:
        if _timeup():
            break
        try:
            obj = await _read_json(p)
            chunks = obj.get('chunks') or []
            for ch in chunks[:64]:
                terms = ch.get('terms') or []
                s = {str(t).lower() for t in terms[:256]}
                present = [t for t in q_terms if t in s]
                if present:
                    N += 1
                    for t in set(present):
                        df[t] = df.get(t, 0) + 1
                    # pair co-occurrence
                    for i in range(len(present)):
                        for j in range(i+1, len(present)):
                            a, b = sorted((present[i], present[j]))
                            co[(a, b)] = co.get((a, b), 0) + 1
        except Exception:
            continue
    if N <= 0:
        return out
    for (a, b), c in co.items():
        try:
            pa = df.get(a, 0) / float(N)
            pb = df.get(b, 0) / float(N)
            pab = c / float(N)
            if pab > 0 and pa > 0 and pb > 0:
                pmi = max(0.0, _m.log(pab / (pa * pb)))
                out[a] = out.get(a, 0.0) + pmi
                out[b] = out.get(b, 0.0) + pmi
        except Exception:
            continue
    return out

async def record_reinforcement(keys: List[str], weight: float = 1.0) -> None:
    """Persist lightweight reinforcement updates for concept nodes.

    keys: list of concept keys like 'path: ...', 'symbol: ...', 'framework: ...', etc.
    """
    try:
        path = os.path.join(BRAIN_ROOT, "reinforce.json")
        obj = await _read_json(path)
        nodes = obj.get("nodes") or {}
        for k in (keys or [])[:64]:
            try:
                nodes[k] = float(nodes.get(k, 0.0)) + float(weight or 0.0)
            except Exception:
                continue
        obj = {"ts": _now_ms(), "nodes": nodes}
        await _write_json(path, obj)
    except Exception:
        pass


async def activate_concepts(query: str, top_k: int = 16, *, max_time_ms: int | None = None) -> List[Tuple[str, float]]:
    """Return top concept keys by activation given a query.

    Activation combines (bounded RT):
    - Base node weight for concepts matching query terms (substring) with recency decay.
    - KG prior (node prior from graph nodes) and query-term edge boosts.
    - Attention boost for recently activated concepts/terms.
    """
    snap = await ensure_concepts_fresh(max_age_sec=120, max_time_ms=max_time_ms)
    nodes: Dict[str, float] = snap.get("nodes", {})
    if not nodes:
        return []
    q_terms = _query_terms(query)
    edges = read_graph_edges()      # {'a||b': w}
    g_nodes = read_graph_nodes()    # {key->w}

    # Recency decay based on snapshot timestamp (half-life sec)
    try:
        half_sec = max(30.0, float(os.getenv("JINX_BRAIN_DECAY_HALF_SEC", "300")))
    except Exception:
        half_sec = 300.0
    try:
        age_ms = max(0.0, float(_now_ms() - float(snap.get("ts") or 0.0)))
    except Exception:
        age_ms = 0.0
    import math as _m
    decay = 0.5 ** (age_ms / 1000.0 / half_sec) if half_sec > 0 else 1.0

    # Attention boost
    try:
        att = _atten_get() or {}
    except Exception:
        att = {}
    try:
        att_gamma = float(os.getenv("EMBED_RERANK_ATTEN_GAMMA", "0.35"))
    except Exception:
        att_gamma = 0.35
    try:
        kg_gamma = float(os.getenv("JINX_BRAIN_KG_PRIOR_GAMMA", "0.2"))
    except Exception:
        kg_gamma = 0.2

    scores: Dict[str, float] = {}
    # Base matching with decay and type weights + IDF
    # Allocate budgets across semantic/IDF if provided
    try:
        idf_budget = int(os.getenv("JINX_BRAIN_IDF_MS", "60"))
    except Exception:
        idf_budget = 60
    idf_map = await _idf_for_terms(q_terms, time_budget_ms=idf_budget if (max_time_ms or 0) > 0 else None)
    # Semantic co-occurrence boost (bounded)
    try:
        sem_budget = int(os.getenv("JINX_BRAIN_SEM_MS", "80"))
    except Exception:
        sem_budget = 80
    sem_map = await _semantic_cooccur_map(q_terms, time_budget_ms=sem_budget)
    try:
        sem_gamma = float(os.getenv("JINX_BRAIN_SEM_GAMMA", "0.25"))
    except Exception:
        sem_gamma = 0.25
    for k, w in nodes.items():
        try:
            low = k.lower()
            base = 0.0
            for t in q_terms:
                if t in low:
                    base += idf_map.get(t, 1.0) + sem_gamma * float(sem_map.get(t, 0.0))
            if base > 0:
                scores[k] = scores.get(k, 0.0) + base * float(w or 0.0) * float(decay) * _type_weight(k)
        except Exception:
            continue

    # KG contributions: node prior and query-term edges (bounded)
    lim_nodes = 20000
    for k in list(nodes.keys())[:lim_nodes]:
        try:
            prior = float(g_nodes.get(k, 0.0))
            if prior != 0.0:
                scores[k] = scores.get(k, 0.0) + kg_gamma * prior
            kg_sum = 0.0
            for t in q_terms:
                ek = _edge_key(k, f"term: {t}")
                w = float(edges.get(ek, 0.0))
                if w != 0.0:
                    kg_sum += w * max(prior, 0.5)
            if kg_sum != 0.0:
                scores[k] = scores.get(k, 0.0) + kg_sum
        except Exception:
            continue

    # Attention boost: exact-match keys or partial term hits
    if att:
        for k in list(scores.keys()):
            try:
                boost = float(att.get(k, 0.0))
                if boost == 0.0:
                    # Approx: sum attention for terms present in key
                    for t in q_terms:
                        akey = f"term: {t}"
                        av = float(att.get(akey, 0.0))
                        if av != 0.0 and t in k.lower():
                            boost += av * 0.5
                if boost != 0.0:
                    scores[k] = scores.get(k, 0.0) * (1.0 + att_gamma * max(0.0, boost))
            except Exception:
                continue

    ranked = sorted(scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[: max(1, top_k)]
    return ranked
