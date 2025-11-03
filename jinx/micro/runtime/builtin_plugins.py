from __future__ import annotations

import asyncio
import os
from typing import Any

from jinx.micro.runtime.plugins import register_plugin, subscribe_event
from jinx.logger.file_logger import append_line as _append
from jinx.log_paths import BLUE_WHISPERS


def register_builtin_plugins() -> None:
    """Register a set of first-party plugins with modern, RT-aware behavior.

    Plugins are off by default unless enabled via JINX_PLUGINS or future toggles.
    - telemetry: lightweight metrics trace to BLUE_WHISPERS (env: 'telemetry:on').
    """

    async def _telemetry_start(ctx) -> None:  # type: ignore[no-redef]
        # local semaphore to avoid log storm
        sem = asyncio.Semaphore(int(os.getenv("JINX_TELEMETRY_CONC", "4")))

        async def _log(topic: str, payload: Any) -> None:
            async with sem:
                try:
                    await _append(BLUE_WHISPERS, f"[telemetry] {topic} {payload}")
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="telemetry", callback=_log)
        subscribe_event("turn.scheduled", plugin="telemetry", callback=_log)
        subscribe_event("turn.finished", plugin="telemetry", callback=_log)
        subscribe_event("turn.metrics", plugin="telemetry", callback=_log)
        subscribe_event("spinner.start", plugin="telemetry", callback=_log)
        subscribe_event("spinner.stop", plugin="telemetry", callback=_log)

    async def _telemetry_stop(ctx) -> None:  # type: ignore[no-redef]
        # No unsubscribe API yet; bus entries will be GC'ed on process exit.
        # Stop hook kept for symmetry and future resource cleanup.
        return None

    register_plugin(
        "telemetry",
        start=_telemetry_start,
        stop=_telemetry_stop,
        enabled=False,  # opt-in via JINX_PLUGINS="telemetry:on"
        priority=50,
        version="1.0.0",
        deps=[],
        features={"telemetry"},
    )

    # Autonomous prefetcher: warms project/dialogue contexts on intake for zero-latency turns
    async def _prefetch_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        from typing import Any

        sem = asyncio.Semaphore(int(os.getenv("JINX_PREFETCH_CONC", "2")))

        async def _do_prefetch(q: str) -> None:
            if not (q or "").strip():
                return
            async with sem:
                try:
                    from jinx.micro.runtime.prefetch_broker import prefetch_project_ctx, prefetch_base_ctx
                except Exception:
                    return
                try:
                    proj_ms = int(os.getenv("JINX_PREFETCH_PROJECT_MS", "260"))
                except Exception:
                    proj_ms = 260
                try:
                    base_ms = int(os.getenv("JINX_PREFETCH_DIALOG_MS", "120"))
                except Exception:
                    base_ms = 120
                await asyncio.gather(
                    prefetch_project_ctx(q, max_time_ms=proj_ms),
                    prefetch_base_ctx(q, max_time_ms=base_ms),
                )

        async def _on_intake(_topic: str, payload: Any) -> None:
            try:
                q = str((payload or {}).get("text") or "").strip()
            except Exception:
                q = ""
            if not q:
                return
            try:
                # Small debounce: yield to allow grouping of burst inputs
                await asyncio.sleep(0)
            except Exception:
                pass
            asyncio.create_task(_do_prefetch(q))

        # Subscribe to queue intake
        subscribe_event("queue.intake", plugin="prefetch", callback=_on_intake)

    async def _prefetch_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "prefetch",
        start=_prefetch_start,
        stop=_prefetch_stop,
        enabled=True,  # autonomous by default
        priority=40,
        version="1.0.0",
        deps=[],
        features={"prefetch"},
    )

    # Cognitive seeds: maintain short-term salient tokens from recent inputs for query expansion
    async def _cog_start(ctx) -> None:  # type: ignore[no-redef]
        import time
        import re
        import jinx.state as jx_state
        from typing import Any

        TOK = re.compile(r"(?u)[\w\.]{3,}")
        try:
            ttl = float(os.getenv("JINX_COG_SEEDS_TTL", "30.0"))
        except Exception:
            ttl = 30.0

        def _update(text: str) -> None:
            s = (text or "").strip()
            if not s:
                return
            seen = set()
            out = []
            for m in TOK.finditer(s):
                t = (m.group(0) or "").strip().lower()
                if len(t) >= 3 and t not in seen:
                    seen.add(t)
                    out.append(t)
                if len(out) >= 12:
                    break
            try:
                jx_state.cog_seeds_terms = out
                jx_state.cog_seeds_ts = float(time.perf_counter())
                jx_state.cog_seeds_ttl = float(ttl)
            except Exception:
                pass

        async def _on_intake(_topic: str, payload: Any) -> None:
            try:
                text = str((payload or {}).get("text") or "")
            except Exception:
                text = ""
            if text:
                _update(text)

        async def _on_finished(_topic: str, payload: Any) -> None:
            # On turn end, expire if TTL elapsed
            try:
                ts = float(getattr(jx_state, "cog_seeds_ts", 0.0) or 0.0)
                tll = float(getattr(jx_state, "cog_seeds_ttl", ttl) or ttl)
            except Exception:
                ts = 0.0; tll = ttl
            if ts > 0.0:
                try:
                    if (time.perf_counter() - ts) >= tll:
                        jx_state.cog_seeds_terms = []
                        jx_state.cog_seeds_ts = 0.0
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="cog", callback=_on_intake)
        subscribe_event("turn.finished", plugin="cog", callback=_on_finished)

    async def _cog_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "cog",
        start=_cog_start,
        stop=_cog_stop,
        enabled=True,  # autonomous by default
        priority=30,
        version="1.0.0",
        deps=[],
        features={"cog"},
    )

    # Autodiscovery: scan micro-packages for auto_init/auto and run auto_start/auto_stop
    async def _auto_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        import importlib
        import os
        from typing import Any

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))  # jinx/micro
        try:
            conc = int(os.getenv("JINX_AUTODISC_MAX_CONC", "3"))
        except Exception:
            conc = 3
        try:
            start_ms = int(os.getenv("JINX_AUTODISC_START_MS", "400"))
        except Exception:
            start_ms = 400
        sem = asyncio.Semaphore(max(1, conc))

        # track for stop
        started: list[tuple[str, Any]] = []

        async def _call_start(mod) -> None:
            try:
                fn = getattr(mod, "auto_start", None)
                if not fn:
                    return
                res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                if asyncio.iscoroutine(res):
                    await asyncio.wait_for(res, timeout=start_ms / 1000.0)  # type: ignore[arg-type]
            except Exception:
                return

        async def _one(pkg: str) -> None:
            async with sem:
                for suffix in ("auto_init", "auto"):
                    modname = f"jinx.micro.{pkg}.{suffix}"
                    try:
                        mod = importlib.import_module(modname)
                    except Exception:
                        continue
                    await _call_start(mod)
                    try:
                        started.append((modname, getattr(mod, "auto_stop", None)))
                    except Exception:
                        started.append((modname, None))
                    break

        # schedule tasks for subpackages
        tasks: list[asyncio.Task] = []
        try:
            for name in os.listdir(base_dir):
                if not name or name.startswith("_"):
                    continue
                p = os.path.join(base_dir, name)
                if os.path.isdir(p):
                    tasks.append(asyncio.create_task(_one(name)))
        except Exception:
            tasks = tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Also execute any autoapi-registered functions (opt-in decorators)
        try:
            from jinx.micro.runtime.autoapi import AUTO_START_FUNCS as _ASF
        except Exception:
            _ASF = []  # type: ignore[assignment]
        if _ASF:
            for fn in list(_ASF):
                try:
                    res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                    if asyncio.iscoroutine(res):
                        await asyncio.wait_for(res, timeout=start_ms / 1000.0)  # type: ignore[arg-type]
                except Exception:
                    continue

        # Store list for stop
        try:
            setattr(ctx, "_autodisc_started", started)
        except Exception:
            pass

    async def _auto_stop(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        try:
            started = list(getattr(ctx, "_autodisc_started", []) or [])
        except Exception:
            started = []
        if not started:
            pass
        try:
            stop_ms = int(os.getenv("JINX_AUTODISC_STOP_MS", "300"))
        except Exception:
            stop_ms = 300
        tasks: list[asyncio.Task] = []
        for _modname, fn in started[::-1]:
            if not fn:
                continue
            try:
                res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                if asyncio.iscoroutine(res):
                    tasks.append(asyncio.create_task(asyncio.wait_for(res, timeout=stop_ms / 1000.0)))  # type: ignore[arg-type]
            except Exception:
                continue
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Also execute autoapi stop functions
        try:
            from jinx.micro.runtime.autoapi import AUTO_STOP_FUNCS as _ZSF
        except Exception:
            _ZSF = []  # type: ignore[assignment]
        if _ZSF:
            ztasks: list[asyncio.Task] = []
            for fn in list(_ZSF):
                try:
                    res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                    if asyncio.iscoroutine(res):
                        ztasks.append(asyncio.create_task(asyncio.wait_for(res, timeout=stop_ms / 1000.0)))  # type: ignore[arg-type]
                except Exception:
                    continue
            if ztasks:
                await asyncio.gather(*ztasks, return_exceptions=True)

    register_plugin(
        "autodiscovery",
        start=_auto_start,
        stop=_auto_stop,
        enabled=True,  # autonomous by default
        priority=20,
        version="1.0.0",
        deps=[],
        features={"autodiscovery"},
    )

    # Foresight: learn token/bigram frequencies and predict likely next tokens
    async def _foresight_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        import os
        import time
        from typing import Any, List
        import jinx.state as jx_state

        try:
            topk = int(os.getenv("JINX_FORESIGHT_TOPK", "4"))
        except Exception:
            topk = 4
        try:
            ttl = float(os.getenv("JINX_FORESIGHT_TTL", "20.0"))
        except Exception:
            ttl = 20.0

        # Lazy imports
        try:
            from jinx.micro.runtime.foresight_store import load_state, save_state, update_tokens, predict_next
        except Exception:
            return

        try:
            from jinx.micro.runtime.prefetch_broker import prefetch_project_ctx, prefetch_base_ctx
        except Exception:
            prefetch_project_ctx = None  # type: ignore
            prefetch_base_ctx = None  # type: ignore

        state = load_state()
        sem = asyncio.Semaphore(int(os.getenv("JINX_FORESIGHT_CONC", "2")))

        async def _prefetch_variants(q: str, terms: List[str]) -> None:
            if not prefetch_project_ctx or not prefetch_base_ctx:
                return
            async with sem:
                proj_ms = int(os.getenv("JINX_FORESIGHT_PROJECT_MS", "240"))
                base_ms = int(os.getenv("JINX_FORESIGHT_DIALOG_MS", "100"))
                tasks = []
                for t in terms[:max(1, topk)]:
                    qv = (q + " " + t).strip()
                    async def _one(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one(qv)))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        async def _on_intake(_topic: str, payload: Any) -> None:
            q = str((payload or {}).get("text") or "").strip()
            if not q:
                return
            # Learn from current input
            update_tokens(state, q, w=1.0)
            save_state(state)
            # Seeds: prefer cognitive seeds when present
            try:
                seeds = list(getattr(jx_state, "cog_seeds_terms", []) or [])
            except Exception:
                seeds = []
            preds = predict_next(state, seeds=seeds, top_k=max(1, topk))
            try:
                jx_state.foresight_terms = preds
                jx_state.foresight_ts = float(time.perf_counter())
                jx_state.foresight_ttl = float(ttl)
            except Exception:
                pass
            # Prefetch predicted variants
            try:
                await _prefetch_variants(q, preds)
            except Exception:
                pass

        # Subscribe to queue intake
        subscribe_event("queue.intake", plugin="foresight", callback=_on_intake)

    async def _foresight_stop(ctx) -> None:  # type: ignore[no-redef]
        # No stateful tasks to cancel here; foresight_store is file-backed
        return None

    register_plugin(
        "foresight",
        start=_foresight_start,
        stop=_foresight_stop,
        enabled=True,  # autonomous by default
        priority=25,
        version="1.0.0",
        deps=[],
        features={"foresight"},
    )

    # Oracle: build a long-horizon term graph and predict next tokens via personalized PageRank
    async def _oracle_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        import os
        from typing import Any, List
        import jinx.state as jx_state

        try:
            topk = int(os.getenv("JINX_ORACLE_TOPK", "6"))
        except Exception:
            topk = 6
        try:
            ttl = float(os.getenv("JINX_ORACLE_TTL", "28.0"))
        except Exception:
            ttl = 28.0

        # Lazy imports
        try:
            from jinx.micro.runtime.oracle_graph import load_graph, save_graph, update_from_text, prune, predict_ppr
        except Exception:
            return
        try:
            from jinx.micro.embeddings.project_retrieval import build_project_context_for as _build_project
            from jinx.micro.embeddings.retrieval import build_context_for as _build_base
            from jinx.micro.embeddings.prefetch_cache import put_project, put_base
        except Exception:
            _build_project = None  # type: ignore
            _build_base = None  # type: ignore
            put_project = None  # type: ignore
            put_base = None  # type: ignore

        adj = load_graph(max_nodes=12000)
        sem = asyncio.Semaphore(int(os.getenv("JINX_ORACLE_CONC", "2")))

        async def _prefetch_variants(q: str, terms: List[str]) -> None:
            if not prefetch_project_ctx or not prefetch_base_ctx:
                return
            async with sem:
                proj_ms = int(os.getenv("JINX_ORACLE_PROJECT_MS", "220"))
                base_ms = int(os.getenv("JINX_ORACLE_DIALOG_MS", "90"))
                tasks = []
                for t in terms[:max(1, topk)]:
                    qv = (q + " " + t).strip()
                    async def _one(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one(qv)))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        async def _on_intake(_topic: str, payload: Any) -> None:
            q = str((payload or {}).get("text") or "").strip()
            if not q:
                return
            # Update long-horizon graph and prune lightly
            update_from_text(adj, q, window=4, w=1.0)
            prune(adj, max_nodes=12000, max_deg=72)
            # Seeds: combine cognitive + foresight
            try:
                seeds = list(getattr(jx_state, "cog_seeds_terms", []) or []) + list(getattr(jx_state, "foresight_terms", []) or [])
            except Exception:
                seeds = []
            preds = predict_ppr(adj, seeds=seeds, top_k=max(1, topk))
            try:
                jx_state.oracle_terms = preds
                jx_state.oracle_ts = float(__import__('time').perf_counter())
                jx_state.oracle_ttl = float(ttl)
            except Exception:
                pass
            # Prefetch predicted variants
            try:
                await _prefetch_variants(q, preds)
            except Exception:
                pass

        async def _on_finished(_topic: str, payload: Any) -> None:
            # Persist graph
            try:
                save_graph(adj)
            except Exception:
                pass

        subscribe_event("queue.intake", plugin="oracle", callback=_on_intake)
        subscribe_event("turn.finished", plugin="oracle", callback=_on_finished)

    async def _oracle_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "oracle",
        start=_oracle_start,
        stop=_oracle_stop,
        enabled=True,  # autonomous by default
        priority=23,
        version="1.0.0",
        deps=[],
        features={"oracle"},
    )

    # Hypersigil: variable-order n-gram model for next-token and short sequence predictions
    async def _hypersigil_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        import os
        import time
        from typing import Any, List
        import jinx.state as jx_state

        try:
            topk = int(os.getenv("JINX_HSIGIL_TOPK", "5"))
        except Exception:
            topk = 5
        try:
            seq_len = int(os.getenv("JINX_HSIGIL_SEQ_LEN", "2"))
        except Exception:
            seq_len = 2
        try:
            ttl = float(os.getenv("JINX_HSIGIL_TTL", "24.0"))
        except Exception:
            ttl = 24.0

        try:
            from jinx.micro.runtime.hypersigil_store import load_store, save_store, update_ngrams, predict_next_tokens, predict_next_sequences
        except Exception:
            return
        try:
            from jinx.micro.embeddings.project_retrieval import build_project_context_for as _build_project
            from jinx.micro.embeddings.retrieval import build_context_for as _build_base
            from jinx.micro.embeddings.prefetch_cache import put_project, put_base
        except Exception:
            _build_project = None  # type: ignore
            _build_base = None  # type: ignore
            put_project = None  # type: ignore
            put_base = None  # type: ignore

        ng = load_store(max_keys=20000)
        sem = asyncio.Semaphore(int(os.getenv("JINX_HSIGIL_CONC", "2")))

        async def _prefetch_variants(q: str, terms: List[str], seqs: List[List[str]]) -> None:
            if not prefetch_project_ctx or not prefetch_base_ctx:
                return
            async with sem:
                proj_ms = int(os.getenv("JINX_HSIGIL_PROJECT_MS", "220"))
                base_ms = int(os.getenv("JINX_HSIGIL_DIALOG_MS", "90"))
                tasks = []
                # single-token variants
                for t in terms[:max(1, topk)]:
                    qv = (q + " " + t).strip()
                    async def _one_tok(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one_tok(qv)))
                # short-sequence variants
                for s in seqs[:max(1, topk)]:
                    if not s:
                        continue
                    qv = (q + " " + " ".join(s)).strip()
                    async def _one_seq(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one_seq(qv)))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        async def _on_intake(_topic: str, payload: Any) -> None:
            q = str((payload or {}).get("text") or "").strip()
            if not q:
                return
            update_ngrams(ng, q, max_order=4, w=1.0)
            # Seeds preference: cog + foresight + oracle
            try:
                seeds = list(getattr(jx_state, "cog_seeds_terms", []) or []) + list(getattr(jx_state, "foresight_terms", []) or []) + list(getattr(jx_state, "oracle_terms", []) or [])
            except Exception:
                seeds = []
            terms = predict_next_tokens(ng, seeds, top_k=max(1, topk))
            seqs = predict_next_sequences(ng, seeds, seq_len=max(1, seq_len), top_k=max(1, topk))
            try:
                jx_state.hsigil_terms = terms
                jx_state.hsigil_seqs = seqs
                jx_state.hsigil_ts = float(time.perf_counter())
                jx_state.hsigil_ttl = float(ttl)
            except Exception:
                pass
            try:
                await _prefetch_variants(q, terms, seqs)
            except Exception:
                pass

        async def _on_finished(_topic: str, payload: Any) -> None:
            try:
                save_store(ng)
            except Exception:
                pass

        subscribe_event("queue.intake", plugin="hypersigil", callback=_on_intake)
        subscribe_event("turn.finished", plugin="hypersigil", callback=_on_finished)

    async def _hypersigil_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "hypersigil",
        start=_hypersigil_start,
        stop=_hypersigil_stop,
        enabled=True,  # autonomous by default
        priority=22,
        version="1.0.0",
        deps=[],
        features={"hypersigil"},
    )

    # Embed Observer: eagerly embed user intake text for global embeddings coverage
    async def _embedobs_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        from typing import Any
        try:
            from jinx.micro.embeddings.pipeline import embed_text as _embed_text
        except Exception:
            _embed_text = None  # type: ignore

        sem = asyncio.Semaphore(int(os.getenv("JINX_EMBEDOBS_CONC", "2")))

        async def _on_intake(_topic: str, payload: Any) -> None:
            if _embed_text is None:
                return
            try:
                text = str((payload or {}).get("text") or "").strip()
            except Exception:
                text = ""
            if not text:
                return
            async with sem:
                try:
                    # Best-effort, background embedding for dialogue-user source
                    await _embed_text(text, source="dialogue", kind="user")
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="embed_observer", callback=_on_intake)

    async def _embedobs_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "embed_observer",
        start=_embedobs_start,
        stop=_embedobs_stop,
        enabled=True,
        priority=45,
        version="1.0.0",
        deps=[],
        features={"embeddings"},
    )

    # Locator semantics: embed-based classification of intake messages (language-agnostic)
    async def _locsem_start(ctx) -> None:  # type: ignore[no-redef]
        from typing import Any
        try:
            from jinx.micro.conversation.locator_semantics import schedule_classify as _sched
        except Exception:
            _sched = None  # type: ignore

        async def _on_intake(_topic: str, payload: Any) -> None:
            if _sched is None:
                return
            try:
                text = str((payload or {}).get("text") or "").strip()
            except Exception:
                text = ""
            if text:
                try:
                    _sched(text)
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="locator_semantics", callback=_on_intake)

    async def _locsem_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "locator_semantics",
        start=_locsem_start,
        stop=_locsem_stop,
        enabled=True,
        priority=44,
        version="1.0.0",
        deps=[],
        features={"embeddings"},
    )


__all__ = ["register_builtin_plugins"]
