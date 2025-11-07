from __future__ import annotations

import asyncio
import os
from typing import Any, List, Dict

import jinx.state as jx_state
from jinx.micro.runtime.plugins import register_plugin, subscribe_event, publish_event


async def _locate_for_text(text: str, *, k: int = 5, budget_ms: int = 120) -> List[Dict[str, Any]]:
    try:
        from jinx.micro.runtime.resource_locator import get_resource_locator
        locator = await get_resource_locator()
        prefer_ext = None
        for tok in (text or '').split():
            if '.' in tok:
                _, ext = tok.rsplit('.', 1)
                if ext:
                    prefer_ext = f".{ext.lower().strip('.,;:')}"
                    break
        results = await locator.locate(text, prefer_ext=prefer_ext, k=k, budget_ms=budget_ms)
        out = [
            {
                'path': r.path,
                'rel': r.rel,
                'score': r.score,
                'reason': r.reason,
            }
            for r in results
        ]
        return out
    except Exception:
        return []


async def _start(ctx) -> None:  # type: ignore[no-redef]
    sem = asyncio.Semaphore(int(os.getenv("JINX_LOCATOR_CONC", "3")))

    async def _on_intake(_topic: str, payload: Any) -> None:
        text = str((payload or {}).get('text') or '')
        group = str((payload or {}).get('group') or 'main')
        if not text:
            return
        async with sem:
            try:
                results = await _locate_for_text(text, k=5, budget_ms=120)
                # Store in state for fast access by conversation modules
                try:
                    setattr(jx_state, 'resolved_resources_last', results)
                    if results:
                        setattr(jx_state, 'primary_resource', results[0])
                except Exception:
                    pass
                # Publish event for any consumer
                try:
                    publish_event('locator.results', {
                        'group': group,
                        'text': text,
                        'results': results,
                        'primary': (results[0] if results else None),
                    })
                except Exception:
                    pass
            except Exception:
                pass

    subscribe_event('queue.intake', plugin='resource_locator', callback=_on_intake)


async def _stop(ctx) -> None:  # type: ignore[no-redef]
    return None


# Registration helper for builtin_plugins

def register_resource_locator_plugin() -> None:
    register_plugin(
        'resource_locator',
        start=_start,
        stop=_stop,
        enabled=True,
        priority=35,
        version='1.0.0',
        deps=set(),
        features={'locator'},
    )


__all__ = [
    'register_resource_locator_plugin',
]
