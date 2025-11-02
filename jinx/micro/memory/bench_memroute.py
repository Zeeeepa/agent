from __future__ import annotations

import asyncio
import os
import time
from statistics import median
from typing import List

from jinx.micro.memory.router import assemble_memroute
from jinx.micro.memory.evergreen_select import select_evergreen_for
from jinx.micro.memory.storage import read_compact


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


async def _gather_queries(max_q: int = 10) -> List[str]:
    """Pick recent lines from compact memory as synthetic queries."""
    try:
        raw = await read_compact()
    except Exception:
        raw = ""
    lines = [ln.strip() for ln in (raw or "").splitlines() if (ln or "").strip()]
    if not lines:
        return [
            "search router logic",
            "project configuration file path",
            "function select evergreen facts",
            "planning and code execution",
        ]
    return lines[-max(4, max_q):]


async def bench_once_memroute(q: str, k: int = 8) -> float:
    t0 = time.perf_counter()
    try:
        _ = await assemble_memroute(q, k=k, preview_chars=int(os.getenv("JINX_MACRO_MEM_PREVIEW_CHARS", "160")))
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000.0


async def bench_once_evg(q: str) -> float:
    t0 = time.perf_counter()
    try:
        _ = await select_evergreen_for(q)
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000.0


async def run_bench(iterations: int = 20) -> None:
    queries = await _gather_queries()
    if not queries:
        print("No queries")
        return
    # Warm-up
    for q in queries[:3]:
        await bench_once_memroute(q)
        await bench_once_evg(q)
    # Measure
    mem_durs: List[float] = []
    evg_durs: List[float] = []
    i = 0
    while i < iterations:
        for q in queries:
            mem_durs.append(await bench_once_memroute(q))
            evg_durs.append(await bench_once_evg(q))
            i += 1
            if i >= iterations:
                break
    # Report
    def fmt(xs: List[float]) -> str:
        if not xs:
            return "n/a"
        return f"n={len(xs)} p50={median(xs):.1f}ms p95={_pct(xs,95):.1f}ms max={max(xs):.1f}ms"

    print("memroute:", fmt(mem_durs))
    print("evergreen_select:", fmt(evg_durs))


if __name__ == "__main__":
    asyncio.run(run_bench())
