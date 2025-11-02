from __future__ import annotations

import asyncio
import os
import time
from typing import List

# Ensure JINX_MEMORY_DIR is set before importing storage-dependent modules
TEST_DIR = os.path.join(os.getcwd(), ".jinx_test_memory")
os.environ["JINX_MEMORY_DIR"] = TEST_DIR

from jinx.micro.memory.router import assemble_memroute
from jinx.micro.memory.evergreen_select import select_evergreen_for
from jinx.micro.memory.storage import memory_dir


def _ensure_dirs() -> None:
    os.makedirs(memory_dir(), exist_ok=True)


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def prepare_fake_memory() -> None:
    _ensure_dirs()
    memdir = memory_dir()
    compact_path = os.path.join(memdir, "compact.md")
    evergreen_path = os.path.join(memdir, "evergreen.md")
    # Build compact with mixed lines
    comp_lines: List[str] = []
    for i in range(1, 201):
        comp_lines.append(f"User: how to open file number {i} in python?")
        comp_lines.append(f"Jinx: Use open('file_{i}.txt', 'r') and read().")
    _write(compact_path, "\n".join(comp_lines) + "\n")
    # Build evergreen with channels
    evg_lines = [
        "path: src/app/main.py",
        "symbol: assemble_memroute",
        "pref: terse answers",
        "decision: use vector stage",
        "setting: OPENAI_MODEL=<redacted>",
    ]
    _write(evergreen_path, "\n".join(evg_lines) + "\n")


async def run_smoke() -> None:
    prepare_fake_memory()
    q = "open file"  # synthetic query
    # Memroute test
    t0 = time.perf_counter()
    lines = await assemble_memroute(q, k=8, preview_chars=160)
    t1 = (time.perf_counter() - t0) * 1000.0
    print(f"memroute: k={len(lines)} dur={t1:.1f}ms")
    for ln in lines:
        print("  ", ln)
    # Evergreen select test
    t0 = time.perf_counter()
    evg = await select_evergreen_for(q)
    t2 = (time.perf_counter() - t0) * 1000.0
    print(f"evergreen_select: len={len(evg)} dur={t2:.1f}ms")
    if evg:
        snippet = evg.splitlines()[:3]
        for ln in snippet:
            print("  ", ln)


if __name__ == "__main__":
    asyncio.run(run_smoke())
