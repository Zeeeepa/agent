from __future__ import annotations

import asyncio
from typing import List, Tuple

from jinx.micro.memory.kb_extract import extract_triplets


LINES: List[str] = [
    # head label
    "label: Value.With.Path",
    # arrows
    "A -> B -> C",
    # equality and assignment
    "Alias == Target",
    "obj.field = some.value",
    # calls & args
    "compute(sum(a, b), item.get(id))",
    # dotted and scope
    "ns::Type.method",
    "pkg.module.symbol",
    # paths
    "src/core/utils/file.py",
    r"C:\\proj\\src\\main.cpp",
    # generics and brackets
    "Map<Key, Value<Nested>>",
    "arr[index]",
    # return types
    "fn(a,b): ReturnType",
    "def g(x,y) -> T[U]",
    # destructuring returns
    "(x, y) = callee(a, b)",
    "[first, second] = build()",
    # object/array literals
    "config = { host: localhost, port: 8080 }",
    "values = [ one, two, three ]",
    # colon types
    "name: Type",
]


async def main() -> None:
    triples: List[Tuple[str, str, str]] = extract_triplets(LINES, max_items=200, max_time_ms=120)
    print(f"triples: {len(triples)}")
    for t in triples:
        print(t)


if __name__ == "__main__":
    asyncio.run(main())
