from __future__ import annotations

import os
from typing import Dict, List

BASE = os.path.join(os.path.dirname(__file__), "micro")


def _list_py_files(dir_path: str) -> List[str]:
    try:
        return [f for f in os.listdir(dir_path) if f.endswith(".py") and not f.startswith("__")]  # noqa: E501
    except FileNotFoundError:
        return []


def _list_dirs(dir_path: str) -> List[str]:
    try:
        return [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    except FileNotFoundError:
        return []


def discover_capabilities() -> Dict[str, Dict[str, int]]:
    """
    Shallow filesystem discovery of micro-module capabilities under `jinx/micro/`.

    Returns a mapping of category -> { files: N, modules: M } where modules are
    subpackages and files are leaf .py modules at that level.
    """
    caps: Dict[str, Dict[str, int]] = {}
    for cat in _list_dirs(BASE):
        path = os.path.join(BASE, cat)
        if not os.path.isdir(path):
            continue
        files = _list_py_files(path)
        submods = [d for d in _list_dirs(path) if os.path.exists(os.path.join(path, d, "__init__.py"))]
        caps[cat] = {
            "files": len(files),
            "modules": len(submods),
        }
    return caps


def dump_capabilities_stdout() -> None:
    print("‖ Capabilities — neon map of the microverse")
    caps = discover_capabilities()
    for cat, stats in sorted(caps.items()):
        print(f"- {cat}: files={stats['files']}, modules={stats['modules']}")
