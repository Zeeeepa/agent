from __future__ import annotations

import os
from typing import Dict, List, Callable
from functools import lru_cache
from pathlib import Path

BASE = os.path.join(os.path.dirname(__file__), "micro")


@lru_cache(maxsize=1024)
def _is_python_module(filename: str) -> bool:
    """Intelligent Python module detection with caching."""
    if not filename:
        return False
    
    # Fast path: extension check
    if not filename.endswith('.py'):
        return False
    
    # Exclude common patterns
    name_lower = filename.lower()
    
    # Exclude special/private modules
    if name_lower.startswith('__'):
        return False
    
    # Exclude test and temporary files
    exclude_patterns = (
        'test_', '_test', '.pytest', '.pyc', '.pyo', '.pyd',
        'temp_', '_temp', 'tmp_', '_tmp'
    )
    
    for pattern in exclude_patterns:
        if pattern in name_lower:
            return False
    
    return True


@lru_cache(maxsize=512)
def _is_valid_package(dir_path: str) -> bool:
    """Check if directory is a valid Python package with caching."""
    if not dir_path or not os.path.isdir(dir_path):
        return False
    
    init_path = os.path.join(dir_path, "__init__.py")
    return os.path.exists(init_path)


def _list_py_files(dir_path: str) -> List[str]:
    """List Python module files with intelligent filtering."""
    try:
        return [f for f in os.listdir(dir_path) if _is_python_module(f)]
    except (FileNotFoundError, PermissionError, OSError):
        return []


def _list_dirs(dir_path: str) -> List[str]:
    """List subdirectories with error handling."""
    try:
        entries = os.listdir(dir_path)
        return [
            d for d in entries 
            if os.path.isdir(os.path.join(dir_path, d)) and not d.startswith('.')
        ]
    except (FileNotFoundError, PermissionError, OSError):
        return []


def discover_capabilities() -> Dict[str, Dict[str, int]]:
    """
    Advanced filesystem discovery of micro-module capabilities under `jinx/micro/`.

    Returns a mapping of category -> { files: N, modules: M } where modules are
    subpackages and files are leaf .py modules at that level.
    
    Uses intelligent filtering and caching for performance.
    """
    caps: Dict[str, Dict[str, int]] = {}
    
    try:
        categories = _list_dirs(BASE)
    except Exception:
        return caps
    
    for cat in categories:
        try:
            path = os.path.join(BASE, cat)
            if not os.path.isdir(path):
                continue
            
            files = _list_py_files(path)
            
            # Find valid submodules (directories with __init__.py)
            subdirs = _list_dirs(path)
            submods = [
                d for d in subdirs 
                if _is_valid_package(os.path.join(path, d))
            ]
            
            caps[cat] = {
                "files": len(files),
                "modules": len(submods),
            }
        except Exception:
            # Skip categories that can't be scanned
            continue
    
    return caps


def dump_capabilities_stdout() -> None:
    print("‖ Capabilities — neon map of the microverse")
    caps = discover_capabilities()
    for cat, stats in sorted(caps.items()):
        print(f"- {cat}: files={stats['files']}, modules={stats['modules']}")
