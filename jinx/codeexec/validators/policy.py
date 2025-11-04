from __future__ import annotations

import os
from typing import Set, FrozenSet, Tuple
from functools import lru_cache

# Shared policy constants for validators
# Keep this module import-light to avoid cycles.


class PolicyConfig:
    """Extensible policy configuration with environment override support."""
    
    # Dynamic evaluation/import bans (immutable baseline)
    _BANNED_DYN_NAMES_BASE: FrozenSet[str] = frozenset({
        "eval", "exec", "compile", "__import__"
    })
    
    _BANNED_DYN_ATTRS_BASE: FrozenSet[Tuple[str, str]] = frozenset({
        ("importlib", "import_module")
    })
    
    # Network/process/system bans (strict baseline)
    _BANNED_NET_MODS_BASE: FrozenSet[str] = frozenset({
        "socket", "ftplib", "telnetlib", "http.client", "urllib3"
    })
    
    _BANNED_NET_FUNCS_BASE: FrozenSet[str] = frozenset({
        "system", "popen", "Popen", "call", "check_call", "check_output"
    })
    
    _BANNED_NET_FROM_BASE: FrozenSet[Tuple[str, str]] = frozenset({
        ("os", "system"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "check_call"),
        ("subprocess", "check_output"),
    })
    
    # Heavy frameworks to avoid under hard RT constraints
    _HEAVY_IMPORTS_BASE: FrozenSet[str] = frozenset({
        "torch", "tensorflow", "jax", "pyspark", "dask", "ray",
        "keras", "mxnet", "chainer"
    })
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_banned_dyn_names() -> Set[str]:
        """Get dynamic call bans with optional environment extension."""
        base = set(PolicyConfig._BANNED_DYN_NAMES_BASE)
        # Allow runtime extension via env var (comma-separated)
        extra = os.getenv("JINX_BANNED_DYN_EXTRA", "").strip()
        if extra:
            base.update(n.strip() for n in extra.split(",") if n.strip())
        return base
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_banned_dyn_attrs() -> Set[Tuple[str, str]]:
        """Get dynamic attribute bans with optional environment extension."""
        return set(PolicyConfig._BANNED_DYN_ATTRS_BASE)
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_banned_net_mods() -> Set[str]:
        """Get network module bans with optional environment extension."""
        base = set(PolicyConfig._BANNED_NET_MODS_BASE)
        extra = os.getenv("JINX_BANNED_NET_MODS_EXTRA", "").strip()
        if extra:
            base.update(m.strip() for m in extra.split(",") if m.strip())
        return base
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_banned_net_funcs() -> Set[str]:
        """Get network function bans."""
        return set(PolicyConfig._BANNED_NET_FUNCS_BASE)
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_banned_net_from() -> Set[Tuple[str, str]]:
        """Get network from-import bans."""
        return set(PolicyConfig._BANNED_NET_FROM_BASE)
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_heavy_imports() -> Set[str]:
        """Get heavy import bans with optional environment extension."""
        base = set(PolicyConfig._HEAVY_IMPORTS_BASE)
        extra = os.getenv("JINX_HEAVY_IMPORTS_EXTRA", "").strip()
        if extra:
            base.update(m.strip() for m in extra.split(",") if m.strip())
        return base


# Backward compatibility: expose as module-level constants
# These now use the extensible config system
BANNED_DYN_NAMES = PolicyConfig.get_banned_dyn_names()
BANNED_DYN_ATTRS = PolicyConfig.get_banned_dyn_attrs()
BANNED_NET_MODS = PolicyConfig.get_banned_net_mods()
BANNED_NET_FUNCS = PolicyConfig.get_banned_net_funcs()
BANNED_NET_FROM = PolicyConfig.get_banned_net_from()
HEAVY_IMPORTS_TOP = PolicyConfig.get_heavy_imports()
