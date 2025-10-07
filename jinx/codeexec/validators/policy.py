from __future__ import annotations

# Shared policy constants for validators
# Keep this module import-light to avoid cycles.

# Dynamic evaluation/import bans
BANNED_DYN_NAMES = {"eval", "exec", "compile", "__import__"}
BANNED_DYN_ATTRS = {("importlib", "import_module")}

# Network/process/system bans (strict baseline)
BANNED_NET_MODS = {"socket", "ftplib", "telnetlib"}
BANNED_NET_FUNCS = {"system", "popen", "Popen", "call", "check_call", "check_output"}
BANNED_NET_FROM = {
    ("os", "system"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
}

# Heavy frameworks to avoid under hard RT constraints
HEAVY_IMPORTS_TOP = {"torch", "tensorflow", "jax", "pyspark", "dask", "ray"}
