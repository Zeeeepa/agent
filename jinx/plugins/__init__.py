"""
Jinx Plugin System - Dynamic tool loading and repository integration.

This package provides a comprehensive plugin architecture for extending Jinx
with external tools and capabilities from other repositories.
"""

from jinx.plugins.base import BasePlugin, Tool, ToolSpec
from jinx.plugins.registry import PluginRegistry, PluginMetadata
from jinx.plugins.loader import PluginLoader
from jinx.plugins.analyzer import RepoAnalyzer, RepoAnalysis, FunctionInfo, ClassInfo
from jinx.plugins.tool_wrapper import ToolWrapper
from jinx.plugins.discovery import PluginDiscovery
from jinx.plugins.runtime_integration import integrate_plugin, list_active_plugins

__all__ = [
    "BasePlugin",
    "Tool",
    "ToolSpec",
    "PluginRegistry",
    "PluginMetadata",
    "PluginLoader",
    "RepoAnalyzer",
    "RepoAnalysis",
    "FunctionInfo",
    "ClassInfo",
    "ToolWrapper",
    "PluginDiscovery",
    "integrate_plugin",
    "list_active_plugins",
]

__version__ = "1.0.0"

