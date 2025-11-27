"""
Runtime integration - integrate plugins with Jinx runtime system.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from jinx.plugins.base import BasePlugin, Tool
from jinx.plugins.registry import PluginRegistry, PluginMetadata


# Global plugin registry integration
_active_plugins: Dict[str, BasePlugin] = {}
_plugin_tools: Dict[str, List[Tool]] = {}


def integrate_plugin(plugin: BasePlugin) -> None:
    """
    Integrate plugin with Jinx runtime.
    
    Makes plugin tools available to the agent runtime and enables
    event bus communication for plugins.
    
    Args:
        plugin: Plugin to integrate
    """
    plugin_name = plugin.name
    
    # Initialize if not already
    if not plugin._initialized:
        plugin.initialize()
        plugin._initialized = True
    
    # Store plugin
    _active_plugins[plugin_name] = plugin
    
    # Get and store tools
    tools = plugin.get_tools()
    _plugin_tools[plugin_name] = tools
    
    # Register with runtime if available
    try:
        from jinx.micro.runtime.api import emit
        emit("plugin_integrated", data={
            "plugin": plugin_name,
            "tools": [t.spec.name for t in tools],
            "version": plugin.version,
        })
    except ImportError:
        # Runtime not available, skip event emission
        pass


def remove_plugin(plugin_name: str) -> None:
    """
    Remove plugin from runtime integration.
    
    Args:
        plugin_name: Name of plugin to remove
    """
    if plugin_name in _active_plugins:
        plugin = _active_plugins[plugin_name]
        plugin.shutdown()
        
        del _active_plugins[plugin_name]
        if plugin_name in _plugin_tools:
            del _plugin_tools[plugin_name]
        
        # Emit removal event
        try:
            from jinx.micro.runtime.api import emit
            emit("plugin_removed", data={"plugin": plugin_name})
        except ImportError:
            pass


def list_active_plugins() -> List[Dict[str, Any]]:
    """
    List all active (integrated) plugins.
    
    Returns:
        List of plugin information dicts
    """
    result = []
    
    for name, plugin in _active_plugins.items():
        tools = _plugin_tools.get(name, [])
        result.append({
            "name": name,
            "version": plugin.version,
            "description": plugin.description,
            "author": plugin.author,
            "tools_count": len(tools),
            "tools": [t.spec.name for t in tools],
        })
    
    return result


def get_plugin_tools(plugin_name: Optional[str] = None) -> List[Tool]:
    """
    Get tools from integrated plugins.
    
    Args:
        plugin_name: Optional plugin name to filter by
    
    Returns:
        List of tools
    """
    if plugin_name:
        return _plugin_tools.get(plugin_name, [])
    
    # Return all tools
    all_tools = []
    for tools in _plugin_tools.values():
        all_tools.extend(tools)
    return all_tools


def execute_plugin_tool(tool_name: str, **kwargs) -> Any:
    """
    Execute tool from any integrated plugin.
    
    Args:
        tool_name: Name of tool to execute
        **kwargs: Arguments to pass to tool
    
    Returns:
        Tool execution result
    
    Raises:
        ValueError: If tool not found
    """
    # Find tool across all plugins
    for plugin_name, tools in _plugin_tools.items():
        for tool in tools:
            if tool.spec.name == tool_name:
                return tool.execute(**kwargs)
    
    raise ValueError(f"Tool '{tool_name}' not found in any integrated plugin")


async def execute_plugin_tool_async(tool_name: str, **kwargs) -> Any:
    """
    Execute async tool from any integrated plugin.
    
    Args:
        tool_name: Name of tool to execute
        **kwargs: Arguments to pass to tool
    
    Returns:
        Tool execution result
    
    Raises:
        ValueError: If tool not found or not async
    """
    # Find tool across all plugins
    for plugin_name, tools in _plugin_tools.items():
        for tool in tools:
            if tool.spec.name == tool_name:
                if not tool.spec.is_async:
                    raise ValueError(f"Tool '{tool_name}' is not async")
                return await tool.execute_async(**kwargs)
    
    raise ValueError(f"Tool '{tool_name}' not found in any integrated plugin")


def discover_and_integrate_all() -> int:
    """
    Discover and integrate all plugins from default locations.
    
    Returns:
        Number of plugins integrated
    """
    from jinx.plugins.discovery import PluginDiscovery
    
    discovery = PluginDiscovery()
    discovered = discovery.discover_all()
    
    registry = PluginRegistry()
    integrated_count = 0
    
    for metadata, loader in discovered:
        try:
            # Register with registry
            registry.register_plugin(metadata, loader)
            
            # Load plugin
            plugin = registry.load_plugin(metadata.name)
            
            # Integrate with runtime
            integrate_plugin(plugin)
            
            integrated_count += 1
        except Exception as e:
            # Skip plugins that fail to load/integrate
            continue
    
    return integrated_count


def load_plugin_from_repo(repo_url: str, branch: str = "main") -> BasePlugin:
    """
    Load and integrate plugin from git repository.
    
    Args:
        repo_url: Git repository URL
        branch: Branch to checkout
    
    Returns:
        Loaded and integrated plugin
    """
    from jinx.plugins.loader import PluginLoader
    
    loader = PluginLoader()
    plugin = loader.load_from_git(repo_url, branch)
    
    # Integrate with runtime
    integrate_plugin(plugin)
    
    return plugin


def hot_reload_plugin(plugin_name: str) -> None:
    """
    Hot reload a plugin (shutdown and reinitialize).
    
    Args:
        plugin_name: Name of plugin to reload
    
    Raises:
        ValueError: If plugin not found
    """
    if plugin_name not in _active_plugins:
        raise ValueError(f"Plugin '{plugin_name}' not active")
    
    plugin = _active_plugins[plugin_name]
    
    # Shutdown
    plugin.shutdown()
    
    # Reinitialize
    plugin.initialize()
    plugin._initialized = True
    
    # Update tools
    tools = plugin.get_tools()
    _plugin_tools[plugin_name] = tools
    
    # Emit reload event
    try:
        from jinx.micro.runtime.api import emit
        emit("plugin_reloaded", data={
            "plugin": plugin_name,
            "tools": [t.spec.name for t in tools],
        })
    except ImportError:
        pass


# Convenience function for quick plugin integration
def quick_integrate(plugin_path: str) -> BasePlugin:
    """
    Quick integrate plugin from path.
    
    Args:
        plugin_path: Path to plugin file or directory
    
    Returns:
        Loaded and integrated plugin
    """
    from jinx.plugins.loader import PluginLoader
    
    loader = PluginLoader()
    plugin = loader.load_from_path(Path(plugin_path))
    integrate_plugin(plugin)
    
    return plugin
