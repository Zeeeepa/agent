"""
Plugin registry - singleton for managing all loaded plugins.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading

from jinx.plugins.base import BasePlugin, Tool


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    
    name: str
    version: str
    author: str
    repo_url: Optional[str] = None
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.name, self.version))


class PluginRegistry:
    """
    Singleton registry for managing all plugins.
    
    Provides centralized plugin discovery, registration, and lifecycle
    management for the Jinx agent framework.
    """
    
    _instance: Optional['PluginRegistry'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugins: Dict[str, BasePlugin] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._loaders: Dict[str, Callable] = {}
        self._initialized = True
    
    def register_plugin(
        self, 
        metadata: PluginMetadata, 
        loader_func: Callable[[], BasePlugin]
    ) -> None:
        """
        Register a plugin with the registry.
        
        Args:
            metadata: Plugin metadata
            loader_func: Function that returns initialized plugin instance
        
        Raises:
            ValueError: If plugin with same name already registered
        """
        if metadata.name in self._metadata:
            existing = self._metadata[metadata.name]
            if existing.version != metadata.version:
                raise ValueError(
                    f"Plugin '{metadata.name}' version conflict: "
                    f"existing={existing.version}, new={metadata.version}"
                )
            # Same version already registered, skip
            return
        
        self._metadata[metadata.name] = metadata
        self._loaders[metadata.name] = loader_func
    
    def load_plugin(self, name: str) -> BasePlugin:
        """
        Load and initialize a registered plugin.
        
        Args:
            name: Plugin name
        
        Returns:
            Initialized plugin instance
        
        Raises:
            ValueError: If plugin not registered
        """
        if name in self._plugins:
            return self._plugins[name]
        
        if name not in self._loaders:
            raise ValueError(f"Plugin '{name}' not registered")
        
        loader = self._loaders[name]
        plugin = loader()
        plugin.initialize()
        
        self._plugins[name] = plugin
        return plugin
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """
        Get loaded plugin by name.
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance if loaded, None otherwise
        """
        return self._plugins.get(name)
    
    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin and cleanup resources.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            plugin.shutdown()
            del self._plugins[name]
    
    def list_plugins(self, loaded_only: bool = False) -> List[PluginMetadata]:
        """
        List all registered or loaded plugins.
        
        Args:
            loaded_only: If True, only return metadata for loaded plugins
        
        Returns:
            List of plugin metadata
        """
        if loaded_only:
            return [
                self._metadata[name] 
                for name in self._plugins.keys()
                if name in self._metadata
            ]
        return list(self._metadata.values())
    
    def discover_plugins(self, search_paths: List[Path]) -> int:
        """
        Discover plugins in specified search paths.
        
        Args:
            search_paths: List of directories to search
        
        Returns:
            Number of plugins discovered
        """
        from jinx.plugins.discovery import PluginDiscovery
        
        discovery = PluginDiscovery()
        discovered = discovery.scan_paths(search_paths)
        
        for metadata, loader in discovered:
            self.register_plugin(metadata, loader)
        
        return len(discovered)
    
    def get_all_tools(self) -> Dict[str, List[Tool]]:
        """
        Get all tools from all loaded plugins.
        
        Returns:
            Dict mapping plugin name to list of tools
        """
        return {
            name: plugin.get_tools()
            for name, plugin in self._plugins.items()
        }
    
    def find_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Find a tool by name across all loaded plugins.
        
        Args:
            tool_name: Name of tool to find
        
        Returns:
            Tool instance if found, None otherwise
        """
        for plugin in self._plugins.values():
            for tool in plugin.get_tools():
                if tool.spec.name == tool_name:
                    return tool
        return None
    
    def clear(self) -> None:
        """Unload all plugins and clear registry."""
        for name in list(self._plugins.keys()):
            self.unload_plugin(name)
        
        self._metadata.clear()
        self._loaders.clear()
    
    @property
    def loaded_count(self) -> int:
        """Number of currently loaded plugins."""
        return len(self._plugins)
    
    @property
    def registered_count(self) -> int:
        """Number of registered plugins (loaded + unloaded)."""
        return len(self._metadata)

