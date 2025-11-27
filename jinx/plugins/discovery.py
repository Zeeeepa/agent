"""
Plugin discovery - auto-discover plugins in standard locations.
"""

import os
from pathlib import Path
from typing import List, Tuple, Callable
import json

from jinx.plugins.base import BasePlugin
from jinx.plugins.registry import PluginMetadata
from jinx.plugins.loader import PluginLoader


class PluginDiscovery:
    """
    Auto-discover plugins from standard search paths.
    
    Scans directories for plugin manifests and Python modules
    implementing the BasePlugin interface.
    """
    
    def __init__(self):
        self.loader = PluginLoader()
        self._default_paths = self._get_default_search_paths()
    
    def scan_paths(
        self, 
        search_paths: List[Path]
    ) -> List[Tuple[PluginMetadata, Callable[[], BasePlugin]]]:
        """
        Scan directories for plugins.
        
        Args:
            search_paths: List of paths to search
        
        Returns:
            List of (metadata, loader_function) tuples
        """
        discovered = []
        
        for path in search_paths:
            if not path.exists():
                continue
            
            if path.is_file():
                # Single plugin file
                result = self._discover_single_plugin(path)
                if result:
                    discovered.append(result)
            else:
                # Directory - scan for plugins
                discovered.extend(self._scan_directory(path))
        
        return discovered
    
    def discover_all(self) -> List[Tuple[PluginMetadata, Callable[[], BasePlugin]]]:
        """
        Discover all plugins in default locations.
        
        Returns:
            List of (metadata, loader_function) tuples
        """
        return self.scan_paths(self._default_paths)
    
    def _scan_directory(
        self, 
        directory: Path
    ) -> List[Tuple[PluginMetadata, Callable[[], BasePlugin]]]:
        """Scan single directory for plugins."""
        discovered = []
        
        # Look for manifest files
        manifest_files = [
            *directory.glob("plugin.json"),
            *directory.glob("plugin.yaml"),
            *directory.glob("*/plugin.json"),
            *directory.glob("*/plugin.yaml"),
        ]
        
        for manifest in manifest_files:
            result = self._load_from_manifest(manifest)
            if result:
                discovered.append(result)
        
        # Look for standalone plugin files
        for py_file in directory.glob("**/*.py"):
            if self._looks_like_plugin(py_file):
                result = self._discover_single_plugin(py_file)
                if result:
                    discovered.append(result)
        
        return discovered
    
    def _discover_single_plugin(
        self, 
        plugin_path: Path
    ) -> Optional[Tuple[PluginMetadata, Callable[[], BasePlugin]]]:
        """Discover single plugin from path."""
        # Extract metadata
        metadata = self.loader.extract_metadata(plugin_path)
        if not metadata:
            return None
        
        # Create loader function
        def loader():
            return self.loader.load_from_path(plugin_path)
        
        return (metadata, loader)
    
    def _load_from_manifest(
        self, 
        manifest_path: Path
    ) -> Optional[Tuple[PluginMetadata, Callable[[], BasePlugin]]]:
        """Load plugin from manifest file."""
        try:
            with open(manifest_path) as f:
                if manifest_path.suffix == ".json":
                    data = json.load(f)
                else:
                    # YAML
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        return None
            
            # Extract metadata
            metadata = PluginMetadata(
                name=data.get("name", "unknown"),
                version=data.get("version", "0.0.0"),
                author=data.get("author", "unknown"),
                repo_url=data.get("repo_url"),
                description=data.get("description", ""),
                capabilities=data.get("capabilities", []),
                dependencies=data.get("dependencies", []),
            )
            
            # Find plugin module
            plugin_file = data.get("module", "plugin.py")
            plugin_path = manifest_path.parent / plugin_file
            
            if not plugin_path.exists():
                return None
            
            # Create loader
            def loader():
                return self.loader.load_from_path(plugin_path)
            
            return (metadata, loader)
        
        except Exception as e:
            return None
    
    def _looks_like_plugin(self, py_file: Path) -> bool:
        """Check if file likely contains a plugin."""
        # Simple heuristic - check filename
        plugin_names = {"plugin", "jinx_plugin", "extension"}
        
        stem_lower = py_file.stem.lower()
        return any(name in stem_lower for name in plugin_names)
    
    def _get_default_search_paths(self) -> List[Path]:
        """Get default plugin search paths."""
        paths = []
        
        # User home directory
        home = Path.home()
        paths.append(home / ".jinx" / "plugins")
        
        # Current working directory
        cwd = Path.cwd()
        paths.append(cwd / "jinx_plugins")
        paths.append(cwd / "plugins")
        
        # Environment variable
        env_path = os.environ.get("JINX_PLUGIN_PATH")
        if env_path:
            for path_str in env_path.split(":"):
                paths.append(Path(path_str))
        
        return paths
