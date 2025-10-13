"""
Plugin loader - dynamic import and initialization of plugins.
"""

import importlib.util
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import inspect
import ast

from jinx.plugins.base import BasePlugin
from jinx.plugins.registry import PluginMetadata


class PluginLoader:
    """
    Loader for dynamically importing and initializing plugins.
    
    Supports loading from local paths and remote git repositories.
    """
    
    def __init__(self):
        self._loaded_modules = {}
    
    def load_from_path(self, plugin_path: Path) -> BasePlugin:
        """
        Load plugin from local filesystem path.
        
        Args:
            plugin_path: Path to plugin module or package
        
        Returns:
            Initialized plugin instance
        
        Raises:
            ValueError: If plugin invalid or not found
        """
        if not plugin_path.exists():
            raise ValueError(f"Plugin path not found: {plugin_path}")
        
        # Determine if path is module or package
        if plugin_path.is_dir():
            # Package - look for __init__.py
            init_file = plugin_path / "__init__.py"
            if not init_file.exists():
                raise ValueError(f"Package missing __init__.py: {plugin_path}")
            module_path = init_file
            module_name = plugin_path.name
        else:
            # Module file
            module_path = plugin_path
            module_name = plugin_path.stem
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            raise ValueError(f"Could not load module from: {plugin_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        self._loaded_modules[module_name] = module
        
        # Find plugin class
        plugin_class = self._find_plugin_class(module)
        if not plugin_class:
            raise ValueError(f"No BasePlugin subclass found in: {plugin_path}")
        
        # Instantiate and return
        return plugin_class()
    
    def load_from_git(
        self, 
        repo_url: str, 
        branch: str = "main",
        plugin_subpath: Optional[str] = None
    ) -> BasePlugin:
        """
        Load plugin from git repository.
        
        Args:
            repo_url: Git repository URL
            branch: Branch to checkout (default: main)
            plugin_subpath: Optional subpath within repo to plugin
        
        Returns:
            Initialized plugin instance
        
        Raises:
            ValueError: If clone fails or plugin not found
        """
        # Clone to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            clone_path = tmppath / "repo"
            
            # Git clone
            result = subprocess.run(
                ["git", "clone", "--depth=1", "--branch", branch, repo_url, str(clone_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise ValueError(f"Git clone failed: {result.stderr}")
            
            # Locate plugin
            if plugin_subpath:
                plugin_path = clone_path / plugin_subpath
            else:
                # Search for plugin in common locations
                plugin_path = self._locate_plugin_in_repo(clone_path)
            
            if not plugin_path:
                raise ValueError(f"No plugin found in repository: {repo_url}")
            
            return self.load_from_path(plugin_path)
    
    def validate_plugin(self, plugin_module) -> bool:
        """
        Validate that module contains valid plugin.
        
        Args:
            plugin_module: Module to validate
        
        Returns:
            True if valid plugin found
        """
        return self._find_plugin_class(plugin_module) is not None
    
    def extract_capabilities(self, plugin_module) -> List[str]:
        """
        Extract list of capabilities from plugin module.
        
        Args:
            plugin_module: Module to analyze
        
        Returns:
            List of capability strings
        """
        capabilities = []
        
        # Check for CAPABILITIES attribute
        if hasattr(plugin_module, "CAPABILITIES"):
            caps = getattr(plugin_module, "CAPABILITIES")
            if isinstance(caps, list):
                capabilities.extend(caps)
        
        # Inspect plugin class methods
        plugin_class = self._find_plugin_class(plugin_module)
        if plugin_class:
            for name, method in inspect.getmembers(plugin_class, inspect.ismethod):
                if not name.startswith("_"):
                    capabilities.append(name)
        
        return capabilities
    
    def extract_metadata(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """
        Extract plugin metadata without loading plugin.
        
        Args:
            plugin_path: Path to plugin
        
        Returns:
            Plugin metadata if extractable, None otherwise
        """
        if not plugin_path.exists():
            return None
        
        # Try to find plugin.yaml or plugin.json
        metadata_files = [
            plugin_path / "plugin.yaml",
            plugin_path / "plugin.json",
            plugin_path.parent / f"{plugin_path.stem}.yaml",
            plugin_path.parent / f"{plugin_path.stem}.json",
        ]
        
        for meta_file in metadata_files:
            if meta_file.exists():
                return self._parse_metadata_file(meta_file)
        
        # Fallback: parse module docstring and attributes
        return self._extract_metadata_from_code(plugin_path)
    
    def _find_plugin_class(self, module):
        """Find BasePlugin subclass in module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BasePlugin) and 
                obj is not BasePlugin):
                return obj
        return None
    
    def _locate_plugin_in_repo(self, repo_path: Path) -> Optional[Path]:
        """Locate plugin in cloned repository."""
        # Common plugin locations
        search_paths = [
            repo_path / "plugin.py",
            repo_path / "src" / "plugin.py",
            repo_path / "plugins" / "main.py",
            repo_path / "__init__.py",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        # Search for any file with BasePlugin subclass
        for py_file in repo_path.rglob("*.py"):
            if self._contains_plugin_class(py_file):
                return py_file
        
        return None
    
    def _contains_plugin_class(self, py_file: Path) -> bool:
        """Check if Python file contains BasePlugin subclass."""
        try:
            with open(py_file, "r") as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "BasePlugin":
                            return True
        except:
            pass
        
        return False
    
    def _parse_metadata_file(self, meta_file: Path) -> PluginMetadata:
        """Parse plugin metadata from YAML/JSON file."""
        import json
        
        if meta_file.suffix == ".json":
            with open(meta_file) as f:
                data = json.load(f)
        else:
            # For YAML, use a simple parser or require pyyaml
            try:
                import yaml
                with open(meta_file) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                # Fallback to JSON if yaml not available
                with open(meta_file) as f:
                    data = json.load(f)
        
        return PluginMetadata(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            author=data.get("author", "unknown"),
            repo_url=data.get("repo_url"),
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            dependencies=data.get("dependencies", []),
        )
    
    def _extract_metadata_from_code(self, plugin_path: Path) -> PluginMetadata:
        """Extract metadata from code docstrings and module attributes."""
        # Simple fallback - load module and inspect
        try:
            spec = importlib.util.spec_from_file_location("_temp_plugin", plugin_path)
            if not spec or not spec.loader:
                return PluginMetadata(name="unknown", version="0.0.0", author="unknown")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            name = getattr(module, "__name__", plugin_path.stem)
            version = getattr(module, "__version__", "0.0.0")
            author = getattr(module, "__author__", "unknown")
            description = getattr(module, "__doc__", "")
            
            return PluginMetadata(
                name=name,
                version=version,
                author=author,
                description=description.strip() if description else ""
            )
        except:
            return PluginMetadata(name=plugin_path.stem, version="0.0.0", author="unknown")

