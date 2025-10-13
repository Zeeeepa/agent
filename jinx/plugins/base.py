"""
Base classes and interfaces for Jinx plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path


@dataclass
class ToolSpec:
    """Specification for a tool exposed by a plugin."""
    
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_type: Optional[type] = None
    is_async: bool = False
    tags: List[str] = field(default_factory=list)
    
    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate arguments against parameter spec."""
        required_params = {
            k: v for k, v in self.parameters.items() 
            if v.get("required", False)
        }
        
        for param_name in required_params:
            if param_name not in args:
                return False
        
        return True


@dataclass
class Tool:
    """A tool instance that can be executed."""
    
    spec: ToolSpec
    plugin_name: str
    enabled: bool = True
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.spec.name}' is disabled")
        
        if not self.spec.validate_args(kwargs):
            raise ValueError(f"Invalid arguments for tool '{self.spec.name}'")
        
        return self.spec.function(**kwargs)
    
    async def execute_async(self, **kwargs) -> Any:
        """Execute async tool with given arguments."""
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.spec.name}' is disabled")
        
        if not self.spec.validate_args(kwargs):
            raise ValueError(f"Invalid arguments for tool '{self.spec.name}'")
        
        if not self.spec.is_async:
            raise RuntimeError(f"Tool '{self.spec.name}' is not async")
        
        return await self.spec.function(**kwargs)


class BasePlugin(ABC):
    """
    Abstract base class that all Jinx plugins must implement.
    
    Plugins extend Jinx with new tools and capabilities from external
    repositories or custom implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with optional configuration."""
        self.config = config or {}
        self._initialized = False
        self._tools: List[Tool] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this plugin."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version string (semver recommended)."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description of plugin capabilities."""
        return f"Plugin: {self.name}"
    
    @property
    def author(self) -> str:
        """Plugin author or maintainer."""
        return "Unknown"
    
    @property
    def repo_url(self) -> Optional[str]:
        """Source repository URL if applicable."""
        return None
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the plugin.
        
        Called once when plugin is loaded. Setup resources, validate
        dependencies, register tools here.
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """
        Return list of tools this plugin provides.
        
        Returns:
            List of Tool instances that can be executed by Jinx.
        """
        pass
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a specific tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool not found or invalid arguments
        """
        tool = self._find_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in plugin '{self.name}'")
        
        return tool.execute(**kwargs)
    
    def shutdown(self) -> None:
        """
        Clean up plugin resources.
        
        Called when plugin is unloaded. Close connections, release
        resources, cleanup temporary files here.
        """
        self._initialized = False
        self._tools.clear()
    
    def _find_tool(self, tool_name: str) -> Optional[Tool]:
        """Find tool by name in plugin's tool list."""
        for tool in self._tools:
            if tool.spec.name == tool_name:
                return tool
        return None
    
    def reload(self) -> None:
        """Reload plugin (shutdown then initialize again)."""
        self.shutdown()
        self.initialize()

