"""
Example plugin demonstrating Jinx plugin system.
"""

from jinx.plugins.base import BasePlugin, Tool, ToolSpec
from typing import List


class ExampleMathPlugin(BasePlugin):
    """Example plugin providing math and string utilities."""
    
    @property
    def name(self) -> str:
        return "example_math"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Example plugin with math and string tools"
    
    @property
    def author(self) -> str:
        return "Jinx Team"
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        # Create tools
        self._tools = [
            self._create_add_tool(),
            self._create_multiply_tool(),
        ]
        self._initialized = True
    
    def get_tools(self) -> List[Tool]:
        return self._tools
    
    def _create_add_tool(self) -> Tool:
        def add(a: float, b: float) -> float:
            return a + b
        
        spec = ToolSpec(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={"a": {"required": True}, "b": {"required": True}},
        )
        return Tool(spec=spec, plugin_name=self.name)
    
    def _create_multiply_tool(self) -> Tool:
        def multiply(a: float, b: float) -> float:
            return a * b
        
        spec = ToolSpec(
            name="multiply",
            description="Multiply two numbers",
            function=multiply,
            parameters={"a": {"required": True}, "b": {"required": True}},
        )
        return Tool(spec=spec, plugin_name=self.name)
