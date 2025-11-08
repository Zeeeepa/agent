# Jinx Plugin System 🔌

Dynamic plugin architecture for extending Jinx with tools from external repositories.

## Overview

The Jinx Plugin System enables you to:
- **Load plugins dynamically** from local paths or git repositories
- **Analyze external repositories** to automatically extract functionality
- **Wrap functions as tools** with automatic parameter detection
- **Hot-reload plugins** during development
- **Integrate seamlessly** with Jinx's runtime and event system

---

## Quick Start

### 1. Create a Simple Plugin

```python
# my_plugin.py
from jinx.plugins.base import BasePlugin, Tool, ToolSpec
from typing import List

class MyAwesomePlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "my_awesome_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def initialize(self) -> None:
        """Initialize plugin and create tools."""
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        spec = ToolSpec(
            name="greet",
            description="Greet someone by name",
            function=greet,
            parameters={"name": {"required": True, "type": "str"}},
        )
        
        self._tools = [Tool(spec=spec, plugin_name=self.name)]
        self._initialized = True
    
    def get_tools(self) -> List[Tool]:
        return self._tools
```

### 2. Load and Use the Plugin

```python
from jinx.plugins.runtime_integration import quick_integrate, execute_plugin_tool

# Load plugin
plugin = quick_integrate("my_plugin.py")

# Use the tool
result = execute_plugin_tool("greet", name="World")
print(result)  # Output: Hello, World!
```

---

## Architecture

```
jinx/plugins/
├── base.py              # BasePlugin, Tool, ToolSpec classes
├── registry.py          # PluginRegistry singleton
├── loader.py            # Dynamic loading (path/git)
├── analyzer.py          # Repository analysis
├── tool_wrapper.py      # Function wrapping utilities
├── discovery.py         # Auto-discovery system
└── runtime_integration.py  # Jinx runtime integration
```

---

## Core Components

### 1. BasePlugin

Abstract base class all plugins must implement:

```python
class BasePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version (semver)."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Setup plugin resources and tools."""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """Return list of tools."""
        pass
    
    def shutdown(self) -> None:
        """Cleanup resources."""
        pass
```

### 2. Tool & ToolSpec

Tools wrap functions with metadata:

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    return_type: Optional[type]
    is_async: bool
    tags: List[str]

@dataclass
class Tool:
    spec: ToolSpec
    plugin_name: str
    enabled: bool
```

### 3. PluginRegistry

Centralized plugin management (singleton):

```python
from jinx.plugins.registry import PluginRegistry, PluginMetadata

registry = PluginRegistry()

# Register plugin
metadata = PluginMetadata(
    name="my_plugin",
    version="1.0.0",
    author="Me",
    description="Does cool stuff"
)
registry.register_plugin(metadata, loader_func)

# Load and use
plugin = registry.load_plugin("my_plugin")
tools = plugin.get_tools()
```

---

## Loading Plugins

### From Local Path

```python
from jinx.plugins.loader import PluginLoader

loader = PluginLoader()
plugin = loader.load_from_path(Path("./plugins/my_plugin.py"))
plugin.initialize()
```

### From Git Repository

```python
plugin = loader.load_from_git(
    repo_url="https://github.com/user/plugin-repo",
    branch="main",
    plugin_subpath="src/plugin.py"  # optional
)
```

### Auto-Discovery

```python
from jinx.plugins.discovery import PluginDiscovery

discovery = PluginDiscovery()
plugins = discovery.discover_all()  # Scans default locations

# Default locations:
# - ~/.jinx/plugins/
# - ./jinx_plugins/
# - ./plugins/
# - $JINX_PLUGIN_PATH
```

---

## Repository Analysis

Analyze external repositories to extract functionality:

```python
from jinx.plugins.analyzer import RepoAnalyzer

analyzer = RepoAnalyzer()
analysis = analyzer.analyze_repo(Path("./external-repo"))

print(f"Found {len(analysis.functions)} functions")
print(f"Found {len(analysis.classes)} classes")
print(f"Dependencies: {analysis.dependencies}")
print(f"Entry points: {analysis.entry_points}")

# Auto-generate tool specs
for tool_spec in analysis.tool_specs:
    print(f"  - {tool_spec.name}: {tool_spec.description}")
```

---

## Runtime Integration

### Integrate with Jinx Runtime

```python
from jinx.plugins.runtime_integration import (
    integrate_plugin,
    list_active_plugins,
    execute_plugin_tool,
    discover_and_integrate_all
)

# Integrate single plugin
integrate_plugin(my_plugin)

# Discover and integrate all
count = discover_and_integrate_all()
print(f"Integrated {count} plugins")

# List active plugins
for plugin_info in list_active_plugins():
    print(f"{plugin_info['name']} v{plugin_info['version']}")
    print(f"  Tools: {', '.join(plugin_info['tools'])}")

# Execute tool
result = execute_plugin_tool("greet", name="Jinx")
```

### Hot Reload

```python
from jinx.plugins.runtime_integration import hot_reload_plugin

# Reload plugin during development
hot_reload_plugin("my_plugin")
```

---

## Advanced Features

### Tool Wrapper

Automatically wrap existing functions:

```python
from jinx.plugins.tool_wrapper import ToolWrapper

def my_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

# Wrap as tool
tool = ToolWrapper.wrap_function(
    my_function,
    metadata={
        "plugin_name": "math_utils",
        "tags": ["math", "arithmetic"]
    }
)

# Execute
result = tool.execute(x=5, y=3)
```

### Async Tools

```python
import asyncio

class AsyncPlugin(BasePlugin):
    def initialize(self) -> None:
        async def fetch_data(url: str) -> str:
            # Async operation
            await asyncio.sleep(1)
            return f"Data from {url}"
        
        spec = ToolSpec(
            name="fetch",
            description="Fetch data asynchronously",
            function=fetch_data,
            parameters={"url": {"required": True}},
            is_async=True
        )
        
        self._tools = [Tool(spec=spec, plugin_name=self.name)]
```

### Plugin Manifests

Create `plugin.json` for metadata:

```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Plugin description",
  "module": "plugin.py",
  "capabilities": ["math", "string"],
  "dependencies": ["numpy", "requests"]
}
```

---

## Examples

### Example 1: Math Plugin

See `examples/example_plugin.py` for a complete example.

### Example 2: Load from GitHub

```python
from jinx.plugins.runtime_integration import load_plugin_from_repo

plugin = load_plugin_from_repo(
    "https://github.com/user/awesome-plugin",
    branch="main"
)

# Plugin is automatically integrated!
tools = plugin.get_tools()
```

### Example 3: Repository Analysis

```python
from jinx.plugins.analyzer import RepoAnalyzer
from pathlib import Path

analyzer = RepoAnalyzer()

# Analyze a Python project
analysis = analyzer.analyze_repo(Path("./some-python-project"))

print(f"📊 Analysis Results:")
print(f"  Functions: {len(analysis.functions)}")
print(f"  Classes: {len(analysis.classes)}")
print(f"  Dependencies: {len(analysis.dependencies)}")
print(f"  Potential tools: {len(analysis.tool_specs)}")

# Build dependency graph
graph = analyzer.build_dependency_graph(Path("./some-python-project"))
print(f"  Modules: {len(graph.nodes())}")
print(f"  Dependencies: {len(graph.edges())}")
```

---

## Best Practices

### 1. Plugin Structure

```python
class WellStructuredPlugin(BasePlugin):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._resources = None
    
    def initialize(self) -> None:
        # Setup resources
        self._resources = self._setup_resources()
        
        # Create tools
        self._tools = self._create_tools()
        
        self._initialized = True
    
    def _setup_resources(self):
        # Initialize expensive resources once
        pass
    
    def _create_tools(self) -> List[Tool]:
        # Create tool list
        pass
    
    def shutdown(self) -> None:
        # Clean up resources
        if self._resources:
            self._resources.close()
        super().shutdown()
```

### 2. Error Handling

```python
def safe_tool_function(x: int) -> int:
    # Validate inputs
    if not isinstance(x, int):
        raise ValueError(f"Expected int, got {type(x)}")
    
    if x < 0:
        raise ValueError("x must be non-negative")
    
    # Process
    return x * 2
```

### 3. Documentation

```python
def my_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Tool description goes here.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Dictionary with results
    
    Example:
        >>> result = my_tool("test", 20)
        >>> print(result)
    """
    pass
```

---

## Troubleshooting

### Plugin Not Loading

1. Check plugin inherits from `BasePlugin`
2. Implement all abstract methods
3. Ensure `initialize()` sets `self._initialized = True`
4. Verify file/path exists

### Tools Not Appearing

1. Check `get_tools()` returns list
2. Verify tools are created in `initialize()`
3. Ensure ToolSpec has valid function

### Import Errors

1. Check plugin dependencies are installed
2. Verify Python path includes plugin location
3. Use absolute imports in plugins

---

## API Reference

### Classes

- **BasePlugin**: Abstract plugin base class
- **Tool**: Executable tool instance
- **ToolSpec**: Tool specification/metadata
- **PluginRegistry**: Singleton plugin manager
- **PluginLoader**: Dynamic plugin loader
- **RepoAnalyzer**: Repository analysis engine
- **ToolWrapper**: Function wrapping utilities
- **PluginDiscovery**: Auto-discovery system

### Functions

- `integrate_plugin(plugin)`: Integrate with runtime
- `quick_integrate(path)`: Fast load and integrate
- `execute_plugin_tool(name, **kwargs)`: Execute tool
- `discover_and_integrate_all()`: Auto-discover plugins
- `hot_reload_plugin(name)`: Reload plugin
- `load_plugin_from_repo(url, branch)`: Load from git

---

## Contributing

To add new plugin capabilities:

1. Extend `BasePlugin` for new plugin types
2. Add analyzers for new languages/frameworks
3. Contribute example plugins
4. Improve auto-discovery heuristics

---

## License

MIT - See main Jinx LICENSE file.

