"""
Tool wrapper - wrap external functions as Jinx tools.
"""

from typing import Any, Callable, Dict, Optional
import inspect
import asyncio

from jinx.plugins.base import Tool, ToolSpec


class ToolWrapper:
    """
    Wrapper for converting external functions into Jinx tools.
    
    Provides safe execution, argument validation, and error handling
    for tools loaded from external sources.
    """
    
    @staticmethod
    def wrap_function(
        func: Callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tool:
        """
        Wrap function as Jinx Tool.
        
        Args:
            func: Function to wrap
            metadata: Optional metadata (name, description, etc)
        
        Returns:
            Tool instance ready for execution
        """
        metadata = metadata or {}
        
        # Extract function signature
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            param_info = {
                "required": param.default == inspect.Parameter.empty,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any"
            }
            parameters[param_name] = param_info
        
        # Create tool spec
        spec = ToolSpec(
            name=metadata.get("name", func.__name__),
            description=metadata.get("description", func.__doc__ or f"Function: {func.__name__}"),
            function=func,
            parameters=parameters,
            return_type=sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None,
            is_async=asyncio.iscoroutinefunction(func),
            tags=metadata.get("tags", []),
        )
        
        return Tool(
            spec=spec,
            plugin_name=metadata.get("plugin_name", "unknown"),
            enabled=True,
        )
    
    @staticmethod
    def create_prompt_template(func: Callable) -> str:
        """
        Create prompt template for tool usage.
        
        Args:
            func: Function to create template for
        
        Returns:
            Prompt template string
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        
        template = f"Tool: {func.__name__}\n"
        template += f"Description: {doc}\n\n"
        template += "Parameters:\n"
        
        for param_name, param in sig.parameters.items():
            required = " (required)" if param.default == inspect.Parameter.empty else " (optional)"
            param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "any"
            template += f"  - {param_name}: {param_type}{required}\n"
        
        return template
    
    @staticmethod
    def validate_args(func: Callable, args: Dict[str, Any]) -> bool:
        """
        Validate arguments for function.
        
        Args:
            func: Function to validate against
            args: Arguments to validate
        
        Returns:
            True if valid, False otherwise
        """
        sig = inspect.signature(func)
        
        # Check required parameters
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                if param_name not in args:
                    return False
        
        return True
    
    @staticmethod
    def execute_safe(func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute function safely with error handling.
        
        Args:
            func: Function to execute
            args: Arguments to pass
        
        Returns:
            Dict with 'success', 'result', and optional 'error'
        """
        if not ToolWrapper.validate_args(func, args):
            return {
                "success": False,
                "error": "Invalid arguments",
                "result": None
            }
        
        try:
            if asyncio.iscoroutinefunction(func):
                # Async function
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(func(**args))
            else:
                result = func(**args)
            
            return {
                "success": True,
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
