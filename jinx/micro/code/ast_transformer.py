"""AST Transformer - intelligent AST-based code transformations with LibCST.

Uses LibCST for production-grade Python AST transformations with full type preservation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import libcst as cst
    from libcst import matchers as m
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False
    cst = None
    m = None


@dataclass
class TransformRule:
    """AST transformation rule."""
    name: str
    matcher: Any  # LibCST matcher
    transformer: Any  # Transformation function
    description: str


class IntelligentTransformer(cst.CSTTransformer if LIBCST_AVAILABLE else object):
    """Intelligent AST transformer with brain integration."""
    
    def __init__(self, rules: List[TransformRule]):
        if LIBCST_AVAILABLE:
            super().__init__()
        self.rules = rules
        self.applied_transformations: List[str] = []
        self.context: Dict[str, Any] = {}
    
    def visit_Import(self, node: cst.Import) -> bool:
        """Visit import statements."""
        return True
    
    def leave_Import(
        self,
        original_node: cst.Import,
        updated_node: cst.Import
    ) -> Union[cst.Import, cst.RemovalSentinel]:
        """Transform import statements."""
        for rule in self.rules:
            if rule.name == 'import_transform':
                if m.matches(updated_node, rule.matcher):
                    result = rule.transformer(updated_node, self.context)
                    if result != updated_node:
                        self.applied_transformations.append(rule.description)
                        return result
        return updated_node
    
    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef
    ) -> Union[cst.FunctionDef, cst.RemovalSentinel]:
        """Transform function definitions."""
        for rule in self.rules:
            if rule.name == 'function_transform':
                if m.matches(updated_node, rule.matcher):
                    result = rule.transformer(updated_node, self.context)
                    if result != updated_node:
                        self.applied_transformations.append(rule.description)
                        return result
        return updated_node
    
    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call
    ) -> Union[cst.Call, cst.RemovalSentinel]:
        """Transform function calls."""
        for rule in self.rules:
            if rule.name == 'call_transform':
                if m.matches(updated_node, rule.matcher):
                    result = rule.transformer(updated_node, self.context)
                    if result != updated_node:
                        self.applied_transformations.append(rule.description)
                        return result
        return updated_node


class ASTTransformEngine:
    """Production-grade AST transformation engine."""
    
    def __init__(self):
        if not LIBCST_AVAILABLE:
            raise ImportError("LibCST not installed. Install: pip install libcst")
    
    async def transform_code(
        self,
        source_code: str,
        rules: List[TransformRule]
    ) -> tuple[str, List[str]]:
        """Transform code using AST rules."""
        
        # Parse code
        try:
            tree = cst.parse_module(source_code)
        except Exception as e:
            raise ValueError(f"Failed to parse code: {e}")
        
        # Create transformer
        transformer = IntelligentTransformer(rules)
        
        # Apply transformations
        modified_tree = tree.visit(transformer)
        
        # Generate code
        new_code = modified_tree.code
        
        return new_code, transformer.applied_transformations
    
    async def add_type_hints(self, source_code: str) -> str:
        """Add type hints to function signatures."""
        
        def add_hints_transformer(node: cst.FunctionDef, ctx: Dict) -> cst.FunctionDef:
            """Add type hints to parameters."""
            # Check if already has type hints
            params = node.params
            
            # Add simple type hints for common patterns
            new_params = []
            for param in params.params:
                if param.annotation is None:
                    # Add str type hint as default
                    new_param = param.with_changes(
                        annotation=cst.Annotation(
                            annotation=cst.Name("str")
                        )
                    )
                    new_params.append(new_param)
                else:
                    new_params.append(param)
            
            new_params_node = params.with_changes(params=new_params)
            
            return node.with_changes(params=new_params_node)
        
        rules = [
            TransformRule(
                name='function_transform',
                matcher=m.FunctionDef(),
                transformer=add_hints_transformer,
                description="Add type hints"
            )
        ]
        
        new_code, _ = await self.transform_code(source_code, rules)
        return new_code
    
    async def convert_to_async(self, source_code: str) -> str:
        """Convert synchronous functions to async."""
        
        def async_transformer(node: cst.FunctionDef, ctx: Dict) -> cst.FunctionDef:
            """Convert def to async def."""
            # Check if already async
            if node.asynchronous is None:
                return node.with_changes(
                    asynchronous=cst.Async(
                        whitespace_after=cst.SimpleWhitespace(" ")
                    )
                )
            return node
        
        rules = [
            TransformRule(
                name='function_transform',
                matcher=m.FunctionDef(),
                transformer=async_transformer,
                description="Convert to async"
            )
        ]
        
        new_code, _ = await self.transform_code(source_code, rules)
        return new_code
    
    async def modernize_imports(
        self,
        source_code: str,
        old_module: str,
        new_module: str
    ) -> str:
        """Modernize import statements."""
        
        def import_transformer(node: cst.Import, ctx: Dict) -> cst.Import:
            """Update module names."""
            new_names = []
            for name in node.names:
                if isinstance(name, cst.ImportAlias):
                    module_name = name.name
                    if isinstance(module_name, cst.Attribute):
                        # Handle: import old.module
                        pass
                    elif isinstance(module_name, cst.Name):
                        if module_name.value == old_module:
                            new_names.append(
                                name.with_changes(
                                    name=cst.Name(new_module)
                                )
                            )
                            continue
                new_names.append(name)
            
            return node.with_changes(names=new_names)
        
        rules = [
            TransformRule(
                name='import_transform',
                matcher=m.Import(),
                transformer=import_transformer,
                description=f"Modernize imports: {old_module} -> {new_module}"
            )
        ]
        
        new_code, _ = await self.transform_code(source_code, rules)
        return new_code


# Singleton
_ast_engine: Optional[ASTTransformEngine] = None
_ast_lock = asyncio.Lock()


async def get_ast_engine() -> ASTTransformEngine:
    """Get singleton AST engine."""
    global _ast_engine
    if _ast_engine is None:
        async with _ast_lock:
            if _ast_engine is None:
                _ast_engine = ASTTransformEngine()
    return _ast_engine


__all__ = [
    "ASTTransformEngine",
    "TransformRule",
    "IntelligentTransformer",
    "get_ast_engine",
]
