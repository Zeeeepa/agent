"""
Repository analyzer - extract functions, classes, and generate tool specs.
"""

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import networkx as nx

from jinx.plugins.base import ToolSpec


@dataclass
class FunctionInfo:
    """Information about a function in a repository."""
    
    name: str
    file_path: Path
    lineno: int
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": str(self.file_path),
            "line": self.lineno,
            "docstring": self.docstring,
            "parameters": self.parameters,
            "return_type": self.return_annotation,
            "async": self.is_async,
            "decorators": self.decorators,
        }


@dataclass
class ClassInfo:
    """Information about a class in a repository."""
    
    name: str
    file_path: Path
    lineno: int
    docstring: Optional[str] = None
    methods: List[str] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": str(self.file_path),
            "line": self.lineno,
            "docstring": self.docstring,
            "methods": self.methods,
            "bases": self.base_classes,
            "attributes": self.attributes,
        }


@dataclass
class RepoAnalysis:
    """Complete analysis of a repository."""
    
    repo_path: Path
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    entry_points: List[str] = field(default_factory=list)
    tool_specs: List[ToolSpec] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_path": str(self.repo_path),
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "dependencies": list(self.dependencies),
            "entry_points": self.entry_points,
            "tool_count": len(self.tool_specs),
        }


class RepoAnalyzer:
    """
    Analyze external repositories to extract functionality.
    
    Scans Python code to find functions, classes, and automatically
    generate tool specifications for Jinx integration.
    """
    
    def __init__(self):
        self._analysis_cache: Dict[Path, RepoAnalysis] = {}
    
    def analyze_repo(self, repo_path: Path) -> RepoAnalysis:
        """
        Analyze complete repository structure and code.
        
        Args:
            repo_path: Path to repository root
        
        Returns:
            Complete repository analysis
        """
        if repo_path in self._analysis_cache:
            return self._analysis_cache[repo_path]
        
        analysis = RepoAnalysis(repo_path=repo_path)
        
        # Extract functions and classes
        analysis.functions = self.extract_functions(repo_path)
        analysis.classes = self.extract_classes(repo_path)
        
        # Find dependencies
        analysis.dependencies = self._extract_imports(repo_path)
        
        # Identify entry points
        analysis.entry_points = self._find_entry_points(repo_path)
        
        # Generate tool specs for public functions
        for func_info in analysis.functions:
            if not func_info.name.startswith("_"):
                tool_spec = self.generate_tool_spec(func_info)
                if tool_spec:
                    analysis.tool_specs.append(tool_spec)
        
        self._analysis_cache[repo_path] = analysis
        return analysis
    
    def extract_functions(self, repo_path: Path) -> List[FunctionInfo]:
        """
        Extract all functions from Python files in repository.
        
        Args:
            repo_path: Path to repository
        
        Returns:
            List of function information
        """
        functions = []
        
        for py_file in repo_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                functions.extend(self._parse_functions_from_file(py_file))
            except Exception as e:
                # Skip files that can't be parsed
                continue
        
        return functions
    
    def extract_classes(self, repo_path: Path) -> List[ClassInfo]:
        """
        Extract all classes from Python files in repository.
        
        Args:
            repo_path: Path to repository
        
        Returns:
            List of class information
        """
        classes = []
        
        for py_file in repo_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                classes.extend(self._parse_classes_from_file(py_file))
            except Exception as e:
                continue
        
        return classes
    
    def generate_tool_spec(self, func_or_class) -> Optional[ToolSpec]:
        """
        Generate tool specification from function or class.
        
        Args:
            func_or_class: FunctionInfo or ClassInfo instance
        
        Returns:
            ToolSpec if generatable, None otherwise
        """
        if isinstance(func_or_class, FunctionInfo):
            return self._function_to_tool_spec(func_or_class)
        elif isinstance(func_or_class, ClassInfo):
            return self._class_to_tool_spec(func_or_class)
        return None
    
    def build_dependency_graph(self, repo_path: Path) -> nx.DiGraph:
        """
        Build dependency graph of modules in repository.
        
        Args:
            repo_path: Path to repository
        
        Returns:
            NetworkX directed graph of dependencies
        """
        graph = nx.DiGraph()
        
        for py_file in repo_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            module_name = self._get_module_name(py_file, repo_path)
            graph.add_node(module_name, file=py_file)
            
            # Extract imports
            imports = self._extract_imports_from_file(py_file)
            for imp in imports:
                graph.add_edge(module_name, imp)
        
        return graph
    
    def _parse_functions_from_file(self, file_path: Path) -> List[FunctionInfo]:
        """Parse functions from single Python file."""
        functions = []
        
        with open(file_path, "r") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                return functions
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_info = FunctionInfo(
                    name=node.name,
                    file_path=file_path,
                    lineno=node.lineno,
                    docstring=ast.get_docstring(node),
                    parameters=[arg.arg for arg in node.args.args],
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                )
                
                # Extract return annotation
                if node.returns:
                    func_info.return_annotation = ast.unparse(node.returns)
                
                functions.append(func_info)
        
        return functions
    
    def _parse_classes_from_file(self, file_path: Path) -> List[ClassInfo]:
        """Parse classes from single Python file."""
        classes = []
        
        with open(file_path, "r") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                return classes
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = ClassInfo(
                    name=node.name,
                    file_path=file_path,
                    lineno=node.lineno,
                    docstring=ast.get_docstring(node),
                    base_classes=[ast.unparse(base) for base in node.bases],
                )
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_info.methods.append(item.name)
                    elif isinstance(item, ast.AnnAssign):
                        if isinstance(item.target, ast.Name):
                            class_info.attributes.append(item.target.id)
                
                classes.append(class_info)
        
        return classes
    
    def _function_to_tool_spec(self, func_info: FunctionInfo) -> Optional[ToolSpec]:
        """Convert function info to tool spec."""
        # Can't create actual callable without loading module
        # This returns a specification that can be used to generate tool later
        return ToolSpec(
            name=func_info.name,
            description=func_info.docstring or f"Function: {func_info.name}",
            function=lambda: None,  # Placeholder
            parameters={
                param: {"required": True, "type": "any"}
                for param in func_info.parameters
            },
            is_async=func_info.is_async,
            tags=["extracted", "function"],
        )
    
    def _class_to_tool_spec(self, class_info: ClassInfo) -> Optional[ToolSpec]:
        """Convert class info to tool spec."""
        # Tool spec for class instantiation
        return ToolSpec(
            name=class_info.name,
            description=class_info.docstring or f"Class: {class_info.name}",
            function=lambda: None,  # Placeholder
            parameters={},
            tags=["extracted", "class"],
        )
    
    def _extract_imports(self, repo_path: Path) -> Set[str]:
        """Extract all imports from repository."""
        imports = set()
        
        for py_file in repo_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            imports.update(self._extract_imports_from_file(py_file))
        
        return imports
    
    def _extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extract imports from single file."""
        imports = set()
        
        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
        except:
            pass
        
        return imports
    
    def _find_entry_points(self, repo_path: Path) -> List[str]:
        """Find entry point files (main.py, __main__.py, etc)."""
        entry_points = []
        
        entry_files = ["main.py", "__main__.py", "cli.py", "app.py"]
        for entry_file in entry_files:
            for found in repo_path.rglob(entry_file):
                entry_points.append(str(found.relative_to(repo_path)))
        
        return entry_points
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", "build", "dist"}
        skip_files = {"setup.py", "conf.py"}
        
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            return True
        
        if file_path.name in skip_files:
            return True
        
        return False
    
    def _get_module_name(self, file_path: Path, repo_path: Path) -> str:
        """Get module name from file path."""
        rel_path = file_path.relative_to(repo_path)
        return str(rel_path.with_suffix("")).replace("/", ".")
    
    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        return ast.unparse(decorator)

