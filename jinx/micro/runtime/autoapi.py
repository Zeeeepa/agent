"""Auto-API - intelligent module registration and lifecycle management.

Automatically discovers and manages:
- Brain systems (27)
- Memory systems (4)
- AI editor components
- ML patching systems
- All custom modules

NO MANUAL REGISTRATION NEEDED!
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Callable, List, Optional, Dict, Any


# Global registries for auto-start/stop callbacks (module-scope, opt-in)
AUTO_START_FUNCS: List[Callable[..., object]] = []
AUTO_STOP_FUNCS: List[Callable[..., object]] = []

# Module registry for intelligent loading
MODULE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def autostart(fn: Callable[..., object]) -> Callable[..., object]:
    """Decorator to register a function as an auto-start hook.

    The function may accept 0 args or a single PluginContext argument.
    """
    AUTO_START_FUNCS.append(fn)
    return fn


def autostop(fn: Callable[..., object]) -> Callable[..., object]:
    """Decorator to register a function as an auto-stop hook.

    The function may accept 0 args or a single PluginContext argument.
    """
    AUTO_STOP_FUNCS.append(fn)
    return fn


def register_module(
    name: str,
    module_path: str,
    *,
    priority: int = 50,
    required: bool = False,
    dependencies: Optional[List[str]] = None
) -> None:
    """Register module for automatic loading.
    
    Args:
        name: Module identifier
        module_path: Import path
        priority: Load priority (0-100, higher = earlier)
        required: If True, fail startup if module can't load
        dependencies: List of module names this depends on
    """
    MODULE_REGISTRY[name] = {
        'module_path': module_path,
        'priority': priority,
        'required': required,
        'dependencies': dependencies or [],
        'loaded': False,
        'instance': None
    }


async def auto_initialize_modules() -> Dict[str, Any]:
    """Automatically initialize all registered modules in priority order."""
    results = {}
    
    # Sort by priority (higher first)
    sorted_modules = sorted(
        MODULE_REGISTRY.items(),
        key=lambda x: x[1]['priority'],
        reverse=True
    )
    
    for name, config in sorted_modules:
        # Check dependencies
        for dep in config['dependencies']:
            if dep not in results or not results[dep]['success']:
                if config['required']:
                    raise RuntimeError(f"Module {name} requires {dep} which failed to load")
                continue
        
        # Try to load module
        try:
            module = importlib.import_module(config['module_path'])
            
            # Try to get singleton/initializer
            instance = None
            for attr_name in ['get_singleton', 'initialize', 'get_instance']:
                if hasattr(module, attr_name):
                    attr = getattr(module, attr_name)
                    if callable(attr):
                        try:
                            if asyncio.iscoroutinefunction(attr):
                                instance = await attr()
                            else:
                                instance = attr()
                            break
                        except Exception:
                            pass
            
            MODULE_REGISTRY[name]['loaded'] = True
            MODULE_REGISTRY[name]['instance'] = instance
            
            results[name] = {
                'success': True,
                'instance': instance,
                'module': module
            }
        
        except Exception as e:
            MODULE_REGISTRY[name]['loaded'] = False
            
            if config['required']:
                raise RuntimeError(f"Failed to load required module {name}: {e}")
            
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def register_builtin_modules() -> None:
    """Register all built-in Jinx modules for auto-loading."""
    
    # ===== BRAIN SYSTEMS (Priority 90-100) =====
    
    # Core orchestrator (highest priority)
    register_module(
        'brain_orchestrator',
        'jinx.micro.brain.brain_orchestrator',
        priority=100,
        required=False
    )
    
    # Memory integration
    register_module(
        'memory_integration',
        'jinx.micro.brain.memory_integration',
        priority=95,
        required=False
    )
    
    # Working memory
    register_module(
        'working_memory',
        'jinx.micro.brain.working_memory',
        priority=90,
        required=False
    )
    
    # Meta-cognitive
    register_module(
        'meta_cognitive',
        'jinx.micro.brain.meta_cognitive',
        priority=90,
        required=False
    )
    
    # ===== MEMORY SYSTEMS (Priority 80-89) =====
    
    register_module(
        'episodic_memory',
        'jinx.micro.brain.episodic_memory',
        priority=85,
        required=False
    )
    
    # ===== CONCURRENT PROCESSING (Priority 70-79) =====
    
    register_module(
        'concurrent_processor',
        'jinx.micro.brain.concurrent_processor',
        priority=75,
        required=False
    )
    
    register_module(
        'context_continuity',
        'jinx.micro.brain.context_continuity',
        priority=75,
        required=False
    )
    
    # ===== AI CODE EDITOR (Priority 60-69) =====
    
    register_module(
        'unified_ai_editor',
        'jinx.micro.code.unified_ai_editor',
        priority=65,
        required=False,
        dependencies=['ai_code_analyzer', 'memory_integrated_patcher']
    )
    
    register_module(
        'ai_code_analyzer',
        'jinx.micro.code.ai_code_analyzer',
        priority=68,
        required=False
    )
    
    register_module(
        'memory_integrated_patcher',
        'jinx.micro.code.memory_integrated_patcher',
        priority=68,
        required=False,
        dependencies=['memory_integration']
    )
    
    # ===== ML PATCHING (Priority 50-59) =====
    
    register_module(
        'semantic_patcher',
        'jinx.micro.code.semantic_patcher',
        priority=55,
        required=False
    )
    
    register_module(
        'ast_transformer',
        'jinx.micro.code.ast_transformer',
        priority=55,
        required=False
    )
    
    # ===== PATCH ORCHESTRATOR (Priority 50) =====
    
    register_module(
        'patch_orchestrator',
        'jinx.micro.code.patch_orchestrator',
        priority=50,
        required=False,
        dependencies=['semantic_patcher', 'ast_transformer']
    )
    
    # ===== ADAPTIVE CONFIGURATION (Priority 95) =====
    
    register_module(
        'adaptive_config',
        'jinx.micro.runtime.adaptive_config',
        priority=95,
        required=False
    )
    
    # ===== PERFORMANCE MONITOR (Priority 94) =====
    
    register_module(
        'performance_monitor',
        'jinx.micro.runtime.performance_monitor',
        priority=94,
        required=False
    )


__all__ = [
    "AUTO_START_FUNCS",
    "AUTO_STOP_FUNCS",
    "MODULE_REGISTRY",
    "autostart",
    "autostop",
    "register_module",
    "auto_initialize_modules",
    "register_builtin_modules",
]
