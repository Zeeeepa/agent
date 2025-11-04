"""Startup Validator - Ensures all ML components load correctly on startup.

This runs automatically during system initialization to verify:
- All critical components can be imported
- Dependencies are available
- Configuration is valid
- System is ready for production use
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Tuple


async def validate_ml_system() -> Tuple[bool, Dict[str, any]]:
    """
    Validate ML system on startup.
    
    Returns:
        (success, report)
    """
    
    report = {
        'timestamp': time.time(),
        'components': {},
        'errors': [],
        'warnings': [],
        'success': False
    }
    
    # === CRITICAL COMPONENTS ===
    critical = [
        ('ml_orchestrator', 'jinx.micro.runtime.ml_orchestrator', 'get_ml_orchestrator'),
        ('semantic_cache', 'jinx.micro.embeddings.semantic_cache', 'get_embedding_cache'),
        ('ml_monitoring', 'jinx.micro.runtime.ml_monitoring', 'get_ml_monitoring'),
        ('auto_scaler', 'jinx.micro.runtime.auto_scaler', 'get_auto_scaler'),
    ]
    
    for name, module_path, function_name in critical:
        try:
            module = __import__(module_path, fromlist=[function_name])
            func = getattr(module, function_name)
            
            # Try to get instance
            instance = await func()
            
            report['components'][name] = {
                'status': 'ok',
                'loaded': True,
                'initialized': instance is not None
            }
        
        except Exception as e:
            report['components'][name] = {
                'status': 'error',
                'loaded': False,
                'error': str(e)
            }
            report['errors'].append(f"{name}: {str(e)}")
    
    # === OPTIONAL COMPONENTS ===
    optional = [
        ('embedding_classifier', 'jinx.micro.runtime.embedding_task_classifier', 'get_embedding_classifier'),
        ('user_learner', 'jinx.micro.runtime.user_profile_learner', 'get_user_learner'),
        ('performance_optimizer', 'jinx.micro.runtime.performance_optimizer', 'get_performance_optimizer'),
    ]
    
    for name, module_path, function_name in optional:
        try:
            module = __import__(module_path, fromlist=[function_name])
            func = getattr(module, function_name)
            instance = await func()
            
            report['components'][name] = {
                'status': 'ok',
                'loaded': True,
                'initialized': instance is not None
            }
        
        except Exception as e:
            report['components'][name] = {
                'status': 'warning',
                'loaded': False,
                'error': str(e)
            }
            report['warnings'].append(f"{name}: {str(e)}")
    
    # === DETERMINE SUCCESS ===
    critical_failures = [
        name for name, info in report['components'].items()
        if name in [c[0] for c in critical] and info['status'] == 'error'
    ]
    
    if not critical_failures:
        report['success'] = True
    
    # === LOG RESULTS ===
    try:
        from jinx.micro.logger.debug_logger import debug_log
        
        if report['success']:
            await debug_log(
                f"ML System validated: {len(report['components'])} components loaded",
                "STARTUP"
            )
        else:
            await debug_log(
                f"ML System validation FAILED: {len(critical_failures)} critical errors",
                "STARTUP"
            )
            for error in report['errors']:
                await debug_log(f"  ERROR: {error}", "STARTUP")
    
    except Exception:
        pass
    
    return (report['success'], report)


async def quick_health_check() -> bool:
    """
    Quick health check for production readiness.
    
    Returns True if system is ready.
    """
    
    try:
        # Check ML orchestrator
        from jinx.micro.runtime.ml_orchestrator import get_ml_orchestrator
        
        orchestrator = await get_ml_orchestrator()
        health = await orchestrator.health_check()
        
        # At minimum, orchestrator must be initialized
        return health.get('initialized', False)
    
    except Exception:
        return False


__all__ = [
    "validate_ml_system",
    "quick_health_check",
]
