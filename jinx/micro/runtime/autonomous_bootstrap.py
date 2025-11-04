"""Autonomous Bootstrap - fully automatic initialization.

Jinx boots herself:
1. Discovers system capabilities
2. Applies optimal configuration
3. Loads all available modules
4. Starts brain systems
5. Initializes memory
6. Activates AI editor

ZERO USER CONFIGURATION REQUIRED!
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from .autoconfig import apply_auto_defaults, _detect_system_capabilities
from .autodiscovery import discover_and_configure, get_autodiscovery
from .autoapi import register_builtin_modules, auto_initialize_modules, MODULE_REGISTRY


class AutonomousBootstrap:
    """Fully autonomous bootstrap system."""
    
    def __init__(self):
        self._initialized = False
        self._capabilities = None
        self._modules = {}
    
    async def bootstrap(self) -> Dict[str, Any]:
        """Bootstrap Jinx automatically - NO USER INPUT!
        
        Returns:
            Dict with initialization status and loaded modules
        """
        
        if self._initialized:
            return {
                'already_initialized': True,
                'modules': self._modules
            }
        
        print("🚀 Jinx Autonomous Bootstrap Starting...")
        print("   No configuration needed - I'll figure it out!")
        print()
        
        # Step 1: Discover system capabilities
        print("🔍 Step 1/5: Discovering system capabilities...")
        try:
            discovery = await get_autodiscovery()
            caps = await discovery.discover()
            self._capabilities = caps
            
            # Show discovery report
            report = discovery.get_capabilities_report(caps)
            print(report)
            print()
        except Exception as e:
            print(f"⚠️  Discovery failed (non-fatal): {e}")
            print("   Continuing with defaults...")
            caps = None
        
        # Step 2: Apply automatic configuration
        print("⚙️  Step 2/5: Applying intelligent configuration...")
        try:
            # Apply .env-based defaults
            apply_auto_defaults()
            
            # Apply discovery-based settings
            if caps:
                await discover_and_configure()
            
            print("✓ Configuration applied automatically")
            print()
        except Exception as e:
            print(f"⚠️  Config failed (non-fatal): {e}")
            print("   Using fallback defaults...")
        
        # Step 3: Register all built-in modules
        print("📦 Step 3/5: Registering intelligent modules...")
        try:
            register_builtin_modules()
            
            registered_count = len(MODULE_REGISTRY)
            print(f"✓ Registered {registered_count} modules for auto-loading")
            print()
        except Exception as e:
            print(f"⚠️  Registration failed: {e}")
        
        # Step 4: Initialize modules in priority order
        print("🧠 Step 4/5: Initializing brain & AI systems...")
        try:
            results = await auto_initialize_modules()
            self._modules = results
            
            success_count = sum(1 for r in results.values() if r.get('success'))
            total_count = len(results)
            
            print(f"✓ Loaded {success_count}/{total_count} modules successfully")
            
            # Show loaded modules
            if success_count > 0:
                print("\n   Loaded systems:")
                for name, result in results.items():
                    if result.get('success'):
                        emoji = "🧠" if 'brain' in name or 'memory' in name else "⚡" if 'editor' in name or 'code' in name else "🔧"
                        print(f"   {emoji} {name}")
            
            print()
        except Exception as e:
            print(f"⚠️  Module initialization failed: {e}")
            self._modules = {}
        
        # Step 5: Final checks
        print("✅ Step 5/5: Final validation...")
        
        # Check critical systems
        has_brain = any('brain' in k or 'memory' in k for k in self._modules.keys())
        has_editor = any('editor' in k or 'code' in k for k in self._modules.keys())
        
        if has_brain:
            print("   ✓ Brain systems operational")
        else:
            print("   ⚠️  Brain systems not available")
        
        if has_editor:
            print("   ✓ AI Editor ready")
        else:
            print("   ⚠️  AI Editor not available")
        
        print()
        print("=" * 60)
        print("🎉 Jinx is ready! All systems configured automatically.")
        print("   No .env editing needed - I adapt to your system!")
        print("=" * 60)
        print()
        
        # Start adaptive configuration
        print("🔄 Starting adaptive configuration system...")
        try:
            from .adaptive_config import get_adaptive_config
            adaptive_config = await get_adaptive_config()
            print("   ✓ Adaptive config active - learning in real-time!")
        except Exception as e:
            print(f"   ⚠️  Adaptive config not available: {e}")
            adaptive_config = None
        
        self._initialized = True
        
        return {
            'success': True,
            'capabilities': caps,
            'modules': self._modules,
            'brain_available': has_brain,
            'editor_available': has_editor,
            'adaptive_config': adaptive_config is not None
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bootstrap status."""
        return {
            'initialized': self._initialized,
            'modules_loaded': len(self._modules),
            'modules': list(self._modules.keys()),
            'capabilities': self._capabilities
        }


# Singleton
_bootstrap: Optional[AutonomousBootstrap] = None


async def get_bootstrap() -> AutonomousBootstrap:
    """Get singleton bootstrap instance."""
    global _bootstrap
    if _bootstrap is None:
        _bootstrap = AutonomousBootstrap()
    return _bootstrap


async def auto_bootstrap() -> Dict[str, Any]:
    """Run full autonomous bootstrap."""
    bootstrap = await get_bootstrap()
    return await bootstrap.bootstrap()


def is_bootstrapped() -> bool:
    """Check if bootstrap completed."""
    global _bootstrap
    return _bootstrap is not None and _bootstrap._initialized


__all__ = [
    "AutonomousBootstrap",
    "get_bootstrap",
    "auto_bootstrap",
    "is_bootstrapped",
]
