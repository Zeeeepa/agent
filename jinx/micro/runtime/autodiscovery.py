"""Auto-Discovery - automatically detect and enable available modules.

Jinx intelligently discovers:
- Available ML libraries (LibCST, sklearn, etc.)
- Brain system capabilities
- Memory system availability
- Hardware resources
- Network connectivity
- API keys presence

NO USER CONFIGURATION NEEDED!
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SystemCapabilities:
    """Detected system capabilities."""
    
    # ML Libraries
    has_libcst: bool = False
    has_sklearn: bool = False
    has_numpy: bool = False
    has_torch: bool = False
    
    # Core dependencies
    has_openai: bool = False
    has_aiofiles: bool = False
    has_psutil: bool = False
    
    # API availability
    has_openai_key: bool = False
    openai_model: Optional[str] = None
    
    # Brain systems
    brain_systems_available: int = 0
    
    # Memory systems
    memory_systems_available: int = 0
    
    # Hardware
    cpu_count: int = 0
    memory_gb: float = 0.0
    
    # Network
    internet_available: bool = False
    
    # Features
    embeddings_available: bool = False
    ast_transforms_available: bool = False
    ml_patching_available: bool = False
    ai_editor_available: bool = False


class AutoDiscovery:
    """Automatically discover available capabilities."""
    
    def __init__(self):
        self._capabilities: Optional[SystemCapabilities] = None
        self._lock = asyncio.Lock()
    
    async def discover(self) -> SystemCapabilities:
        """Discover all available capabilities."""
        
        if self._capabilities is not None:
            return self._capabilities
        
        async with self._lock:
            if self._capabilities is not None:
                return self._capabilities
            
            caps = SystemCapabilities()
            
            # Discover ML libraries
            caps.has_libcst = self._check_import('libcst')
            caps.has_sklearn = self._check_import('sklearn')
            caps.has_numpy = self._check_import('numpy')
            caps.has_torch = self._check_import('torch')
            
            # Discover core dependencies
            caps.has_openai = self._check_import('openai')
            caps.has_aiofiles = self._check_import('aiofiles')
            caps.has_psutil = self._check_import('psutil')
            
            # Check API keys
            caps.has_openai_key = bool(os.getenv('OPENAI_API_KEY'))
            caps.openai_model = os.getenv('OPENAI_MODEL') or 'gpt-4o'
            
            # Discover brain systems
            caps.brain_systems_available = await self._count_brain_systems()
            
            # Discover memory systems
            caps.memory_systems_available = await self._count_memory_systems()
            
            # Hardware capabilities
            if caps.has_psutil:
                try:
                    import psutil
                    caps.cpu_count = psutil.cpu_count(logical=True) or 4
                    caps.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
                except Exception:
                    caps.cpu_count = 4
                    caps.memory_gb = 8.0
            else:
                caps.cpu_count = 4
                caps.memory_gb = 8.0
            
            # Network availability
            caps.internet_available = await self._check_internet()
            
            # Feature availability
            caps.embeddings_available = caps.has_openai and caps.has_openai_key
            caps.ast_transforms_available = caps.has_libcst
            caps.ml_patching_available = caps.has_numpy and caps.has_sklearn
            caps.ai_editor_available = caps.ast_transforms_available
            
            self._capabilities = caps
            return caps
    
    def _check_import(self, module_name: str) -> bool:
        """Check if module can be imported."""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    async def _count_brain_systems(self) -> int:
        """Count available brain systems by checking file existence."""
        count = 0
        
        # Check by file existence instead of import to avoid circular deps
        try:
            from pathlib import Path
            brain_dir = Path(__file__).parent.parent / 'brain'
            
            if brain_dir.exists():
                brain_files = list(brain_dir.glob('*.py'))
                # Count python files (excluding __init__ and __pycache__)
                count = len([f for f in brain_files if f.stem not in ('__init__', '__pycache__')])
        except Exception:
            # Fallback: assume 27 systems available
            count = 27
        
        return count
    
    async def _count_memory_systems(self) -> int:
        """Count available memory systems by checking file existence."""
        count = 0
        
        # Check by file existence to avoid circular deps
        try:
            from pathlib import Path
            
            # Check key memory files
            brain_dir = Path(__file__).parent.parent / 'brain'
            memory_dir = Path(__file__).parent.parent / 'memory'
            
            memory_files = [
                brain_dir / 'working_memory.py',
                brain_dir / 'episodic_memory.py',
                brain_dir / 'memory_integration.py',
                memory_dir / '__init__.py',
            ]
            
            count = sum(1 for f in memory_files if f.exists())
        except Exception:
            # Fallback: assume 4 systems
            count = 4
        
        return count
    
    async def _check_internet(self) -> bool:
        """Check if internet is available."""
        try:
            # Try to resolve openai.com
            import socket
            socket.create_connection(("api.openai.com", 443), timeout=2)
            return True
        except (socket.error, socket.timeout):
            return False
        except Exception:
            return False
    
    def get_recommended_settings(self, caps: SystemCapabilities) -> Dict[str, str]:
        """Get recommended settings based on capabilities."""
        settings = {}
        
        # Core settings
        if caps.has_openai_key:
            settings['OPENAI_MODEL'] = caps.openai_model
        
        # Brain systems
        if caps.brain_systems_available >= 20:
            settings['JINX_BRAIN_ENABLE'] = '1'
            settings['JINX_BRAIN_ALL_SYSTEMS'] = '1'
        elif caps.brain_systems_available >= 10:
            settings['JINX_BRAIN_ENABLE'] = '1'
            settings['JINX_BRAIN_CORE_ONLY'] = '1'
        
        # Memory systems
        if caps.memory_systems_available >= 3:
            settings['JINX_MEMORY_FULL'] = '1'
        elif caps.memory_systems_available >= 1:
            settings['JINX_MEMORY_BASIC'] = '1'
        
        # AI Editor
        if caps.ai_editor_available:
            settings['JINX_AI_EDITOR_ENABLE'] = '1'
            settings['JINX_AST_TRANSFORMS'] = '1'
        
        # ML Patching
        if caps.ml_patching_available:
            settings['JINX_ML_PATCHING'] = '1'
            settings['JINX_SEMANTIC_PATCH_ML'] = '1'
        
        # Embeddings
        if caps.embeddings_available:
            settings['EMBED_PROJECT_ENABLE'] = '1'
            settings['EMBED_BRAIN_ENABLE'] = '1'
        
        # Concurrency based on hardware
        if caps.cpu_count >= 8 and caps.memory_gb >= 16:
            settings['JINX_MAX_CONCURRENT'] = '5'
            settings['JINX_FRAME_MAX_CONC'] = '4'
        elif caps.cpu_count >= 4:
            settings['JINX_MAX_CONCURRENT'] = '3'
            settings['JINX_FRAME_MAX_CONC'] = '2'
        else:
            settings['JINX_MAX_CONCURRENT'] = '2'
            settings['JINX_FRAME_MAX_CONC'] = '1'
        
        return settings
    
    async def apply_discovered_settings(self) -> None:
        """Apply settings based on discovery."""
        caps = await self.discover()
        settings = self.get_recommended_settings(caps)
        
        for key, value in settings.items():
            if os.getenv(key) in (None, ""):
                os.environ[key] = value
    
    def get_capabilities_report(self, caps: SystemCapabilities) -> str:
        """Generate human-readable capabilities report."""
        report = ["🔍 Jinx Autonomous Discovery Report", "=" * 50, ""]
        
        # API Status
        report.append("📡 API & Connectivity:")
        report.append(f"  {'✓' if caps.has_openai_key else '✗'} OpenAI API Key")
        report.append(f"  {'✓' if caps.internet_available else '✗'} Internet Connection")
        report.append(f"  Model: {caps.openai_model}")
        report.append("")
        
        # ML Libraries
        report.append("🔬 ML Libraries:")
        report.append(f"  {'✓' if caps.has_libcst else '✗'} LibCST (AST transforms)")
        report.append(f"  {'✓' if caps.has_sklearn else '✗'} scikit-learn (ML)")
        report.append(f"  {'✓' if caps.has_numpy else '✗'} NumPy")
        report.append(f"  {'✓' if caps.has_torch else '✗'} PyTorch")
        report.append("")
        
        # Core Dependencies
        report.append("📦 Core Dependencies:")
        report.append(f"  {'✓' if caps.has_openai else '✗'} OpenAI SDK")
        report.append(f"  {'✓' if caps.has_aiofiles else '✗'} aiofiles")
        report.append(f"  {'✓' if caps.has_psutil else '✗'} psutil")
        report.append("")
        
        # Brain Systems
        report.append("🧠 Brain Systems:")
        report.append(f"  Available: {caps.brain_systems_available}/27")
        status = "Full" if caps.brain_systems_available >= 20 else "Partial" if caps.brain_systems_available >= 10 else "Limited"
        report.append(f"  Status: {status}")
        report.append("")
        
        # Memory Systems
        report.append("💾 Memory Systems:")
        report.append(f"  Available: {caps.memory_systems_available}/4")
        report.append("")
        
        # Hardware
        report.append("🖥️ Hardware:")
        report.append(f"  CPU Cores: {caps.cpu_count}")
        report.append(f"  Memory: {caps.memory_gb:.1f} GB")
        perf = "High" if caps.cpu_count >= 8 and caps.memory_gb >= 16 else "Medium" if caps.cpu_count >= 4 else "Low"
        report.append(f"  Performance: {perf}")
        report.append("")
        
        # Features
        report.append("✨ Available Features:")
        report.append(f"  {'✓' if caps.embeddings_available else '✗'} Embeddings & Retrieval")
        report.append(f"  {'✓' if caps.ast_transforms_available else '✗'} AST Transformations")
        report.append(f"  {'✓' if caps.ml_patching_available else '✗'} ML-Powered Patching")
        report.append(f"  {'✓' if caps.ai_editor_available else '✗'} AI Code Editor")
        report.append("")
        
        report.append("=" * 50)
        report.append("🚀 Jinx configured automatically!")
        
        return "\n".join(report)


# Singleton
_discovery: Optional[AutoDiscovery] = None
_discovery_lock = asyncio.Lock()


async def get_autodiscovery() -> AutoDiscovery:
    """Get singleton auto-discovery instance."""
    global _discovery
    if _discovery is None:
        async with _discovery_lock:
            if _discovery is None:
                _discovery = AutoDiscovery()
    return _discovery


async def discover_and_configure() -> SystemCapabilities:
    """Discover capabilities and apply configuration."""
    discovery = await get_autodiscovery()
    caps = await discovery.discover()
    await discovery.apply_discovered_settings()
    return caps


__all__ = [
    "AutoDiscovery",
    "SystemCapabilities",
    "get_autodiscovery",
    "discover_and_configure",
]
