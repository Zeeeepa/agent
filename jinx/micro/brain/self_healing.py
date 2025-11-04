"""Self-healing code system with automatic error correction.

Automatically fixes common errors using learned patterns and ML.
"""

from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class HealingStrategy:
    """Code healing strategy."""
    strategy_id: str
    error_pattern: str
    fix_template: str
    success_rate: float
    uses: int
    successes: int


@dataclass
class HealingResult:
    """Result of healing attempt."""
    success: bool
    fixed_code: Optional[str]
    strategy_used: Optional[str]
    confidence: float
    changes: List[str]


class SelfHealingSystem:
    """Automatically fix code errors using ML-learned patterns."""
    
    def __init__(self, state_path: str = "log/self_healing.json"):
        self.state_path = state_path
        
        # Learned healing strategies
        self.strategies: Dict[str, HealingStrategy] = {}
        
        # Healing history
        self.history: deque[Tuple[str, str, bool]] = deque(maxlen=200)  # (error, strategy_id, success)
        
        # Built-in healing rules (bootstrap)
        self._register_builtin_strategies()
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _register_builtin_strategies(self) -> None:
        """Register built-in healing strategies."""
        builtin = {
            'none_attribute': HealingStrategy(
                strategy_id='none_attribute',
                error_pattern=r"AttributeError.*'NoneType'.*'(\w+)'",
                fix_template="Add None check before attribute access",
                success_rate=0.7,
                uses=0,
                successes=0
            ),
            'key_error': HealingStrategy(
                strategy_id='key_error',
                error_pattern=r"KeyError.*'(\w+)'",
                fix_template="Replace dict[key] with dict.get(key, default)",
                success_rate=0.8,
                uses=0,
                successes=0
            ),
            'index_error': HealingStrategy(
                strategy_id='index_error',
                error_pattern=r"IndexError.*list index",
                fix_template="Add bounds check before indexing",
                success_rate=0.6,
                uses=0,
                successes=0
            ),
            'undefined_variable': HealingStrategy(
                strategy_id='undefined_variable',
                error_pattern=r"NameError.*'(\w+)'.*not defined",
                fix_template="Initialize variable before use",
                success_rate=0.5,
                uses=0,
                successes=0
            ),
            'type_error_operand': HealingStrategy(
                strategy_id='type_error_operand',
                error_pattern=r"TypeError.*unsupported operand",
                fix_template="Add type conversion or check",
                success_rate=0.6,
                uses=0,
                successes=0
            ),
            'import_error': HealingStrategy(
                strategy_id='import_error',
                error_pattern=r"ImportError|ModuleNotFoundError",
                fix_template="Add package installation or fix import path",
                success_rate=0.7,
                uses=0,
                successes=0
            ),
        }
        
        self.strategies.update(builtin)
    
    def _load_state(self) -> None:
        """Load persisted strategies."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore strategies
                for strategy_data in data.get('strategies', []):
                    strategy = HealingStrategy(
                        strategy_id=strategy_data['strategy_id'],
                        error_pattern=strategy_data['error_pattern'],
                        fix_template=strategy_data['fix_template'],
                        success_rate=strategy_data['success_rate'],
                        uses=strategy_data['uses'],
                        successes=strategy_data['successes']
                    )
                    # Update or add
                    if strategy.strategy_id in self.strategies:
                        self.strategies[strategy.strategy_id].success_rate = strategy.success_rate
                        self.strategies[strategy.strategy_id].uses = strategy.uses
                        self.strategies[strategy.strategy_id].successes = strategy.successes
                    else:
                        self.strategies[strategy.strategy_id] = strategy
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist strategies."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                strategies_data = [
                    {
                        'strategy_id': s.strategy_id,
                        'error_pattern': s.error_pattern,
                        'fix_template': s.fix_template,
                        'success_rate': s.success_rate,
                        'uses': s.uses,
                        'successes': s.successes
                    }
                    for s in self.strategies.values()
                ]
                
                data = {
                    'strategies': strategies_data,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _match_error_pattern(self, error_msg: str) -> List[Tuple[HealingStrategy, float]]:
        """Match error message to strategies."""
        matches: List[Tuple[HealingStrategy, float]] = []
        
        for strategy in self.strategies.values():
            try:
                if re.search(strategy.error_pattern, error_msg, re.IGNORECASE):
                    # Confidence based on success rate
                    confidence = strategy.success_rate * (1 + strategy.successes / max(1, strategy.uses + 1))
                    matches.append((strategy, confidence))
            except re.error:
                continue
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    async def heal(
        self,
        code: str,
        error_msg: str,
        error_type: str
    ) -> HealingResult:
        """Attempt to heal code automatically."""
        async with self._lock:
            # Match error to strategies
            matches = self._match_error_pattern(error_msg)
            
            if not matches:
                return HealingResult(
                    success=False,
                    fixed_code=None,
                    strategy_used=None,
                    confidence=0.0,
                    changes=[]
                )
            
            # Try best strategy
            strategy, confidence = matches[0]
            strategy.uses += 1
            
            # Apply healing
            fixed_code, changes = await self._apply_strategy(code, error_msg, error_type, strategy)
            
            if fixed_code:
                strategy.successes += 1
                
                # Update success rate (EMA)
                alpha = 0.1
                new_rate = 1.0
                strategy.success_rate = alpha * new_rate + (1 - alpha) * strategy.success_rate
                
                # Record history
                self.history.append((error_msg, strategy.strategy_id, True))
                
                # Save periodically
                if strategy.uses % 5 == 0:
                    await self._save_state()
                
                return HealingResult(
                    success=True,
                    fixed_code=fixed_code,
                    strategy_used=strategy.strategy_id,
                    confidence=confidence,
                    changes=changes
                )
            else:
                # Failed
                alpha = 0.1
                new_rate = 0.0
                strategy.success_rate = alpha * new_rate + (1 - alpha) * strategy.success_rate
                
                self.history.append((error_msg, strategy.strategy_id, False))
                
                return HealingResult(
                    success=False,
                    fixed_code=None,
                    strategy_used=strategy.strategy_id,
                    confidence=confidence,
                    changes=[]
                )
    
    async def _apply_strategy(
        self,
        code: str,
        error_msg: str,
        error_type: str,
        strategy: HealingStrategy
    ) -> Tuple[Optional[str], List[str]]:
        """Apply healing strategy to code."""
        changes: List[str] = []
        
        try:
            if strategy.strategy_id == 'none_attribute':
                return await self._fix_none_attribute(code, error_msg, changes)
            
            elif strategy.strategy_id == 'key_error':
                return await self._fix_key_error(code, error_msg, changes)
            
            elif strategy.strategy_id == 'index_error':
                return await self._fix_index_error(code, error_msg, changes)
            
            elif strategy.strategy_id == 'undefined_variable':
                return await self._fix_undefined_variable(code, error_msg, changes)
            
            elif strategy.strategy_id == 'type_error_operand':
                return await self._fix_type_error(code, error_msg, changes)
            
            elif strategy.strategy_id == 'import_error':
                return await self._fix_import_error(code, error_msg, changes)
            
            else:
                # Generic strategy
                return None, []
        
        except Exception:
            return None, []
    
    async def _fix_none_attribute(self, code: str, error_msg: str, changes: List[str]) -> Tuple[Optional[str], List[str]]:
        """Fix NoneType attribute access."""
        # Extract attribute name from error
        match = re.search(r"'(\w+)'", error_msg)
        if not match:
            return None, changes
        
        attr_name = match.group(1)
        
        # Find attribute accesses
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if f".{attr_name}" in line and "if " not in line and "is not None" not in line:
                # Add None check
                indent = len(line) - len(line.lstrip())
                var_match = re.search(rf'(\w+)\.{attr_name}', line)
                if var_match:
                    var_name = var_match.group(1)
                    fixed_lines.append(' ' * indent + f'if {var_name} is not None:')
                    fixed_lines.append(' ' * (indent + 4) + line.strip())
                    changes.append(f"Added None check for {var_name}.{attr_name}")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if changes:
            return '\n'.join(fixed_lines), changes
        return None, []
    
    async def _fix_key_error(self, code: str, error_msg: str, changes: List[str]) -> Tuple[Optional[str], List[str]]:
        """Fix KeyError by replacing dict[key] with dict.get(key, default)."""
        # Extract key from error
        match = re.search(r"'(\w+)'", error_msg)
        if not match:
            return None, changes
        
        key_name = match.group(1)
        
        # Replace dict[key] with dict.get(key, None)
        fixed = code
        pattern = rf'(\w+)\[(["\']){key_name}\2\]'
        
        def replacer(m):
            dict_name = m.group(1)
            changes.append(f"Replaced {dict_name}['{key_name}'] with .get()")
            return f'{dict_name}.get("{key_name}", None)'
        
        fixed = re.sub(pattern, replacer, fixed)
        
        if changes:
            return fixed, changes
        return None, []
    
    async def _fix_index_error(self, code: str, error_msg: str, changes: List[str]) -> Tuple[Optional[str], List[str]]:
        """Fix IndexError by adding bounds check."""
        # Find direct index accesses
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            match = re.search(r'(\w+)\[(\d+)\]', line)
            if match and 'if ' not in line and 'len(' not in line:
                list_name = match.group(1)
                index = match.group(2)
                indent = len(line) - len(line.lstrip())
                
                fixed_lines.append(' ' * indent + f'if len({list_name}) > {index}:')
                fixed_lines.append(' ' * (indent + 4) + line.strip())
                changes.append(f"Added bounds check for {list_name}[{index}]")
            else:
                fixed_lines.append(line)
        
        if changes:
            return '\n'.join(fixed_lines), changes
        return None, []
    
    async def _fix_undefined_variable(self, code: str, error_msg: str, changes: List[str]) -> Tuple[Optional[str], List[str]]:
        """Fix undefined variable by initializing."""
        match = re.search(r"'(\w+)'", error_msg)
        if not match:
            return None, changes
        
        var_name = match.group(1)
        
        # Add initialization at the beginning
        lines = code.split('\n')
        
        # Find first non-import line
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(('import ', 'from ', '#')):
                insert_index = i
                break
        
        lines.insert(insert_index, f'{var_name} = None  # Auto-initialized by self-healing')
        changes.append(f"Initialized {var_name} = None")
        
        return '\n'.join(lines), changes
    
    async def _fix_type_error(self, code: str, error_msg: str, changes: List[str]) -> Tuple[Optional[str], List[str]]:
        """Fix type error by adding conversion."""
        # Try to add str() or int() conversions
        # This is a simplified heuristic
        if 'str' in error_msg.lower() and 'int' in error_msg.lower():
            # Add type conversions where needed
            fixed = code
            # Look for arithmetic operations on potential strings
            fixed = re.sub(r'(\w+)\s*\+\s*(\w+)', r'str(\1) + str(\2)', fixed)
            changes.append("Added type conversions")
            return fixed, changes
        
        return None, []
    
    async def _fix_import_error(self, code: str, error_msg: str, changes: List[str]) -> Tuple[Optional[str], List[str]]:
        """Fix import error by adding package installation."""
        # Extract module name
        match = re.search(r"No module named '(\w+)'", error_msg)
        if not match:
            return None, changes
        
        module_name = match.group(1)
        
        # Add installation code at the beginning
        install_code = f"""
# Auto-install missing module
import subprocess
import sys
try:
    import {module_name}
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '{module_name}'])
    import {module_name}
"""
        
        fixed = install_code + '\n' + code
        changes.append(f"Added auto-install for {module_name}")
        
        return fixed, changes
    
    def get_stats(self) -> Dict[str, object]:
        """Get healing statistics."""
        total_uses = sum(s.uses for s in self.strategies.values())
        total_successes = sum(s.successes for s in self.strategies.values())
        
        success_rate = total_successes / total_uses if total_uses > 0 else 0.0
        
        strategy_stats = {
            s.strategy_id: {
                'success_rate': s.success_rate,
                'uses': s.uses,
                'successes': s.successes
            }
            for s in self.strategies.values()
            if s.uses > 0
        }
        
        return {
            'total_healing_attempts': total_uses,
            'total_successes': total_successes,
            'overall_success_rate': success_rate,
            'strategies': strategy_stats
        }


# Singleton
_healer: Optional[SelfHealingSystem] = None
_healer_lock = asyncio.Lock()


async def get_self_healer() -> SelfHealingSystem:
    """Get singleton self-healing system."""
    global _healer
    if _healer is None:
        async with _healer_lock:
            if _healer is None:
                _healer = SelfHealingSystem()
    return _healer


async def heal_code(code: str, error_msg: str, error_type: str) -> HealingResult:
    """Attempt to heal code automatically."""
    healer = await get_self_healer()
    return await healer.heal(code, error_msg, error_type)


__all__ = [
    "SelfHealingSystem",
    "HealingResult",
    "get_self_healer",
    "heal_code",
]
