"""ML-based error prediction and prevention system.

Predicts potential errors before execution using pattern learning.
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ErrorPattern:
    """Learned error pattern."""
    pattern_id: str
    error_type: str  # TypeError, AttributeError, etc.
    indicators: Dict[str, float]  # code_features -> importance
    frequency: int
    last_seen: float


@dataclass
class ErrorPrediction:
    """Predicted error with confidence."""
    error_type: str
    confidence: float
    indicators: List[str]
    suggested_fix: Optional[str]


class IntelligentErrorPredictor:
    """Predict errors before execution using ML pattern recognition."""
    
    def __init__(self, state_path: str = "log/error_predictor.json"):
        self.state_path = state_path
        
        # Learned error patterns
        self.patterns: Dict[str, ErrorPattern] = {}
        
        # Error history for learning
        self.error_history: deque[Tuple[str, str, Dict[str, float]]] = deque(maxlen=500)  # (code, error_type, features)
        
        # Code feature extractors (language-agnostic)
        self.feature_weights: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Common error types and their signatures
        self.error_signatures: Dict[str, List[str]] = {
            'TypeError': ['type mismatch', 'none attribute', 'unsupported operand'],
            'AttributeError': ['no attribute', 'nonetype', 'module has no'],
            'KeyError': ['key not found', 'missing key', 'dict key'],
            'IndexError': ['out of range', 'list index', 'index out'],
            'ValueError': ['invalid value', 'cannot convert', 'invalid literal'],
            'ImportError': ['no module', 'cannot import', 'import failed'],
            'NameError': ['not defined', 'name not found', 'undefined'],
        }
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted patterns."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore patterns
                for pattern_data in data.get('patterns', []):
                    pattern = ErrorPattern(
                        pattern_id=pattern_data['pattern_id'],
                        error_type=pattern_data['error_type'],
                        indicators=pattern_data['indicators'],
                        frequency=pattern_data['frequency'],
                        last_seen=pattern_data['last_seen']
                    )
                    self.patterns[pattern.pattern_id] = pattern
                
                # Restore feature weights
                self.feature_weights = defaultdict(lambda: 0.5, data.get('feature_weights', {}))
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist patterns."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                patterns_data = [
                    {
                        'pattern_id': p.pattern_id,
                        'error_type': p.error_type,
                        'indicators': p.indicators,
                        'frequency': p.frequency,
                        'last_seen': p.last_seen
                    }
                    for p in self.patterns.values()
                ]
                
                data = {
                    'patterns': patterns_data,
                    'feature_weights': dict(self.feature_weights),
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _extract_code_features(self, code: str) -> Dict[str, float]:
        """Extract language-agnostic code features."""
        features: Dict[str, float] = {}
        
        if not code:
            return features
        
        c = code.strip()
        
        # Structural features
        features['has_none_check'] = 1.0 if 'is None' in c or 'is not None' in c else 0.0
        features['has_try_except'] = 1.0 if 'try:' in c or 'except' in c else 0.0
        features['has_type_check'] = 1.0 if 'isinstance(' in c or 'type(' in c else 0.0
        features['has_dict_get'] = 1.0 if '.get(' in c else 0.0
        features['has_index_access'] = 1.0 if '[' in c and ']' in c else 0.0
        features['has_attribute_access'] = float(c.count('.')) / max(1, len(c)) * 100
        
        # Dangerous patterns
        features['has_eval'] = 1.0 if 'eval(' in c else 0.0
        features['has_exec'] = 1.0 if 'exec(' in c else 0.0
        features['direct_index'] = float(sum(1 for m in re.finditer(r'\[\d+\]', c))) / max(1, len(c.split('\n')))
        
        # Variable usage patterns
        lines = c.split('\n')
        undefined_usage = 0
        for i, line in enumerate(lines):
            # Simple heuristic: variable used before potential assignment
            if '=' not in line and any(op in line for op in ['.', '[', '(']):
                undefined_usage += 1
        features['potential_undefined'] = min(1.0, undefined_usage / max(1, len(lines)))
        
        # None-related risks
        none_count = c.lower().count('none')
        features['none_density'] = min(1.0, none_count / max(1, len(lines)))
        
        # Import patterns
        features['has_imports'] = 1.0 if 'import ' in c else 0.0
        features['has_from_import'] = 1.0 if 'from ' in c and 'import' in c else 0.0
        
        # AST-based features (if parseable)
        try:
            tree = ast.parse(c)
            
            # Count risky nodes
            risky_nodes = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.Subscript, ast.Attribute)):
                    risky_nodes += 1
            
            features['risky_node_density'] = min(1.0, risky_nodes / max(1, len(lines)))
        except Exception:
            features['parse_failed'] = 1.0
        
        return features
    
    async def predict(self, code: str) -> List[ErrorPrediction]:
        """Predict potential errors in code."""
        async with self._lock:
            features = self._extract_code_features(code)
            
            predictions: List[ErrorPrediction] = []
            
            # Match against learned patterns
            for pattern in self.patterns.values():
                # Compute similarity score
                score = 0.0
                matched_indicators: List[str] = []
                
                for indicator, importance in pattern.indicators.items():
                    if indicator in features:
                        feature_value = features[indicator]
                        score += feature_value * importance
                        if feature_value > 0.5:
                            matched_indicators.append(indicator)
                
                # Normalize by number of indicators
                if pattern.indicators:
                    score /= len(pattern.indicators)
                
                # Frequency boost
                score *= (1 + pattern.frequency / 100.0)
                
                # Threshold for prediction
                if score > 0.6:
                    # Generate suggested fix
                    suggested_fix = self._suggest_fix(pattern.error_type, matched_indicators)
                    
                    predictions.append(ErrorPrediction(
                        error_type=pattern.error_type,
                        confidence=min(1.0, score),
                        indicators=matched_indicators,
                        suggested_fix=suggested_fix
                    ))
            
            # Sort by confidence
            predictions.sort(key=lambda p: p.confidence, reverse=True)
            
            return predictions[:5]  # Top 5 predictions
    
    def _suggest_fix(self, error_type: str, indicators: List[str]) -> Optional[str]:
        """Suggest fix based on error type and indicators."""
        suggestions = {
            'TypeError': "Add type checks: isinstance(obj, expected_type)",
            'AttributeError': "Check if attribute exists: hasattr(obj, 'attr') or use getattr(obj, 'attr', default)",
            'KeyError': "Use dict.get(key, default) instead of dict[key]",
            'IndexError': "Check list length before indexing: if len(lst) > index",
            'ValueError': "Validate input before conversion: check format and type",
            'ImportError': "Ensure module is installed and path is correct",
            'NameError': "Check variable is defined before use",
        }
        
        base_suggestion = suggestions.get(error_type, "Review code for potential issues")
        
        # Add specific guidance based on indicators
        if 'has_none_check' in indicators and error_type == 'AttributeError':
            return f"{base_suggestion} (detected None-related pattern)"
        
        return base_suggestion
    
    async def learn_from_error(
        self,
        code: str,
        error_type: str,
        error_message: str
    ) -> None:
        """Learn from actual error."""
        async with self._lock:
            features = self._extract_code_features(code)
            
            # Record in history
            self.error_history.append((code, error_type, features))
            
            # Create or update pattern
            pattern_id = hashlib.md5(
                f"{error_type}|{json.dumps(sorted(features.items()))}".encode()
            ).hexdigest()[:12]
            
            if pattern_id in self.patterns:
                # Update existing pattern
                pattern = self.patterns[pattern_id]
                pattern.frequency += 1
                pattern.last_seen = time.time()
                
                # Update indicators using EMA
                alpha = 0.2
                for feature, value in features.items():
                    if value > 0:
                        current = pattern.indicators.get(feature, 0.5)
                        pattern.indicators[feature] = alpha * value + (1 - alpha) * current
            else:
                # Create new pattern
                pattern = ErrorPattern(
                    pattern_id=pattern_id,
                    error_type=error_type,
                    indicators={k: v for k, v in features.items() if v > 0},
                    frequency=1,
                    last_seen=time.time()
                )
                self.patterns[pattern_id] = pattern
            
            # Update global feature weights
            for feature, value in features.items():
                if value > 0:
                    # Features that appear in errors are important
                    self.feature_weights[feature] = min(1.0, self.feature_weights[feature] + 0.05)
            
            # Periodically save
            if len(self.error_history) % 10 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, object]:
        """Get predictor statistics."""
        if not self.patterns:
            return {
                'patterns_learned': 0,
                'errors_seen': len(self.error_history)
            }
        
        error_type_counts = defaultdict(int)
        for pattern in self.patterns.values():
            error_type_counts[pattern.error_type] += pattern.frequency
        
        return {
            'patterns_learned': len(self.patterns),
            'errors_seen': len(self.error_history),
            'error_distribution': dict(error_type_counts),
            'top_features': sorted(
                self.feature_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Singleton
_predictor: Optional[IntelligentErrorPredictor] = None
_predictor_lock = asyncio.Lock()


async def get_error_predictor() -> IntelligentErrorPredictor:
    """Get singleton error predictor."""
    global _predictor
    if _predictor is None:
        async with _predictor_lock:
            if _predictor is None:
                _predictor = IntelligentErrorPredictor()
    return _predictor


async def predict_errors(code: str) -> List[ErrorPrediction]:
    """Predict potential errors in code."""
    predictor = await get_error_predictor()
    return await predictor.predict(code)


async def learn_from_error(code: str, error_type: str, error_message: str) -> None:
    """Learn from execution error."""
    predictor = await get_error_predictor()
    await predictor.learn_from_error(code, error_type, error_message)


__all__ = [
    "IntelligentErrorPredictor",
    "ErrorPrediction",
    "get_error_predictor",
    "predict_errors",
    "learn_from_error",
]
