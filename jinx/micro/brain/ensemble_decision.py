"""Ensemble Decision Maker - объединяет решения от множества систем.

Использует voting, weighting и confidence-based fusion для best decisions.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Decision:
    """Decision from a single system."""
    system: str
    value: Any
    confidence: float
    reasoning: Optional[str] = None
    weight: float = 1.0


@dataclass
class EnsembleDecision:
    """Final ensemble decision."""
    value: Any
    confidence: float
    contributing_systems: List[str]
    agreement_score: float  # How much systems agreed (0-1)
    method: str  # 'voting', 'weighted', 'confidence_based'


class EnsembleDecisionMaker:
    """Объединяет решения от множества ML-систем для лучших результатов."""
    
    def __init__(self, state_path: str = "log/ensemble_decision.json"):
        self.state_path = state_path
        
        # System reliability scores (learned)
        self.reliability: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Decision history for learning
        self.history: deque[Tuple[str, EnsembleDecision, bool]] = deque(maxlen=500)
        
        # Confidence calibration (learned per system)
        self.calibration: Dict[str, Tuple[float, float]] = {}  # (scale, offset)
        
        # Ensemble strategies performance
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'uses': 0,
            'successes': 0,
            'avg_confidence': 0.5
        })
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load ensemble state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore reliability scores
                self.reliability = defaultdict(lambda: 1.0, data.get('reliability', {}))
                
                # Restore calibration
                for system, calib in data.get('calibration', {}).items():
                    self.calibration[system] = tuple(calib)
                
                # Restore strategy performance
                for strategy, perf in data.get('strategies', {}).items():
                    self.strategy_performance[strategy] = perf
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist ensemble state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize calibration
                calibration_data = {k: list(v) for k, v in self.calibration.items()}
                
                data = {
                    'reliability': dict(self.reliability),
                    'calibration': calibration_data,
                    'strategies': dict(self.strategy_performance),
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _calibrate_confidence(self, system: str, raw_confidence: float) -> float:
        """Calibrate confidence based on learned parameters."""
        if system not in self.calibration:
            return raw_confidence
        
        scale, offset = self.calibration[system]
        calibrated = raw_confidence * scale + offset
        
        return max(0.0, min(1.0, calibrated))
    
    async def make_decision(
        self,
        decisions: List[Decision],
        context: Optional[Dict[str, Any]] = None
    ) -> EnsembleDecision:
        """Make ensemble decision from multiple system decisions."""
        async with self._lock:
            if not decisions:
                return EnsembleDecision(
                    value=None,
                    confidence=0.0,
                    contributing_systems=[],
                    agreement_score=0.0,
                    method='none'
                )
            
            # Calibrate confidences
            for decision in decisions:
                decision.confidence = self._calibrate_confidence(
                    decision.system,
                    decision.confidence
                )
                
                # Apply reliability weight
                decision.weight = self.reliability[decision.system]
            
            # Select best ensemble strategy
            strategy = await self._select_strategy(decisions, context)
            
            # Apply strategy
            if strategy == 'voting':
                result = await self._voting_decision(decisions)
            elif strategy == 'weighted':
                result = await self._weighted_decision(decisions)
            elif strategy == 'confidence_based':
                result = await self._confidence_based_decision(decisions)
            else:
                result = await self._hybrid_decision(decisions)
            
            result.method = strategy
            
            return result
    
    async def _select_strategy(
        self,
        decisions: List[Decision],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Select best ensemble strategy based on context."""
        # If high agreement among systems, use voting
        values = [d.value for d in decisions]
        unique_values = len(set(str(v) for v in values))
        
        if unique_values <= 2:
            return 'voting'
        
        # If high confidence variation, use confidence-based
        confidences = [d.confidence for d in decisions]
        conf_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        
        if conf_variance > 0.1:
            return 'confidence_based'
        
        # If reliability scores vary, use weighted
        weights = [d.weight for d in decisions]
        weight_variance = sum((w - sum(weights)/len(weights))**2 for w in weights) / len(weights)
        
        if weight_variance > 0.1:
            return 'weighted'
        
        # Default: hybrid
        return 'hybrid'
    
    async def _voting_decision(self, decisions: List[Decision]) -> EnsembleDecision:
        """Simple majority voting."""
        # Count votes for each value
        votes: Dict[str, List[Decision]] = defaultdict(list)
        
        for d in decisions:
            key = str(d.value)
            votes[key].append(d)
        
        # Find majority
        if not votes:
            return EnsembleDecision(None, 0.0, [], 0.0, 'voting')
        
        winner = max(votes.items(), key=lambda x: len(x[1]))
        winner_key, winner_decisions = winner
        
        # Compute confidence as fraction of votes
        confidence = len(winner_decisions) / len(decisions)
        
        # Agreement score
        agreement = confidence
        
        # Get actual value from first winner decision
        value = winner_decisions[0].value
        
        return EnsembleDecision(
            value=value,
            confidence=confidence,
            contributing_systems=[d.system for d in winner_decisions],
            agreement_score=agreement,
            method='voting'
        )
    
    async def _weighted_decision(self, decisions: List[Decision]) -> EnsembleDecision:
        """Weighted voting by system reliability."""
        # Count weighted votes
        weighted_votes: Dict[str, Tuple[float, List[Decision]]] = defaultdict(lambda: (0.0, []))
        
        for d in decisions:
            key = str(d.value)
            current_weight, current_decisions = weighted_votes[key]
            weighted_votes[key] = (
                current_weight + d.weight,
                current_decisions + [d]
            )
        
        if not weighted_votes:
            return EnsembleDecision(None, 0.0, [], 0.0, 'weighted')
        
        # Find winner by weight
        winner = max(weighted_votes.items(), key=lambda x: x[1][0])
        winner_key, (winner_weight, winner_decisions) = winner
        
        # Compute confidence from weights
        total_weight = sum(d.weight for d in decisions)
        confidence = winner_weight / max(0.1, total_weight)
        
        # Agreement score
        agreement = len(winner_decisions) / len(decisions)
        
        value = winner_decisions[0].value
        
        return EnsembleDecision(
            value=value,
            confidence=min(1.0, confidence),
            contributing_systems=[d.system for d in winner_decisions],
            agreement_score=agreement,
            method='weighted'
        )
    
    async def _confidence_based_decision(self, decisions: List[Decision]) -> EnsembleDecision:
        """Select based on highest confidence."""
        if not decisions:
            return EnsembleDecision(None, 0.0, [], 0.0, 'confidence')
        
        # Sort by confidence
        sorted_decisions = sorted(decisions, key=lambda d: d.confidence, reverse=True)
        
        # Take highest confidence
        best = sorted_decisions[0]
        
        # Check for agreement at high confidence
        high_conf_decisions = [d for d in decisions if d.confidence > 0.7]
        if high_conf_decisions:
            # Count agreements
            values = [str(d.value) for d in high_conf_decisions]
            most_common = max(set(values), key=values.count)
            agreeing = [d for d in high_conf_decisions if str(d.value) == most_common]
            
            if len(agreeing) >= len(high_conf_decisions) * 0.6:
                # Strong agreement - use it
                avg_confidence = sum(d.confidence for d in agreeing) / len(agreeing)
                
                return EnsembleDecision(
                    value=agreeing[0].value,
                    confidence=avg_confidence,
                    contributing_systems=[d.system for d in agreeing],
                    agreement_score=len(agreeing) / len(decisions),
                    method='confidence_based'
                )
        
        # No strong agreement - use highest confidence
        return EnsembleDecision(
            value=best.value,
            confidence=best.confidence,
            contributing_systems=[best.system],
            agreement_score=1.0 / len(decisions),
            method='confidence_based'
        )
    
    async def _hybrid_decision(self, decisions: List[Decision]) -> EnsembleDecision:
        """Hybrid approach combining multiple strategies."""
        # Compute all strategies
        voting_result = await self._voting_decision(decisions)
        weighted_result = await self._weighted_decision(decisions)
        confidence_result = await self._confidence_based_decision(decisions)
        
        # Score each result
        results = [
            (voting_result, voting_result.confidence * voting_result.agreement_score),
            (weighted_result, weighted_result.confidence * weighted_result.agreement_score * 1.1),
            (confidence_result, confidence_result.confidence * 0.9)
        ]
        
        # Select best
        best_result, _ = max(results, key=lambda x: x[1])
        best_result.method = 'hybrid'
        
        return best_result
    
    async def record_outcome(
        self,
        decision: EnsembleDecision,
        success: bool,
        actual_value: Optional[Any] = None
    ) -> None:
        """Record decision outcome for learning."""
        async with self._lock:
            # Record history
            self.history.append((str(decision.value), decision, success))
            
            # Update strategy performance
            strategy = decision.method
            perf = self.strategy_performance[strategy]
            perf['uses'] += 1
            if success:
                perf['successes'] += 1
            
            # Update average confidence (EMA)
            alpha = 0.1
            perf['avg_confidence'] = (
                alpha * decision.confidence +
                (1 - alpha) * perf['avg_confidence']
            )
            
            # Update system reliability
            for system in decision.contributing_systems:
                current = self.reliability[system]
                
                if success:
                    # Increase reliability
                    self.reliability[system] = min(2.0, current * 1.05)
                else:
                    # Decrease reliability
                    self.reliability[system] = max(0.5, current * 0.95)
            
            # Update confidence calibration
            if actual_value is not None:
                correct = (str(decision.value) == str(actual_value))
                
                for system in decision.contributing_systems:
                    if system not in self.calibration:
                        self.calibration[system] = (1.0, 0.0)
                    
                    scale, offset = self.calibration[system]
                    
                    # Simple calibration adjustment
                    if correct and decision.confidence < 0.9:
                        scale *= 1.02  # Boost confidence
                    elif not correct and decision.confidence > 0.5:
                        scale *= 0.98  # Reduce confidence
                    
                    self.calibration[system] = (
                        max(0.5, min(1.5, scale)),
                        offset
                    )
            
            # Periodically save
            if len(self.history) % 20 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            'decisions_made': len(self.history),
            'reliability_scores': dict(self.reliability),
            'strategy_performance': {
                k: {
                    'success_rate': v['successes'] / max(1, v['uses']),
                    'uses': v['uses']
                }
                for k, v in self.strategy_performance.items()
                if v['uses'] > 0
            },
            'calibrated_systems': len(self.calibration)
        }


# Singleton
_ensemble: Optional[EnsembleDecisionMaker] = None
_ensemble_lock = asyncio.Lock()


async def get_ensemble_decision_maker() -> EnsembleDecisionMaker:
    """Get singleton ensemble decision maker."""
    global _ensemble
    if _ensemble is None:
        async with _ensemble_lock:
            if _ensemble is None:
                _ensemble = EnsembleDecisionMaker()
    return _ensemble


async def make_ensemble_decision(
    decisions: List[Decision],
    context: Optional[Dict[str, Any]] = None
) -> EnsembleDecision:
    """Make ensemble decision from multiple systems."""
    ensemble = await get_ensemble_decision_maker()
    return await ensemble.make_decision(decisions, context)


async def record_ensemble_outcome(decision: EnsembleDecision, success: bool, actual: Optional[Any] = None) -> None:
    """Record ensemble decision outcome."""
    ensemble = await get_ensemble_decision_maker()
    await ensemble.record_outcome(decision, success, actual)


__all__ = [
    "Decision",
    "EnsembleDecision",
    "EnsembleDecisionMaker",
    "get_ensemble_decision_maker",
    "make_ensemble_decision",
    "record_ensemble_outcome",
]
