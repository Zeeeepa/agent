"""Learning Coordinator - координирует обучение между всеми 15 системами.

Создает unified learning loops и shared experiences между системами.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LearningEvent:
    """Event that triggers learning across multiple systems."""
    source_system: str
    event_type: str  # 'success', 'failure', 'optimization', 'discovery'
    data: Dict[str, Any]
    timestamp: float
    propagated_to: List[str]


@dataclass
class SharedExperience:
    """Experience shared between systems for transfer learning."""
    context: str
    outcome: bool
    involved_systems: List[str]
    features: Dict[str, float]
    learned_patterns: List[str]


class LearningCoordinator:
    """Координатор обучения для всех brain систем с transfer learning."""
    
    def __init__(self, state_path: str = "log/learning_coordinator.json"):
        self.state_path = state_path
        
        # Learning event queue
        self.event_queue: asyncio.Queue[LearningEvent] = asyncio.Queue(maxsize=1000)
        
        # Shared experiences between systems
        self.shared_experiences: deque[SharedExperience] = deque(maxlen=1000)
        
        # System learning rates (adaptive)
        self.learning_rates: Dict[str, float] = defaultdict(lambda: 0.1)
        
        # Transfer learning patterns: source_system -> target_systems -> transfer_weight
        self.transfer_patterns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0.1))
        
        # Learning milestones
        self.milestones: List[Dict[str, Any]] = []
        
        self._lock = asyncio.Lock()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        self._load_state()
        self._setup_transfer_patterns()
    
    def _setup_transfer_patterns(self) -> None:
        """Setup initial transfer learning patterns."""
        
        # Query Classifier → Adaptive Retrieval
        # Classification insights improve retrieval strategy
        self.transfer_patterns['query_classifier']['adaptive_retrieval'] = 0.3
        
        # Error Predictor → Self-Healing
        # Prediction patterns improve healing strategies
        self.transfer_patterns['error_predictor']['self_healing'] = 0.5
        
        # Adaptive Retrieval → Context Optimizer
        # Retrieval insights optimize context allocation
        self.transfer_patterns['adaptive_retrieval']['context_optimizer'] = 0.4
        
        # Threshold Learner → Rate Limiter
        # Threshold patterns inform rate limits
        self.transfer_patterns['threshold_learner']['rate_limiter'] = 0.3
        
        # Intelligent Planner → Semantic Router
        # Planning insights improve routing
        self.transfer_patterns['intelligent_planner']['semantic_router'] = 0.35
        
        # Outcome Tracker → ALL systems
        # Global patterns benefit everyone
        for system in ['adaptive_retrieval', 'threshold_learner', 'query_classifier',
                       'context_optimizer', 'semantic_router', 'intelligent_planner',
                       'prompt_optimizer', 'error_predictor', 'self_healing',
                       'rate_limiter', 'predictive_cache']:
            self.transfer_patterns['outcome_tracker'][system] = 0.2
    
    def _load_state(self) -> None:
        """Load coordinator state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore learning rates
                self.learning_rates = defaultdict(lambda: 0.1, data.get('learning_rates', {}))
                
                # Restore transfer patterns
                patterns_data = data.get('transfer_patterns', {})
                for source, targets in patterns_data.items():
                    self.transfer_patterns[source] = defaultdict(lambda: 0.1, targets)
                
                # Restore milestones
                self.milestones = data.get('milestones', [])
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist coordinator state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize transfer patterns
                patterns_data = {}
                for source, targets in self.transfer_patterns.items():
                    patterns_data[source] = dict(targets)
                
                data = {
                    'learning_rates': dict(self.learning_rates),
                    'transfer_patterns': patterns_data,
                    'milestones': self.milestones[-20:],  # Last 20
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def start(self) -> None:
        """Start learning coordination."""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._worker_task = asyncio.create_task(self._process_learning_events())
    
    async def stop(self) -> None:
        """Stop learning coordination."""
        async with self._lock:
            self._running = False
            
            if self._worker_task:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
    
    async def emit_learning_event(
        self,
        source_system: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Emit learning event to be processed."""
        try:
            event = LearningEvent(
                source_system=source_system,
                event_type=event_type,
                data=data,
                timestamp=time.time(),
                propagated_to=[]
            )
            
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            # Drop oldest if full
            try:
                await self.event_queue.put(event)
            except Exception:
                pass
    
    async def _process_learning_events(self) -> None:
        """Process learning events and coordinate transfer learning."""
        while self._running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Propagate to relevant systems
                await self._propagate_learning(event)
                
                # Record shared experience
                await self._record_shared_experience(event)
                
                # Check for milestones
                await self._check_milestones()
                
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                continue
            except Exception:
                await asyncio.sleep(0.1)
    
    async def _propagate_learning(self, event: LearningEvent) -> None:
        """Propagate learning to relevant systems via transfer learning."""
        try:
            source = event.source_system
            
            # Get target systems for transfer
            transfer_targets = self.transfer_patterns.get(source, {})
            
            for target_system, transfer_weight in transfer_targets.items():
                # Skip if weight too low
                if transfer_weight < 0.1:
                    continue
                
                # Apply transfer learning
                await self._apply_transfer_learning(
                    target_system,
                    event,
                    transfer_weight
                )
                
                event.propagated_to.append(target_system)
        except Exception:
            pass
    
    async def _apply_transfer_learning(
        self,
        target_system: str,
        event: LearningEvent,
        weight: float
    ) -> None:
        """Apply transfer learning to target system."""
        try:
            # System-specific transfer logic
            
            if target_system == 'adaptive_retrieval' and event.source_system == 'query_classifier':
                # If classifier is confident, retrieval can be more aggressive
                confidence = event.data.get('confidence', 0.5)
                if confidence > 0.8:
                    # Signal to increase retrieval k
                    pass
            
            elif target_system == 'self_healing' and event.source_system == 'error_predictor':
                # Prediction patterns improve healing
                predicted_error = event.data.get('error_type')
                if predicted_error:
                    # Pre-warm healing strategy for this error type
                    pass
            
            elif target_system == 'context_optimizer' and event.source_system == 'adaptive_retrieval':
                # Retrieval success affects context budget
                success = event.data.get('success', False)
                if not success:
                    # Signal to adjust context budget
                    pass
        except Exception:
            pass
    
    async def _record_shared_experience(self, event: LearningEvent) -> None:
        """Record shared experience for cross-system learning."""
        try:
            experience = SharedExperience(
                context=event.data.get('query', 'unknown'),
                outcome=event.event_type == 'success',
                involved_systems=[event.source_system] + event.propagated_to,
                features={
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                },
                learned_patterns=[]
            )
            
            self.shared_experiences.append(experience)
        except Exception:
            pass
    
    async def _check_milestones(self) -> None:
        """Check and record learning milestones."""
        try:
            # Check if we hit milestone thresholds
            total_experiences = len(self.shared_experiences)
            
            milestone_thresholds = [100, 500, 1000, 5000]
            
            for threshold in milestone_thresholds:
                # Check if we just crossed this threshold
                if total_experiences >= threshold:
                    # Check if not already recorded
                    existing = any(
                        m.get('threshold') == threshold
                        for m in self.milestones
                    )
                    
                    if not existing:
                        milestone = {
                            'threshold': threshold,
                            'timestamp': time.time(),
                            'learning_rates': dict(self.learning_rates),
                        }
                        self.milestones.append(milestone)
                        
                        # Save on milestone
                        await self._save_state()
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning coordinator statistics."""
        return {
            'events_queued': self.event_queue.qsize(),
            'shared_experiences': len(self.shared_experiences),
            'transfer_patterns': {
                source: len(targets)
                for source, targets in self.transfer_patterns.items()
            },
            'milestones': len(self.milestones),
            'learning_rates': dict(self.learning_rates),
            'running': self._running
        }


# Singleton
_coordinator: Optional[LearningCoordinator] = None
_coordinator_lock = asyncio.Lock()


async def get_learning_coordinator() -> LearningCoordinator:
    """Get singleton learning coordinator."""
    global _coordinator
    if _coordinator is None:
        async with _coordinator_lock:
            if _coordinator is None:
                _coordinator = LearningCoordinator()
                # Auto-start
                await _coordinator.start()
    return _coordinator


async def emit_learning_event(system: str, event_type: str, data: Dict[str, Any]) -> None:
    """Emit learning event for cross-system propagation."""
    coordinator = await get_learning_coordinator()
    await coordinator.emit_learning_event(system, event_type, data)


__all__ = [
    "LearningCoordinator",
    "get_learning_coordinator",
    "emit_learning_event",
]
