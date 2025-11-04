"""Brain Orchestrator - координирует все 13 ML-систем автоматически.

Создает feedback loops и cross-system optimization для максимальной автоматизации.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SystemState:
    """State snapshot of a brain system."""
    system_name: str
    performance: float  # 0-1
    load: float  # 0-1
    last_update: float
    metrics: Dict[str, Any]


@dataclass
class CrossSystemInsight:
    """Insight learned from multiple systems working together."""
    systems_involved: List[str]
    pattern: str
    effectiveness: float
    uses: int


class BrainOrchestrator:
    """Meta-learning orchestrator that coordinates all brain systems automatically."""
    
    def __init__(self, state_path: str = "log/brain_orchestrator.json"):
        self.state_path = state_path
        
        # System registry
        self.systems: Dict[str, Any] = {}
        self.system_states: Dict[str, SystemState] = {}
        
        # Cross-system patterns learned
        self.cross_patterns: List[CrossSystemInsight] = []
        
        # Automatic triggers: condition -> action
        self.triggers: Dict[str, List[Tuple[callable, callable]]] = defaultdict(list)
        
        # System dependencies and feedback loops
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.feedback_loops: List[Tuple[str, str, callable]] = []
        
        # Performance history for meta-optimization
        self.performance_history: deque[Dict[str, float]] = deque(maxlen=100)
        
        # Optimization strategies learned
        self.optimization_strategies: Dict[str, float] = {}
        
        self._lock = asyncio.Lock()
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        self._setup_automatic_connections()
        self._load_state()
    
    def _setup_automatic_connections(self) -> None:
        """Setup automatic connections between all systems."""
        
        # Query Classifier -> Adaptive Retrieval
        # When intent changes, adjust retrieval strategy
        self.dependencies['query_classifier'].add('adaptive_retrieval')
        
        # Adaptive Retrieval -> Context Optimizer
        # Retrieval results influence context budget
        self.dependencies['adaptive_retrieval'].add('context_optimizer')
        
        # Error Predictor -> Self-Healing
        # Predictions trigger healing preemptively
        self.dependencies['error_predictor'].add('self_healing')
        
        # Outcome Tracker -> ALL systems
        # All systems report to outcome tracker
        for system in ['adaptive_retrieval', 'threshold_learner', 'query_classifier',
                       'context_optimizer', 'semantic_router', 'intelligent_planner',
                       'prompt_optimizer', 'error_predictor', 'self_healing',
                       'rate_limiter', 'predictive_cache']:
            self.dependencies[system].add('outcome_tracker')
        
        # Semantic Router -> Intelligent Planner
        # Routing decisions influence planning
        self.dependencies['semantic_router'].add('intelligent_planner')
        
        # Threshold Learner -> Rate Limiter
        # Learned thresholds affect rate limiting
        self.dependencies['threshold_learner'].add('rate_limiter')
        
        # Predictive Cache -> Adaptive Retrieval
        # Cache hits reduce retrieval load
        self.dependencies['predictive_cache'].add('adaptive_retrieval')
    
    def _load_state(self) -> None:
        """Load orchestrator state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore optimization strategies
                self.optimization_strategies = data.get('strategies', {})
                
                # Restore cross patterns
                for pattern_data in data.get('cross_patterns', []):
                    self.cross_patterns.append(CrossSystemInsight(
                        systems_involved=pattern_data['systems'],
                        pattern=pattern_data['pattern'],
                        effectiveness=pattern_data['effectiveness'],
                        uses=pattern_data['uses']
                    ))
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist orchestrator state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                data = {
                    'strategies': self.optimization_strategies,
                    'cross_patterns': [
                        {
                            'systems': p.systems_involved,
                            'pattern': p.pattern,
                            'effectiveness': p.effectiveness,
                            'uses': p.uses
                        }
                        for p in self.cross_patterns
                    ],
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def start(self) -> None:
        """Start automatic orchestration."""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            
            # Start background tasks
            self._background_tasks = [
                asyncio.create_task(self._monitor_systems()),
                asyncio.create_task(self._auto_optimize()),
                asyncio.create_task(self._detect_patterns()),
                asyncio.create_task(self._execute_feedback_loops()),
            ]
    
    async def stop(self) -> None:
        """Stop orchestration."""
        async with self._lock:
            self._running = False
            
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self._background_tasks.clear()
    
    async def _monitor_systems(self) -> None:
        """Continuously monitor all systems."""
        while self._running:
            try:
                # Collect metrics from all systems (with yield points)
                for i, system_name in enumerate(self.dependencies.keys()):
                    await self._update_system_state(system_name)
                    
                    # Yield every 3 systems to prevent blocking
                    if i % 3 == 0:
                        await asyncio.sleep(0)
                
                # Record performance snapshot
                snapshot = {
                    name: state.performance
                    for name, state in self.system_states.items()
                }
                self.performance_history.append(snapshot)
                
                await asyncio.sleep(5.0)  # Every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)
    
    async def _update_system_state(self, system_name: str) -> None:
        """Update state for a specific system."""
        try:
            # Import and get stats dynamically
            if system_name == 'adaptive_retrieval':
                from jinx.micro.brain import get_adaptive_manager
                mgr = await get_adaptive_manager()
                metrics = {'arms': len(mgr.k_arms)}
                performance = 0.7  # Placeholder
            
            elif system_name == 'outcome_tracker':
                from jinx.micro.brain import get_outcome_tracker
                tracker = await get_outcome_tracker()
                metrics = tracker.get_all_metrics()
                
                # Compute average performance
                perf_sum = sum(
                    m.get('success_rate', 0.5)
                    for m in metrics.values()
                )
                performance = perf_sum / max(1, len(metrics))
            
            elif system_name == 'query_classifier':
                from jinx.micro.brain import get_query_classifier
                classifier = await get_query_classifier()
                metrics = classifier.get_stats()
                performance = 0.75  # Placeholder
            
            else:
                metrics = {}
                performance = 0.5
            
            self.system_states[system_name] = SystemState(
                system_name=system_name,
                performance=performance,
                load=0.5,  # Can be computed from metrics
                last_update=time.time(),
                metrics=metrics
            )
        except Exception:
            pass
    
    async def _auto_optimize(self) -> None:
        """Automatically optimize system parameters."""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                continue
            
            try:
                
                # Analyze recent performance
                if len(self.performance_history) < 10:
                    continue
                
                # Detect degradation
                recent = list(self.performance_history)[-10:]
                older = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else []
                
                if older:
                    for system_name in recent[0].keys():
                        recent_avg = sum(s.get(system_name, 0.5) for s in recent) / len(recent)
                        older_avg = sum(s.get(system_name, 0.5) for s in older) / len(older)
                        
                        # If performance dropped significantly
                        if recent_avg < older_avg - 0.15:
                            await self._trigger_optimization(system_name)
            except Exception:
                pass
    
    async def _trigger_optimization(self, system_name: str) -> None:
        """Trigger optimization for a degrading system."""
        try:
            # System-specific optimizations
            if system_name == 'adaptive_retrieval':
                # Increase exploration rate temporarily
                from jinx.micro.brain import get_adaptive_manager
                mgr = await get_adaptive_manager()
                # Reset some arms to explore new strategies
                # (Would need to add this method to the manager)
            
            elif system_name == 'rate_limiter':
                # Reset adaptive limits
                from jinx.micro.brain import get_rate_limiter
                limiter = await get_rate_limiter()
                # Could trigger a reset or adjustment
            
            # Record strategy
            self.optimization_strategies[f"{system_name}_reset"] = time.time()
        except Exception:
            pass
    
    async def _detect_patterns(self) -> None:
        """Detect cross-system patterns automatically."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Every minute
            except asyncio.CancelledError:
                break
            except Exception:
                continue
            
            try:
                
                # Analyze correlations between systems
                if len(self.performance_history) < 20:
                    continue
                
                recent = list(self.performance_history)[-20:]
                
                # Look for correlations
                systems = list(recent[0].keys())
                
                for i, sys1 in enumerate(systems):
                    for sys2 in systems[i+1:]:
                        correlation = self._compute_correlation(recent, sys1, sys2)
                        
                        if abs(correlation) > 0.7:  # Strong correlation
                            pattern = f"{sys1}_affects_{sys2}" if correlation > 0 else f"{sys1}_inverse_{sys2}"
                            
                            # Check if pattern already exists
                            existing = next((p for p in self.cross_patterns if p.pattern == pattern), None)
                            
                            if existing:
                                existing.uses += 1
                                existing.effectiveness = (existing.effectiveness * 0.9 + abs(correlation) * 0.1)
                            else:
                                self.cross_patterns.append(CrossSystemInsight(
                                    systems_involved=[sys1, sys2],
                                    pattern=pattern,
                                    effectiveness=abs(correlation),
                                    uses=1
                                ))
            except Exception:
                pass
    
    def _compute_correlation(self, data: List[Dict[str, float]], sys1: str, sys2: str) -> float:
        """Compute correlation between two systems."""
        try:
            values1 = [d.get(sys1, 0.5) for d in data]
            values2 = [d.get(sys2, 0.5) for d in data]
            
            if len(values1) != len(values2):
                return 0.0
            
            # Simple Pearson correlation
            n = len(values1)
            sum1 = sum(values1)
            sum2 = sum(values2)
            sum1sq = sum(x * x for x in values1)
            sum2sq = sum(x * x for x in values2)
            psum = sum(x * y for x, y in zip(values1, values2))
            
            num = psum - (sum1 * sum2 / n)
            den = ((sum1sq - sum1**2 / n) * (sum2sq - sum2**2 / n)) ** 0.5
            
            if den == 0:
                return 0.0
            
            return num / den
        except Exception:
            return 0.0
    
    async def _execute_feedback_loops(self) -> None:
        """Execute automatic feedback loops between systems."""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                continue
            
            try:
                
                # Example: Query Classifier performance affects Retrieval exploration
                try:
                    from jinx.micro.brain import get_query_classifier, get_adaptive_manager
                    
                    classifier = await get_query_classifier()
                    stats = classifier.get_stats()
                    
                    # If classifier has low confidence, increase retrieval exploration
                    if stats.get('total_queries', 0) > 50:
                        # Could adjust retrieval parameters
                        pass
                except Exception:
                    pass
                
                # Example: Cache hit rate affects retrieval aggressiveness
                try:
                    from jinx.micro.brain import get_predictive_cache
                    
                    cache = await get_predictive_cache()
                    cache_stats = cache.get_stats()
                    
                    hit_rate = cache_stats.get('hit_rate', 0.5)
                    
                    # If high cache hit rate, can reduce retrieval frequency
                    if hit_rate > 0.8:
                        # Signal to adaptive retrieval to be more conservative
                        pass
                except Exception:
                    pass
            except Exception:
                pass
    
    async def process_query_pipeline(self, query: str) -> Dict[str, Any]:
        """Process query through full orchestrated pipeline with COGNITIVE THINKING."""
        t0 = time.time()
        results = {}
        
        # MEMORY + THINKING: Search memories and think with context
        try:
            from jinx.micro.brain import think_with_memory, remember, remember_episode
            
            # Search all memory systems for relevant context
            enhanced_thinking = await think_with_memory(query)
            
            if enhanced_thinking:
                results['thought'] = enhanced_thinking.get('thought', '')
                results['reasoning'] = enhanced_thinking.get('reasoning', '')
                results['memories_used'] = enhanced_thinking.get('memories_used', 0)
            
            # Store in working memory
            await remember('current_query', query, priority=10)
            await remember('last_thought', results.get('thought', ''), priority=8)
        except Exception:
            pass
        
        try:
            # 1. Route semantically
            from jinx.micro.brain import route_query, Decision, make_ensemble_decision
            route = await route_query(query)
            results['route'] = route.route_name
            results['route_confidence'] = route.confidence
            
            # 2. Classify intent
            from jinx.micro.brain import classify_query
            intent = await classify_query(query)
            results['intent'] = intent.intent
            results['intent_confidence'] = intent.confidence
            
            # 3. Get adaptive retrieval params from multiple systems
            from jinx.micro.brain import select_retrieval_params
            k, timeout = await select_retrieval_params(query)
            results['k'] = k
            results['timeout'] = timeout
            
            # 4. Allocate context budget
            from jinx.micro.brain import allocate_context_budget
            allocation = await allocate_context_budget(query)
            results['context_budget'] = allocation.total_budget
            
            # 5. Create intelligent plan
            from jinx.micro.brain import create_intelligent_plan
            plan = await create_intelligent_plan(query)
            results['plan_steps'] = len(plan.steps)
            results['plan_strategy'] = plan.strategy
            
            # 6. Use ensemble decision for final strategy
            # Collect decisions from multiple systems
            decisions = [
                Decision('semantic_router', route.route_name, route.confidence, weight=1.2),
                Decision('query_classifier', intent.intent, intent.confidence, weight=1.0),
                Decision('intelligent_planner', plan.strategy, plan.confidence, weight=0.9)
            ]
            
            ensemble = await make_ensemble_decision(decisions, {'query': query})
            results['ensemble_decision'] = ensemble.value
            results['ensemble_confidence'] = ensemble.confidence
            results['ensemble_agreement'] = ensemble.agreement_score
            
            # 7. Add experience to knowledge graph
            from jinx.micro.brain import add_to_knowledge_graph
            await add_to_knowledge_graph(
                'orchestrator',
                'process_query',
                True,
                {
                    'route': route.route_name,
                    'intent': intent.intent,
                    'plan': plan.strategy,
                    'involved_systems': ['router', 'classifier', 'planner']
                }
            )
            
            # 8. Check cache
            from jinx.micro.brain import get_predictive_cache
            cache = await get_predictive_cache()
            # Could check if similar query was cached
            
            elapsed = time.time() - t0
            results['total_time_ms'] = elapsed * 1000
            
            # REASONING: System reasons about the results
            try:
                from jinx.micro.brain import reason, self_reflect, remember_episode
                
                # Collect evidence
                evidence = [
                    f"Route: {results.get('route')}",
                    f"Intent: {results.get('intent')}",
                    f"Ensemble confidence: {results.get('ensemble_confidence', 0)}",
                    f"Time: {elapsed * 1000:.1f}ms"
                ]
                
                # Reason about query processing
                conclusion, conf = await reason("query processing", evidence)
                results['reasoning'] = conclusion
                results['reasoning_confidence'] = conf
                
                # Store successful processing as episode in long-term memory
                success_score = results.get('ensemble_confidence', 0.5)
                if success_score > 0.7:
                    await remember_episode(
                        content=f"Processed query: {query[:100]}. Route: {results.get('route')}, Intent: {results.get('intent')}",
                        episode_type='experience',
                        context={
                            'query': query,
                            'route': results.get('route'),
                            'intent': results.get('intent'),
                            'time_ms': elapsed * 1000
                        },
                        importance=success_score
                    )
                
                # Periodically reflect on self and consolidate memories
                if int(time.time()) % 60 == 0:  # Every minute
                    reflection = await self_reflect()
                    results['self_reflection'] = reflection.performance_assessment
                    
                    # Consolidate memories
                    from jinx.micro.brain import consolidate_all_memories
                    consolidation_stats = await consolidate_all_memories()
                    results['memory_consolidation'] = consolidation_stats
            except Exception:
                pass
            
            # Record metrics
            from jinx.micro.brain import record_system_metrics
            await record_system_metrics('orchestrator', elapsed * 1000, True)
            
            # Record cross-system execution
            await self._record_pipeline_execution(query, results, True, elapsed)
            
        except Exception as e:
            results['error'] = str(e)
            elapsed = time.time() - t0
            
            # Record failure metrics
            from jinx.micro.brain import record_system_metrics
            await record_system_metrics('orchestrator', elapsed * 1000, False)
            
            await self._record_pipeline_execution(query, results, False, elapsed)
        
        return results
    
    async def _record_pipeline_execution(
        self,
        query: str,
        results: Dict[str, Any],
        success: bool,
        elapsed: float
    ) -> None:
        """Record full pipeline execution for meta-learning."""
        try:
            # Store pattern for future optimization
            pattern_key = f"{results.get('route', 'unknown')}_{results.get('intent', 'unknown')}"
            
            if pattern_key not in self.optimization_strategies:
                self.optimization_strategies[pattern_key] = 0.0
            
            # Update strategy effectiveness (EMA)
            alpha = 0.1
            reward = 1.0 if success else 0.0
            
            # Time penalty
            expected_time = 1.0  # 1 second baseline
            if elapsed > expected_time:
                reward -= min(0.5, (elapsed - expected_time) / expected_time)
            
            current = self.optimization_strategies[pattern_key]
            self.optimization_strategies[pattern_key] = alpha * reward + (1 - alpha) * current
            
            # Periodically save
            if len(self.optimization_strategies) % 10 == 0:
                await self._save_state()
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'systems_monitored': len(self.system_states),
            'cross_patterns_learned': len(self.cross_patterns),
            'optimization_strategies': len(self.optimization_strategies),
            'dependencies': {k: list(v) for k, v in self.dependencies.items()},
            'running': self._running,
            'performance_history_size': len(self.performance_history)
        }


# Singleton
_orchestrator: Optional[BrainOrchestrator] = None
_orchestrator_lock = asyncio.Lock()


async def get_brain_orchestrator() -> BrainOrchestrator:
    """Get singleton brain orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        async with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = BrainOrchestrator()
                # Auto-start
                await _orchestrator.start()
    return _orchestrator


async def process_with_full_intelligence(query: str) -> Dict[str, Any]:
    """Process query through all 13 brain systems automatically."""
    orchestrator = await get_brain_orchestrator()
    return await orchestrator.process_query_pipeline(query)


__all__ = [
    "BrainOrchestrator",
    "get_brain_orchestrator",
    "process_with_full_intelligence",
]
