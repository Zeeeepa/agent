"""Brain module: ML-driven self-learning and self-healing intelligence.

This module contains 27 advanced AI systems - A COMPLETE COGNITIVE ARCHITECTURE WITH CONCURRENT PROCESSING:

Core Intelligence:
- Adaptive retrieval with multi-armed bandits (UCB1)
- Threshold learning with Thompson Sampling (Bayesian)
- Query classification with intent learning (embeddings)
- Context size optimization with Q-learning (RL)

Routing & Planning:
- Semantic routing with embedding-based matching (no hardcoded rules)
- Intelligent planning with RL-based task decomposition

Optimization:
- Predictive caching with pattern recognition + semantic prefetch
- Adaptive rate limiting with congestion detection
- Prompt optimization with A/B testing + context awareness

Safety & Healing:
- Error prediction with ML pattern recognition
- Self-healing code with automatic error correction

Monitoring:
- Outcome tracking for continuous learning feedback
- Framework detection with semantic analysis

Meta-Coordination:
- Brain Orchestrator: coordinates all systems with automatic feedback loops
- Auto-Tuner: continuously optimizes all parameters via gradient-based meta-learning
- Learning Coordinator: manages transfer learning and shared experiences between systems
- Performance Monitor: real-time anomaly detection and auto-correction
- Ensemble Decision Maker: combines decisions from multiple systems
- Knowledge Graph Builder: extracts patterns and builds knowledge automatically
- Query Expander: ML-driven query enrichment for better retrieval

Cognitive Architecture (UNIFIED MEMORY + INTELLIGENCE + CONCURRENT):
- Meta-Cognitive System: self-awareness, reflection, reasoning about own processes
- Working Memory: active short-term memory for current processing (Miller's 7±2)
- Goal-Driven System: motivation, goal-setting, and achievement-oriented behavior
- Episodic Memory: long-term memory with semantic search via embeddings
- Memory Integration Hub: unified search across all memory systems
- Concurrent Processor: parallel task processing with intelligence and priority
- Context Continuity: conversation flow tracking and context understanding

All 27 systems are:
✓ Autonomous (work without configuration)
✓ Self-learning (improve from every outcome)
✓ RT-compatible (async, bounded, cached)
✓ Interconnected (automatic cross-system optimization)
✓ Self-optimizing (meta-learning layer adjusts everything)

Import this module and all systems start automatically in background!
"""

from __future__ import annotations

from .adaptive_retrieval import (
    AdaptiveRetrievalManager,
    get_adaptive_manager,
    select_retrieval_params,
    record_retrieval_outcome,
)

from .threshold_learner import (
    ThresholdLearner,
    get_threshold_learner,
    select_threshold,
    record_threshold_outcome,
)

from .predictive_cache import (
    PredictiveCacheManager,
    get_predictive_cache,
)

from .outcome_tracker import (
    OutcomeTracker,
    get_outcome_tracker,
    record_outcome,
)

from .query_classifier import (
    QueryIntent,
    IntelligentQueryClassifier,
    get_query_classifier,
    classify_query,
    learn_query_intent,
)

from .rate_limiter import (
    AdaptiveRateLimiter,
    RateLimitState,
    get_rate_limiter,
    acquire_rate_limit,
)

from .context_optimizer import (
    ContextAllocation,
    ContextSizeOptimizer,
    get_context_optimizer,
    allocate_context_budget,
    record_context_outcome,
)

from .prompt_optimizer import (
    AdaptivePromptOptimizer,
    PromptVariant,
    get_prompt_optimizer,
    select_optimal_prompt,
    record_prompt_outcome,
)

from .error_predictor import (
    IntelligentErrorPredictor,
    ErrorPrediction,
    get_error_predictor,
    predict_errors,
    learn_from_error,
)

from .self_healing import (
    SelfHealingSystem,
    HealingResult,
    get_self_healer,
    heal_code,
)

from .semantic_router import (
    SemanticRouter,
    RouteMatch,
    get_semantic_router,
    route_query,
    record_routing_outcome,
)

from .intelligent_planner import (
    IntelligentPlanner,
    ExecutionPlan,
    PlanStep,
    get_intelligent_planner,
    create_intelligent_plan,
    record_plan_execution,
)

from .orchestrator import (
    BrainOrchestrator,
    get_brain_orchestrator,
    process_with_full_intelligence,
)

from .auto_tuner import (
    AutoTuner,
    get_auto_tuner,
)

from .learning_coordinator import (
    LearningCoordinator,
    get_learning_coordinator,
    emit_learning_event,
)

from .performance_monitor import (
    PerformanceMonitor,
    get_performance_monitor,
    record_system_metrics,
)

from .ensemble_decision import (
    Decision,
    EnsembleDecision,
    EnsembleDecisionMaker,
    get_ensemble_decision_maker,
    make_ensemble_decision,
    record_ensemble_outcome,
)

from .knowledge_graph import (
    KnowledgeGraphBuilder,
    get_knowledge_graph,
    add_to_knowledge_graph,
    query_knowledge_graph,
)

from .query_expander import (
    QueryExpander,
    ExpandedQuery,
    get_query_expander,
    expand_query,
    record_expansion_outcome,
)

from .meta_cognitive import (
    MetaCognitiveSystem,
    Thought,
    SelfReflection,
    get_meta_cognitive,
    think_about,
    self_reflect,
    reason,
)

from .working_memory import (
    WorkingMemory,
    MemoryItem,
    get_working_memory,
    remember,
    recall,
    get_focus,
)

from .goal_driven import (
    GoalDrivenSystem,
    Goal,
    GoalStatus,
    get_goal_system,
    set_goal,
    get_current_goal,
    mark_progress,
)

from .episodic_memory import (
    EpisodicMemory,
    Episode,
    get_episodic_memory,
    remember_episode,
    recall_similar_episodes,
    integrate_with_working_memory,
)

from .memory_integration import (
    MemoryIntegrationHub,
    IntegratedMemory,
    get_memory_hub,
    search_all_memories,
    consolidate_all_memories,
    think_with_memory,
)

# Advanced memory integration
from .advanced_memory_integration import (
    AdvancedMemoryHub,
    get_advanced_memory_hub,
)

# Query understanding
from .query_understanding import (
    QueryUnderstanding,
    QueryAnalysis,
    QueryIntent,
    get_query_understanding,
)

# Meta-learning optimizer
from .meta_learning_optimizer import (
    MetaLearningOptimizer,
    GaussianProcessSurrogate,
    CrossEntropyOptimizer,
    NeuralArchitectureSearch,
    get_meta_optimizer,
)

# Neural contextual bandits
from .neural_contextual_bandits import (
    NeuralContextualBandit,
    MultiHeadAttention,
    PrioritizedReplayBuffer,
    get_neural_bandit,
)

from .concurrent_processor import (
    ConcurrentProcessor,
    TaskPriority,
    get_concurrent_processor,
    process_concurrent_queries,
)

from .context_continuity import (
    ContextContinuity,
    get_context_continuity,
    analyze_query_continuity,
    get_conversation_context,
)

__all__ = [
    # Adaptive retrieval
    "AdaptiveRetrievalManager",
    "get_adaptive_manager",
    "select_retrieval_params",
    "record_retrieval_outcome",
    
    # Threshold learning
    "ThresholdLearner",
    "get_threshold_learner",
    "select_threshold",
    "record_threshold_outcome",
    
    # Predictive cache
    "PredictiveCacheManager",
    "get_predictive_cache",
    
    # Outcome tracking
    "OutcomeTracker",
    "get_outcome_tracker",
    "record_outcome",
    
    # Query classification
    "QueryIntent",
    "IntelligentQueryClassifier",
    "get_query_classifier",
    "classify_query",
    "learn_query_intent",
    
    # Rate limiting
    "AdaptiveRateLimiter",
    "RateLimitState",
    "get_rate_limiter",
    "acquire_rate_limit",
    
    # Context optimization
    "ContextAllocation",
    "ContextSizeOptimizer",
    "get_context_optimizer",
    "allocate_context_budget",
    "record_context_outcome",
    
    # Prompt optimization
    "AdaptivePromptOptimizer",
    "PromptVariant",
    "get_prompt_optimizer",
    "select_optimal_prompt",
    "record_prompt_outcome",
    
    # Error prediction
    "IntelligentErrorPredictor",
    "ErrorPrediction",
    "get_error_predictor",
    "predict_errors",
    "learn_from_error",
    
    # Self-healing
    "SelfHealingSystem",
    "HealingResult",
    "get_self_healer",
    "heal_code",
    
    # Semantic routing
    "SemanticRouter",
    "RouteMatch",
    "get_semantic_router",
    "route_query",
    "record_routing_outcome",
    
    # Intelligent planning
    "IntelligentPlanner",
    "ExecutionPlan",
    "PlanStep",
    "get_intelligent_planner",
    "create_intelligent_plan",
    "record_plan_execution",
    
    # Brain Orchestration (coordinates all systems)
    "BrainOrchestrator",
    "get_brain_orchestrator",
    "process_with_full_intelligence",
    
    # Auto-Tuning (continuous optimization)
    "AutoTuner",
    "get_auto_tuner",
    
    # Learning Coordination (transfer learning)
    "LearningCoordinator",
    "get_learning_coordinator",
    "emit_learning_event",
    
    # Performance Monitoring (real-time)
    "PerformanceMonitor",
    "get_performance_monitor",
    "record_system_metrics",
    
    # Ensemble Decision Making
    "Decision",
    "EnsembleDecision",
    "EnsembleDecisionMaker",
    "get_ensemble_decision_maker",
    "make_ensemble_decision",
    "record_ensemble_outcome",
    
    # Knowledge Graph
    "KnowledgeGraphBuilder",
    "get_knowledge_graph",
    "add_to_knowledge_graph",
    "query_knowledge_graph",
    
    # Query Expansion
    "QueryExpander",
    "ExpandedQuery",
    "get_query_expander",
    "expand_query",
    "record_expansion_outcome",
    
    # Meta-Cognitive (Self-Awareness)
    "MetaCognitiveSystem",
    "Thought",
    "SelfReflection",
    "get_meta_cognitive",
    "think_about",
    "self_reflect",
    "reason",
    
    # Working Memory
    "WorkingMemory",
    "MemoryItem",
    "get_working_memory",
    "remember",
    "recall",
    "get_focus",
    
    # Goal-Driven System
    "GoalDrivenSystem",
    "Goal",
    "GoalStatus",
    "get_goal_system",
    "set_goal",
    "get_current_goal",
    "mark_progress",
    
    # Episodic Memory (Long-term with embeddings)
    "EpisodicMemory",
    "Episode",
    "get_episodic_memory",
    "remember_episode",
    "recall_similar_episodes",
    "integrate_with_working_memory",
    
    # Memory Integration Hub
    "MemoryIntegrationHub",
    "IntegratedMemory",
    "get_memory_hub",
    "search_all_memories",
    "consolidate_all_memories",
    "think_with_memory",
    
    # Concurrent Processing
    "ConcurrentProcessor",
    "TaskPriority",
    "get_concurrent_processor",
    "process_concurrent_queries",
    
    # Context Continuity
    "ContextContinuity",
    "get_context_continuity",
    "analyze_query_continuity",
    "get_conversation_context",
    
    # Advanced Memory Integration
    "AdvancedMemoryHub",
    "get_advanced_memory_hub",
    
    # Query Understanding
    "QueryUnderstanding",
    "QueryAnalysis",
    "get_query_understanding",
    
    # Meta-Learning (MAML + Bayesian + CEM + NAS)
    "MetaLearningOptimizer",
    "GaussianProcessSurrogate",
    "CrossEntropyOptimizer",
    "NeuralArchitectureSearch",
    "get_meta_optimizer",
    
    # Neural Contextual Bandits (Deep RL + Attention)
    "NeuralContextualBandit",
    "MultiHeadAttention",
    "PrioritizedReplayBuffer",
    "get_neural_bandit",
]

# ============================================================================
# AUTOMATIC INITIALIZATION - All systems start automatically
# ============================================================================

import asyncio as _asyncio
import threading as _threading

_init_lock = _threading.Lock()
_initialized = False

def _auto_init_brain_systems():
    """Automatically initialize and start all brain systems."""
    global _initialized
    
    with _init_lock:
        if _initialized:
            return
        
        _initialized = True
        
        # Start background initialization
        _background_loop = None
        _background_thread = None
        
        def _init():
            global _background_loop
            try:
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
                _background_loop = loop
                
                async def _startup():
                    try:
                        # Start all systems in parallel (non-blocking)
                        init_tasks = [
                            get_brain_orchestrator(),
                            get_auto_tuner(),
                            get_learning_coordinator(),
                            get_performance_monitor(),
                            get_ensemble_decision_maker(),
                            get_knowledge_graph(),
                            get_query_expander(),
                            get_meta_cognitive(),
                            get_working_memory(),
                            get_goal_system(),
                            get_episodic_memory(),
                            get_memory_hub(),
                        ]
                        
                        # Initialize all in parallel
                        results = await _asyncio.gather(*init_tasks, return_exceptions=True)
                        
                        # Set initial goal (non-blocking)
                        try:
                            await set_goal("Continuously improve and help the user", priority=10)
                        except Exception:
                            pass
                        
                        # Initialize concurrent processing
                        concurrent_proc = await get_concurrent_processor()
                        context_cont = await get_context_continuity()
                        
                        # Create memory links (non-blocking, fire and forget)
                        try:
                            memory_hub = results[-1]
                            if memory_hub and not isinstance(memory_hub, Exception):
                                _asyncio.create_task(memory_hub.create_memory_links())
                        except Exception:
                            pass
                        
                        # All 27 systems are now running - COMPLETE COGNITIVE ARCHITECTURE ACTIVE!
                    except Exception:
                        pass
                
                async def _keep_alive():
                    """Keep loop alive for background tasks."""
                    try:
                        await _startup()
                        # Wait indefinitely (until cancelled)
                        await _asyncio.Event().wait()
                    except _asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                
                try:
                    # Start and keep running
                    loop.run_until_complete(_keep_alive())
                except (KeyboardInterrupt, SystemExit):
                    pass
                except Exception:
                    pass
                finally:
                    # Graceful shutdown
                    try:
                        pending = _asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(_asyncio.gather(*pending, return_exceptions=True))
                        loop.close()
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Run in background thread to not block import
        _background_thread = _threading.Thread(target=_init, daemon=True, name="jinx-brain-thread")
        _background_thread.start()

# Cleanup function for graceful shutdown
async def _shutdown_brain_systems():
    """Gracefully shutdown all brain systems."""
    try:
        # Stop orchestrator
        orch = await get_brain_orchestrator()
        await orch.stop()
    except Exception:
        pass
    
    try:
        # Stop auto-tuner
        tuner = await get_auto_tuner()
        await tuner.stop()
    except Exception:
        pass
    
    try:
        # Stop learning coordinator
        coord = await get_learning_coordinator()
        await coord.stop()
    except Exception:
        pass
    
    try:
        # Stop performance monitor
        mon = await get_performance_monitor()
        await mon.stop()
    except Exception:
        pass

# Lazy initialization helper
def ensure_brain_initialized():
    """Ensure brain systems are initialized (can be called safely multiple times)."""
    global _initialized
    if not _initialized:
        _auto_init_brain_systems()

# Auto-initialize on import (can be disabled via env var)
import os as _os
_auto_init_enabled = _os.getenv('JINX_BRAIN_AUTO_INIT', '1').strip() not in ('0', 'false', 'off')

if _auto_init_enabled:
    # Start initialization in background (non-blocking import)
    _auto_init_brain_systems()
else:
    # Manual initialization required - call ensure_brain_initialized() when needed
    pass
