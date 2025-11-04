"""Dynamic Configuration Plugin - Real-time adaptive configuration system.

Automatically adjusts ALL system parameters on-the-fly based on:
- Task type detection (using embeddings)
- Performance metrics
- Historical success patterns
- Resource availability

This is the BRAIN of the configuration system - it orchestrates everything.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, Optional, Any
from jinx.micro.logger.debug_logger import debug_log


class DynamicConfigPlugin:
    """Self-adapting configuration plugin - the smartest part of Jinx."""
    
    def __init__(self):
        self._enabled = True
        self._lock = asyncio.Lock()
        
        # Current active profile
        self._current_profile: Optional[str] = None
        self._profile_start_time = time.time()
        
        # Performance tracking for current profile
        self._profile_operations = 0
        self._profile_successes = 0
        self._profile_total_latency = 0.0
        
        # Learning: profile effectiveness scores
        self._profile_scores: Dict[str, float] = {}
        
        # All tunable parameters with their ranges and defaults
        self._all_parameters = self._initialize_parameter_registry()
    
    def _initialize_parameter_registry(self) -> Dict[str, Dict[str, Any]]:
        """Complete registry of ALL tunable parameters."""
        
        return {
            # ===== EMBEDDINGS PARAMETERS =====
            'EMBED_PROJECT_ENABLE': {
                'type': 'bool',
                'default': '1',
                'description': 'Enable project embeddings'
            },
            'EMBED_PROJECT_TOP_K': {
                'type': 'int',
                'min': 10,
                'max': 150,
                'default': 50,
                'adaptive': True,  # Can be auto-tuned
                'description': 'Number of embedding results'
            },
            'EMBED_PROJECT_EXHAUSTIVE': {
                'type': 'bool',
                'default': '1',
                'adaptive': True,
                'description': 'Use all retrieval stages'
            },
            'EMBED_PROJECT_SCORE_THRESHOLD': {
                'type': 'float',
                'min': 0.05,
                'max': 0.40,
                'default': 0.15,
                'adaptive': True,
                'description': 'Minimum similarity score'
            },
            'EMBED_PROJECT_TOTAL_CODE_BUDGET': {
                'type': 'int',
                'min': 10000,
                'max': 150000,
                'default': 50000,
                'adaptive': True,
                'description': 'Max code characters in context'
            },
            'EMBED_PROJECT_CALLGRAPH': {
                'type': 'bool',
                'default': '1',
                'adaptive': True,
                'description': 'Enable callgraph enrichment'
            },
            'EMBED_PROJECT_CALLGRAPH_TOP_HITS': {
                'type': 'int',
                'min': 1,
                'max': 10,
                'default': 5,
                'adaptive': True,
                'description': 'Callgraph top hits'
            },
            'EMBED_PROJECT_ALWAYS_FULL_PY_SCOPE': {
                'type': 'bool',
                'default': '1',
                'description': 'Extract full Python scope'
            },
            'EMBED_PROJECT_FULL_SCOPE_TOP_N': {
                'type': 'int',
                'min': 0,
                'max': 20,
                'default': 10,
                'adaptive': True,
                'description': 'How many get full scope'
            },
            
            # ===== TIMEOUTS =====
            'JINX_STAGE_BASECTX_MS': {
                'type': 'int',
                'min': 200,
                'max': 2000,
                'default': 500,
                'adaptive': True,
                'description': 'Base context timeout'
            },
            'JINX_STAGE_PROJCTX_MS': {
                'type': 'int',
                'min': 1000,
                'max': 15000,
                'default': 5000,
                'adaptive': True,
                'description': 'Project context timeout'
            },
            'JINX_STAGE_MEMCTX_MS': {
                'type': 'int',
                'min': 200,
                'max': 2000,
                'default': 500,
                'adaptive': True,
                'description': 'Memory context timeout'
            },
            'EMBED_UNIFIED_MAX_TIME_MS': {
                'type': 'int',
                'min': 1000,
                'max': 10000,
                'default': 3000,
                'adaptive': True,
                'description': 'Unified context timeout'
            },
            
            # ===== CONCURRENCY =====
            'JINX_MAX_CONCURRENT': {
                'type': 'int',
                'min': 1,
                'max': 16,
                'default': 6,
                'adaptive': True,
                'description': 'Max concurrent operations'
            },
            'JINX_FRAME_MAX_CONC': {
                'type': 'int',
                'min': 1,
                'max': 10,
                'default': 3,
                'adaptive': True,
                'description': 'Frame concurrency'
            },
            
            # ===== BRAIN SYSTEMS =====
            'JINX_BRAIN_ENABLE': {
                'type': 'bool',
                'default': '1',
                'description': 'Enable brain systems'
            },
            'EMBED_BRAIN_ENABLE': {
                'type': 'bool',
                'default': '1',
                'description': 'Enable embeddings brain'
            },
            'JINX_BRAIN_ADAPTIVE_RETRIEVAL': {
                'type': 'bool',
                'default': '1',
                'adaptive': True,
                'description': 'Adaptive retrieval'
            },
            
            # ===== QUALITY/PERFORMANCE =====
            'JINX_CTX_COMPACT': {
                'type': 'bool',
                'default': '1',
                'adaptive': True,
                'description': 'Compact context'
            },
            'JINX_AUTOMACROS': {
                'type': 'bool',
                'default': '1',
                'description': 'Auto-inject macros'
            },
        }
    
    async def start(self):
        """Start the dynamic configuration plugin."""
        
        if not self._enabled:
            return
        
        await debug_log("Dynamic Config Plugin started - AI-powered auto-tuning active", "PLUGIN")
        
        # Start background monitoring task
        asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background loop that monitors and adapts configuration."""
        
        while self._enabled:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check if adaptation is needed
                await self._check_and_adapt()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                await debug_log(f"Monitoring loop error: {e}", "PLUGIN")
                await asyncio.sleep(60)
    
    async def _check_and_adapt(self):
        """Check performance and adapt if needed."""
        
        async with self._lock:
            if self._profile_operations < 5:
                # Not enough data yet
                return
            
            # Calculate current profile effectiveness
            success_rate = self._profile_successes / self._profile_operations
            avg_latency = self._profile_total_latency / self._profile_operations
            
            # Score current profile
            score = self._calculate_profile_score(success_rate, avg_latency)
            
            if self._current_profile:
                self._profile_scores[self._current_profile] = score
            
            await debug_log(
                f"Profile '{self._current_profile}' score: {score:.2f} "
                f"(success={success_rate:.2%}, latency={avg_latency:.0f}ms)",
                "PLUGIN"
            )
            
            # If score is low, consider switching profile
            if score < 0.6:
                await debug_log(
                    f"Profile performance low - adapting parameters",
                    "PLUGIN"
                )
                await self._optimize_current_profile()
    
    def _calculate_profile_score(self, success_rate: float, avg_latency_ms: float) -> float:
        """Calculate effectiveness score for profile."""
        
        # Normalize latency to 0-1 (lower is better)
        latency_score = max(0, 1.0 - (avg_latency_ms / 5000))
        
        # Weighted combination
        score = 0.6 * success_rate + 0.4 * latency_score
        
        return max(0, min(1, score))
    
    async def _optimize_current_profile(self):
        """Optimize current profile parameters."""
        
        # Get current success rate
        success_rate = self._profile_successes / max(1, self._profile_operations)
        avg_latency = self._profile_total_latency / max(1, self._profile_operations)
        
        # Decide optimization strategy
        if success_rate < 0.8:
            # Low success - need more thorough search
            await self._increase_thoroughness()
        
        if avg_latency > 3000:
            # High latency - need speed optimization
            await self._increase_speed()
        
        # Reset counters for new profile
        self._profile_operations = 0
        self._profile_successes = 0
        self._profile_total_latency = 0.0
    
    async def _increase_thoroughness(self):
        """Increase thoroughness (more results, longer timeouts)."""
        
        await debug_log("Increasing thoroughness - boosting search depth", "PLUGIN")
        
        # Increase TOP_K
        current = int(os.getenv('EMBED_PROJECT_TOP_K', '50'))
        new_value = min(150, int(current * 1.3))
        os.environ['EMBED_PROJECT_TOP_K'] = str(new_value)
        
        # Increase timeouts
        current_timeout = int(os.getenv('JINX_STAGE_PROJCTX_MS', '5000'))
        new_timeout = min(15000, int(current_timeout * 1.5))
        os.environ['JINX_STAGE_PROJCTX_MS'] = str(new_timeout)
        
        # Ensure exhaustive mode
        os.environ['EMBED_PROJECT_EXHAUSTIVE'] = '1'
        
        # Lower threshold
        current_thresh = float(os.getenv('EMBED_PROJECT_SCORE_THRESHOLD', '0.15'))
        new_thresh = max(0.05, current_thresh * 0.8)
        os.environ['EMBED_PROJECT_SCORE_THRESHOLD'] = f'{new_thresh:.2f}'
    
    async def _increase_speed(self):
        """Increase speed (fewer results, shorter timeouts)."""
        
        await debug_log("Increasing speed - optimizing for latency", "PLUGIN")
        
        # Decrease TOP_K
        current = int(os.getenv('EMBED_PROJECT_TOP_K', '50'))
        new_value = max(20, int(current * 0.7))
        os.environ['EMBED_PROJECT_TOP_K'] = str(new_value)
        
        # Decrease timeouts slightly
        current_timeout = int(os.getenv('JINX_STAGE_PROJCTX_MS', '5000'))
        new_timeout = max(2000, int(current_timeout * 0.8))
        os.environ['JINX_STAGE_PROJCTX_MS'] = str(new_timeout)
        
        # Increase concurrency
        current_conc = int(os.getenv('JINX_MAX_CONCURRENT', '6'))
        new_conc = min(12, current_conc + 2)
        os.environ['JINX_MAX_CONCURRENT'] = str(new_conc)
    
    async def adapt_for_request(self, request_text: str, context: Optional[Dict] = None, user_id: Optional[str] = None):
        """
        Adapt configuration for specific request using ML Orchestrator.
        
        This is called BEFORE each user request is processed.
        
        Uses production ML pipeline:
        - Semantic cache (if available)
        - ML task detection with user adaptation
        - Bayesian optimizer
        - RL agent
        - Monitoring integration
        """
        
        if not self._enabled:
            return
        
        async with self._lock:
            try:
                # === USE ML ORCHESTRATOR ===
                from .ml_orchestrator import get_ml_orchestrator
                
                orchestrator = await get_ml_orchestrator()
                
                # Get ML prediction with full pipeline
                result = await orchestrator.predict_task(
                    query=request_text,
                    context=context,
                    user_id=user_id,
                    explain=False  # Don't need explanation for config
                )
                
                task_type = result['task_type']
                confidence = result['confidence']
                
                await debug_log(
                    f"ML Orchestrator: {task_type} (conf={confidence:.2f}, "
                    f"latency={result['latency_ms']:.0f}ms, cache={result['from_cache']})",
                    "PLUGIN"
                )
                
                # === LEVEL 2: BAYESIAN OPTIMIZATION ===
                from .task_detector import get_optimal_config_for_task
                # Try to get optimized config from Bayesian optimizer
                try:
                    from .bayesian_config_optimizer import get_bayesian_optimizer
                    
                    optimizer = await get_bayesian_optimizer()
                    
                    # Get best known config or suggest new one
                    best_config = await optimizer.get_best_config(task_type)
                    
                    if best_config:
                        await debug_log(
                            f"Bayesian: Using best known config for {task_type}",
                            "PLUGIN"
                        )
                        optimal_config = best_config
                    else:
                        # No history - use baseline profile
                        optimal_config = get_optimal_config_for_task(task_type)
                        
                        # But suggest improvement
                        suggested = await optimizer.suggest_config(
                            task_type,
                            optimal_config
                        )
                        
                        # Blend baseline with suggestion
                        optimal_config = {
                            k: (optimal_config.get(k, v) + suggested.get(k, v)) / 2
                            for k, v in suggested.items()
                        }
                        
                        await debug_log(
                            f"Bayesian: Suggesting exploratory config",
                            "PLUGIN"
                        )
                
                except Exception as e:
                    # Fallback to profile-based
                    optimal_config = get_optimal_config_for_task(task_type)
                    await debug_log(f"Bayesian fallback: {e}", "PLUGIN")
                
                # === LEVEL 3: RL AGENT REFINEMENT ===
                # Let RL agent adjust config based on long-term learning
                try:
                    from .rl_config_agent import get_rl_agent
                    
                    rl_agent = await get_rl_agent()
                    
                    # Get recent performance
                    recent_perf = {
                        'success_rate': self._profile_successes / max(1, self._profile_operations),
                        'avg_latency': self._profile_total_latency / max(1, self._profile_operations),
                        'quality': 0.5
                    }
                    
                    # RL agent suggests adjustments
                    rl_adjusted = await rl_agent.select_action(
                        optimal_config,
                        task_type,
                        recent_perf
                    )
                    
                    # Apply RL adjustments with small weight (20%)
                    final_config = {}
                    for key in optimal_config:
                        base = optimal_config.get(key, 0)
                        rl = rl_adjusted.get(key, base)
                        final_config[key] = 0.8 * base + 0.2 * rl
                    
                    optimal_config = final_config
                    
                    await debug_log(
                        f"RL: Applied learned adjustments",
                        "PLUGIN"
                    )
                
                except Exception as e:
                    # RL is optional - continue without it
                    await debug_log(f"RL not available: {e}", "PLUGIN")
                
                # === APPLY FINAL CONFIGURATION ===
                if task_type != self._current_profile or confidence > 0.8:
                    await self._apply_profile(task_type, optimal_config)
                    self._current_profile = task_type
                    self._profile_start_time = time.time()
            
            except Exception as e:
                await debug_log(f"Adaptation error: {e}", "PLUGIN")
    
    async def _apply_profile(self, profile_name: str, config: Dict[str, str]):
        """Apply configuration profile."""
        
        await debug_log(f"Applying profile: {profile_name}", "PLUGIN")
        
        # Apply each parameter
        applied_count = 0
        for key, value in config.items():
            if key in self._all_parameters:
                os.environ[key] = str(value)
                applied_count += 1
        
        await debug_log(f"Applied {applied_count} parameters for '{profile_name}'", "PLUGIN")
    
    async def record_result(
        self,
        success: bool,
        latency_ms: float,
        request_text: Optional[str] = None
    ):
        """Record operation result and update ALL learning systems."""
        
        async with self._lock:
            self._profile_operations += 1
            if success:
                self._profile_successes += 1
            self._profile_total_latency += latency_ms
            
            # Compute quality score
            success_rate = self._profile_successes / max(1, self._profile_operations)
            latency_score = max(0, 1.0 - (latency_ms / 5000))
            quality_score = 0.6 * success_rate + 0.4 * latency_score
            
            # === UPDATE ML CLASSIFIER ===
            if request_text and self._current_profile:
                try:
                    from .ml_task_classifier import get_ml_classifier
                    
                    ml_classifier = await get_ml_classifier()
                    
                    # Learn from feedback
                    await ml_classifier.learn_from_feedback(
                        text=request_text,
                        predicted_task=self._current_profile,
                        actual_task=None,  # We don't have ground truth
                        outcome_quality=quality_score
                    )
                
                except Exception:
                    pass
            
            # === UPDATE BAYESIAN OPTIMIZER ===
            if self._current_profile:
                try:
                    from .bayesian_config_optimizer import get_bayesian_optimizer
                    import os
                    
                    optimizer = await get_bayesian_optimizer()
                    
                    # Current config
                    current_config = {
                        'EMBED_PROJECT_TOP_K': int(os.getenv('EMBED_PROJECT_TOP_K', '50')),
                        'EMBED_PROJECT_SCORE_THRESHOLD': float(os.getenv('EMBED_PROJECT_SCORE_THRESHOLD', '0.15')),
                        'JINX_STAGE_PROJCTX_MS': int(os.getenv('JINX_STAGE_PROJCTX_MS', '5000')),
                        'JINX_MAX_CONCURRENT': int(os.getenv('JINX_MAX_CONCURRENT', '6')),
                        'EMBED_PROJECT_EXHAUSTIVE': int(os.getenv('EMBED_PROJECT_EXHAUSTIVE', '1'))
                    }
                    
                    # Record observation
                    await optimizer.record_observation(
                        config=current_config,
                        context=self._current_profile,
                        latency_ms=latency_ms,
                        success_rate=success_rate
                    )
                
                except Exception:
                    pass
            
            # === UPDATE RL AGENT ===
            if self._current_profile:
                try:
                    from .rl_config_agent import get_rl_agent
                    import os
                    
                    rl_agent = await get_rl_agent()
                    
                    # Prepare next state
                    next_config = {
                        'EMBED_PROJECT_TOP_K': int(os.getenv('EMBED_PROJECT_TOP_K', '50')),
                        'EMBED_PROJECT_SCORE_THRESHOLD': float(os.getenv('EMBED_PROJECT_SCORE_THRESHOLD', '0.15')),
                        'JINX_STAGE_PROJCTX_MS': int(os.getenv('JINX_STAGE_PROJCTX_MS', '5000')),
                        'JINX_MAX_CONCURRENT': int(os.getenv('JINX_MAX_CONCURRENT', '6')),
                        'EMBED_PROJECT_EXHAUSTIVE': int(os.getenv('EMBED_PROJECT_EXHAUSTIVE', '1'))
                    }
                    
                    next_perf = {
                        'success_rate': success_rate,
                        'avg_latency': self._profile_total_latency / max(1, self._profile_operations),
                        'quality': quality_score
                    }
                    
                    # Observe reward
                    await rl_agent.observe_reward(
                        reward=quality_score,
                        next_config=next_config,
                        next_task_type=self._current_profile,
                        next_performance=next_perf,
                        done=False
                    )
                
                except Exception:
                    pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        
        async with self._lock:
            return {
                'enabled': self._enabled,
                'current_profile': self._current_profile,
                'profile_uptime_seconds': time.time() - self._profile_start_time,
                'operations_in_profile': self._profile_operations,
                'success_rate_in_profile': (
                    self._profile_successes / max(1, self._profile_operations)
                ),
                'avg_latency_in_profile': (
                    self._profile_total_latency / max(1, self._profile_operations)
                ),
                'profile_scores': self._profile_scores,
                'total_parameters': len(self._all_parameters),
                'adaptive_parameters': sum(
                    1 for p in self._all_parameters.values()
                    if p.get('adaptive', False)
                )
            }
    
    def stop(self):
        """Stop the plugin."""
        self._enabled = False


# Singleton
_dynamic_plugin: Optional[DynamicConfigPlugin] = None
_plugin_lock = asyncio.Lock()


async def get_dynamic_config_plugin() -> DynamicConfigPlugin:
    """Get singleton dynamic config plugin."""
    global _dynamic_plugin
    if _dynamic_plugin is None:
        async with _plugin_lock:
            if _dynamic_plugin is None:
                _dynamic_plugin = DynamicConfigPlugin()
                await _dynamic_plugin.start()
    return _dynamic_plugin


async def adapt_config_for_request(request_text: str, context: Optional[Dict] = None):
    """Adapt configuration for incoming request."""
    plugin = await get_dynamic_config_plugin()
    await plugin.adapt_for_request(request_text, context)


async def record_request_result(success: bool, latency_ms: float, request_text: Optional[str] = None):
    """Record request result for learning."""
    plugin = await get_dynamic_config_plugin()
    await plugin.record_result(success, latency_ms, request_text)


__all__ = [
    "DynamicConfigPlugin",
    "get_dynamic_config_plugin",
    "adapt_config_for_request",
    "record_request_result",
]
