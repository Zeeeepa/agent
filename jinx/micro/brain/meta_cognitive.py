"""Meta-Cognitive System - самосознание и рефлексия AI.

Система размышляет о своих собственных процессах и принимает решения высшего уровня.
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
class Thought:
    """Мысль системы."""
    content: str
    thought_type: str  # 'observation', 'question', 'hypothesis', 'decision', 'reflection'
    confidence: float
    reasoning: List[str]
    timestamp: float
    context: Dict[str, Any]


@dataclass
class SelfReflection:
    """Рефлексия о собственном состоянии."""
    performance_assessment: str  # 'excellent', 'good', 'degrading', 'poor'
    identified_issues: List[str]
    proposed_solutions: List[str]
    learning_insights: List[str]
    confidence_in_assessment: float


class MetaCognitiveSystem:
    """Система метакогнитивных процессов - думает о том, как она думает."""
    
    def __init__(self, state_path: str = "log/meta_cognitive.json"):
        self.state_path = state_path
        
        # Stream of consciousness
        self.thought_stream: deque[Thought] = deque(maxlen=1000)
        
        # Self-awareness state
        self.current_state: Dict[str, Any] = {
            'confidence': 0.5,
            'uncertainty': 0.5,
            'attention_focus': None,
            'goals': [],
            'concerns': []
        }
        
        # Hypotheses about the world
        self.hypotheses: Dict[str, Tuple[float, List[str]]] = {}  # hypothesis -> (confidence, evidence)
        
        # Self-monitoring
        self.performance_history: deque[float] = deque(maxlen=100)
        
        # Meta-strategies (learned)
        self.strategies: Dict[str, Dict[str, Any]] = {}
        
        # Reasoning chains
        self.reasoning_chains: List[List[Thought]] = []
        
        self._lock = asyncio.Lock()
        self._load_state()
    
    def _load_state(self) -> None:
        """Load cognitive state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.current_state = data.get('state', self.current_state)
                
                # Restore hypotheses
                for hyp, (conf, evidence) in data.get('hypotheses', {}).items():
                    self.hypotheses[hyp] = (conf, evidence)
                
                # Restore strategies
                self.strategies = data.get('strategies', {})
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist cognitive state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize hypotheses
                hypotheses_data = {
                    k: [v[0], v[1]] for k, v in self.hypotheses.items()
                }
                
                data = {
                    'state': self.current_state,
                    'hypotheses': hypotheses_data,
                    'strategies': self.strategies,
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    async def think(self, stimulus: str, context: Dict[str, Any]) -> Thought:
        """Думает о stimulus и генерирует мысль."""
        async with self._lock:
            # Observe
            observation = await self._observe(stimulus, context)
            
            # Question
            questions = await self._generate_questions(observation, context)
            
            # Hypothesize
            if questions:
                hypothesis = await self._form_hypothesis(observation, questions[0], context)
                
                # Decide
                decision = await self._make_decision(hypothesis, context)
                
                # Record thought
                thought = Thought(
                    content=decision,
                    thought_type='decision',
                    confidence=self.current_state['confidence'],
                    reasoning=[observation, questions[0], hypothesis],
                    timestamp=time.time(),
                    context=context
                )
            else:
                # Simple observation
                thought = Thought(
                    content=observation,
                    thought_type='observation',
                    confidence=0.7,
                    reasoning=[],
                    timestamp=time.time(),
                    context=context
                )
            
            self.thought_stream.append(thought)
            
            # Update state
            await self._update_state(thought)
            
            return thought
    
    async def _observe(self, stimulus: str, context: Dict[str, Any]) -> str:
        """Наблюдает и интерпретирует stimulus."""
        # Analyze stimulus
        stim_lower = stimulus.lower()
        
        observations = []
        
        # Check for patterns
        if 'error' in stim_lower or 'fail' in stim_lower:
            observations.append("System is experiencing failures")
        
        if 'slow' in stim_lower or 'latency' in stim_lower:
            observations.append("Performance degradation detected")
        
        if 'success' in stim_lower or context.get('success'):
            observations.append("Operation completed successfully")
        
        # Check context
        if context.get('performance'):
            perf = context['performance']
            if perf < 0.5:
                observations.append("Low performance in recent operations")
            elif perf > 0.8:
                observations.append("High performance in recent operations")
        
        if not observations:
            observations.append(f"Stimulus received: {stimulus[:50]}")
        
        return "; ".join(observations)
    
    async def _generate_questions(self, observation: str, context: Dict[str, Any]) -> List[str]:
        """Генерирует вопросы для investigation."""
        questions = []
        
        obs_lower = observation.lower()
        
        if 'failure' in obs_lower or 'error' in obs_lower:
            questions.append("What is causing the failures?")
            questions.append("How can we prevent this?")
        
        if 'degradation' in obs_lower or 'slow' in obs_lower:
            questions.append("What changed to cause performance degradation?")
            questions.append("Which system is the bottleneck?")
        
        if 'success' in obs_lower:
            questions.append("What factors contributed to success?")
            questions.append("Can we replicate this?")
        
        if not questions:
            questions.append("What does this mean for the system?")
        
        return questions
    
    async def _form_hypothesis(self, observation: str, question: str, context: Dict[str, Any]) -> str:
        """Формирует гипотезу."""
        obs_lower = observation.lower()
        q_lower = question.lower()
        
        # Generate hypothesis based on observation and question
        if 'causing' in q_lower and 'failure' in obs_lower:
            # Check recent system states
            hypotheses = [
                "Hypothesis: Configuration issue in one of the brain systems",
                "Hypothesis: Resource exhaustion (memory or CPU)",
                "Hypothesis: External dependency failure"
            ]
            
            # Select based on context
            if context.get('cpu_high'):
                return hypotheses[1]
            return hypotheses[0]
        
        elif 'prevent' in q_lower:
            return "Hypothesis: Implementing additional validation will prevent failures"
        
        elif 'degradation' in obs_lower:
            return "Hypothesis: Increased load or inefficient parameters causing slowdown"
        
        elif 'success' in obs_lower and 'factors' in q_lower:
            return "Hypothesis: Optimal parameter combination and good data quality"
        
        return "Hypothesis: Current behavior is within normal operating parameters"
    
    async def _make_decision(self, hypothesis: str, context: Dict[str, Any]) -> str:
        """Принимает решение на основе hypothesis."""
        hyp_lower = hypothesis.lower()
        
        decisions = []
        
        if 'configuration' in hyp_lower:
            decisions.append("DECISION: Review and adjust system configurations")
        
        if 'resource exhaustion' in hyp_lower:
            decisions.append("DECISION: Reduce load and optimize resource usage")
        
        if 'validation' in hyp_lower:
            decisions.append("DECISION: Add validation checks to critical paths")
        
        if 'parameters' in hyp_lower:
            decisions.append("DECISION: Trigger parameter re-optimization")
        
        if 'optimal' in hyp_lower:
            decisions.append("DECISION: Record and replicate successful patterns")
        
        if not decisions:
            decisions.append("DECISION: Continue monitoring and gather more data")
        
        return "; ".join(decisions)
    
    async def _update_state(self, thought: Thought) -> None:
        """Обновляет internal state на основе thought."""
        # Update confidence based on thought type
        if thought.thought_type == 'decision':
            self.current_state['confidence'] *= 0.95  # Slight decrease (uncertainty)
        elif thought.thought_type == 'observation' and thought.confidence > 0.8:
            self.current_state['confidence'] = min(1.0, self.current_state['confidence'] * 1.05)
        
        # Update attention focus
        if 'error' in thought.content.lower() or 'failure' in thought.content.lower():
            self.current_state['attention_focus'] = 'error_handling'
            if 'failure' not in self.current_state.get('concerns', []):
                self.current_state.setdefault('concerns', []).append('failure')
        
        elif 'performance' in thought.content.lower():
            self.current_state['attention_focus'] = 'optimization'
        
        # Update uncertainty
        self.current_state['uncertainty'] = 1.0 - self.current_state['confidence']
    
    async def reflect(self) -> SelfReflection:
        """Проводит глубокую рефлексию о собственном состоянии."""
        async with self._lock:
            # Analyze recent performance
            recent_thoughts = list(self.thought_stream)[-20:]
            
            # Count thought types
            thought_types = defaultdict(int)
            for t in recent_thoughts:
                thought_types[t.thought_type] += 1
            
            # Assess performance
            avg_confidence = sum(t.confidence for t in recent_thoughts) / max(1, len(recent_thoughts))
            
            if avg_confidence > 0.8:
                assessment = 'excellent'
            elif avg_confidence > 0.6:
                assessment = 'good'
            elif avg_confidence > 0.4:
                assessment = 'degrading'
            else:
                assessment = 'poor'
            
            # Identify issues
            issues = []
            if thought_types['decision'] > len(recent_thoughts) * 0.7:
                issues.append("Too many decisions without sufficient observation")
            
            if avg_confidence < 0.5:
                issues.append("Low confidence in recent judgments")
            
            if len(self.current_state.get('concerns', [])) > 3:
                issues.append("Multiple unresolved concerns")
            
            # Propose solutions
            solutions = []
            if issues:
                if "observation" in issues[0].lower():
                    solutions.append("Increase observation time before making decisions")
                
                if "confidence" in issues[0].lower():
                    solutions.append("Gather more evidence before forming hypotheses")
                
                if "concerns" in issues[0].lower():
                    solutions.append("Prioritize and address top concerns systematically")
            
            # Learning insights
            insights = []
            
            # Analyze reasoning chains
            if len(self.thought_stream) > 10:
                # Pattern: successful chains
                successful = [t for t in recent_thoughts if t.confidence > 0.7]
                if successful:
                    insights.append(f"Successful reasoning pattern: {successful[0].reasoning[:2]}")
            
            # Check hypothesis accuracy
            if len(self.hypotheses) > 5:
                insights.append(f"Maintaining {len(self.hypotheses)} active hypotheses")
            
            reflection = SelfReflection(
                performance_assessment=assessment,
                identified_issues=issues,
                proposed_solutions=solutions,
                learning_insights=insights,
                confidence_in_assessment=min(1.0, len(recent_thoughts) / 20.0)
            )
            
            # Save state after reflection
            await self._save_state()
            
            return reflection
    
    async def reason_about(self, topic: str, evidence: List[str]) -> Tuple[str, float]:
        """Логический вывод на основе evidence."""
        async with self._lock:
            # Build reasoning chain
            chain = []
            
            # Premises
            for e in evidence:
                chain.append(Thought(
                    content=f"Premise: {e}",
                    thought_type='observation',
                    confidence=0.8,
                    reasoning=[],
                    timestamp=time.time(),
                    context={'evidence': True}
                ))
            
            # Inference
            if len(evidence) >= 2:
                # Combine evidence
                if all('success' in e.lower() for e in evidence):
                    conclusion = f"Conclusion: {topic} is functioning well"
                    confidence = 0.9
                
                elif all('fail' in e.lower() or 'error' in e.lower() for e in evidence):
                    conclusion = f"Conclusion: {topic} requires immediate attention"
                    confidence = 0.85
                
                elif any('success' in e.lower() for e in evidence) and any('fail' in e.lower() for e in evidence):
                    conclusion = f"Conclusion: {topic} shows mixed results, needs investigation"
                    confidence = 0.7
                
                else:
                    conclusion = f"Conclusion: Insufficient evidence about {topic}"
                    confidence = 0.5
            else:
                conclusion = f"Conclusion: Need more evidence about {topic}"
                confidence = 0.4
            
            # Add conclusion to chain
            chain.append(Thought(
                content=conclusion,
                thought_type='hypothesis',
                confidence=confidence,
                reasoning=[e for e in evidence],
                timestamp=time.time(),
                context={'topic': topic}
            ))
            
            self.reasoning_chains.append(chain)
            
            return conclusion, confidence
    
    def get_state(self) -> Dict[str, Any]:
        """Получить текущее cognitive state."""
        return {
            'confidence': self.current_state['confidence'],
            'uncertainty': self.current_state['uncertainty'],
            'attention_focus': self.current_state.get('attention_focus'),
            'active_hypotheses': len(self.hypotheses),
            'recent_thoughts': len(self.thought_stream),
            'reasoning_chains': len(self.reasoning_chains),
            'concerns': self.current_state.get('concerns', [])
        }


# Singleton
_meta_cognitive: Optional[MetaCognitiveSystem] = None
_mc_lock = asyncio.Lock()


async def get_meta_cognitive() -> MetaCognitiveSystem:
    """Get singleton meta-cognitive system."""
    global _meta_cognitive
    if _meta_cognitive is None:
        async with _mc_lock:
            if _meta_cognitive is None:
                _meta_cognitive = MetaCognitiveSystem()
    return _meta_cognitive


async def think_about(stimulus: str, context: Optional[Dict[str, Any]] = None) -> Thought:
    """System thinks about stimulus."""
    mc = await get_meta_cognitive()
    return await mc.think(stimulus, context or {})


async def self_reflect() -> SelfReflection:
    """System reflects on itself."""
    mc = await get_meta_cognitive()
    return await mc.reflect()


async def reason(topic: str, evidence: List[str]) -> Tuple[str, float]:
    """System reasons about topic."""
    mc = await get_meta_cognitive()
    return await mc.reason_about(topic, evidence)


__all__ = [
    "MetaCognitiveSystem",
    "Thought",
    "SelfReflection",
    "get_meta_cognitive",
    "think_about",
    "self_reflect",
    "reason",
]
