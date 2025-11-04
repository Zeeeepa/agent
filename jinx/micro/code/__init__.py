"""Code module - Complete AI Editor with unified brain integration.

MODERN AI EDITOR FEATURES (like Cursor/Copilot):
- Real-time code intelligence
- Intelligent completions with memory
- Context-aware suggestions
- Automated refactoring
- Code quality analysis
- Natural language chat with codebase

MEMORY INTEGRATION (ALL 4 SYSTEMS):
- Working Memory (active context)
- Episodic Memory (past experiences)
- Semantic Memory (knowledge)
- Jinx Memory (permanent patterns)

ADVANCED CAPABILITIES:
- AST transformations (LibCST)
- Semantic patching (embeddings)
- Pattern learning (ML)
- Code smell detection (AI)
- Brain-powered intelligence (27 systems)
- Real-time diagnostics (LSP-compatible)
"""

from .smart_patcher import (
    SmartPatcher,
    PatchBuilder,
    PatchRule,
    PatchResult,
    PatchMethod,
    PatchStatus,
    CommonPatches,
    get_smart_patcher,
    apply_intelligent_patch,
)

from .patch_validator import (
    PatchValidator,
    ValidationResult,
    get_patch_validator,
    validate_patched_file,
)

from .patch_orchestrator import (
    PatchOrchestrator,
    PatchOperation,
    PatchReport,
    get_patch_orchestrator,
    apply_intelligent_patch_operation,
)

from .ast_transformer import (
    ASTTransformEngine,
    TransformRule,
    IntelligentTransformer,
    get_ast_engine,
)

from .semantic_patcher import (
    SemanticPatcher,
    CodePattern,
    SemanticMatch,
    get_semantic_patcher,
    find_semantic_matches,
    suggest_intelligent_patches,
)

from .memory_integrated_patcher import (
    MemoryIntegratedPatcher,
    MemoryEnhancedPatch,
    get_memory_integrated_patcher,
)

from .ai_code_analyzer import (
    AICodeAnalyzer,
    CodeDiagnostic,
    CodeSuggestion,
    CodeAction,
    DiagnosticSeverity,
    get_ai_analyzer,
)

from .unified_ai_editor import (
    UnifiedAIEditor,
    EditorContext,
    IntelligentEdit,
    get_unified_ai_editor,
)

__all__ = [
    # Smart Patcher
    "SmartPatcher",
    "PatchBuilder",
    "PatchRule",
    "PatchResult",
    "PatchMethod",
    "PatchStatus",
    "CommonPatches",
    "get_smart_patcher",
    "apply_intelligent_patch",
    
    # Validator
    "PatchValidator",
    "ValidationResult",
    "get_patch_validator",
    "validate_patched_file",
    
    # Orchestrator (Brain-powered)
    "PatchOrchestrator",
    "PatchOperation",
    "PatchReport",
    "get_patch_orchestrator",
    "apply_intelligent_patch_operation",
    
    # AST Transformer (LibCST)
    "ASTTransformEngine",
    "TransformRule",
    "IntelligentTransformer",
    "get_ast_engine",
    
    # Semantic Patcher (ML + Embeddings)
    "SemanticPatcher",
    "CodePattern",
    "SemanticMatch",
    "get_semantic_patcher",
    "find_semantic_matches",
    "suggest_intelligent_patches",
    
    # Memory Integrated Patcher (All 4 memory systems)
    "MemoryIntegratedPatcher",
    "MemoryEnhancedPatch",
    "get_memory_integrated_patcher",
    
    # AI Code Analyzer (Real-time intelligence)
    "AICodeAnalyzer",
    "CodeDiagnostic",
    "CodeSuggestion",
    "CodeAction",
    "DiagnosticSeverity",
    "get_ai_analyzer",
    
    # Unified AI Editor (Complete modern editor)
    "UnifiedAIEditor",
    "EditorContext",
    "IntelligentEdit",
    "get_unified_ai_editor",
]
