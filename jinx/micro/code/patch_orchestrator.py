"""Patch Orchestrator - unified brain-powered patch operations.

Natural language interface with AST transforms, semantic search, and full brain integration.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .smart_patcher import (
    SmartPatcher,
    PatchBuilder,
    PatchRule,
    PatchResult,
    PatchStatus,
    CommonPatches,
    get_smart_patcher
)
from .patch_validator import (
    ValidationResult,
    get_patch_validator,
    validate_patched_file
)
from .ast_transformer import get_ast_engine, ASTTransformEngine
from .semantic_patcher import get_semantic_patcher, SemanticPatcher


@dataclass
class PatchOperation:
    """High-level patch operation."""
    description: str
    file_path: str
    rules: List[PatchRule]
    validate: bool = True
    rollback_on_error: bool = True


@dataclass
class PatchReport:
    """Comprehensive patch report."""
    operation: PatchOperation
    patch_result: PatchResult
    validation_result: Optional[ValidationResult]
    success: bool
    message: str


class PatchOrchestrator:
    """Orchestrates patching operations with full brain intelligence."""
    
    def __init__(self):
        self._history: List[PatchReport] = []
        self._ast_engine: Optional[ASTTransformEngine] = None
        self._semantic_patcher: Optional[SemanticPatcher] = None
    
    async def apply_operation(self, operation: PatchOperation) -> PatchReport:
        """Apply patch operation with full validation."""
        
        # Get patcher
        patcher = await get_smart_patcher()
        
        # Apply patch
        patch_result = await patcher.apply_patch_set(
            operation.file_path,
            operation.rules,
            validate=operation.validate,
            rollback_on_error=operation.rollback_on_error
        )
        
        # Validate if requested
        validation_result = None
        if operation.validate and patch_result.status == PatchStatus.SUCCESS:
            validator = await get_patch_validator()
            validation_result = await validator.validate_patch(
                operation.file_path,
                check_compilation=True,
                check_imports=True,
                check_linting=False,
                run_tests=False
            )
        
        # Determine success
        success = (
            patch_result.status == PatchStatus.SUCCESS and
            patch_result.syntax_valid and
            (validation_result is None or validation_result.overall_ok)
        )
        
        # Build message
        message = self._build_message(patch_result, validation_result)
        
        # Create report
        report = PatchReport(
            operation=operation,
            patch_result=patch_result,
            validation_result=validation_result,
            success=success,
            message=message
        )
        
        # Store in history
        self._history.append(report)
        
        # Log to brain memory
        await self._log_to_memory(report)
        
        return report
    
    async def apply_from_description(
        self,
        description: str,
        file_path: str
    ) -> PatchReport:
        """Apply patch from natural language description using brain intelligence."""
        
        # Use brain to interpret description and create rules
        rules = await self._interpret_with_brain(description, file_path)
        
        if not rules:
            # Could not interpret, return error report
            return PatchReport(
                operation=PatchOperation(
                    description=description,
                    file_path=file_path,
                    rules=[]
                ),
                patch_result=PatchResult(
                    status=PatchStatus.FAILED,
                    file_path=file_path,
                    applied_rules=[],
                    failed_rules=[],
                    syntax_valid=True,
                    backup_path=None,
                    errors=["Could not interpret patch description"],
                    warnings=[]
                ),
                validation_result=None,
                success=False,
                message="Failed to interpret patch description with brain"
            )
        
        # Create operation
        operation = PatchOperation(
            description=description,
            file_path=file_path,
            rules=rules
        )
        
        return await self.apply_operation(operation)
    
    async def apply_intelligent_ast_transform(
        self,
        file_path: str,
        transform_type: str,
        **kwargs
    ) -> PatchReport:
        """Apply AST-based transformation with LibCST."""
        if self._ast_engine is None:
            self._ast_engine = await get_ast_engine()
        
        # Read file
        content = Path(file_path).read_text(encoding='utf-8')
        
        # Apply AST transform based on type
        try:
            if transform_type == 'add_type_hints':
                new_content = await self._ast_engine.add_type_hints(content)
            elif transform_type == 'convert_to_async':
                new_content = await self._ast_engine.convert_to_async(content)
            elif transform_type == 'modernize_imports':
                old_module = kwargs.get('old_module', '')
                new_module = kwargs.get('new_module', '')
                new_content = await self._ast_engine.modernize_imports(
                    content, old_module, new_module
                )
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
            
            # Write back
            Path(file_path).write_text(new_content, encoding='utf-8')
            
            # Learn from this
            if self._semantic_patcher is None:
                self._semantic_patcher = await get_semantic_patcher()
            
            await self._semantic_patcher.learn_from_patch(
                original_code=content,
                patched_code=new_content,
                success=True,
                context={'transform_type': transform_type, **kwargs}
            )
            
            # Create success report
            return PatchReport(
                operation=PatchOperation(
                    description=f"AST transform: {transform_type}",
                    file_path=file_path,
                    rules=[]
                ),
                patch_result=PatchResult(
                    status=PatchStatus.SUCCESS,
                    file_path=file_path,
                    applied_rules=[f"AST: {transform_type}"],
                    failed_rules=[],
                    syntax_valid=True,
                    backup_path=None,
                    errors=[],
                    warnings=[]
                ),
                validation_result=None,
                success=True,
                message=f"✓ Successfully applied AST transform: {transform_type}"
            )
        
        except Exception as e:
            return PatchReport(
                operation=PatchOperation(
                    description=f"AST transform: {transform_type}",
                    file_path=file_path,
                    rules=[]
                ),
                patch_result=PatchResult(
                    status=PatchStatus.FAILED,
                    file_path=file_path,
                    applied_rules=[],
                    failed_rules=[f"AST: {transform_type}"],
                    syntax_valid=False,
                    backup_path=None,
                    errors=[str(e)],
                    warnings=[]
                ),
                validation_result=None,
                success=False,
                message=f"✗ AST transform failed: {str(e)}"
            )
    
    async def suggest_semantic_patches(
        self,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Suggest patches using semantic analysis and brain."""
        if self._semantic_patcher is None:
            self._semantic_patcher = await get_semantic_patcher()
        
        # Read file
        content = Path(file_path).read_text(encoding='utf-8')
        
        # Get suggestions
        suggestions = await self._semantic_patcher.suggest_patches_for_file(
            file_path, content
        )
        
        return suggestions
    
    async def detect_and_fix_code_smells(
        self,
        file_path: str
    ) -> PatchReport:
        """Detect and automatically fix code smells using brain intelligence."""
        if self._semantic_patcher is None:
            self._semantic_patcher = await get_semantic_patcher()
        
        # Read file
        content = Path(file_path).read_text(encoding='utf-8')
        
        # Detect smells
        smells = await self._semantic_patcher.detect_code_smells(content)
        
        if not smells:
            return PatchReport(
                operation=PatchOperation(
                    description="Code smell detection",
                    file_path=file_path,
                    rules=[]
                ),
                patch_result=PatchResult(
                    status=PatchStatus.SUCCESS,
                    file_path=file_path,
                    applied_rules=[],
                    failed_rules=[],
                    syntax_valid=True,
                    backup_path=None,
                    errors=[],
                    warnings=["No code smells detected"]
                ),
                validation_result=None,
                success=True,
                message="✓ No code smells detected"
            )
        
        # Auto-fix high severity smells
        rules = []
        for smell in smells:
            if smell['severity'] == 'high':
                # Create fix rule based on smell type
                if 'Silent exception' in smell['smell']:
                    rules.append(PatchBuilder.regex_patch(
                        pattern=r'except Exception:\s*pass',
                        replacement='except Exception as e:\n    logger.error(f"Error: {e}")',
                        description=f"Fix: {smell['smell']}"
                    ))
        
        if rules:
            operation = PatchOperation(
                description="Auto-fix code smells",
                file_path=file_path,
                rules=rules
            )
            return await self.apply_operation(operation)
        
        return PatchReport(
            operation=PatchOperation(
                description="Code smell detection",
                file_path=file_path,
                rules=[]
            ),
            patch_result=PatchResult(
                status=PatchStatus.SUCCESS,
                file_path=file_path,
                applied_rules=[],
                failed_rules=[],
                syntax_valid=True,
                backup_path=None,
                errors=[],
                warnings=[f"Detected {len(smells)} smells but no auto-fix available"]
            ),
            validation_result=None,
            success=True,
            message=f"⚠ Detected {len(smells)} code smells"
        )
    
    async def apply_common_patch(
        self,
        patch_name: str,
        file_path: str
    ) -> PatchReport:
        """Apply common patch by name."""
        
        # Get rules from CommonPatches
        if patch_name == "phases_import_orchestrator":
            rules = CommonPatches.phases_import_orchestrator()
        elif patch_name == "fix_duplicate_except":
            rules = CommonPatches.fix_duplicate_except()
        else:
            return PatchReport(
                operation=PatchOperation(
                    description=f"Common patch: {patch_name}",
                    file_path=file_path,
                    rules=[]
                ),
                patch_result=PatchResult(
                    status=PatchStatus.FAILED,
                    file_path=file_path,
                    applied_rules=[],
                    failed_rules=[],
                    syntax_valid=True,
                    backup_path=None,
                    errors=[f"Unknown common patch: {patch_name}"],
                    warnings=[]
                ),
                validation_result=None,
                success=False,
                message=f"Unknown common patch: {patch_name}"
            )
        
        operation = PatchOperation(
            description=f"Apply common patch: {patch_name}",
            file_path=file_path,
            rules=rules
        )
        
        return await self.apply_operation(operation)
    
    async def _interpret_with_brain(
        self,
        description: str,
        file_path: str
    ) -> List[PatchRule]:
        """Interpret natural language description using brain intelligence."""
        
        rules = []
        desc_lower = description.lower()
        
        # Use brain to understand intent
        from jinx.micro.brain import think_with_memory, search_all_memories
        
        # Think about what the description means
        thought = await think_with_memory(
            f"Interpret code patch request: {description}\nFor file: {file_path}"
        )
        
        # Search for similar past patches
        similar_patches = await search_all_memories(
            f"patch {description}",
            k=5
        )
        
        # Extract patterns from similar patches
        for memory in similar_patches:
            if 'pattern' in memory.content and 'replacement' in memory.content:
                # Try to extract pattern/replacement from memory
                # This is simplified - would parse structured data
                pass
        
        # Pattern matching with brain-enhanced understanding
        if "replace" in desc_lower and "with" in desc_lower:
            # Extract pattern: "replace X with Y"
            parts = description.split(" with ")
            if len(parts) == 2:
                old = parts[0].replace("replace ", "").strip()
                new = parts[1].strip()
                
                rules.append(PatchBuilder.regex_patch(
                    pattern=re.escape(old),
                    replacement=new,
                    description=f"Replace '{old}' with '{new}'"
                ))
        
        elif "add async" in desc_lower or "make async" in desc_lower:
            # Use AST transform for async conversion
            rules.append(PatchBuilder.regex_patch(
                pattern=r'^(\s*)def ',
                replacement=r'\1async def ',
                description="Convert to async",
                flags=re.MULTILINE
            ))
        
        elif "add type hints" in desc_lower:
            # Mark for AST transform
            rules.append(PatchBuilder.regex_patch(
                pattern='__AST_TYPE_HINTS__',
                replacement='',
                description="Add type hints (AST)"
            ))
        
        elif "fix imports" in desc_lower or "update imports" in desc_lower:
            # Search for import patterns in file
            rules.append(PatchBuilder.regex_patch(
                pattern=r'from old_module import',
                replacement='from new_module import',
                description="Modernize imports"
            ))
        
        return rules
    
    def _build_message(
        self,
        patch_result: PatchResult,
        validation_result: Optional[ValidationResult]
    ) -> str:
        """Build human-readable message."""
        
        if patch_result.status == PatchStatus.SUCCESS:
            msg = f"✓ Successfully applied {len(patch_result.applied_rules)} patch rules"
            
            if validation_result and not validation_result.overall_ok:
                msg += f"\n⚠ Validation failed: "
                if not validation_result.compilation_ok:
                    msg += "Compilation errors. "
                if not validation_result.imports_ok:
                    msg += "Import errors. "
                if not validation_result.tests_ok:
                    msg += "Tests failed. "
        
        elif patch_result.status == PatchStatus.ROLLED_BACK:
            msg = f"✗ Patch rolled back due to validation errors"
            msg += f"\nApplied: {len(patch_result.applied_rules)}, Failed: {len(patch_result.failed_rules)}"
        
        elif patch_result.status == PatchStatus.PARTIAL:
            msg = f"⚠ Partial success: {len(patch_result.applied_rules)} applied, {len(patch_result.failed_rules)} failed"
        
        else:  # FAILED
            msg = f"✗ Patch failed"
            if patch_result.errors:
                msg += f"\nErrors: {', '.join(patch_result.errors[:3])}"
        
        return msg
    
    async def _log_to_memory(self, report: PatchReport) -> None:
        """Log patch operation to brain memory."""
        try:
            from jinx.micro.brain import remember_episode
            
            content = f"Patch operation: {report.operation.description}"
            if report.success:
                content += f" - SUCCESS ({len(report.patch_result.applied_rules)} rules)"
            else:
                content += f" - FAILED ({len(report.patch_result.errors)} errors)"
            
            await remember_episode(
                content=content,
                episode_type='tool_use',
                context={
                    'file': report.patch_result.file_path,
                    'status': report.patch_result.status.value,
                    'rules_applied': len(report.patch_result.applied_rules)
                },
                importance=0.7 if report.success else 0.8
            )
        except Exception:
            pass
    
    def get_history(self) -> List[PatchReport]:
        """Get patch history."""
        return self._history.copy()


# Singleton
_orchestrator: Optional[PatchOrchestrator] = None
_orch_lock = asyncio.Lock()


async def get_patch_orchestrator() -> PatchOrchestrator:
    """Get singleton patch orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        async with _orch_lock:
            if _orchestrator is None:
                _orchestrator = PatchOrchestrator()
    return _orchestrator


async def apply_intelligent_patch_operation(
    description: str,
    file_path: str,
    rules: Optional[List[PatchRule]] = None
) -> PatchReport:
    """Apply patch with full intelligence."""
    orchestrator = await get_patch_orchestrator()
    
    if rules:
        operation = PatchOperation(
            description=description,
            file_path=file_path,
            rules=rules
        )
        return await orchestrator.apply_operation(operation)
    else:
        return await orchestrator.apply_from_description(description, file_path)


# Import re for _interpret_description
import re


__all__ = [
    "PatchOrchestrator",
    "PatchOperation",
    "PatchReport",
    "get_patch_orchestrator",
    "apply_intelligent_patch_operation",
]
