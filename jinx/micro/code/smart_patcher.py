"""Smart Patcher - intelligent code patching with validation and rollback.

Applies regex or AST-based patches with automatic validation and error detection.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class PatchMethod(Enum):
    """Patch application method."""
    REGEX = "regex"
    AST = "ast"
    MULTILINE = "multiline"


class PatchStatus(Enum):
    """Patch application status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"


@dataclass
class PatchRule:
    """Single patch rule."""
    pattern: str
    replacement: str
    method: PatchMethod
    description: str
    flags: int = 0
    validator: Optional[Callable[[str], bool]] = None


@dataclass
class PatchResult:
    """Result of patch application."""
    status: PatchStatus
    file_path: str
    applied_rules: List[str]
    failed_rules: List[str]
    syntax_valid: bool
    backup_path: Optional[str]
    errors: List[str]
    warnings: List[str]


class SmartPatcher:
    """Intelligent code patcher with validation."""
    
    def __init__(self):
        self._backup_dir = Path(tempfile.gettempdir()) / "jinx_patch_backups"
        self._backup_dir.mkdir(exist_ok=True)
    
    async def apply_patch_set(
        self,
        file_path: str,
        rules: List[PatchRule],
        *,
        validate: bool = True,
        rollback_on_error: bool = True
    ) -> PatchResult:
        """Apply set of patch rules with validation."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return PatchResult(
                status=PatchStatus.FAILED,
                file_path=str(file_path),
                applied_rules=[],
                failed_rules=[rule.description for rule in rules],
                syntax_valid=False,
                backup_path=None,
                errors=[f"File not found: {file_path}"],
                warnings=[]
            )
        
        # Create backup
        backup_path = await self._create_backup(file_path)
        
        # Read original content
        original_content = file_path.read_text(encoding='utf-8')
        current_content = original_content
        
        applied_rules = []
        failed_rules = []
        errors = []
        warnings = []
        
        # Apply each rule
        for rule in rules:
            try:
                new_content, applied = await self._apply_single_rule(
                    current_content,
                    rule
                )
                
                if applied:
                    current_content = new_content
                    applied_rules.append(rule.description)
                else:
                    failed_rules.append(rule.description)
                    warnings.append(f"Rule did not match: {rule.description}")
            except Exception as e:
                failed_rules.append(rule.description)
                errors.append(f"Error applying {rule.description}: {str(e)}")
        
        # Check if any changes were made
        if current_content == original_content:
            return PatchResult(
                status=PatchStatus.FAILED,
                file_path=str(file_path),
                applied_rules=[],
                failed_rules=failed_rules,
                syntax_valid=True,
                backup_path=str(backup_path),
                errors=errors or ["No changes applied"],
                warnings=warnings
            )
        
        # Validate syntax if requested
        syntax_valid = True
        if validate:
            syntax_valid, syntax_errors = await self._validate_syntax(
                current_content,
                file_path.suffix
            )
            
            if not syntax_valid:
                errors.extend(syntax_errors)
                
                if rollback_on_error:
                    # Rollback
                    file_path.write_text(original_content, encoding='utf-8')
                    return PatchResult(
                        status=PatchStatus.ROLLED_BACK,
                        file_path=str(file_path),
                        applied_rules=applied_rules,
                        failed_rules=failed_rules,
                        syntax_valid=False,
                        backup_path=str(backup_path),
                        errors=errors,
                        warnings=warnings
                    )
        
        # Write patched content
        file_path.write_text(current_content, encoding='utf-8')
        
        # Determine status
        if failed_rules:
            status = PatchStatus.PARTIAL if applied_rules else PatchStatus.FAILED
        else:
            status = PatchStatus.SUCCESS
        
        return PatchResult(
            status=status,
            file_path=str(file_path),
            applied_rules=applied_rules,
            failed_rules=failed_rules,
            syntax_valid=syntax_valid,
            backup_path=str(backup_path),
            errors=errors,
            warnings=warnings
        )
    
    async def _apply_single_rule(
        self,
        content: str,
        rule: PatchRule
    ) -> Tuple[str, bool]:
        """Apply single patch rule."""
        if rule.method == PatchMethod.REGEX:
            return await self._apply_regex(content, rule)
        elif rule.method == PatchMethod.MULTILINE:
            return await self._apply_multiline(content, rule)
        elif rule.method == PatchMethod.AST:
            return await self._apply_ast(content, rule)
        else:
            return content, False
    
    async def _apply_regex(
        self,
        content: str,
        rule: PatchRule
    ) -> Tuple[str, bool]:
        """Apply regex-based patch."""
        # Compile pattern
        pattern = re.compile(rule.pattern, rule.flags or re.DOTALL)
        
        # Check if pattern matches
        if not pattern.search(content):
            return content, False
        
        # Apply replacement
        new_content = pattern.sub(rule.replacement, content)
        
        # Check if actually changed
        if new_content == content:
            return content, False
        
        # Validate if validator provided
        if rule.validator and not rule.validator(new_content):
            return content, False
        
        return new_content, True
    
    async def _apply_multiline(
        self,
        content: str,
        rule: PatchRule
    ) -> Tuple[str, bool]:
        """Apply multiline pattern patch with whitespace normalization."""
        # Normalize whitespace in pattern and content for matching
        pattern_normalized = re.sub(r'\s+', r'\\s+', rule.pattern)
        pattern = re.compile(pattern_normalized, re.DOTALL)
        
        if not pattern.search(content):
            return content, False
        
        new_content = pattern.sub(rule.replacement, content)
        
        if new_content == content:
            return content, False
        
        return new_content, True
    
    async def _apply_ast(
        self,
        content: str,
        rule: PatchRule
    ) -> Tuple[str, bool]:
        """Apply AST-based patch (for Python files)."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content, False
        
        # AST transformation would go here
        # For now, fallback to regex
        return await self._apply_regex(content, rule)
    
    async def _validate_syntax(
        self,
        content: str,
        file_extension: str
    ) -> Tuple[bool, List[str]]:
        """Validate syntax of patched content."""
        errors = []
        
        if file_extension == '.py':
            # Python syntax validation
            try:
                ast.parse(content)
                return True, []
            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
                return False, errors
        
        # For other files, just check it's not empty
        if not content.strip():
            errors.append("File is empty after patching")
            return False, errors
        
        return True, []
    
    async def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file before patching."""
        import time
        timestamp = int(time.time())
        backup_name = f"{file_path.name}.{timestamp}.backup"
        backup_path = self._backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        
        return backup_path
    
    async def restore_backup(self, backup_path: str, target_path: str) -> bool:
        """Restore file from backup."""
        try:
            shutil.copy2(backup_path, target_path)
            return True
        except Exception:
            return False


class PatchBuilder:
    """Builder for creating patch rules."""
    
    @staticmethod
    def regex_patch(
        pattern: str,
        replacement: str,
        description: str,
        *,
        flags: int = 0
    ) -> PatchRule:
        """Create regex patch rule."""
        return PatchRule(
            pattern=pattern,
            replacement=replacement,
            method=PatchMethod.REGEX,
            description=description,
            flags=flags
        )
    
    @staticmethod
    def multiline_patch(
        pattern: str,
        replacement: str,
        description: str
    ) -> PatchRule:
        """Create multiline patch rule."""
        return PatchRule(
            pattern=pattern,
            replacement=replacement,
            method=PatchMethod.MULTILINE,
            description=description
        )
    
    @staticmethod
    def import_replacement(
        old_imports: str,
        new_imports: str,
        description: str = "Update imports"
    ) -> PatchRule:
        """Create import replacement patch."""
        # Escape special regex chars in old_imports
        pattern = re.escape(old_imports)
        pattern = pattern.replace(r'\(', r'\(').replace(r'\)', r'\)')
        pattern = pattern.replace(r'\ ', r'\s*')
        
        return PatchRule(
            pattern=pattern,
            replacement=new_imports,
            method=PatchMethod.REGEX,
            description=description,
            flags=re.DOTALL
        )
    
    @staticmethod
    def function_call_replacement(
        old_call: str,
        new_call: str,
        description: str = "Update function call"
    ) -> PatchRule:
        """Create function call replacement patch."""
        # Create flexible pattern for function call
        pattern = old_call.replace('(', r'\(').replace(')', r'\)')
        
        return PatchRule(
            pattern=pattern,
            replacement=new_call,
            method=PatchMethod.REGEX,
            description=description
        )


# Example patch sets for common operations
class CommonPatches:
    """Common patch patterns."""
    
    @staticmethod
    def phases_import_orchestrator() -> List[PatchRule]:
        """Patch for orchestrator.py phases import."""
        return [
            PatchBuilder.regex_patch(
                pattern=r'from jinx\.micro\.conversation\.phases import \([\s\S]*?\)',
                replacement=(
                    "from jinx.micro.conversation.phases import (\n"
                    "    call_llm as _phase_llm,\n"
                    "    execute_blocks as _phase_exec,\n"
                    "    build_runtime_base_ctx as _phase_base_ctx,\n"
                    "    build_runtime_mem_ctx as _phase_mem_ctx,\n"
                    "    build_project_context_enriched as _phase_proj_ctx,\n"
                    ")"
                ),
                description="Update phases import"
            ),
            PatchBuilder.regex_patch(
                pattern=r'base_ctx_task\s*=\s*asyncio\.create_task\(build_context_for\(eff_q\)\)',
                replacement='base_ctx_task = asyncio.create_task(_phase_base_ctx(eff_q))',
                description="Update base_ctx_task creation"
            ),
            PatchBuilder.regex_patch(
                pattern=r'mem_ctx_task\s*=\s*asyncio\.create_task\(_build_mem_ctx\(eff_q\)\)',
                replacement='mem_ctx_task = asyncio.create_task(_phase_mem_ctx(eff_q))',
                description="Update mem_ctx_task creation"
            ),
            PatchBuilder.regex_patch(
                pattern=r'proj_ctx_task:\s*asyncio\.Task\[str\]\s*=\s*asyncio\.create_task\(_build_proj_ctx_enriched\([^\)]*\)\)',
                replacement='proj_ctx_task: asyncio.Task[str] = asyncio.create_task(_phase_proj_ctx(_q, user_text=x or "", synth=synth or ""))',
                description="Update proj_ctx_task creation"
            )
        ]
    
    @staticmethod
    def fix_duplicate_except() -> List[PatchRule]:
        """Fix duplicate except blocks."""
        return [
            PatchBuilder.multiline_patch(
                pattern=r'except Exception:\s*pass\s*pairs = parse_tagged_blocks\(out, code_id\)\s*except Exception:\s*pairs = \[\]',
                replacement=(
                    'try:\n'
                    '        pairs = parse_tagged_blocks(out, code_id)\n'
                    '    except Exception:\n'
                    '        pairs = []'
                ),
                description="Fix duplicate except blocks"
            )
        ]


# Singleton
_patcher: Optional[SmartPatcher] = None
_patcher_lock = asyncio.Lock()


async def get_smart_patcher() -> SmartPatcher:
    """Get singleton smart patcher."""
    global _patcher
    if _patcher is None:
        async with _patcher_lock:
            if _patcher is None:
                _patcher = SmartPatcher()
    return _patcher


async def apply_intelligent_patch(
    file_path: str,
    rules: List[PatchRule],
    **kwargs
) -> PatchResult:
    """Apply intelligent patch with validation."""
    patcher = await get_smart_patcher()
    return await patcher.apply_patch_set(file_path, rules, **kwargs)


__all__ = [
    "SmartPatcher",
    "PatchBuilder",
    "PatchRule",
    "PatchResult",
    "PatchMethod",
    "PatchStatus",
    "CommonPatches",
    "get_smart_patcher",
    "apply_intelligent_patch",
]
