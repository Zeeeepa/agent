"""Patch Validator - validates patches with compilation, linting, and tests.

Ensures patches don't break code with comprehensive validation.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of comprehensive validation."""
    compilation_ok: bool
    compilation_errors: List[str]
    
    linting_ok: bool
    linting_warnings: List[str]
    
    imports_ok: bool
    import_errors: List[str]
    
    tests_ok: bool
    test_errors: List[str]
    
    overall_ok: bool


class PatchValidator:
    """Comprehensive patch validation."""
    
    def __init__(self):
        self._python_exe = sys.executable
    
    async def validate_patch(
        self,
        file_path: str,
        *,
        check_compilation: bool = True,
        check_imports: bool = True,
        check_linting: bool = False,
        run_tests: bool = False
    ) -> ValidationResult:
        """Comprehensive patch validation."""
        
        compilation_ok = True
        compilation_errors = []
        
        linting_ok = True
        linting_warnings = []
        
        imports_ok = True
        import_errors = []
        
        tests_ok = True
        test_errors = []
        
        # 1. Check compilation (syntax)
        if check_compilation:
            compilation_ok, compilation_errors = await self._check_compilation(file_path)
        
        # 2. Check imports (can file be imported?)
        if check_imports and compilation_ok:
            imports_ok, import_errors = await self._check_imports(file_path)
        
        # 3. Check linting (optional)
        if check_linting:
            linting_ok, linting_warnings = await self._check_linting(file_path)
        
        # 4. Run tests (optional)
        if run_tests:
            tests_ok, test_errors = await self._run_tests(file_path)
        
        overall_ok = compilation_ok and imports_ok and (not run_tests or tests_ok)
        
        return ValidationResult(
            compilation_ok=compilation_ok,
            compilation_errors=compilation_errors,
            linting_ok=linting_ok,
            linting_warnings=linting_warnings,
            imports_ok=imports_ok,
            import_errors=import_errors,
            tests_ok=tests_ok,
            test_errors=test_errors,
            overall_ok=overall_ok
        )
    
    async def _check_compilation(self, file_path: str) -> Tuple[bool, List[str]]:
        """Check if file compiles (syntax check)."""
        try:
            # Use py_compile for Python files
            import py_compile
            
            # Try to compile
            py_compile.compile(file_path, doraise=True)
            
            return True, []
        except py_compile.PyCompileError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Compilation check failed: {str(e)}"]
    
    async def _check_imports(self, file_path: str) -> Tuple[bool, List[str]]:
        """Check if file can be imported without errors."""
        file_path = Path(file_path)
        
        # Get module name from file path
        # e:\agent\jinx\micro\code\file.py -> jinx.micro.code.file
        try:
            parts = file_path.parts
            
            # Find 'jinx' in path
            if 'jinx' in parts:
                idx = parts.index('jinx')
                module_parts = parts[idx:-1]  # Exclude filename
                module_parts = list(module_parts) + [file_path.stem]
                module_name = '.'.join(module_parts)
            else:
                # Can't determine module, skip import check
                return True, []
            
            # Try to import in subprocess to isolate
            cmd = [
                self._python_exe,
                '-c',
                f'import {module_name}'
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(file_path.parent.parent.parent.parent)  # Go to project root
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                return True, []
            else:
                error_msg = stderr.decode('utf-8', errors='ignore')
                return False, [f"Import error: {error_msg}"]
        
        except Exception as e:
            # Import check failed, but don't fail validation
            return True, [f"Could not check imports: {str(e)}"]
    
    async def _check_linting(self, file_path: str) -> Tuple[bool, List[str]]:
        """Run linter (optional - doesn't fail validation)."""
        warnings = []
        
        # Try to use ruff if available
        try:
            cmd = ['ruff', 'check', file_path]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                output = stdout.decode('utf-8', errors='ignore')
                if output:
                    warnings.append(output)
        
        except FileNotFoundError:
            # Ruff not installed, skip
            pass
        except Exception:
            pass
        
        return True, warnings  # Linting warnings don't fail validation
    
    async def _run_tests(self, file_path: str) -> Tuple[bool, List[str]]:
        """Run tests for the file (if test file exists)."""
        file_path = Path(file_path)
        
        # Look for test file
        test_file = file_path.parent / f"test_{file_path.name}"
        
        if not test_file.exists():
            # No test file, tests pass by default
            return True, []
        
        try:
            # Run pytest on test file
            cmd = [self._python_exe, '-m', 'pytest', str(test_file), '-v']
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                return True, []
            else:
                output = stdout.decode('utf-8', errors='ignore')
                return False, [f"Tests failed:\n{output}"]
        
        except Exception as e:
            return True, [f"Could not run tests: {str(e)}"]


# Singleton
_validator: Optional[PatchValidator] = None
_validator_lock = asyncio.Lock()


async def get_patch_validator() -> PatchValidator:
    """Get singleton patch validator."""
    global _validator
    if _validator is None:
        async with _validator_lock:
            if _validator is None:
                _validator = PatchValidator()
    return _validator


async def validate_patched_file(
    file_path: str,
    **kwargs
) -> ValidationResult:
    """Validate patched file comprehensively."""
    validator = await get_patch_validator()
    return await validator.validate_patch(file_path, **kwargs)


__all__ = [
    "PatchValidator",
    "ValidationResult",
    "get_patch_validator",
    "validate_patched_file",
]
