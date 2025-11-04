"""CLI tool for applying intelligent patches.

Usage:
    python -m jinx.tools.apply_patch --file path/to/file.py --patch phases_import
    python -m jinx.tools.apply_patch --file path/to/file.py --pattern "old" --replacement "new"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jinx.micro.code import (
    PatchBuilder,
    PatchOperation,
    get_patch_orchestrator,
    CommonPatches,
)


async def apply_common_patch(file_path: str, patch_name: str):
    """Apply common patch by name."""
    orchestrator = await get_patch_orchestrator()
    report = await orchestrator.apply_common_patch(patch_name, file_path)
    
    print(f"\n{'='*60}")
    print(f"PATCH REPORT: {report.operation.description}")
    print(f"{'='*60}")
    print(f"File: {report.patch_result.file_path}")
    print(f"Status: {report.patch_result.status.value}")
    print(f"\n{report.message}")
    
    if report.patch_result.applied_rules:
        print(f"\n✓ Applied rules:")
        for rule in report.patch_result.applied_rules:
            print(f"  - {rule}")
    
    if report.patch_result.failed_rules:
        print(f"\n✗ Failed rules:")
        for rule in report.patch_result.failed_rules:
            print(f"  - {rule}")
    
    if report.patch_result.errors:
        print(f"\n✗ Errors:")
        for error in report.patch_result.errors:
            print(f"  - {error}")
    
    if report.patch_result.warnings:
        print(f"\n⚠ Warnings:")
        for warning in report.patch_result.warnings:
            print(f"  - {warning}")
    
    if report.validation_result:
        print(f"\nValidation:")
        print(f"  Compilation: {'✓' if report.validation_result.compilation_ok else '✗'}")
        print(f"  Imports: {'✓' if report.validation_result.imports_ok else '✗'}")
        
        if report.validation_result.compilation_errors:
            print(f"  Compilation errors:")
            for error in report.validation_result.compilation_errors:
                print(f"    - {error}")
    
    if report.patch_result.backup_path:
        print(f"\nBackup saved: {report.patch_result.backup_path}")
    
    print(f"{'='*60}\n")
    
    return 0 if report.success else 1


async def apply_regex_patch(file_path: str, pattern: str, replacement: str, description: str):
    """Apply regex patch."""
    rules = [
        PatchBuilder.regex_patch(
            pattern=pattern,
            replacement=replacement,
            description=description or f"Replace '{pattern}' with '{replacement}'"
        )
    ]
    
    operation = PatchOperation(
        description=description or "Custom regex patch",
        file_path=file_path,
        rules=rules
    )
    
    orchestrator = await get_patch_orchestrator()
    report = await orchestrator.apply_operation(operation)
    
    print(f"\n{'='*60}")
    print(f"PATCH REPORT: {report.operation.description}")
    print(f"{'='*60}")
    print(f"File: {report.patch_result.file_path}")
    print(f"Status: {report.patch_result.status.value}")
    print(f"\n{report.message}")
    
    if report.patch_result.backup_path:
        print(f"\nBackup saved: {report.patch_result.backup_path}")
    
    print(f"{'='*60}\n")
    
    return 0 if report.success else 1


async def main():
    parser = argparse.ArgumentParser(
        description="Apply intelligent patches to code files"
    )
    
    parser.add_argument(
        '--file',
        required=True,
        help='Path to file to patch'
    )
    
    parser.add_argument(
        '--patch',
        help='Name of common patch to apply'
    )
    
    parser.add_argument(
        '--pattern',
        help='Regex pattern to replace'
    )
    
    parser.add_argument(
        '--replacement',
        help='Replacement string'
    )
    
    parser.add_argument(
        '--description',
        help='Description of the patch'
    )
    
    parser.add_argument(
        '--list-patches',
        action='store_true',
        help='List available common patches'
    )
    
    args = parser.parse_args()
    
    if args.list_patches:
        print("\nAvailable common patches:")
        print("  - phases_import_orchestrator: Update orchestrator.py imports")
        print("  - fix_duplicate_except: Fix duplicate except blocks")
        print()
        return 0
    
    if args.patch:
        return await apply_common_patch(args.file, args.patch)
    
    elif args.pattern and args.replacement:
        return await apply_regex_patch(
            args.file,
            args.pattern,
            args.replacement,
            args.description or "Custom patch"
        )
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
