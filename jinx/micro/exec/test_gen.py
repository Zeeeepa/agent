from __future__ import annotations

import os
import ast
from typing import List, Tuple

from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.async_utils.fs import read_text_abs_thread


def _to_module(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").rstrip("/")
    if rel.endswith("/__init__.py"):
        rel = rel[:-12]
    elif rel.endswith(".py"):
        rel = rel[:-3]
    parts = [p for p in rel.split("/") if p]
    return ".".join(parts)


async def gen_unit_test_stub(symbol: str, defs: List[Tuple[str, int]], *, preview_chars: int = 1200) -> str:
    """Generate a minimal unittest stub for a symbol using its first definition location.

    Returns a test module text (string). This is a conservative stub intended for human augmentation.
    """
    if not defs:
        return f"# test stub for {symbol}\nimport unittest\n\nclass Test{symbol.capitalize()}(unittest.TestCase):\n    def test_smoke(self):\n        self.assertTrue(callable({repr(symbol)}))\n\nif __name__ == '__main__':\n    unittest.main()\n"
    root = _resolve_root()
    rel, _line = defs[0]
    ap = os.path.join(root, rel)
    try:
        src = await read_text_abs_thread(ap)
    except Exception:
        src = ""
    mod_name = _to_module(rel)
    # Inspect signature if possible
    args_hint = ""
    try:
        tree = ast.parse(src or "")
        for node in getattr(tree, "body", []) or []:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
                # Build placeholder args for non-default parameters
                params = []
                for arg in node.args.args[len(node.args.posonlyargs):]:
                    if arg.arg == "self":
                        continue
                    params.append("None")
                args_hint = ", ".join(params)
                break
    except Exception:
        args_hint = ""
    test_code = (
        f"import unittest\n"
        f"from {mod_name} import {symbol}\n\n"
        f"class Test{symbol.capitalize()}(unittest.TestCase):\n"
        f"    def test_smoke(self):\n"
        f"        try:\n"
        f"            _ = {symbol}({args_hint})\n"
        f"        except Exception as e:\n"
        f"            self.fail('raised: ' + str(e))\n\n"
        f"if __name__ == '__main__':\n"
        f"    unittest.main()\n"
    )
    return test_code[:preview_chars]
