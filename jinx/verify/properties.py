from __future__ import annotations

from typing import Tuple


def run_basic_properties(code: str) -> Tuple[bool, str]:
    """Run lightweight property-based checks if Hypothesis is available.

    - Parsing the code should be stable under round-trip (parse -> unparse -> parse).
    - The function returns (ok, message).
    """
    try:
        import ast
        try:
            import astunparse  # type: ignore
        except Exception:
            astunparse = None  # type: ignore
        try:
            from hypothesis import given, settings  # type: ignore
            from hypothesis.strategies import text  # type: ignore
        except Exception:
            # Hypothesis not available: soft pass
            return True, "hypothesis not available"

        # Deterministic round-trip for the edited code itself
        try:
            m1 = ast.parse(code)
            s1 = ast.unparse(m1) if hasattr(ast, "unparse") else (astunparse.unparse(m1) if astunparse else code)
            m2 = ast.parse(s1)
            _ = m2
        except Exception as e:
            return False, f"roundtrip failed: {e}"

        # A tiny generated check to exercise parser stability
        @settings(max_examples=5, deadline=None)
        @given(text(min_size=0, max_size=20))
        def _prop_roundtrip(s: str) -> None:
            src = f"def _f():\n    x = 1\n    # {s}"
            t1 = ast.parse(src)
            s2 = ast.unparse(t1) if hasattr(ast, "unparse") else (astunparse.unparse(t1) if astunparse else src)
            t2 = ast.parse(s2)
            assert isinstance(t2, ast.AST)

        try:
            _prop_roundtrip()  # type: ignore[misc]
        except Exception as e:
            return False, f"hypothesis property failed: {e}"
        return True, ""
    except Exception as e:
        return False, f"properties error: {e}"
