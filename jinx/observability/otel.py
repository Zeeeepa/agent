from __future__ import annotations

from contextlib import contextmanager


try:
    from opentelemetry import trace as _otel_trace  # type: ignore
except Exception:  # opentelemetry is optional
    _otel_trace = None  # type: ignore


@contextmanager
def span(name: str, attrs: dict | None = None):
    """Best-effort OpenTelemetry span context manager.

    If OpenTelemetry is not installed or errors occur, behaves as a no-op.
    """
    if _otel_trace is None:
        # No OTEL: no-op span
        yield
        return
    ctx = None
    try:
        tracer = _otel_trace.get_tracer("jinx")
        ctx = tracer.start_as_current_span(name)
        sp = ctx.__enter__()
        # Set attributes if provided (best-effort)
        if attrs:
            try:
                cur = _otel_trace.get_current_span()
                target = cur or sp
                for k, v in (attrs or {}).items():
                    try:
                        target.set_attribute(str(k), v)
                    except Exception:
                        pass
            except Exception:
                pass
        yield
    except Exception:
        # Any OTEL error -> no-op
        yield
    finally:
        try:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        except Exception:
            pass
