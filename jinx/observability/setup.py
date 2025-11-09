from __future__ import annotations

"""
Optional OpenTelemetry setup. If opentelemetry SDK is installed and JINX_OTEL_SETUP=1,
configures a simple console exporter for traces.
"""

def setup_otel() -> None:
    try:
        import os
        if str(os.getenv("JINX_OTEL_SETUP", "0")).lower() in ("", "0", "false", "off", "no"):
            return
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        # Exporter choice
        exporter_name = str(os.getenv("JINX_OTEL_EXPORTER", "console")).lower()
        if exporter_name == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # type: ignore
            exporter = ConsoleSpanExporter()
        else:
            # Default to console if unknown
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # type: ignore
            exporter = ConsoleSpanExporter()
        resource = Resource.create({"service.name": "jinx"})
        provider = TracerProvider(resource=resource)
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    except Exception:
        return
