"""Crash Diagnostics - Advanced crash detection and analysis.

Automatically detects and logs:
- Unexpected exits
- Uncaught exceptions
- Signal handlers (SIGTERM, SIGINT, etc.)
- State dumps before crash
- Last operations trace
- Memory/CPU at crash time
- Full stack traces
"""

from __future__ import annotations

import asyncio
import atexit
import os
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OperationTrace:
    """Single operation in trace buffer."""
    timestamp: float
    operation: str
    details: Dict[str, Any] = field(default_factory=dict)
    success: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class CrashReport:
    """Complete crash report."""
    crash_time: float
    exit_code: int
    reason: str
    
    # Traces
    last_operations: List[OperationTrace]
    stack_traces: List[str]
    
    # State
    pulse: int
    shutdown_requested: bool
    active_tasks: int
    
    # System
    memory_mb: float
    cpu_percent: float
    
    # Extra
    exception_info: Optional[Dict[str, Any]] = None
    signal_info: Optional[Dict[str, Any]] = None


class CrashDiagnostics:
    """Advanced crash diagnostics system."""
    
    def __init__(self):
        self._trace_buffer: deque[OperationTrace] = deque(maxlen=100)
        self._installed = False
        self._crash_log_path = ".jinx/crash_reports"
        self._last_operation: Optional[str] = None
        self._operation_start: Optional[float] = None
        
        # Statistics
        self._total_operations = 0
        self._failed_operations = 0
        
        # Shutdown detection
        self._normal_shutdown = False
        self._exit_reason: Optional[str] = None
    
    def install_handlers(self):
        """Install all crash detection handlers."""
        if self._installed:
            return
        
        self._installed = True
        
        # Exit handler
        atexit.register(self._on_exit)
        
        # Exception hook
        sys.excepthook = self._exception_hook
        
        # Signal handlers (Unix-like)
        # DON'T intercept SIGINT - let it work naturally
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            # signal.signal(signal.SIGINT, self._signal_handler)  # ← COMMENTED OUT
        except (AttributeError, ValueError):
            # Windows doesn't have these signals
            pass
        
        # Ensure crash log directory exists
        os.makedirs(self._crash_log_path, exist_ok=True)
    
    def record_operation(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None
    ):
        """Record operation in trace buffer."""
        trace = OperationTrace(
            timestamp=time.time(),
            operation=operation,
            details=details or {},
            success=success,
            error=error
        )
        
        self._trace_buffer.append(trace)
        self._total_operations += 1
        
        if success is False or error:
            self._failed_operations += 1
        
        if success is not False and not error:
            self._last_operation = operation
    
    def start_operation(self, operation: str):
        """Mark start of long-running operation."""
        self._last_operation = operation
        self._operation_start = time.time()
    
    def end_operation(self, success: bool = True, error: Optional[str] = None):
        """Mark end of long-running operation."""
        if self._last_operation and self._operation_start:
            duration = time.time() - self._operation_start
            
            self.record_operation(
                self._last_operation,
                details={'duration_ms': duration * 1000},
                success=success,
                error=error
            )
        
        self._last_operation = None
        self._operation_start = None
    
    def mark_normal_shutdown(self, reason: str = "user_requested"):
        """Mark shutdown as normal (not a crash)."""
        self._normal_shutdown = True
        self._exit_reason = reason
    
    def _exception_hook(
        self,
        exc_type: type,
        exc_value: BaseException,
        exc_traceback: Any
    ):
        """Handle uncaught exceptions."""
        
        # Format exception
        exc_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        exc_str = ''.join(exc_lines)
        
        print(f"\n{'='*70}")
        print("UNCAUGHT EXCEPTION DETECTED!")
        print(f"{'='*70}")
        print(exc_str)
        print(f"{'='*70}\n")
        
        # Record
        self.record_operation(
            "uncaught_exception",
            details={
                'type': exc_type.__name__,
                'message': str(exc_value)
            },
            success=False,
            error=exc_str
        )
        
        # Attempt self-healing
        try:
            import asyncio
            from jinx.micro.runtime.self_healing import auto_heal_error
            
            # Try to heal in background
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(auto_heal_error(
                    exc_type.__name__,
                    str(exc_value),
                    exc_str
                ))
            else:
                # Run synchronously if no loop
                asyncio.run(auto_heal_error(
                    exc_type.__name__,
                    str(exc_value),
                    exc_str
                ))
        except Exception:
            pass
        
        # Generate crash report
        report = self._generate_crash_report(
            reason=f"Uncaught {exc_type.__name__}: {exc_value}",
            exception_info={
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': exc_str
            }
        )
        
        self._save_crash_report(report)
        
        # Call original excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    def _signal_handler(self, signum: int, frame: Any):
        """Handle signals (SIGTERM, SIGINT, etc.)."""
        
        signal_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT"
        }
        
        sig_name = signal_names.get(signum, f"Signal {signum}")
        
        print(f"\n{'='*70}")
        print(f"SIGNAL RECEIVED: {sig_name}")
        print(f"{'='*70}\n")
        
        self.record_operation(
            "signal_received",
            details={'signal': sig_name, 'number': signum},
            success=False
        )
        
        self.mark_normal_shutdown(f"signal_{sig_name}")
        
        # Re-raise to allow normal signal handling
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
    
    def _on_exit(self):
        """Called at program exit."""
        
        # Check if this was a crash or normal exit
        if self._normal_shutdown:
            print(f"\n[Diagnostics] Normal shutdown: {self._exit_reason}")
            return
        
        # Abnormal exit - generate report
        print(f"\n{'='*70}")
        print("ABNORMAL PROGRAM TERMINATION DETECTED!")
        print(f"{'='*70}")
        
        report = self._generate_crash_report(
            reason="Abnormal termination (no shutdown signal)"
        )
        
        self._save_crash_report(report)
        
        # Print summary
        print(f"\nCrash Report:")
        print(f"  Reason: {report.reason}")
        print(f"  Last operation: {self._last_operation or 'unknown'}")
        print(f"  Total operations: {self._total_operations}")
        print(f"  Failed operations: {self._failed_operations}")
        print(f"  Pulse: {report.pulse}")
        print(f"\nLast 10 operations:")
        
        for trace in list(self._trace_buffer)[-10:]:
            status = "✓" if trace.success else "✗" if trace.success is False else "?"
            duration = trace.details.get('duration_ms', '')
            duration_str = f" ({duration:.1f}ms)" if duration else ""
            print(f"  {status} {trace.operation}{duration_str}")
            if trace.error:
                print(f"    Error: {trace.error[:100]}")
        
        print(f"\nFull report saved to: {self._crash_log_path}")
        print(f"{'='*70}\n")
    
    def _generate_crash_report(
        self,
        reason: str,
        exception_info: Optional[Dict[str, Any]] = None,
        signal_info: Optional[Dict[str, Any]] = None
    ) -> CrashReport:
        """Generate complete crash report."""
        
        # Get system info
        memory_mb = 0.0
        cpu_percent = 0.0
        
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent(interval=0.1)
        except Exception:
            pass
        
        # Get state
        pulse = 0
        shutdown_requested = False
        active_tasks = 0
        
        try:
            import jinx.state as jx_state
            pulse = getattr(jx_state, 'pulse', 0)
            shutdown_requested = getattr(jx_state, 'shutdown_event', None) is not None
            if shutdown_requested:
                shutdown_requested = jx_state.shutdown_event.is_set()
        except Exception:
            pass
        
        try:
            active_tasks = len(asyncio.all_tasks())
        except Exception:
            pass
        
        # Get stack traces
        stack_traces = []
        
        try:
            for thread_id, frame in sys._current_frames().items():
                stack = ''.join(traceback.format_stack(frame))
                stack_traces.append(f"Thread {thread_id}:\n{stack}")
        except Exception:
            pass
        
        return CrashReport(
            crash_time=time.time(),
            exit_code=0,  # Will be set by exit handler
            reason=reason,
            last_operations=list(self._trace_buffer)[-20:],
            stack_traces=stack_traces,
            pulse=pulse,
            shutdown_requested=shutdown_requested,
            active_tasks=active_tasks,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            exception_info=exception_info,
            signal_info=signal_info
        )
    
    def _save_crash_report(self, report: CrashReport):
        """Save crash report to file."""
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"crash_{timestamp}.txt"
            filepath = os.path.join(self._crash_log_path, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("JINX CRASH REPORT\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Crash Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.crash_time))}\n")
                f.write(f"Reason: {report.reason}\n")
                f.write(f"Exit Code: {report.exit_code}\n\n")
                
                f.write("-"*70 + "\n")
                f.write("STATE\n")
                f.write("-"*70 + "\n")
                f.write(f"Pulse: {report.pulse}\n")
                f.write(f"Shutdown Requested: {report.shutdown_requested}\n")
                f.write(f"Active Tasks: {report.active_tasks}\n")
                f.write(f"Memory: {report.memory_mb:.1f} MB\n")
                f.write(f"CPU: {report.cpu_percent:.1f}%\n\n")
                
                if report.exception_info:
                    f.write("-"*70 + "\n")
                    f.write("EXCEPTION\n")
                    f.write("-"*70 + "\n")
                    f.write(f"Type: {report.exception_info.get('type', 'Unknown')}\n")
                    f.write(f"Message: {report.exception_info.get('message', 'None')}\n\n")
                    f.write(report.exception_info.get('traceback', '') + "\n")
                
                f.write("-"*70 + "\n")
                f.write("LAST OPERATIONS\n")
                f.write("-"*70 + "\n")
                
                for trace in report.last_operations:
                    ts = time.strftime('%H:%M:%S', time.localtime(trace.timestamp))
                    status = "SUCCESS" if trace.success else "FAILED" if trace.success is False else "RUNNING"
                    f.write(f"[{ts}] {status} - {trace.operation}\n")
                    
                    if trace.details:
                        for key, value in trace.details.items():
                            f.write(f"  {key}: {value}\n")
                    
                    if trace.error:
                        f.write(f"  Error: {trace.error}\n")
                    
                    f.write("\n")
                
                if report.stack_traces:
                    f.write("-"*70 + "\n")
                    f.write("STACK TRACES\n")
                    f.write("-"*70 + "\n")
                    
                    for stack in report.stack_traces:
                        f.write(stack + "\n")
                        f.write("-"*70 + "\n\n")
                
                f.write("="*70 + "\n")
                f.write("END OF CRASH REPORT\n")
                f.write("="*70 + "\n")
            
            print(f"\n[Diagnostics] Crash report saved: {filepath}")
        
        except Exception as e:
            print(f"\n[Diagnostics] Failed to save crash report: {e}")


# Global instance
_diagnostics: Optional[CrashDiagnostics] = None


def get_diagnostics() -> CrashDiagnostics:
    """Get singleton diagnostics instance."""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = CrashDiagnostics()
    return _diagnostics


def install_crash_diagnostics():
    """Install crash diagnostics handlers."""
    diag = get_diagnostics()
    diag.install_handlers()


def record_operation(
    operation: str,
    details: Optional[Dict[str, Any]] = None,
    success: Optional[bool] = None,
    error: Optional[str] = None
):
    """Record operation for diagnostics."""
    diag = get_diagnostics()
    diag.record_operation(operation, details, success, error)


def start_operation(operation: str):
    """Start tracking long operation."""
    diag = get_diagnostics()
    diag.start_operation(operation)


def end_operation(success: bool = True, error: Optional[str] = None):
    """End tracking long operation."""
    diag = get_diagnostics()
    diag.end_operation(success, error)


def mark_normal_shutdown(reason: str = "user_requested"):
    """Mark shutdown as intentional."""
    diag = get_diagnostics()
    diag.mark_normal_shutdown(reason)


__all__ = [
    "CrashDiagnostics",
    "get_diagnostics",
    "install_crash_diagnostics",
    "record_operation",
    "start_operation",
    "end_operation",
    "mark_normal_shutdown",
]
