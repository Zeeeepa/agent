from __future__ import annotations

from typing import List

# Keep list semantics for compatibility with existing containment checks.
chaos_taboo: List[str] = [
    # Unbounded loops / busy-wait
    "while True:",
    "while 1:",
    "for _ in iter(int, 1):",

    # Sleeping / blocking
    "time.sleep(",

    # Threading / background execution
    "threading.Thread(",

    # Subprocess and shell execution
    "subprocess.run(",
    "subprocess.call(",
    "subprocess.Popen(",
    "os.system(",

    # Potentially dangerous dynamic execution
    "eval(",
    "exec(",

    # Destructive filesystem operations
    "shutil.rmtree(",
    "os.remove(",
    "os.rmdir(",
]
