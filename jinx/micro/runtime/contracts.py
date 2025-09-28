from __future__ import annotations

# Event topic names for micro-program interactions
TASK_REQUEST = "task.request"      # payload: {id, name, args, kwargs}
TASK_PROGRESS = "task.progress"    # payload: {id, pct, msg}
TASK_RESULT = "task.result"        # payload: {id, ok, result|error}
PROGRAM_SPAWN = "program.spawn"     # payload: {id, name}
PROGRAM_EXIT = "program.exit"       # payload: {id, name, ok}
PROGRAM_HEARTBEAT = "program.heartbeat"  # payload: {id, name}
PROGRAM_LOG = "program.log"         # payload: {id, name, level, msg}
