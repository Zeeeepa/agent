from __future__ import annotations

import os

# Conversation transcript (was: log/soul_fragment.txt)
INK_SMEARED_DIARY: str = os.path.join("log", "ink_smeared_diary.txt")

# General/default log (was: log/cortex_wail.txt)
BLUE_WHISPERS: str = os.path.join("log", "blue_whispers.txt")

# User input and executed code logs (was: log/detonator.txt)
TRIGGER_ECHOES: str = os.path.join("log", "trigger_echoes.txt")

# Sandbox output summary (was: log/nano_doppelganger.txt)
CLOCKWORK_GHOST: str = os.path.join("log", "clockwork_ghost.txt")

# Sandbox streaming logs directory and index
SANDBOX_DIR: str = os.path.join("log", "sandbox")
SANDBOX_INDEX: str = os.path.join(SANDBOX_DIR, "index.jsonl")
