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

# Evergreen durable memory store
EVERGREEN_MEMORY: str = os.path.join("log", "evergreen_memory.txt")

# Directory for general OpenAI request dumps (one file per request)
OPENAI_REQUESTS_DIR_GENERAL: str = os.path.join("log", "openai", "general")

# Directory for memory optimizer OpenAI request dumps (one file per request)
OPENAI_REQUESTS_DIR_MEMORY: str = os.path.join("log", "openai", "memory")

# Autotune persisted state
AUTOTUNE_STATE: str = os.path.join("log", "autotune_state.json")

# Planner/reflector trace (JSONL records for debugging planner chain)
PLAN_TRACE: str = os.path.join("log", "plan_trace.jsonl")

# Chain resilience persistent state (auto-disable windows, failure counters)
CHAIN_STATE: str = os.path.join("log", "chain_state.json")
