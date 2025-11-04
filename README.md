<img width="1024" height="1536" alt="01" src="https://github.com/user-attachments/assets/3032cb55-f086-4fae-bdc5-826a9a7ebb2c" />

# Jinx — Autonomous Engineering Agent

I’m **Jinx** — an autonomous engineering agent built for teams that ship. I turn intent into execution: understand goals, generate code, validate, sandbox, and deliver — all auditable and reproducible by design.

> Enterprise-grade. Minimal surface area. Maximum signal.

---

## 🚀 Features

* **Autonomous loop** — understand → generate → verify → execute → refine.
* **Sandboxed runtime** — isolated async process for secure code execution.
* **Durable memory** — persistent `<evergreen>` store + rolling context compression.
* **Semantic embeddings** — retrieve relevant dialogue or code context.
* **Cognitive core (Brain)** — concept tracking, framework detection, adaptive reasoning.
* **Structured logging** — full trace of model inputs, outputs, and execution results.
* **Micro‑modular architecture** — lightweight, extendable, fault‑tolerant.

> Designed for reliability. Built for regulated and production‑grade environments.

---

## 🧩 How It Works

```
User Intent → [jinx.py Entrypoint]
                ↓
 [Conversation Orchestrator] → Injects Memory + Embeddings
                ↓
             [LLM / Model]
                ↓
        Generated Code → [Sandbox Execution]
                ↓
         Validation → Update Memory → Log
```

### Core Components

* **Entrypoint:** `jinx.py` — initializes async orchestrator.
* **Orchestrator:** `jinx/conversation/orchestrator.py` — fuses context, memory, and embeddings.
* **Memory System:** `jinx/memory/storage.py`, `jinx/memory/optimizer.py` — handles `<evergreen>` and transcript compaction.
* **Embeddings Engine:** `jinx/embeddings/retrieval.py`, `jinx/micro/embeddings/*` — semantic slicing and ANN retrieval.
* **Brain Module:** `jinx/micro/brain/*` — concept attention and cognitive linking.
* **Sandbox:** `jinx/sandbox/*` — non‑blocking subprocess for executing generated code.
* **Logging:** `jinx/log_paths.py` — structured logs, audit‑ready.

Together, these layers form Jinx’s autonomous reasoning cycle.

---

## ⭐ Star History

<p align="center">
  <a href="https://star-history.com/#machinegpt/agent&Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=machinegpt/agent&type=Date&theme=dark" />
  </a>
</p>

---

## 🔧 Environment Setup

### Python Virtual Environment
Before setting up the project, it's recommended to create a virtual environment. Follow these steps:

Learn about virtual environments: [Python Packaging Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

Before running Jinx, create a virtual environment:

**Windows:**

```
py -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
```

### Project Setup
- Runtime ensures optional dependencies when needed (e.g., `aiofiles`, `prompt_toolkit`). No `requirements.txt` necessary.
- Provide an OpenAI API key and configuration via `.env` at project root. See `.env.example` for all keys:

Required:
```
OPENAI_API_KEY=
```

Optional (defaults in code / example):
```
PULSE=120           # initial error-tolerance pulse
TIMEOUT=300         # seconds before autonomous thinking
OPENAI_MODEL=gpt-5  # model override; service falls back to gpt-5 if unset
# PROXY=socks5://127.0.0.1:12334
```

Create `.env` from the example:

Windows (PowerShell):
```
Copy-Item .env.example .env
```

macOS/Linux:
```
cp .env.example .env
```

## 🧠 Quick Start

From a local clone:

```bash
python jinx.py
```

This launches an interactive REPL. Describe a goal — Jinx plans, writes code, tests it in sandbox, and returns results.

---

## 📚 Example Interaction

**User:** “Write a Python function to compute factorial and add tests.”

**Jinx:**

* Generates `factorial(n)` implementation.
* Creates test (`assert factorial(5) == 120`).
* Executes safely in sandbox.
* If failed — refines until success.

> Result: *Function implemented, validated, and logged.*

---

## 🏗️ Architecture Overview

The runtime is async‑first and auditable:

* **Entrypoint** → `jinx.py`
* **Conversation Orchestrator** → dialogue + embeddings + memory injection
* **Memory Layer** → persistent + compacted context
* **Embeddings Engine** → ANN‑based semantic recall
* **Brain Module** → concept recognition and linking
* **Sandbox Runtime** → secure subprocess for isolated execution
* **Logging** → complete audit trail under `/log/`

---

## 🔐 Security & Compliance

* **Secrets:** Managed via `.env` (never logged).
* **Sandbox:** All model code runs isolated; not a hard boundary, but a safety layer.
* **Logging:** Structured; avoid sensitive data in prompts.

---

## ✨ Contributions

Contributions, suggestions, and feature proposals are welcome.

* Open issues for discussion before PRs.
* Follow modular design — see `jinx/contracts.py`.

<a href="https://github.com/machinegpt/agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=machinegpt/agent" />
</a>

---

## 📄 License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 💬 Support

* File issues and feature requests in [GitHub Issues](https://github.com/machinegpt/agent/issues)
* Start a Discussion for architectural or design topics.

---

**Jinx — a system learning to build and evolve itself.**
