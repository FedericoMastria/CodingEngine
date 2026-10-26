# CodingEngine

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows 11](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](https://www.microsoft.com/windows)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Local First](https://img.shields.io/badge/architecture-local--first-success.svg)]()

**CodingEngine** is a dual-channel, local-first AI coding orchestrator and CLI router native to Windows 11. 

It routes software engineering tasks seamlessly between a local **Tier-0 local model** (Qwen 35B MoE via Ollama on consumer GPU) and cloud CLI channels (**Claude Code CLI** and **Google Antigravity CLI**), providing zero-cost conversational triage, external safety checkpoints, automated verification, and strict API-billing immunity.

---

## Key Highlights

- **Local-First Tier-0 Routing:** Conversational queries, fast edits, and small refactors run locally on GPU with zero latency and zero cost.
- **Dual-Channel Cloud Orchestration:**
  - **Channel A:** Claude Code CLI (`claude.exe`) for deep reasoning and architectural decisions.
  - **Channel B:** Antigravity CLI (`agy.exe`) for Gemini Pro, Argon, and Claude models.
- **Strict Budget & Quota Tracking:** Real-time 5h and 7d quota monitoring via ConPTY statusLine hooks; dynamic failover and cooldown rotation.
- **Deterministic Verification Engine:** Automated multi-level verification (L1 syntax, L2 typecheck/build, L3 targeted tests) with pre-task baseline diffing.
- **Non-Negotiable Safety Invariants:** Sandboxed checkpoints stored outside the workspace, safe selective undo, and environment scrubbing.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │          User CLI / REPL (`ai`)        │
                      └──────────────────┬─────────────────────┘
                                         │
                                   Phase 0: Triage
                        (Conversational / Target Inference)
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │    AST Repo-Map & Target Validation    │
                      └──────────────────┬─────────────────────┘
                                         │
                                   Phase 1 & 2:
                        JSON Scoring via Local Qwen
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │       Adaptive Decision Matrix         │
                      └───────┬──────────────┬──────────────┬──┘
                              │              │              │
                     Offline / Low-Risk      │              │
                              │       Channel A (Claude)    │
                              ▼              │      Channel B (Antigravity)
                      ┌───────────────┐      ▼              ▼
                      │ Local Tier-0  │ ┌──────────┐ ┌───────────────┐
                      │  Ollama Qwen  │ │claude.exe│ │    agy.exe    │
                      └───────────────┘ └──────────┘ └───────────────┘
                              ▲              ▲              ▲
                              └──────────────┼──────────────┘
                                             │
                                   Phase 4: Verification
                              (L1 Syntax, L2 Typecheck, L3 Tests)
                                             │
                                             ▼
                      ┌────────────────────────────────────────┐
                      │  Atomic Snapshot & Safe Rollback Engine│
                      └────────────────────────────────────────┘
```

---

## Non-Negotiable Invariants

1. **`--offline` Absolute:** When `--offline` is set, cloud channels are physically disabled. No token overflow or model degradation can touch external networks.
2. **Data Safety & External Checkpoints:** Plain-copy transactional snapshots are **never** written inside the project workspace; they live externally in `~/.ai_checkpoints/<project_hash>/<checkpoint_id>/`. Safe undo targets only `touched_files` with pre/post SHA-256 verification.
3. **Zero API Billing:** Subprocesses invoking cloud CLIs rigorously sanitize `os.environ`, stripping all raw API tokens (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, etc.) to enforce subscription-backed execution only.
4. **Windows Native:** Path case-insensitivity handling (`path.lower()`), backslash normalization, WinError 5 / 32 handling, ConPTY pseudo-terminals, same-volume NTFS Vault for secret isolation, and PowerShell-safe `+file` syntax to avoid splatting.
5. **Deterministic Verification:** Process exit code `0` never implies completion; tasks must satisfy the verification gate (L1 syntax, L2 build/types, L3 targeted unit tests).

---

## CLI Reference

### Basic Execution & REPL

```powershell
# Run a single task (auto-detects target files or use +file)
ai "fix off-by-one error in +src/calculator.py"

# Launch the interactive multi-turn REPL
ai

# Continue the previous session with context delta
ai -c "now add unit tests for that edge case"

# Continue a specific task session
ai -c --task-id 20261026-114500-a1b2 "proceed with step 2"
```

### Self-Healing & Debugging

```powershell
# Run a test or build command, capture failures, and fix automatically
ait pytest tests/test_core.py

# Or using the long form
ai --fix "npm test"
```

### Review & Commits

```powershell
# Perform a multi-perspective code review (local, cross-model, or ponytail mode)
ai --review
ai --review --ponytail

# Generate conventional commits with automatic secret scanning & verification
ai --commit
```

### Inspection & Rollback

```powershell
# Check current system status, model allocation, and Ollama server health
ai --status

# Inspect real-time 5h/7d quotas across Claude and Antigravity
ai --usage --refresh

# Undo changes from the most recent task (or specific task ID)
ai --undo
ai --undo --task-id 20261026-114500-a1b2
```

---

## Installation & Setup

### Prerequisites

- **OS:** Windows 11 (64-bit)
- **Runtime:** Python 3.14+
- **Local LLM Server:** [Ollama](https://ollama.com/) with `qwen3.6:35b-a3b` (or configured dense alternative `qwen3.8:27b`)
- **Cloud CLIs (Optional for Cloud Dispatch):**
  - Claude Code CLI (`claude.exe`)
  - Google Antigravity CLI (`agy.exe`)

### 1. Clone & Install

```powershell
git clone https://github.com/FedericoMastria/CodingEngine.git
cd CodingEngine

# Create virtual environment and install in editable mode
uv venv
uv pip install -e .
```

This registers native `ai.exe` and `ait.exe` console launchers in your `.venv\Scripts` directory.

### 2. Configure Environment

Review and adjust project settings in `.ai_context/config.toml`:

```toml
[models]
ollama_model = "qwen3.6:35b-a3b"
ollama_num_ctx = 32768
local_max_input_tokens = 26000

[cloud]
claude_credit_reserve_pct = 15
adaptive_routing = true

[security]
secret_scanner_enabled = true
secret_file_patterns = [".env*", "*.pem", "*.key"]
```

### 3. Setup Agent Skills

To install the `ai-orchestrator` skill for Claude Code and Antigravity:

```powershell
python skills/install_skills.py
# Or run the batch script:
skills\ai-skill-setup.cmd
```

---

## Directory Structure

```
CodingEngine/
├── ai_core/                    # Core orchestration logic
│   ├── adapters/               # Cloud CLI adapters (Claude & Antigravity)
│   ├── adr.py                  # Architectural Decision Record lifecycle
│   ├── checkpoint.py           # External transactional snapshots & safe undo
│   ├── config.py               # Typed configuration schema (config.toml)
│   ├── conpty.py               # Windows ConPTY terminal runner
│   ├── credit.py               # Quota, rate limits, and 5h/7d window tracking
│   ├── patcher.py              # Atomic 2-stage SEARCH/REPLACE patcher
│   ├── repomap.py              # AST PageRank repository map
│   ├── review.py               # Multi-perspective code review
│   ├── router.py               # 4-phase decision engine & dispatch loop
│   ├── runner.py               # Task execution & `ait` / `--fix` loops
│   ├── security.py             # Environment sanitization & secret scrubber
│   ├── tui.py                  # Terminal UI, markdown rendering, & ANSI styling
│   ├── vault.py                # NTFS volume-isolated secret vault
│   └── verification.py         # Multi-level verification (L1/L2/L3)
├── docs/                       # Architecture specs and plans
│   ├── CodingEngineArchitecture_v1.0.md # Complete v1.0 specification
│   └── superpowers/            # Implementation plans and design specs
├── skills/                     # Agent skills (ai-orchestrator)
├── tests/                      # Full test suite (170+ unit & integration tests)
├── pyproject.toml              # Project metadata and CLI entrypoints
└── README.md
```

---

## License

This project is licensed under the [MIT License](LICENSE).
