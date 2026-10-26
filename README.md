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
- **WSL Projects:** Projects that live inside a WSL distro (`\\wsl.localhost\<distro>\...`) are detected automatically. Git, verification and the secret vault run inside the distro; both cloud channels work (see [Working with WSL projects](#4-optional-working-with-wsl-projects)).

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

**Inside the REPL** (full-screen, wheel scrolls, drag selects and copies):

| Key | Action |
|---|---|
| `Enter` / `Alt+Enter` | send / new line (the box wraps and grows to 8 rows) |
| `Esc` | interrupt the running task (kills its agent processes, drops queued lines); with a draft: clear it |
| `Esc Esc` (idle) | undo the last task (`/undo`, asks first) |
| `Ctrl+C` | interrupt; with a draft clear it; twice to quit |
| `Ctrl+R` | search history (again for older matches) |
| `Tab` on `+path` | complete project files |
| `Ctrl+L` | clear the output pane |

Large pastes collapse to `[Pasted text #1 +40 lines]` and are sent in full. After a task that changed files, a coloured diff summary is shown (`/diff` for all of it).

Commands: `/review [local|cross|ultra|ponytail] [target]`, `/commit`, `/diff`, `/fix [cmd]`, `/plan <task>` (read-only plan from a cloud agent: claude plan mode / `agy --mode plan`), `/adr`, `/channel auto|anthropic|agy|local`, `/offline` (one-way for the session), `/dry-run`, `/gc`, plus `/status /usage /history /new /clear /undo /exit`. The status bar shows the live channel and `OFFLINE` / `dry-run`.

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

### Projects in WSL

```powershell
# From Windows, point --dir at the project's UNC path (use ai.exe, not ai.cmd)
ai --dir \\wsl.localhost\Ubuntu\home\<user>\<project> "fix the failing test" +src\app.py

# Or cd there first (PowerShell can) and run ai without --dir
cd \\wsl.localhost\Ubuntu\home\<user>\<project>
ai --status --agent              # JSON status, includes "wsl": "<distro>:/home/<user>/<project>"
ai --dry-run --verbose "..."     # prints "→ WSL: <distro>:/home/<user>/<project>"
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
- **WSL projects (Optional):** WSL2 with a distro such as Ubuntu — see [step 4](#4-optional-working-with-wsl-projects)

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

### 4. (Optional) Working with WSL projects

There is no switch to turn on: when the project root is under `\\wsl.localhost\<distro>\...` or `\\wsl$\<distro>\...` the engine switches to WSL mode by itself (both spellings are normalised to the first one, so checkpoints and `--undo` always find each other). The engine, Ollama and the checkpoint store (`~/.ai_checkpoints`) stay on Windows; every *project* command runs inside the distro through `wsl.exe -d <distro> --cd <dir> -e ...`.

**What you need to set up**

1. **WSL2 and a distro** (for example Ubuntu), with `git` installed in it.
2. **The project's own toolchain inside the distro.** Verification runs there: L3 uses the project's `.venv/bin/python` if it exists, otherwise `python3` (the distro has no `python` on its bare `PATH`): a `verify_l3_command` that starts with `python` gets that substitution, any other command you set in `config.toml` runs as-is inside the distro. L2 (`tsc --noEmit`) is detected only when `tsconfig.json` and `node_modules/typescript` are in the project root (a sub-folder such as `ui/` needs `verify_l2_command`).
3. **Claude Code inside the distro (Channel A).** Claude Code refuses every edit when its working directory is a Windows network path, so the engine runs the distro's own install. Install it in the distro and log in with your **subscription account, not an API key**:

   ```bash
   # inside the distro
   curl -fsSL https://claude.ai/install.sh | bash   # or your usual installer
   claude                                           # log in once
   ```

   The engine finds it with a login shell, so `~/.local/bin` does not need to be on a non-interactive `PATH`. API keys are removed from the environment on the Windows side and again inside the distro (`env -u`).
4. **Antigravity (Channel B): nothing to install.** The Windows `agy.exe` is used: it asks for a login when started in a network folder, so the engine starts it from `%USERPROFILE%\.ai_agy_cwd` and adds the project with `--add-dir`.
5. **Use `ai.exe`** (installed by step 1), not `ai.cmd`: `cmd.exe` cannot hold a network folder as its working directory and falls back to `C:\Windows`. The engine detects this and stops with a clear message.

**Check it works**

```powershell
ai --dir \\wsl.localhost\<distro>\home\<user>\<project> --dry-run --verbose "describe the project" 
```

It should print `→ WSL: <distro>:/home/<user>/<project>`. A task with `--offline` exercises everything except the cloud channels.

**What behaves differently in WSL mode**

| Area | Behaviour |
|---|---|
| Git, checkpoints, undo | Git runs inside the distro (LF line endings and Linux file modes are preserved, byte-exact undo). Snapshots stay in `~/.ai_checkpoints` on Windows, never in the project. |
| Secret files (Channel B) | The vault moves them into the distro's home (`~/.ai_vault_<id>`, mode `700`) for the duration of the task and restores them afterwards. |
| File names | Case-sensitive, as on Linux (`Foo.py` and `foo.py` are different files). |
| First call after idle | WSL shuts its VM down when idle (`vmIdleTimeout` in `.wslconfig`); the first call then takes a few seconds longer. |

**Troubleshooting**

| Message | Fix |
|---|---|
| `claude is not installed in WSL distro '<distro>': install it there and log in` | Do step 3. Until then Channel A is skipped for WSL projects (a warning is printed once). |
| `WSL distro '<distro>' not found (installed: ...)` | Check the name with `wsl -l -q`. |
| `cmd.exe cannot run in a UNC folder` | Use `ai.exe`, or pass `--dir \\wsl.localhost\...`. |
| `Git refuses this repository ... dubious ownership` | Applies to Windows repositories owned by another user; WSL projects use the distro's own Git and do not hit it. |

**Not verified yet:** `--cross` / `--ultra` reviews and resuming a cloud session (`ai -c`) on WSL projects. Mirrored networking is not required.

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
│   ├── mdrender.py             # Markdown replies -> ANSI (rich): code highlighting, tables, quotes
│   ├── patcher.py              # Atomic 2-stage SEARCH/REPLACE patcher
│   ├── repomap.py              # AST PageRank repository map
│   ├── review.py               # Multi-perspective code review
│   ├── router.py               # 4-phase decision engine & dispatch loop
│   ├── runner.py               # Task execution & `ait` / `--fix` loops
│   ├── security.py             # Environment sanitization & secret scrubber
│   ├── slash.py                # REPL slash commands (/review, /plan, /channel, /offline, /diff ...)
│   ├── tui.py                  # Terminal UI, markdown rendering, & ANSI styling
│   ├── vault.py                # NTFS volume-isolated secret vault
│   ├── verification.py         # Multi-level verification (L1/L2/L3)
│   └── wsl.py                  # WSL mode: detection, canonical root, `project_argv`, distro `claude`
├── docs/                       # Architecture specs and plans
│   ├── CodingEngineArchitecture_v1.0.md # Complete v1.0 specification
│   └── superpowers/            # Implementation plans and design specs
├── skills/                     # Agent skills (ai-orchestrator)
├── tests/                      # Full test suite (220+ unit & integration tests)
├── pyproject.toml              # Project metadata and CLI entrypoints
└── README.md
```

---

## License

This project is licensed under the [MIT License](LICENSE).
