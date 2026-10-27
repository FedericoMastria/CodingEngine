# SYSTEM SPECIFICATION: Local-First Multi-CLI AI Orchestrator & Router (v1.0 — Audit-Hardened)

> **Version Note v1.0 (2026-10-26):** This specification defines CodingEngine v1.0, aligned with verified real-world behavior and production test suites. Notes labeled **[v1.0]** highlight where implementation evolved from the initial v0.9 baseline with user approval. The implementation status, approved divergences, and forward roadmap are detailed in **Section 14**.

---

## 1. System Goal and Non-Negotiable Invariants

Build a native command-line orchestrator for **Windows 11** (in Python) that routes software engineering tasks between a local model (Tier-0 on a dedicated consumer GPU) and two cloud CLIs (`claude` and `agy`), operating **strictly within the boundaries of consumer flat-rate Pro subscriptions** (**Claude Pro** and **Gemini Pro** authenticated via OAuth on their respective CLIs), without **ever** using raw API keys or generating metered pay-per-token charges, minimizing duplicate token usage, and maintaining a shared architectural memory across turns.

> **Design Precedence Rules (5 Non-Negotiable Invariants):** No optimization for speed, quota conservation, prompt caching, or developer convenience can override these rules:
> 1. **Invariant #1 — Absolute `--offline` Mode:** When `--offline` is set, under no circumstances can an escalation, fallback, retry, or context overflow trigger `claude.exe` or `agy.exe`.
> 2. **Invariant #2 — File Integrity & Recoverability:** No patch or rollback may silently overwrite external concurrent edits or leave files in a partially modified state.
> 3. **Invariant #3 — Real Security Boundaries:** Protection of runtime state and sensitive files never relies on prompt engineering alone, but on enforceable file permissions, external checkpoints outside the workspace, process environment sanitization, verified tool deny-policies, and volume-isolated file vaults.
> 4. **Invariant #4 — Verifiable Task Outcome:** A subprocess exit code `0` never implies task verification; `task_status`, `verification_status`, `verification_level`, and `process_exit_code` are always assessed and reported independently against a pre-task baseline.
> 5. **Invariant #5 — Zero API Billing / Pro Plans Only (`B01`):** No execution path may consume pay-as-you-go API credits or trigger metered billing. Subprocess environments are completely stripped of API keys prior to launch. For `claude -p`, execution is governed by the 5-hour and 7-day Pro usage windows. If a preflight check cannot guarantee zero external billing, dispatch to that channel is blocked.

The system serves two complementary operational modes:
1. **Interactive Developer CLI ("Zero-Flag Cognitive Load"):** The developer simply runs `ai <prompt>` (or launches the interactive multi-turn REPL `ai >`). The orchestrator transparently handles supported CLI flags, secret detection, transactional checkpointing, and post-task verification.
2. **Headless Skill / Tool for Personal AI Agents ("Jarvis" / Hermes / MCP):** A non-blocking machine mode (`--agent` / `--json`) enabling autonomous assistants to invoke CodingEngine as their **software engineering skill**, coordinating VRAM usage (`OLLAMA_NUM_PARALLEL=1`) over the shared local model.

The system utilizes an **Asymmetric Dual-Channel Cloud Architecture (`B01`, `Q01`, `Q02`)**:
- **Default Channel B (`agy.exe` — Gemini Pro Subscription, including _"Claude and GPT models"_):** Primary cloud channel for **Sonnet** and **Opus** tasks due to generous rolling rate limits (5-hour and weekly reset windows).
- **Reserved Channel A (`claude.exe` — Claude Pro Subscription):** Treated as a scarce resource reserved for **Haiku** (available exclusively on Channel A), cross-channel reviews (`ai --review --cross`), and emergency failovers when Channel B is experiencing quota exhaustion or cooldown.

---

## 2. Hardware Infrastructure, Local Runtime, and CLI Adapter Architecture

### A. Local Hardware & Runtime (Tier-0 Dedicated GPU)

- **GPU Requirements:** Dedicated modern GPU with 16–24 GB VRAM (sufficient to hold the entire MoE model weights and a quantized KV cache in video memory).
- **Runtime:** Native Windows Ollama instance (ROCm/HIP or CUDA) on `http://localhost:11434`.
- **Unified Model (Router + Tier-0 Executor + Memory Manager + Agent Brain):** Configurable in `config.toml` (primary target: `qwen3.6:35b-a3b`, 35B total / 3B active parameters MoE, `Q4_K_M`; dense alternative: `qwen3.8:27b`).
- **VRAM Budget & Anti-CPU-Spill Preflight (`L03`):**
  - With 24 GB total VRAM, subtracting ~19.5 GB for model weights and **~1.0–1.5 GB reserved by Windows 11 DWM / desktop**, **~3.0–3.5 GB of VRAM** remains for the `q8_0` KV cache and compute buffers.
  - At startup and during benchmarks (§12.3), the orchestrator verifies via `ollama ps` (or `/api/ps`) that the model is allocated **100% on GPU (`size_vram == size`)**. If CPU/RAM spill is detected, the system transitions to `local_degraded` mode, notifying the user and automatically reducing `ollama_num_ctx` (e.g., to `16384`) to prevent severe throughput collapse.
- **Coordinated VRAM Sharing with External Agents (`L02`):**
  - Required Windows user environment variables:
    - `OLLAMA_FLASH_ATTENTION=1`
    - `OLLAMA_KV_CACHE_TYPE="q8_0"`
    - `OLLAMA_KEEP_ALIVE=-1`
    - `OLLAMA_NUM_PARALLEL=1` (single allocated KV slot to prevent VRAM multiplication).
  - **Server Config Verification (`ai_core/ollama_env.py`):** Variables belong to the server process; they are validated against the latest `msg="server config"` line in `%LOCALAPPDATA%\Ollama\server.log`. Discrepancies are flagged in `ai --status` and logged to `stderr`.
  - **Shared Context Constraint:** All callers (`ai`, ADR workers, external assistants) send identical `"num_ctx"` (`32768`), avoiding model reloads.
  - **Best-Effort Prefix Caching:** A deterministic prompt order (`System + Global Profile` → `Repo-Map` → `Blackboard` → `User Request`) promotes KV cache reuse between consecutive tasks.
- **Ollama API Payload (`/api/chat` / `/api/generate`) Top-Level vs `options` (`L04`, `L05`):**
  - **Top-Level JSON Fields:**
    - `"model"`: From `config.toml`.
    - `"think": false | true`: Top-level field in Ollama API (not inside options).
    - `"format": <json_schema>`: Constrained grammar for Phase 2 scoring and clarification requests.
  - **`"options"` Field & Truncation Guard (`L04`):**
    - `"num_ctx": 32768` (calibrated for 0% CPU spill).
    - `"num_predict": 4096` (for `"think": false`) or `"num_predict": 6144` (for `"think": true`).
    - **Local Input Budget:** `local_max_input_tokens = 26000`, reserving ~6,768 tokens for thinking and output.
    - **Truncation Detection:** The orchestrator checks `done_reason` on every response; if `done_reason == "length"`, the output was truncated and is discarded immediately as `patch_failed` without applying broken edits.
  - **Sampling Profiles (`L05`):**
    - *Scoring / Routing / Triage / Smart Commit:* `"think": false`, `options: {"temperature": 0.0, "num_ctx": 32768, "num_predict": 512}`.
    - *Tier-0 Execution (Attempt 1 SEARCH/REPLACE):* `"think": false`, `options: {"temperature": 0.1, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05, "num_ctx": 32768, "num_predict": 4096}`.
    - *Tier-0 Retry (Attempt 2) & Local Review (`--review`):* `"think": true`, `options: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_ctx": 32768, "num_predict": 6144}`.
- **Mandatory Windows UTF-8 Encoding (`W03`):**
  - Startup sets `PYTHONUTF8 = "1"` and reconfigures `sys.stdout` / `sys.stderr` to UTF-8.
  - All `subprocess.Popen` / `subprocess.run` calls enforce `encoding="utf-8", errors="replace"` to eliminate Windows code page corruption (`cp1252` / `cp850`).

---

### B. Dual CLI Adapter Architecture and Anti-API-Key Guard (`B01`–`B05`)

> **Subprocess Environment Sanitization (`B01`):** Prior to launching `claude.exe` or `agy.exe`, the adapter clones `os.environ` and **strictly removes** `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, and all related variables. If any key was detected, a warning is emitted on `stderr`.

1. **Claude Code CLI (`claude.exe`) — Channel A (`ClaudeAdapter`):**
   - **Binary:** Detected via PATH or `~/.local/bin/claude.exe`.
   - **Command Line:** `claude -p --output-format stream-json --verbose --permission-mode acceptEdits|plan --permission-prompts none --model <m> --max-budget-usd <x> --disallowedTools <list> [--strict-mcp-config] [--resume <id>] [--append-system-prompt <static>]`.
   - **Critical Flag Rule:** When using `--print/-p`, `--output-format stream-json` **mandatorily requires `--verbose`** (otherwise the CLI exits with error code 1).
   - **Usage Tracking:** Monitored via the real 5-hour and 7-day rate-limit windows (`rate_limit_event` stream events and statusline hooks in `~/.ai_usage/claude.json`).
   - **Prompt Injection Defense:** Static rules (`Global Profile + ADRs`) are passed via `--append-system-prompt`; dynamic working context is piped via `stdin`.

2. **Antigravity CLI (`agy.exe`) — Channel B (`AntigravityAdapter`):**
   - **Binary:** Verified via `agy --version` (requires version $\ge 1.1.8$).
   - **Command Line:** `agy --print=<prompt> --output-format stream-json --model <id> --mode accept-edits|plan --disable-slash-commands`.
   - **CLI Nuances (v1.0):**
     - Prompt is delivered as the **value** of `--print=<prompt>` (stdin is ignored in non-interactive mode).
     - Model effort is selected via model ID suffix (e.g., `-low`, `-medium`, `-high`), never `--effort`.
     - Headless slash commands (such as `/usage`) reach the LLM and consume quota (32k+ tokens); `refuse_slash_command` intercepts and blocks them. Real quota is extracted for free from the statusline hook.
     - Post-task filesystem hash verification validates working tree integrity even in read-only mode (`--mode plan`).

---

### C. Flag and Capability State Table (`B03`–`B05`)

| Component | Flag / Command / ID | Verified State | Notes |
| :--- | :--- | :--- | :--- |
| `claude.exe` | `-p --output-format stream-json --verbose` | **VERIFIED** | `--verbose` is strictly mandatory with `--print` and `stream-json`. |
| `claude.exe` | `--permission-prompts none` | **VERIFIED** | Instantly denies unapproved tools without blocking the event stream. |
| `claude.exe` | `--max-budget-usd <amount>` | **VERIFIED** | Halts session with `"terminal_reason": "budget_exhausted"` if cap is hit. |
| `claude.exe` | `--strict-mcp-config` | **VERIFIED** | Loads zero MCP servers; used when project contains sensitive secret patterns. |
| `agy.exe` | `--print=<prompt> --output-format stream-json` | **VERIFIED** | Prompt is passed directly in the argument; stdin is ignored. |
| `agy.exe` | Headless `/usage` | **PROHIBITED** | Reaches model and wastes 32k tokens. Monitored via statusline hook instead. |
| `agy` / `claude` | Statusline capture hook | **VERIFIED** | `ai.exe --statusline-capture agy\|claude` saves quotas to `~/.ai_usage/<cli>.json`. |
| Local Model | `qwen3.6:35b-a3b` | **VERIFIED** | 100% GPU allocation, 32k context, cold start ~54s. |

---

## 3. Project Awareness, Shared Memory, and Security

### A. AST Repo-Map (Standard Library AST on CPU)

- Uses Python's native `ast` module (and robust regex extractors for JS/TS, Rust, Go) to build a structural skeleton of the repository: file trees, class declarations, public method signatures, and imports.
- Computes PageRank across the internal import graph to surface top-ranked architectural files.
- Incremental caching in `.ai_context/repo_map.txt` (with metadata index in `.ai_context/repo_map_index.json`), skipping unchanged files by checking `(mtime_ns, file_size)` and SHA-256.
- Dynamic token budget: 2,048 tokens when target files are explicit, up to 8,192 tokens for open-ended queries.

---

### B. Blackboard Pattern (`.ai_context/blackboard.json`) & Anti-Injection Architecture (`M01`, `M02`)

1. **Global Profile & Ponytail Decision Ladder (`~/.ai_profile.md`):**
   A compact markdown profile (~150–220 tokens) defining user standards and the **Ponytail Ladder**:
   1. *YAGNI:* Minimal viable solution only; no unneeded abstractions.
   2. *Reuse:* Reuse existing patterns and helpers from the AST Repo-Map.
   3. *Standard Library:* Prefer native language libraries.
   4. *Native Platform:* Utilize Windows/OS capabilities directly.
   5. *Installed Dependency:* Avoid adding new dependencies for minor helpers.
   6. *One-Liner:* Prefer concise, readable solutions.
   7. *Minimum Viable Code:* Keep it lean, but never compromise on security, boundary trust, or error recovery.

2. **Project Blackboard (`.ai_context/blackboard.json`):**
   - `system_identity`: Tracks the active and recent session pointers (`ai -c`).
   - `architectural_decisions` (ADRs): New decisions proposed by local Qwen enter as `"status": "proposed"`. Only user-confirmed decisions become `"status": "active"`.
   - `recent_tasks`: Rolling ring buffer of the last $N \le 3$ completed tasks (`{task_id, prompt, channel, status, touched_files, checkpoint_id}`).
   - Follow-up continuity: Detects follow-up intent (`_is_followup_prompt`) and inherits target files and differential diffs (`<recent_task_diff>`).
   - `chat_history`: Multi-turn conversational memory for the interactive REPL.
   - `routing_stats`: Adaptive per-project performance tracker (`model|task_type|depth` -> rolling outcomes).

---

### C. Git Invisibility & State Isolation (`S01`, `S02`)

1. **Silent Git Exclusion:** Initializes `.ai_context/.gitignore` and appends `.ai_context/` to `.git/info/exclude`.
2. **External Checkpoints (`S01`):** Checkpoints and file snapshots live outside the repository in `~/.ai_checkpoints/<project_hash>/`.
3. **State Integrity Guard (`R06`):** Captures SHA-256 of `config.toml` and `blackboard.json` before cloud dispatches, restoring them in a `finally` block if modified by cloud subprocesses.

---

### D. Privacy Firewall, Scrubber, and NTFS Vault (`S01`–`S05`)

1. **Level 1 Content Scrubber:** Deterministic regex and entropy scanner checks all outbound cloud prompts, diffs, and attachments for private keys (`BEGIN * PRIVATE KEY`), API tokens, and connection strings. Dispatches are aborted or diverted to local execution if secrets are present.
2. **Level 2 Pattern-Based Isolation:** Case-insensitive path checking against `secret_file_patterns` (`.env*`, `*.pem`, `*.key`, etc.), excluding safe templates (`.env.example`).
3. **Same-Volume NTFS Vault (`S05`):** When isolating secret files for Channel B, files are atomically moved (`os.replace`) to a hidden directory on the **same NTFS volume** (`<drive>:\.ai_vault_<sid>\`), ensuring zero cross-drive copying. Concurrency file locks (`WinError 32`) trigger graceful fallback rather than file manipulation.

---

## 4. Four-Phase Routing and Decision Engine

### Phase 0: Invariants & Fast-Path Triage
- Git state check: Halts if repository has unresolved merges or rebases (`MERGE_HEAD`, `rebase-apply`).
- Absolute `--offline`: Rejects all cloud routing; exits with code `11` if Ollama is unreachable.
- Conversational Fast-Path: Friendly greetings, explanations, and project Q&A without file targets run locally on Tier-0 Qwen with zero cloud latency.

### Phase 1: Context Estimation & Target Resolution
- Supports PowerShell-safe `+path/file` syntax to prevent `@` splatting.
- Conservative token estimation (~3 characters per token).
- Caps local input at `local_max_input_tokens = 26000` (within 32k window).

### Phase 2: Constrained JSON Scoring via Local Qwen
A single structured Ollama call produces:
```json
{
  "files_scope": 1 | 2 | 3,
  "reasoning_depth": 1 | 2 | 3 | 4,
  "needs_repo_search": true | false,
  "creates_new_files": true | false,
  "task_type": "algorithmic" | "architecture" | "debug" | "refactor" | "boilerplate" | "docs",
  "target_files": ["src/calculator.py"],
  "routing_notes": "AST signatures and constraints" | null
}
```
Validated with anti-traversal rules and parent-directory existence checks.

### Phase 3: Adaptive Routing Matrix

1. **Local Tier-0 (Qwen 35B MoE):**
   - Selected when: `--offline` is set, OR ($\le 3$ target files, `needs_repo_search == false`, `reasoning_depth <= 2`, and $\le 26	ext{k}$ input tokens).
   - Adaptive routing skips local execution if recent success rate on this task type is below 40%.
2. **Channel B (Antigravity — Gemini Pro / Argon):**
   - High volume, broad context (>26k tokens), or large multi-file refactoring (>3 files).
   - Higher reasoning depth ($\ge 3$) routes to Argon or Claude on Channel B.
3. **Channel A (Claude Code CLI — Pro Subscription):**
   - Reserved for Haiku (`depth == 1`), architectural cross-reviews, or failovers when Channel B is on cooldown.

---

## 5. Execution, Checkpointing, and Verification

### A. External Checkpointing & Selective Undo (`R01`–`R03`)
- Pre-task Git tree hash recorded via `git write-tree`.
- Snapshot of targeted files stored in `~/.ai_checkpoints/<project_hash>/<checkpoint_id>/`.
- `ai --undo [--task-id <id>]` performs safe selective rollback on `touched_files` only, preserving any concurrent staging done by the developer on other files.

### B. Atomic 2-Stage SEARCH/REPLACE Patcher (`L08`)
- Stage 1: Exact byte match (preserving exact indentation and CRLF line endings).
- Stage 2: Line-trimmed whitespace-tolerant match (strictly requiring unique single occurrence `occurrences == 1`).
- Fallback: Full function or file replacement on small files if SEARCH/REPLACE fails.
- Deletion: a `<<<<<<< DELETE` / `>>>>>>> REPLACE` block under the file path removes a provided file (not combinable with other blocks on the same file). Checkpoints record the post-task state as absent, so `--undo` restores it. Cloud agents have no headless delete tool and are told to ask for `--channel local`.

### C. Multi-Level Verification Engine (L1, L2, L3)
1. **L1 Syntax Check:** In-process AST compilation for Python (`compile`), JSON (`json.loads`), and TOML (`tomllib.loads`).
2. **L2 Build & Typecheck:** Runs project typechecks (e.g., `tsc --noEmit`, `cargo check`, `pyright`) with a 120s timeout.
3. **L3 Targeted Tests:** Executes relevant test suites discovered for the touched modules (`pytest`, `unittest`).
4. **Pre-Task Baseline:** Compares test results against pre-existing failures to avoid blaming existing bugs on the current task (`completed_with_preexisting_failures`).

---

## 6. Disambiguated Exit Codes

| Exit Code | Identifier | Meaning |
| :---: | :--- | :--- |
| `0` | `OK` | Task executed and successfully passed verification gates. |
| `1` | `FAILED` | Execution or tests failed. |
| `2` | `USAGE` | CLI argument or syntax error. |
| `10` | `NEEDS_CLARIFICATION` | Ambiguous request; interactive answer needed (`ai -c`). |
| `11` | `OFFLINE_UNAVAILABLE` | Local model down or context exceeded while `--offline` is active. |
| `12` | `QUOTA_EXHAUSTED` | Rate limits hit across all eligible channels. |
| `13` | `SECRET_BLOCKED` | Privacy Firewall intercepted sensitive credentials. |

---

## 7. Economic Model, Rate Limits, and Session Rotation

- **Claude Pro (Channel A):** Monitored via the 5-hour and 7-day rate-limit windows (`rate_limit_event` stream telemetry and statusline captures). Usage $\ge 85\%$ triggers `exhausted` state.
- **Antigravity (Channel B):** Quotas tracked across `Gemini Models`, `Claude and GPT models`, and `Gemini Argon` via zero-cost statusline hooks.
- **Controlled Rotation (`Q03`):** If a session in progress (`ai -c`) encounters a cooldown on its primary channel, it smoothly rotates to the available alternate channel, re-injecting the blackboard and recent diffs.

---

## 8. CLI Reference and Interactive REPL

```powershell
# Run a single task
ai "fix syntax error in +src/utils.py"

# Launch multi-turn interactive REPL
ai

# Continue session with context delta
ai -c "add unit tests for that edge case"

# Self-repair terminal command or test failure
ait pytest tests/test_core.py
ai --fix "npm test"

# Code reviews and conventional commits
ai --review --ponytail
ai --commit

# Quota inspection and safe rollback
ai --usage --refresh
ai --undo --task-id <id>
```

---

## 9. Monorepo Discovery & Directory Structure

Project root discovery order:
1. Explicit `--dir <path>`.
2. Upward traversal for `.git` or `.ai_context`, stopping below the user's home folder and drive roots (a marker there would turn every project under it into one disk-wide root).
3. First build manifest (`pyproject.toml`, `package.json`, `Cargo.toml`).
4. Current working directory.

```
<project-root>
├── .ai_context/                # Local runtime blackboard & configuration
│   ├── config.toml             # Project-level configuration (tracked)
│   ├── blackboard.json         # Runtime state (git-ignored)
│   └── repo_map.txt            # Cached AST skeleton (git-ignored)
├── ai_core/                    # Orchestration engine modules
├── docs/                       # Specifications and design plans
├── skills/                     # Agent skills (ai-orchestrator)
└── tests/                      # Full test suite
```

---

## 10. Concurrency & Stale-Lock Recovery (`R06`, `R07`)

- Enforces single-task execution per worktree via `.ai_context/.lock`.
- Locks are released when tasks enter `needs_clarification`.
- Anti-PID reuse verification validates `{pid, process_create_time, task_id}` using Win32 process timestamps.

---

## 11. Project Configuration Schema (`config.toml`)

Key defaults from `.ai_context/config.toml`:
```toml
[models]
ollama_model = "qwen3.6:35b-a3b"
ollama_num_ctx = 32768
local_max_input_tokens = 26000
scoring_full_content_max_tokens = 5500

[cloud]
claude_credit_reserve_pct = 15
claude_allow_usage_credits = false
adaptive_routing = true

[security]
secret_scanner_enabled = true
secret_file_patterns = [".env*", "*.pem", "*.key", "*.pfx"]
vault_isolation_mode = "auto"
```

---

## 12. Empirical Benchmarks & Skills Setup

- **Local GPU Benchmark (§12.3):** Verified 100% GPU VRAM allocation for `qwen3.6:35b-a3b` (32k context, Flash Attention enabled, Q8_0 KV cache). Prefill speeds: ~686 tok/s (1.7k) to 325 tok/s (21k). Generation: 26–59 tok/s.
- **Pinned Context7 MCP:** Installed with pinned lockfile in `~/.ai_mcp/context7`, automatically excluded on sensitive tasks.
- **Skill Deployment:** Universal skill installer in `skills/install_skills.py` configures Claude Code and Antigravity agents.

---

## 13. Audit Resolution Changelog (v0.8 → v0.9)

Summary of critical architectural hardening resolved between v0.8 and the v0.9 baseline:
- Invariant #5 established: environment key sanitization and zero-billing guarantees.
- External checkpointing outside the repository (`~/.ai_checkpoints/`).
- Same-volume NTFS Vault for atomic file isolation.
- Pre-task baseline verification to prevent false-negative rollbacks.
- PowerShell-safe `+file` argument parsing.

---

## 14. Implementation Status and Forward Roadmap (v1.0)

### A. Approved Divergences from Initial v0.9 Baseline

| v0.9 Initial Assumption | Implemented v1.0 Reality | Engineering Reason |
| :--- | :--- | :--- |
| AST Repo-Map with `tree-sitter` | Native Python `ast` + regex extractors with PageRank | Zero external C-bindings; resilient cross-platform stdlib |
| Qwen Tokenizer dependency | Conservative `chars / 3` estimate | Zero external runtime dependencies |
| Compressed `.zst` snapshots | Fast plain-copy transactional snapshots | `zstandard` is not part of Python 3.12 standard library |
| Claude `-p` monthly credit counter | Real 5h and 7d rolling quota tracking via statusline | Claude Pro CLI accounts enforce 5h/7d limits rather than monthly pool |
| PowerShell `$PROFILE` wrapper for `ait` | Native `ait.exe` launcher registered in PATH | Avoids mutating user PowerShell profiles |
| Static Phase 3 routing matrix | Project-level adaptive routing on `routing_stats` | Prevents repeated local failures on complex domain tasks |
| Manual L2 configuration only | Automatic `tsc --noEmit` discovery on TypeScript projects | Ensures TypeScript projects have automated build verification |

### B. Delivered and Verified in v1.0

- **Phases 0–3 Routing:** Constrained JSON scoring on local Qwen, path validation, 3-channel matrix, adaptive routing.
- **Safety Invariants #1–#5:** Full offline latch, external checkpoints, safe undo (`--undo`, `--task-id`), secret scrubber, environment sanitization.
- **Channel Adapters:** Channel A (Claude Code) and Channel B (Antigravity) with real rate-limit windows.
- **Developer Features:** Multi-turn REPL `ai >`, session continuation `ai -c`, ADR lifecycle, same-volume NTFS Vault, `--review` (local/cross/ponytail), `--commit`, `ait` / `--fix`, `--recover`, and `--gc`.
- **Terminal UX:** Quiet background routing, ANSI markdown rendering, and structured summary cards.
- **WSL Mode:** projects under `\\wsl.localhost\<distro>` are detected automatically; git, verification (L2/L3 with the project `.venv`; L1 stays in-process), `ait` and the vault (distro home, `chmod 700`) run inside the distro through `ai_core/wsl.py`; case-sensitive path identity. **Cloud edits:** Channel A runs the distro's own `claude` (`wsl.exe -d <distro> --cd <dir> -e env -u <API keys> <claude> ...`, resolved with a login shell) because Claude Code refuses every Edit on a UNC working directory; Channel B runs the Windows `agy` started from `~/.ai_agy_cwd` with the project added through `--add-dir`, because agy asks for a login when its cwd is a UNC path. Verified on throwaway repos: local task + undo on a clone of a real FastAPI project (`completed_verified (L3)`), Claude/haiku inside Ubuntu ($0.0033) and agy/gemini-flash with a `.env` isolated by the vault and restored byte-identical (both `completed_verified (L3)`, undo identical).
- **Engine runtime:** CPython 3.14.
- **Test Suite:** 394 passing unit and integration tests (WSL live tests skip without the Ubuntu distro).

### C. Open Verifications and Forward Roadmap

- [ ] **First Statusline Hook Capture in Clean Session:** Confirm first live event capture of `~/.ai_usage/claude.json` following CLI daemon restart.
- [ ] **Real Task Smoke Test with `--strict-mcp-config`:** Validate `--strict-mcp-config` during an end-to-end task containing an isolated secret file.
- [ ] **Cloud CLIs on a UNC working directory:** both fail there (Claude Code: every Edit denied, even with `--add-dir`, forward slashes, a `\\wsl$` cwd or a blanket `--allowedTools Edit`; agy: "Authentication required" with the same command and environment that work from a Windows folder). The engine works around it (Claude inside the distro, agy from a Windows folder + `--add-dir`); a release of either CLI that accepts UNC would let both go back to a plain cwd.
- [x] **Background Refresh Worker:** after a task, a detached `ai --usage --refresh` runs when the agy capture is older than `usage_max_age_min` (60 min); attempts are throttled to one per 15 min and `ai --usage` refreshes stale data itself.
