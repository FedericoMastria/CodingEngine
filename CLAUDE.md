# CodingEngine — Project Instructions for Claude Code

## Project Overview
CodingEngine is a dual-channel, local/cloud orchestrator and CLI router native for Windows 11 (Python 3.14). It also drives projects that live in WSL (`\\wsl.localhost\<distro>\...`): see `ai_core/wsl.py`.
Full architecture specification: [CodingEngineArchitecture_v1.0.md](CodingEngineArchitecture_v1.0.md) (v1.0). **Delivered and verified features: §14.B. Open checks: §14.C. Approved divergences from v0.9 baseline: §14.A.**

## Non-Negotiable Invariants
1. **Invariant #1 (`--offline` Absolute):** When `--offline` is set, cloud channels are physically disabled. No token overflow or Ollama failure can touch the network.
2. **Invariant #2 & #3 (Data Safety & External Checkpoints):** Checkpoints and snapshots (plain copies, no `.zst`) are NEVER written inside the workspace; they live in `~/.ai_checkpoints/<project_hash>/<checkpoint_id>/`. Never overwrite user changes; safe undo targets only `touched_files` with pre/post SHA verification.
3. **Invariant #5 (Zero API Billing):** Subprocesses invoking cloud CLIs MUST sanitize `os.environ`, stripping `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`.
4. **Windows Native:** Handle path case-insensitivity (compare via `security.path_key` / `wsl.project_key`, never bare `.lower()`: WSL projects are case-sensitive), backslashes, `WinError 5` (Access Denied), and `WinError 32` (File in Use). Primary target syntax is `+file` to prevent PowerShell splatting.
5. **Claude CLI Integration:** Non-interactive calls use `claude -p`. When `--output-format stream-json` is used, `--verbose` is MANDATORY (otherwise exit code 1).

## Engineering Principles (Ponytail & Superpowers)
- **Ponytail Ladder:** YAGNI -> Reuse -> Stdlib -> Native Platform -> Installed Dep -> One-Liner -> Minimum Viable Code.
- **Never Cut:** Security, trust boundaries, error handling, data protection, explicit Windows exceptions.
- **Systematic Debugging:** Root cause analysis before proposing or applying fixes. No blind trial-and-error.
- **Test-Driven:** Critical logic must be verified with self-contained tests before marking complete.

## Directory Structure
- `ai_core/config.py`: Typed dataclass schema for `config.toml` (unknown keys rejected).
- `ai_core/security.py`: Environment sanitization (anti-API-key, `cloud=True` for CLIs), offline latch, secret scrubber, path validator.
- `ai_core/checkpoint.py`: Git tree hash + external transactional snapshot (`~/.ai_checkpoints/`), safe undo, `diff_since`.
- `ai_core/verification.py`: L1/L2/L3 verification engine with pre-task baseline.
- `ai_core/router.py`: Decision engine (Phases 0-3), dispatch with Q03 rotation, `--agent` JSON-lines output.
- `ai_core/adapters/`: `base.py` (contract + shared streaming runner), `claude_adapter.py` (Channel A), `antigravity_adapter.py` (Channel B).
- `ai_core/credit.py`: Channel A real 5h/7d windows (+ list-price $ counter) + Channel B cooldowns/real quotas (`~/.ai_state.json`).
- `ai_core/session.py`: Blackboard I/O, `pending_sessions` (`ai -c`), single-task lock (PID-reuse safe).
- `ai_core/adr.py`: ADR lifecycle (`proposed` / `active` / `rejected`, source tracking).
- `ai_core/repomap.py`: AST repo-map with import-graph PageRank and incremental cache.
- `ai_core/tui.py` + `ai_core/slash.py`: full-screen REPL (prompt_toolkit) and its slash commands. Esc/Ctrl+C interrupt = `cancel_active()` (kills every child registered with `ai_core.track`) + `KeyboardInterrupt` injected into the task thread, i.e. the same path as a console Ctrl+C: the `finally` blocks in `run_cloud` finalize the checkpoint, so `/undo` works after an interrupt. Any new long-running `subprocess.Popen` must be wrapped in `track(...)` and any `input()` in REPL-reachable code must use `router.INPUT_FN`. `/offline` calls `enforce_offline()` (one-way latch). Tests drive it headless through a prompt_toolkit pipe: `tests/test_tui.py`, `tests/test_slash.py`.
- `ai_core/mdrender.py`: reply markdown -> ANSI via rich (left-aligned headings, tables, quotes, highlighted code, no OSC 8 links, no padding). Always reached through `router.render_markdown` / `router.say_md`, which fall back to the regex `render_terminal_markdown` when colour is off, rich is missing or rendering raises; `--agent` output stays raw. In the TUI `router.MD_DEFER` makes `render_markdown` return a one-line placeholder (`\x00MD<n>\x00`); `Tui._text()` lays each reply out at the current pane width (text before the placeholder, i.e. the reply bullet, becomes the continuation indent), so replies re-wrap on resize. Scrollback cuts must stay on line boundaries (a half placeholder would show as control characters).
- `ai_core/wsl.py`: WSL mode. `wsl_target`/`canonical_root` detect and normalise `\\wsl.localhost` paths; `project_argv` runs every *project* command (git, L2/L3 verification, `ait`, vault chmod, and Claude itself) inside the distro via `wsl.exe -d <distro> --cd <dir> -e`; `linux_claude` finds the distro's `claude`. `agy` and Ollama stay Windows processes.
- `ai_core/vault.py`: Same-volume NTFS vault for secret files (Channel B), journal + `ai --recover`.
- `ai_core/patcher.py`: Atomic 2-stage SEARCH/REPLACE for the local branch.
- `ai_core/review.py`, `commit.py`, `runner.py`: `--review`, `--commit`, `ait` / `--fix`.
- `ai_router.py`: Main CLI entrypoint. `pyproject.toml`: only for native `ai.exe`/`ait.exe` launchers.
- `ai.cmd`, `ait.cmd`: Batch fallbacks (cmd.exe re-parses `> | &`: prefer the `.exe` launchers).
- `ai_core/ollama_env.py`, `mcp.py`, `benchmark.py`, `statusline.py`, `conpty.py`: Ollama server-env check, pinned Context7, §12.3 benchmark, usage hook, ConPTY.
- `skills/`: `ai-orchestrator` skill + `install_skills.py` / `ai-skill-setup.cmd` (all agents).

## Verified CLI Facts (2026-10-25)
- `agy` 1.4.7: the prompt must be the VALUE of `--print` (stdin is ignored; `-p -` sends a literal "-"). Effort is the model-ID suffix. Headless auto-denies shell commands; file tool is `write_file{Path}`.
- `/usage` is interactive-only on BOTH CLIs: `claude -p "/usage"` was answered by Opus ($0.1273) and `agy --print=/usage` burned 32,760 tokens. Never send bare slash commands headlessly (`refuse_slash_command`).
- REAL usage, free: (1) `statusLine` hook (`ai.exe --statusline-capture claude|agy`, same `{"type":"command"}` schema in `~/.claude/settings.json` and `~/.gemini/antigravity-cli/settings.json`) → `~/.ai_usage/<cli>.json`; agy sends `quota.{3p-5h,3p-weekly,gemini-5h,gemini-weekly,gemini-argon-wtu-weekly}.remaining_fraction/reset_time`, claude sends `rate_limits.five_hour/seven_day.used_percentage/resets_at`. (2) claude `-p` stream `rate_limit_event.rate_limit_info`. `ai --usage --refresh` opens agy in ConPTY with NO prompt (verified 0 tokens). Claude Pro limits here are 5h/7d windows (usage credits off); `total_cost_usd` is list-price (`costBasis: list`), not a charge.
- `--max-budget-usd` is checked between turns: one turn can overshoot it (cap $0.01 → spent $0.1273). The 15% reserve absorbs it.
- Cold start of `qwen3.6:35b-a3b`: ~49-54 s load + ~20 s warm-up; a client timeout aborts the load.
- Benchmark (`ai --benchmark` → `~/.ai_benchmark.json`): prefill 686 tok/s @1.7k → 325 @21k, gen 59 → 26 tok/s, ~3.7 chars/token. Prefill is ~4x slower than first assumed: never raise `scoring_full_content_max_tokens` without re-measuring.
- Ollama server env is checked from the `server config` line of `%LOCALAPPDATA%\Ollama\server.log` (`ai_core/ollama_env.py`), not from our own environment. Required: FLASH_ATTENTION=1, KV_CACHE_TYPE=q8_0, KEEP_ALIVE=-1, NUM_PARALLEL=1 (user env vars set 2026-10-25).
- WSL mode (2026-10-26): inside `wsl.exe -e` PATH has `python3` but no `python`; `.venv` is probed in the distro (`test -x`) because a symlinked venv is dangling from Windows. Git for Windows ships its own `sh.exe`, so tests must prove a command ran in the distro (`uname -r` contains `microsoft`). Git Bash rewrites `/home/...` arguments into Windows paths: wrap Linux commands in `sh -c`. `st_dev` is 0 on every UNC share.
- Cloud on WSL projects (2026-10-26): `claude -p` with a `\\wsl.localhost` cwd reads files but its Edit is ALWAYS denied (probed: --add-dir, forward slashes, `\\wsl$`, blanket --allowedTools Edit). So Channel A runs the distro's own `claude` (`ClaudeAdapter._launch` -> `wsl.exe ... -e env -u <API keys> <path> ...`; the path comes from `wsl.linux_claude`, a login shell, because `wsl.exe -e` has a bare PATH without `~/.local/bin`). `agy` with a UNC cwd answers "Authentication required" (same command and env work from a Windows folder), so Channel B starts it in `~/.ai_agy_cwd` with `--add-dir <project>` and a note giving the project root (relative paths in the task are resolved against it). `wsl.exe -e` passes multi-line / 22 KB / quoted arguments byte-exact. The distro's claude was checked on the stream (not the credential files): `apiKeySource: none`, 5h/7d rate-limit windows, overage disabled = Pro subscription, not API billing; it has no statusLine hook, so `~/.ai_usage/claude.json` is not refreshed by WSL runs.
- Context7 MCP: `ai --install-mcp` → pinned `@upstash/context7-mcp@4.3.0` in `~/.ai_mcp/context7` (`ai_core/mcp.py`); `--strict-mcp-config` on Claude calls when the project has secret files.
