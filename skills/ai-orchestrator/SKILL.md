---
name: ai-orchestrator
description: >-
  Delegates a software-engineering task (fix, refactor, implement, review, commit) to the local-first
  CodingEngine router, which picks the cheapest safe executor (local Qwen on the GPU, Antigravity, or
  Claude Code), checkpoints the repo first and verifies the result. Use when the user asks to change code
  in a local repository and wants it done headlessly, offline (--offline), reviewed, or committed safely.
---

# AI Orchestrator (`ai --agent`)

Run from the **project directory**. Command: `{{AI_CMD}}` (or `ai` if it is on PATH).

```
{{AI_CMD}} --agent "<task in plain words>" +path/to/file.py +path/to/other.ts
```

- Mark target files with **`+path`** (never `@path` in PowerShell: `@` is splatting and silently drops the argument).
- Put the whole task in one quoted string; pass the user's wording verbatim.
- Add `--offline` when the user wants nothing to leave the machine (cloud is then physically disabled).
- Never paste secrets, tokens or `.env` contents into the task: the router blocks them (exit 13).

## Reading the result

`stdout` is JSON Lines; the **last** line is `{"event": "result", ...}`. Progress/diagnostics go to `stderr`.
Use `task_status` and the exit code, never just "the process exited":

| Exit | `task_status` | What to do |
|---|---|---|
| 0 | `completed_verified` / `completed_syntax_only` / `completed_unverified` / `completed_with_preexisting_failures` | Done. Report `touched_files` and `verification_level`. |
| 1 | `tests_failed`, `patch_failed`, `provider_error`, `permission_denied`, `project_busy` | Report `message`; offer `ai --undo`. Do not retry blindly. |
| 2 | usage error | Fix the command line. |
| **10** | `needs_clarification` | Ask the user `clarification.question`, then run `clarification.resume` with the answer. |
| 11 | `local_model_unavailable` / `context_exceeded_offline` | Offline impossible: Ollama down or task too large. Ask before retrying without `--offline`. |
| 12 | `quota_exhausted` | Every allowed channel is out of quota. Tell the user; do not loop. |
| 13 | `secret_detected_aborted` | A secret was found (locations only in `message`). Never retry with the secret. |

Answering a clarification (exit 10):

```
{{AI_CMD}} -c --task-id <task_id> --agent "<the user's answer>"
```

## Other commands

| Need | Command |
|---|---|
| Revert the last task (only the files it touched, never user edits) | `{{AI_CMD}} --undo --agent` (whole multi-turn task: `--undo --task-id <id>`) |
| Read-only review on the other channel | `{{AI_CMD}} --review --cross --agent` |
| Local review / over-engineering audit | `{{AI_CMD}} --review --agent` / `{{AI_CMD}} --review --ponytail --agent` |
| Secret-checked commit | `{{AI_CMD}} --commit --agent` (only what is already staged) |
| Repair a failing command | `{{AI_CMD}} --fix --agent pytest -x` (`--fix` must be last) |
| Channels, credit, pending questions | `{{AI_CMD}} --status --json` |
| Real plan usage (Claude 5h/7d, Antigravity quotas) | `{{AI_CMD}} --usage --json` (`--refresh` re-reads agy, 0 tokens) |
| Architectural decisions | `{{AI_CMD}} --adr list --agent` (only the user approves: `--adr approve <id>`) |

## Rules

- Never send a bare slash command such as `/usage` as a task: headless CLIs forward it to the model and it costs.
- One task at a time per repository (the router holds a lock; `project_busy` means wait).
- Never approve ADRs, run `--review --ultra`, or confirm Opus spending on the user's behalf: those need the human.
- Do not edit `.ai_context/`, `~/.ai_checkpoints/` or `~/.ai_state.json` yourself.
