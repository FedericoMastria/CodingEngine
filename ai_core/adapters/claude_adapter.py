"""Channel A: Claude Code CLI (`claude.exe` v2.1.310, verified flags in spec §2.C)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_core import TaskStatus
from ai_core.adapters.base import (
    CHANNEL_QUOTA_RE, NEEDS_CLARIFICATION_RE, AdapterError, AdapterResult, BaseAdapter, TaskRequest,
    refuse_slash_command, stream_process,
)
from ai_core.config import HOME, Config
from ai_core.security import API_KEY_VARS, require_cloud_allowed, sanitize_subprocess_env
from ai_core.wsl import WslError, linux_claude, project_argv, wsl_target

# Flags this adapter relies on; checked against `claude --help` before the first dispatch.
REQUIRED_FLAGS = (
    "--print", "--output-format", "--verbose", "--permission-mode", "--permission-prompts",
    "--disallowedTools", "--max-budget-usd", "--model", "--append-system-prompt", "--resume",
    "--strict-mcp-config",
)
_EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
_PROTECTED_PATHS = (
    "./.ai_context/**", "~/.ai_vault/**", "~/.ai_checkpoints/**", "~/.ai_state*", "~/.ai_usage/**", "~/.ssh/**",
    "~/.aws/**",
)
_STDERR_TAIL = 2000
_PROBE_TIMEOUT_S = 60


@dataclass(slots=True)
class StreamState:
    """Incremental parser for `--output-format stream-json` (one JSON event per line)."""

    session_id: str | None = None
    result: dict[str, Any] | None = None
    edited_paths: list[str] = field(default_factory=list)
    assistant_text: list[str] = field(default_factory=list)
    rate_limit: dict[str, Any] | None = None
    malformed_lines: int = 0

    def feed(self, line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self.malformed_lines += 1  # stray non-JSON output (warnings, progress) is tolerated
            return None
        if not isinstance(event, dict):
            self.malformed_lines += 1
            return None
        if isinstance(sid := event.get("session_id"), str) and sid:
            self.session_id = sid
        if event.get("type") == "assistant":
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            for block in content if isinstance(content, list) else ():
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    self.assistant_text.append(block["text"])
                elif block.get("type") == "tool_use" and block.get("name") in _EDIT_TOOLS:
                    tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                    path = tool_input.get("file_path") or tool_input.get("notebook_path")
                    if isinstance(path, str):
                        self.edited_paths.append(path)
        elif event.get("type") == "result":
            self.result = event
        elif event.get("type") == "rate_limit_event" and isinstance(event.get("rate_limit_info"), dict):
            self.rate_limit = event["rate_limit_info"]  # real 5h/7d windows from the API headers, free
        return event

    def to_result(self, exit_code: int | None, stderr: str, timed_out: bool = False) -> AdapterResult:
        res = self.result
        out = AdapterResult(process_exit_code=exit_code, status=None, session_id=self.session_id,
                            edited_paths=list(dict.fromkeys(self.edited_paths)), rate_limit=self.rate_limit)
        if res is None:
            reason = "timed out" if timed_out else f"exited with code {exit_code}"
            out.error = f"claude {reason} without a result event: {stderr[-_STDERR_TAIL:].strip()}"
            out.channel_exhausted = bool(CHANNEL_QUOTA_RE.search(stderr))
            out.status = TaskStatus.QUOTA_EXHAUSTED if out.channel_exhausted else TaskStatus.PROVIDER_ERROR
            return out
        cost = res.get("total_cost_usd")
        out.total_cost_usd = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
        out.model_usage = res["modelUsage"] if isinstance(res.get("modelUsage"), dict) else {}
        out.result_text = res["result"] if isinstance(res.get("result"), str) else ""
        denials = res.get("permission_denials")
        out.permission_denials = len(denials) if isinstance(denials, list) else 0
        subtype = res.get("subtype")
        if res.get("terminal_reason") == "budget_exhausted" or subtype == "error_max_budget_usd":
            # Per-task --max-budget-usd cap: the channel itself is still usable.
            out.status, out.error = TaskStatus.QUOTA_EXHAUSTED, "--max-budget-usd reached"
        elif res.get("is_error") or subtype not in (None, "success"):
            out.error = out.result_text or f"claude result subtype {subtype!r}"
            out.channel_exhausted = bool(CHANNEL_QUOTA_RE.search(out.error)) or \
                (self.rate_limit or {}).get("status") == "rejected"
            out.status = TaskStatus.QUOTA_EXHAUSTED if out.channel_exhausted else TaskStatus.PROVIDER_ERROR
        elif m := NEEDS_CLARIFICATION_RE.search(out.result_text or "\n".join(self.assistant_text)):
            out.status, out.clarification = TaskStatus.NEEDS_CLARIFICATION, m[1].strip()
        return out


class ClaudeAdapter(BaseAdapter):
    name = "claude"
    is_cloud = True

    def __init__(self, cfg: Config, binary: str | None = None) -> None:
        require_cloud_allowed(self.name)
        if cfg.claude_allow_usage_credits:
            raise AdapterError("claude_allow_usage_credits=true is forbidden (Invariant #5): Channel A disabled")
        self.cfg = cfg
        self.binary = binary or shutil.which("claude") or str(HOME / ".local" / "bin" / "claude.exe")
        self.version: str | None = None
        self._wsl_versions: dict[str, str] = {}  # per distro: the Linux claude is a separate install

    def _launch(self, root: Path | None, args: Sequence[str]) -> list[str]:
        """The claude command line: the Windows binary, or the distro's own for a WSL project.

        Claude Code refuses every Edit on a UNC cwd (probed 2026-10-26), so a WSL project runs it inside the distro.
        `env -u` removes the API keys on the Linux side too (Invariant #5), whatever the distro's environment holds.
        """
        target = wsl_target(root) if root else None
        if target is None:
            return [self.binary, *args]
        strip = [arg for var in sorted(API_KEY_VARS) for arg in ("-u", var)]
        try:
            return project_argv(root, ["env", *strip, linux_claude(target.distro), *args])
        except WslError as exc:
            raise AdapterError(str(exc)) from exc

    def _run_quick(self, *args: str, root: Path | None = None) -> str:
        argv = self._launch(root, args)
        try:
            cp = subprocess.run(
                argv, capture_output=True, encoding="utf-8", errors="replace",
                timeout=_PROBE_TIMEOUT_S, env=sanitize_subprocess_env(os.environ, warn=False, cloud=True),
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise AdapterError(f"claude CLI not found at {self.binary}") from exc
        except PermissionError as exc:  # WinError 5: blocked by ACL / SmartScreen / AV
            raise AdapterError(f"cannot execute {self.binary}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(f"claude {' '.join(args)} timed out") from exc
        if cp.returncode != 0:
            raise AdapterError(f"claude {' '.join(args)} exited {cp.returncode}: {cp.stderr.strip()}")
        return cp.stdout

    def _verify(self, root: Path | None) -> str:
        version = self._run_quick("--version", root=root).strip()
        help_text = self._run_quick("--help", root=root)
        if missing := [f for f in REQUIRED_FLAGS if f not in help_text]:
            raise AdapterError(f"claude {version} lacks required flags: {', '.join(missing)}")
        return version

    def probe(self, root: Path | None = None) -> str:
        """`claude --version` + `--help` capability check (B02/B03): refuse unverified CLIs (per install)."""
        if target := wsl_target(root) if root else None:
            if target.distro not in self._wsl_versions:
                self._wsl_versions[target.distro] = self._verify(root)
            return self._wsl_versions[target.distro]
        if self.version is None:
            self.version = self._verify(None)
        return self.version

    def disallowed_tools(self) -> list[str]:
        """Real access control (Invariant #3): deny rules for secrets and orchestrator state."""
        patterns = [f"**/{p}" for p in self.cfg.secret_file_patterns] + list(_PROTECTED_PATHS)
        return [f"{tool}({p})" for p in patterns for tool in ("Read", "Edit")]

    def build_command(self, request: TaskRequest) -> list[str]:
        budget = self.cfg.claude_max_budget_usd_per_task if request.max_budget_usd is None else request.max_budget_usd
        cmd = [
            self.binary, "-p",
            # stream-json under -p REQUIRES --verbose, otherwise claude exits 1 (smoke test §12.1).
            "--output-format", "stream-json", "--verbose",
            "--permission-mode", "plan" if request.read_only else "acceptEdits", "--permission-prompts", "none",
            "--model", request.model,
            "--max-budget-usd", f"{budget:.2f}",
            # one comma-separated value: a variadic option can't swallow later argv
            "--disallowedTools", ",".join(self.disallowed_tools()),
        ]
        if request.disable_mcp:  # strict with no --mcp-config = zero MCP servers (S06)
            cmd.append("--strict-mcp-config")
        if request.resume_session_id:
            cmd += ["--resume", request.resume_session_id]
        if request.system_append:
            cmd += ["--append-system-prompt", request.system_append]
        if self.cfg.claude_use_advisor_on_depth_3 and request.reasoning_depth == 3:
            cmd += ["--advisor", "opus"]
        if request.allowed_tools:
            cmd += ["--allowedTools", ",".join(request.allowed_tools)]
        if request.exclude_dynamic_system_prompt_sections or getattr(self.cfg, "claude_exclude_dynamic_system_prompt_sections", False):
            cmd.append("--exclude-dynamic-system-prompt-sections")
        return cmd

    def _launch_task(self, request: TaskRequest) -> list[str]:
        cmd = self.build_command(request)
        return self._launch(request.project_root, cmd[1:]) if wsl_target(request.project_root) else cmd

    def run_task(self, request: TaskRequest) -> AdapterResult:
        require_cloud_allowed(self.name)  # re-checked at dispatch time: the latch may have closed
        refuse_slash_command(request.prompt, "claude")
        self.probe(request.project_root)
        payload = f"{request.context_block}\n\n{request.prompt}" if request.context_block else request.prompt
        state = StreamState()

        def on_line(line: str) -> None:
            event = state.feed(line)
            if event is not None and request.verbose:
                _echo_event(event)

        try:
            code, stderr, timed_out = stream_process(
                self._launch_task(request), request.project_root, payload,
                sanitize_subprocess_env(os.environ, cloud=True), self.cfg.cloud_timeout_s, on_line)
        except AdapterError as exc:
            raise AdapterError(f"{exc} (binary: {self.binary})") from exc
        return state.to_result(code, stderr, timed_out)


def _echo_event(event: dict[str, Any]) -> None:
    if event.get("type") == "assistant" and isinstance(event.get("message"), dict):
        for block in event["message"].get("content") or ():
            if isinstance(block, dict) and block.get("type") == "text":
                print(f"  claude │ {block.get('text', '')}".rstrip(), file=sys.stderr)
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                detail = json.dumps(block.get("input", {}), ensure_ascii=False)[:160]
                print(f"  claude │ ⚙ {block.get('name')} {detail}", file=sys.stderr)
    elif event.get("type") == "system" and event.get("subtype") == "init":
        print(f"  claude │ session {event.get('session_id')} model {event.get('model')}", file=sys.stderr)
