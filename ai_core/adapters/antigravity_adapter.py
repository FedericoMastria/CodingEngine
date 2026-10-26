"""Channel B: Antigravity CLI (`agy.exe` >= 1.1.8; stream format verified on 1.4.7, spec §2.B.2).

Verified behaviour (2026-10-25): the prompt must be the VALUE of `--print`; stdin is ignored
(`-p -` sends the literal prompt "-", and `--input-format stream-json` produced no result). It is
passed as one `--print=<prompt>` token so a prompt starting with "-" is never read as a flag;
agy.exe is a real executable, so no cmd.exe re-parsing happens, but CreateProcess caps the whole
command line at 32,767 chars. Headless mode auto-denies any tool permission it cannot prompt for;
effort is the model-ID suffix (`claude-sonnet-5-5-medium`), so `--effort` is never passed (B04).
There is no per-file deny flag: secret files are isolated by the NTFS vault before dispatch (S05).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from ai_core import TaskStatus
from ai_core.adapters.base import (
    CHANNEL_QUOTA_RE, NEEDS_CLARIFICATION_RE, AdapterError, AdapterResult, BaseAdapter, TaskRequest,
    refuse_slash_command, stream_process,
)
from ai_core.config import Config
from ai_core.security import require_cloud_allowed, sanitize_subprocess_env

MIN_VERSION = (1, 1, 8)
REQUIRED_FLAGS = ("--print", "--output-format", "--model", "--mode", "--conversation", "--print-timeout")
_EDIT_TOOLS = frozenset({"write_file", "write_to_file", "replace_file_content", "multi_replace_file_content",
                         "sed_file", "notebook_edit"})  # write_file{Path} observed on 1.4.7
# Headless agy auto-denies shell commands; saying so up front avoids turns wasted on denied actions.
HEADLESS_NOTE = ("Headless session: shell/terminal commands are disabled and will be denied. Use only the file "
                 "tools (view, write, replace) inside the current workspace to complete the task.")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_PROBE_TIMEOUT_S = 60
_STDERR_TAIL = 2000
_TIMEOUT_GRACE_S = 30  # our watchdog fires after agy's own --print-timeout
_MAX_CMDLINE = 32_000  # CreateProcess limit is 32,767 chars for the whole command line


def bucket_for_model(model: str) -> str:
    """Channel B quota buckets of the Gemini Pro plan (§7.A.2)."""
    m = model.lower()
    if m.startswith(("claude-", "gpt-")):
        return "claude_gpt"
    return "argon" if "argon" in m else "gemini"


@dataclass(slots=True)
class AgyStreamState:
    conversation_id: str | None = None
    result: dict[str, Any] | None = None
    edited_paths: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    model: str | None = None
    malformed_lines: int = 0

    def feed(self, line: str) -> dict[str, Any] | None:
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self.malformed_lines += 1
            return None
        if not isinstance(event, dict):
            self.malformed_lines += 1
            return None
        if isinstance(cid := event.get("conversation_id"), str) and cid:
            self.conversation_id = cid
        kind = event.get("event")
        if kind == "init" and isinstance(event.get("init"), dict):
            self.model = event["init"].get("model")
        elif kind == "step_update" and isinstance(step := event.get("step_update"), dict):
            info = step.get("tool_info") if isinstance(step.get("tool_info"), dict) else {}
            params = info.get("parameters") if isinstance(info.get("parameters"), dict) else {}
            tool = step.get("tool_name")
            if tool in _EDIT_TOOLS:
                self.edited_paths += [v for k, v in params.items()
                                      if isinstance(v, str) and ("file" in k.lower() or "path" in k.lower())]
            elif tool == "ask_question":  # headless can't answer it: surface it as a clarification
                self.questions += [v for v in params.values() if isinstance(v, str) and v.strip()][:1]
        elif kind == "result" and isinstance(event.get("result"), dict):
            self.result = event["result"]
        return event

    def to_result(self, exit_code: int | None, stderr: str, timed_out: bool = False) -> AdapterResult:
        res = self.result
        out = AdapterResult(process_exit_code=exit_code, status=None, session_id=self.conversation_id,
                            edited_paths=list(dict.fromkeys(self.edited_paths)))
        if res is None:
            reason = "timed out" if timed_out else f"exited with code {exit_code}"
            out.error = f"agy {reason} without a result event: {stderr[-_STDERR_TAIL:].strip()}"
            out.channel_exhausted = bool(CHANNEL_QUOTA_RE.search(stderr))
            out.status = TaskStatus.QUOTA_EXHAUSTED if out.channel_exhausted else TaskStatus.PROVIDER_ERROR
            return out
        out.result_text = res["response"] if isinstance(res.get("response"), str) else ""
        denied = res.get("denied_actions")
        out.permission_denials = len(denied) if isinstance(denied, list) else 0
        if isinstance(usage := res.get("usage"), dict):
            out.model_usage = {self.model or "agy": usage}
        if res.get("status") != "SUCCESS":
            detail = " ".join(str(res.get(k, "")) for k in ("error", "message", "status")).strip()
            out.error = f"agy status {res.get('status')!r}: {detail or out.result_text} {stderr[-500:]}".strip()
            out.channel_exhausted = bool(CHANNEL_QUOTA_RE.search(out.error))
            out.status = TaskStatus.QUOTA_EXHAUSTED if out.channel_exhausted else TaskStatus.PROVIDER_ERROR
        elif m := NEEDS_CLARIFICATION_RE.search(out.result_text):
            out.status, out.clarification = TaskStatus.NEEDS_CLARIFICATION, m[1].strip()
        elif self.questions:
            out.status, out.clarification = TaskStatus.NEEDS_CLARIFICATION, self.questions[0].strip()
        return out


class AntigravityAdapter(BaseAdapter):
    name = "agy"
    is_cloud = True

    def __init__(self, cfg: Config, binary: str | None = None) -> None:
        require_cloud_allowed(self.name)
        self.cfg = cfg
        self.binary = binary or shutil.which("agy") or "agy.exe"
        self.version: str | None = None

    def probe(self) -> str:
        """Minimum version (B04) and verified flags; refuses anything older or different."""
        if self.version is None:
            env = sanitize_subprocess_env(os.environ, warn=False, cloud=True)
            try:
                ver = subprocess.run([self.binary, "--version"], capture_output=True, encoding="utf-8",
                                     errors="replace", timeout=_PROBE_TIMEOUT_S, env=env, stdin=subprocess.DEVNULL)
                hlp = subprocess.run([self.binary, "--help"], capture_output=True, encoding="utf-8",
                                     errors="replace", timeout=_PROBE_TIMEOUT_S, env=env, stdin=subprocess.DEVNULL)
            except FileNotFoundError as exc:
                raise AdapterError(f"agy CLI not found at {self.binary}") from exc
            except (PermissionError, subprocess.TimeoutExpired) as exc:
                raise AdapterError(f"cannot run {self.binary}: {exc}") from exc
            m = _VERSION_RE.search(ver.stdout)
            if ver.returncode != 0 or not m:
                raise AdapterError(f"agy --version failed: {(ver.stderr or ver.stdout).strip()}")
            if tuple(map(int, m.groups())) < MIN_VERSION:
                raise AdapterError(f"agy {m[0]} is older than the required {'.'.join(map(str, MIN_VERSION))}")
            help_text = hlp.stdout + hlp.stderr
            if missing := [f for f in REQUIRED_FLAGS if f not in help_text]:
                raise AdapterError(f"agy {m[0]} lacks required flags: {', '.join(missing)}")
            self.version = m[0]
        return self.version

    def build_command(self, request: TaskRequest) -> list[str]:
        cmd = [
            self.binary, "--output-format", "stream-json", "--model", request.model,
            "--mode", "plan" if request.read_only else "accept-edits",
            "--print-timeout", f"{self.cfg.cloud_timeout_s}s",
        ]
        if not self.cfg.agy_enable_headless_slash_commands:
            cmd.append("--disable-slash-commands")
        if request.resume_session_id:
            cmd += ["--conversation", request.resume_session_id]
        # No --append-system-prompt equivalent: static part, then dynamic context, then the prompt.
        for parts in ((HEADLESS_NOTE, request.system_append, request.context_block, request.prompt),
                      (HEADLESS_NOTE, request.system_append, request.prompt)):
            full = [*cmd, "--print=" + "\n\n".join(p for p in parts if p)]
            if len(subprocess.list2cmdline(full)) <= _MAX_CMDLINE:
                if len(parts) == 3 and request.context_block:
                    print("⚠ agy: repo-map/project memory dropped to fit the Windows command-line limit",
                          file=sys.stderr)
                return full
        raise AdapterError(f"prompt too long for agy's command line ({_MAX_CMDLINE} chars): use Channel A")

    def run_task(self, request: TaskRequest) -> AdapterResult:
        require_cloud_allowed(self.name)
        refuse_slash_command(request.prompt, "agy")
        self.probe()
        state = AgyStreamState()

        def on_line(line: str) -> None:
            event = state.feed(line)
            if event is not None and request.verbose:
                _echo_event(event)

        code, stderr, timed_out = stream_process(
            self.build_command(request), request.project_root, "",  # stdin is ignored by agy -p
            sanitize_subprocess_env(os.environ, cloud=True), self.cfg.cloud_timeout_s + _TIMEOUT_GRACE_S, on_line)
        return state.to_result(code, stderr, timed_out)


def _echo_event(event: dict[str, Any]) -> None:
    if event.get("event") == "init":
        print(f"  agy │ conversation {event.get('conversation_id')} model {event.get('init', {}).get('model')}",
              file=sys.stderr)
    elif event.get("event") == "step_update" and isinstance(step := event.get("step_update"), dict) \
            and step.get("tool_name"):
        params = json.dumps((step.get("tool_info") or {}).get("parameters", {}), ensure_ascii=False)[:160]
        print(f"  agy │ ⚙ {step['tool_name']} [{step.get('state')}] {params}", file=sys.stderr)
