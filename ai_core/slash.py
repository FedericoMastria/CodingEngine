"""REPL slash commands that wrap the CLI flags (--review, --commit, --fix, --adr, --gc) plus live mode switches
(/channel, /offline, /dry-run) and the coloured `/diff` of the last task. Used by `ai >` (TUI and plain)."""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from ai_core import ExitCode, router
from ai_core.checkpoint import diff_since
from ai_core.config import Config
from ai_core.security import enforce_offline, is_offline, require_loopback_url
from ai_core.session import load_blackboard

CHANNELS = {"auto": None, "anthropic": "anthropic", "claude": "anthropic", "a": "anthropic",
            "agy": "agy", "b": "agy", "local": "local"}
REVIEW_MODES = ("local", "cross", "ultra", "ponytail")
_CHANGES_LINES = 40  # lines of the automatic summary after a task; /diff shows everything

# name -> one-line help; the TUI uses it for completion and /help
COMMANDS = {
    "review": "review changes: /review [local|cross|ultra|ponytail] [target]",
    "commit": "secret-checked commit with a generated message",
    "diff": "everything the last task changed",
    "fix": "repair a failing command: /fix [command ...]",
    "plan": "implementation plan from a cloud agent (read-only): /plan <task>",
    "adr": "decisions: /adr [list|approve <id>|reject <id>|add <title>]",
    "channel": "force a channel: /channel auto|anthropic|agy|local",
    "offline": "cloud off for the rest of this session (cannot be undone)",
    "dry-run": "toggle: show the routing decision without running",
    "gc": "delete expired checkpoints",
}


@dataclass
class ReplState:
    root: Path
    cfg: Config
    channel: str | None = None
    offline: bool = False
    dry_run: bool = False
    verbose: bool = False

    def options(self, **extra: object) -> router.TaskOptions:
        return router.TaskOptions("", self.root, offline=self.offline, dry_run=self.dry_run, verbose=self.verbose,
                                  agent=False, channel=self.channel, **extra)  # type: ignore[arg-type]

    def label(self) -> str:
        flags = (" · OFFLINE" if self.offline or is_offline() else "") + (" · dry-run" if self.dry_run else "")
        return f"channel: {self.channel or 'auto'}{flags}"


def colorize_diff(text: str, max_lines: int | None = None) -> str:
    lines = text.splitlines()
    shown = lines if max_lines is None else lines[:max_lines]
    out = []
    for ln in shown:
        if ln.startswith(("+++", "---", "diff ", "index ")):
            out.append(router._bold(ln))
        elif ln.startswith("@@"):
            out.append(router._cyan(ln))
        elif ln.startswith("+"):
            out.append(router._green(ln))
        elif ln.startswith("-"):
            out.append(router._red(ln))
        else:
            out.append(ln)
    if len(shown) < len(lines):
        out.append(router._dim(f"… {len(lines) - len(shown)} more line(s): /diff shows all"))
    return "\n".join(out)


def last_task(root: Path) -> dict | None:
    recent, _ = load_blackboard(root)
    tasks = recent.get("recent_tasks") or []
    return tasks[0] if tasks else None


def show_changes(st: ReplState, before_task_id: str | None, max_lines: int | None = _CHANGES_LINES) -> None:
    """Print the coloured diff of the task that just ran; silent if it changed nothing (or never ran)."""
    task = last_task(st.root)
    if not task or task.get("task_id") == before_task_id or not task.get("touched_files") or not task.get("checkpoint_id"):
        return
    try:
        diff = diff_since(st.root, task["checkpoint_id"], st.cfg)
    except Exception:  # a summary must never turn a finished task into an error
        return
    if diff.strip():
        print(router._dim(f"── changes: {', '.join(task['touched_files'])} ──"))
        print(colorize_diff(diff, max_lines))


def _diff(st: ReplState, _rest: str) -> ExitCode:
    task = last_task(st.root)
    if not task or not task.get("checkpoint_id"):
        print(router._dim("no task with a checkpoint yet"))
        return ExitCode.OK
    diff = diff_since(st.root, task["checkpoint_id"], st.cfg)
    print(colorize_diff(diff) if diff.strip() else router._dim("no changes since the last checkpoint"))
    return ExitCode.OK


def _review(st: ReplState, rest: str) -> ExitCode:
    from ai_core.review import review
    words = rest.split()
    mode = words.pop(0).lower() if words and words[0].lower() in REVIEW_MODES else "local"
    return review(st.root, st.cfg, mode, st.options(), " ".join(words) or None)


def _commit(st: ReplState, _rest: str) -> ExitCode:
    from ai_core.commit import commit
    return commit(st.root, st.cfg, st.options())


def _fix(st: ReplState, rest: str) -> ExitCode:
    from ai_core.runner import fix
    return fix(st.root, st.cfg, rest.split(), st.options())


def _plan(st: ReplState, rest: str) -> ExitCode:
    from ai_core.review import plan
    return plan(st.root, st.cfg, st.options(), rest)


def _adr(st: ReplState, rest: str) -> ExitCode:
    from ai_core.adr import run_cli
    return ExitCode(run_cli(st.root, shlex.split(rest), False))


def _channel(st: ReplState, rest: str) -> ExitCode:
    name = rest.strip().lower()
    if not name:
        print(f"{router._bold('channel')}: {st.channel or 'auto'}  ({' | '.join(sorted(set(CHANNELS)))})")
        return ExitCode.OK
    if name not in CHANNELS:
        print(f"✖ unknown channel {name!r}: use auto, anthropic, agy or local", flush=True)
        return ExitCode.USAGE
    st.channel = CHANNELS[name]
    print(f"→ channel: {st.channel or 'auto'}")
    return ExitCode.OK


def _offline(st: ReplState, _rest: str) -> ExitCode:
    require_loopback_url(st.cfg.ollama_url)  # refuse before latching if the local endpoint is not loopback
    enforce_offline()
    st.offline = True
    print("→ offline: cloud channels are physically disabled until you restart `ai`")
    return ExitCode.OK


def _dry_run(st: ReplState, rest: str) -> ExitCode:
    arg = rest.strip().lower()
    st.dry_run = (arg in ("on", "true", "1")) if arg else not st.dry_run
    print(f"→ dry-run {'on: tasks only show their routing' if st.dry_run else 'off'}")
    return ExitCode.OK


def _gc(st: ReplState, _rest: str) -> ExitCode:
    return router.gc(st.root, st.cfg)


_HANDLERS = {"review": _review, "commit": _commit, "diff": _diff, "fix": _fix, "plan": _plan, "adr": _adr,
             "channel": _channel, "offline": _offline, "dry-run": _dry_run, "dryrun": _dry_run, "gc": _gc}


def run(st: ReplState, line: str) -> bool:
    """Handle `/name args`; False when `line` is not a slash command (it is a task)."""
    if not line.startswith("/"):
        return False
    name, _, rest = line[1:].partition(" ")
    handler = _HANDLERS.get(name.lower())
    if handler is None:
        print(f"✖ unknown command /{name}: /help lists the commands")
        return True
    try:
        handler(st, rest)
    except Exception as exc:  # KeyboardInterrupt (the TUI interrupt) is a BaseException and passes through
        print(f"✖ /{name}: {exc}")
    return True
