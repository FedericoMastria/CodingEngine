"""`ait <command>` tracked runner and `ai --fix [<command>]` auto-repair (spec §5.C.2, W04).

`ait` runs the real command (never an alias over python/npm/pytest), tees its output live and,
on failure, saves the tail to `.ai_context/last_terminal_error.json`. `ai --fix` turns that
failure into a debugging task (checkpointed by the router before dispatch) and re-runs the
command only if it is in `safe_verify_allowlist`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from ai_core import ExitCode, kill_process_tree, track
from ai_core.adapters.base import SYSTEMATIC_DEBUGGING_DIRECTIVE
from ai_core.checkpoint import write_json_atomic
from ai_core.config import CONTEXT_DIR_NAME, Config, ensure_context_dir
from ai_core.router import TaskOptions, emit, run, say
from ai_core.security import is_within, sanitize_subprocess_env
from ai_core.wsl import project_argv, wsl_target

LAST_ERROR_FILE = "last_terminal_error.json"
_TAIL_CHARS = 64 * 1024
_PROMPT_TAIL = 6000
_TARGETS_CAP = 3
_TRACE_PATHS = (
    re.compile(r'File "([^"]+)", line \d+'),                       # Python tracebacks
    re.compile(r"\(?((?:[A-Za-z]:)?[\w./\\-]+\.[A-Za-z]{1,5}):\d+(?::\d+)?\)?"),  # path:line[:col] (JS, TS, Go, Rust, gcc)
)
_IGNORED_TRACE_PARTS = {"site-packages", "dist-packages", "node_modules", ".venv", "venv"}


def resolve_command(argv: list[str]) -> list[str]:
    """Windows CreateProcess only appends .exe; which() honours PATHEXT (npm.cmd, pytest.exe...)."""
    return [shutil.which(argv[0]) or argv[0], *argv[1:]]


def _pump(src: IO[str], sink: IO[str], tail: deque[str]) -> None:
    for line in src:  # line-by-line: live output, unlike read(n) which buffers until n chars
        sink.write(line)
        sink.flush()
        tail.append(line)


def _tail(chunks: deque[str]) -> str:
    return "".join(chunks)[-_TAIL_CHARS:]


def run_tracked(root: Path, argv: list[str], *, stdout_to_stderr: bool = False) -> tuple[int, str, str]:
    """Run, tee live, capture tails. Saves last_terminal_error.json on failure, clears it on success."""
    out_tail: deque[str] = deque(maxlen=2000)
    err_tail: deque[str] = deque(maxlen=2000)
    try:
        proc = track(subprocess.Popen(project_argv(root, argv) if wsl_target(root) else resolve_command(argv), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      encoding="utf-8", errors="replace", env=sanitize_subprocess_env(os.environ, warn=False)))
    except (FileNotFoundError, PermissionError) as exc:  # missing tool / WinError 5
        code, out, err = 127, "", f"cannot run {argv[0]}: {exc}"
        print(err, file=sys.stderr)
    else:
        with proc:  # closes both pipes on exit: no leaked handles
            threads = [threading.Thread(target=_pump, args=(proc.stdout, sys.stderr if stdout_to_stderr else sys.stdout,
                                                             out_tail), daemon=True),
                       threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_tail), daemon=True)]
            for t in threads:
                t.start()
            try:
                code = proc.wait()
            except KeyboardInterrupt:
                kill_process_tree(proc)
                raise
            for t in threads:
                t.join(timeout=5)
        out, err = _tail(out_tail), _tail(err_tail)
    path = root / CONTEXT_DIR_NAME / LAST_ERROR_FILE
    if code != 0:
        ensure_context_dir(root)
        write_json_atomic(path, {"command": argv, "cwd": str(root), "exit_code": code, "stdout_tail": out,
                                 "stderr_tail": err, "at": datetime.now(UTC).isoformat(timespec="seconds")})
    else:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("command") == argv:
                path.unlink()
        except (OSError, ValueError):
            pass
    return code, out, err


def is_safe_to_rerun(argv: list[str], cfg: Config) -> bool:
    if not argv:
        return False
    exe = Path(argv[0]).name.lower()
    for ext in (".exe", ".cmd", ".bat"):
        exe = exe.removesuffix(ext)
    command = " ".join([exe, *argv[1:]]).lower()
    return any(command == e.lower() or command.startswith(e.lower() + " ") for e in cfg.safe_verify_allowlist)


def extract_targets(text: str, root: Path, cfg: Config) -> list[str]:
    """Project files named in a stack trace, innermost frame first (most likely culprit)."""
    root_r = root.resolve()
    found: list[str] = []
    for pattern in _TRACE_PATHS:
        for raw in reversed(pattern.findall(text)):
            candidate = Path(raw)
            path = (candidate if candidate.is_absolute() else root / candidate)
            try:
                path = path.resolve()
            except OSError:
                continue
            parts = {p.lower() for p in path.parts}
            if not path.is_file() or not is_within(path, root_r) or parts & _IGNORED_TRACE_PARTS \
                    or CONTEXT_DIR_NAME in parts:
                continue
            rel = Path(os.path.relpath(path, root_r)).as_posix()
            if rel not in found:
                found.append(rel)
    return found[:_TARGETS_CAP]


def load_last_error(root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((root / CONTEXT_DIR_NAME / LAST_ERROR_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("command"), list) else None
    except (OSError, ValueError):
        return None


def fix(root: Path, cfg: Config, argv: list[str], base: TaskOptions) -> ExitCode:
    if argv:
        code, out, err = run_tracked(root, argv, stdout_to_stderr=base.agent)
        if code == 0:
            say(base, f"✔ `{' '.join(argv)}` passes: nothing to fix")
            return ExitCode.OK
        record: dict[str, Any] = {"command": argv, "exit_code": code, "stdout_tail": out, "stderr_tail": err}
    elif (loaded := load_last_error(root)) is None:
        say(base, "✖ nothing to fix: pass a command (ai --fix pytest -x) or run it first with `ait <command>`")
        return ExitCode.USAGE
    else:
        record = loaded
    command = [str(c) for c in record["command"]]
    stdout, stderr = str(record.get("stdout_tail", "")), str(record.get("stderr_tail", ""))
    prompt = (f"The command `{' '.join(command)}` exits with code {record.get('exit_code')}. Find and fix the root "
              f"cause in the project code.\n<stderr>\n{stderr[-_PROMPT_TAIL:]}\n</stderr>\n"
              f"<stdout>\n{stdout[-_PROMPT_TAIL:]}\n</stdout>")
    targets = extract_targets(stderr + "\n" + stdout, root, cfg)
    opts = TaskOptions(prompt, root, tuple(targets), offline=base.offline, verbose=base.verbose, agent=base.agent,
                       extra_directives=SYSTEMATIC_DEBUGGING_DIRECTIVE, strict_verification=True)
    result = run(opts, cfg)
    if result != ExitCode.OK:
        return result
    if not is_safe_to_rerun(command, cfg):
        say(base, f"ℹ `{command[0]}` is not in safe_verify_allowlist: re-run it yourself to confirm the fix")
        return result
    say(base, f"↻ re-running `{' '.join(command)}`")
    code, _, _ = run_tracked(root, command, stdout_to_stderr=base.agent)
    say(base, f"{'✔ fixed' if code == 0 else '✖ still failing'}: `{' '.join(command)}` exited {code}")
    emit(base, "fix_rerun", command=command, exit_code=code, fixed=code == 0)  # --agent: the real verdict
    return ExitCode.OK if code == 0 else ExitCode.FAILED
