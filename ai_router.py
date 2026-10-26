"""`ai` CLI entrypoint (spec §8). Usage: ai "refactor the parser" +src/parser.py [--offline] [--dry-run]."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importable from any cwd via ai.cmd

from ai_core import ExitCode
from ai_core.config import ConfigError, find_project_root, load_config
from ai_core.wsl import WslError, canonical_root, wsl_target

_TRAILING_PUNCT = ",;:)]}'\""
# ai's own flags allowed right after --fix; the first other token starts the command line.
_FIX_OWN_FLAGS = frozenset({"--offline", "--agent", "--json", "--verbose", "-v", "--dry-run"})


def _looks_like_path(token: str) -> bool:
    return "/" in token or "\\" in token or "." in token.strip(".")



def detect_powershell_splatting(words: list[str]) -> str | None:
    """Detect truncated or bare @ arguments caused by PowerShell splatting (W01, §4 Fase 1)."""
    for arg in words:
        for token in arg.split(" "):
            bare = token.rstrip(_TRAILING_PUNCT)
            if bare == "@":
                return "bare '@' detected"
            if bare.startswith("@") and len(bare) > 1 and not _looks_like_path(bare[1:]):
                return f"malformed splatting token '{bare}'"
    return None

def split_prompt(words: list[str], root: Path) -> tuple[str, list[str]]:
    """Extract target files: `+path` (primary, PowerShell-safe), quoted `@path`, or existing paths in text (W01).

    Each argument is tokenised too: `ai "fix the bug in calc.py"` arrives as ONE argument from a shell.
    A standalone `+x` argument is always a target; inside a sentence `+x` must look like a path
    (so "add +1 to the counter" never creates a file named `1`). The marker is dropped but the path
    stays in the prompt, so the prompt reaches the model verbatim.
    """
    prompt_words: list[str] = []
    files: list[str] = []
    for arg in words:
        standalone = " " not in arg.strip()
        tokens = []
        for token in arg.split(" "):
            bare = token.rstrip(_TRAILING_PUNCT)
            if len(bare) > 1 and bare[0] in "+@" and (standalone or _looks_like_path(bare[1:])):
                token, bare = token[1:], bare[1:]
                files.append(bare.rstrip(".") or bare)
            elif _looks_like_path(bare):
                # "... in calc.py." ends a sentence. Undotted first: Win32 strips trailing dots, so
                # "calc.py." would also pass is_file() and become a bogus target name.
                for candidate in (bare.rstrip("."), bare):
                    if candidate and (root / candidate).is_file():
                        files.append(candidate)
                        break
            tokens.append(token)
        prompt_words.append(" ".join(tokens))
    return " ".join(prompt_words).strip(), list(dict.fromkeys(files))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai", description="Local-first router: Ollama (Tier-0) + Claude Code (Channel A) + Antigravity (Channel B).",
        epilog="ait <command>  runs a command and records its failure for `ai --fix`.")
    p.add_argument("words", nargs="*", help="task text; mark target files with +path/to/file")
    p.add_argument("--channel", choices=["anthropic", "agy", "local"], help="force execution channel, bypassing routing matrix (§4 Fase 3)")
    p.add_argument("--dir", type=Path, default=None, help="project root directory (priority 1 in root discovery, §9.1)")
    p.add_argument("--file", action="append", default=[], help="attach a target file (alternative to +file, §4 Fase 1)")
    p.add_argument("--offline", action="store_true", help="100%% local: cloud channels physically disabled")
    p.add_argument("--dry-run", action="store_true", help="show routing decision and command, dispatch nothing")
    p.add_argument("--verbose", "-v", action="store_true", help="stream agent events to stderr")
    p.add_argument("--agent", "--json", dest="agent", action="store_true",
                   help="headless: JSON-lines events on stdout, never prompts, exit codes 0/1/2/10/11/12/13")
    p.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                   help="continue a session (answer a clarification); rotates channel on cooldown")
    p.add_argument("--task-id", help="session/task for -c and --undo")
    p.add_argument("--undo", nargs="?", const="", metavar="CHECKPOINT_ID",
                   help="selectively revert the last (or given) checkpoint; with --task-id every turn of that task")
    p.add_argument("--status", action="store_true", help="Ollama, channels, credit, checkpoints, sessions")
    p.add_argument("--usage", action="store_true", help="consumption per channel/model (Channel A $, Channel B tokens)")
    p.add_argument("--sync-claude", type=float, metavar="USD",
                   help="with --usage: set Channel A spend to what the interactive claude /usage shows")
    p.add_argument("--b-cooldown", nargs=2, metavar=("BUCKET", "MINUTES"),
                   help="with --usage: mark an agy bucket (gemini|argon|claude_gpt) exhausted; 0 clears")
    p.add_argument("--refresh", action="store_true",
                   help="with --usage: open agy (no prompt, no tokens) so its status line reports real quotas")
    p.add_argument("--install-hooks", action="store_true",
                   help="with --usage: register the usage status-line hook in claude and agy settings")
    p.add_argument("--benchmark", action="store_true",
                   help="measure local prefill/generation speed (§12.3) -> ~/.ai_benchmark.json")
    p.add_argument("--install-mcp", action="store_true",
                   help="install Context7 MCP at a pinned version and register it in claude and agy")
    p.add_argument("--review", action="store_true", help="code review of the working tree vs HEAD")
    review = p.add_mutually_exclusive_group()
    review.add_argument("--cross", action="store_true", help="with --review: read-only review on the other channel")
    review.add_argument("--ultra", action="store_true", help="with --review: claude ultrareview (asks first)")
    review.add_argument("--ponytail", action="store_true", help="with --review: Ponytail over-engineering audit")
    p.add_argument("--commit", action="store_true", help="secret-checked commit with a generated message")
    p.add_argument("--adr", nargs="*", metavar="ACTION", help="list | approve <id> | reject <id> | add <title> [why]")
    p.add_argument("--recover", action="store_true", help="restore files left in the NTFS vault")
    p.add_argument("--gc", action="store_true", help="delete expired checkpoints and their Git refs")
    p.add_argument("--fix", action="store_true", help="--fix [<command> ...]: repair a failing command (must be last)")
    return p



def run_repl(root: Path, cfg: Any, args: argparse.Namespace) -> int:
    """Interactive multi-turn command loop for `ai >` (spec §8, inter-prompt continuity)."""
    from ai_core import router, session, slash, tui
    last_task_id: str | None = None
    st = slash.ReplState(root, cfg, args.channel, args.offline, args.dry_run, args.verbose)  # /channel, /offline, /dry-run edit it

    def step(line: str) -> bool:
        """Handle one REPL line; True = exit."""
        nonlocal last_task_id
        line = line.strip()
        if not line:
            return False
        cmd = line.lower().lstrip("/") if line.startswith("/") else line.lower()
        if cmd in ("exit", "quit", "q"):
            return True
        if cmd in ("new", "clear", "reset"):
            last_task_id = None
            bb, bb_writable = session.load_blackboard(root)
            if bb_writable and "chat_history" in bb:
                bb["chat_history"] = []
                session.save_blackboard(root, bb)
            print("→ Started fresh conversation session (blackboard rolling task memory preserved).")
            return False
        if cmd == "history":
            bb, _ = session.load_blackboard(root)
            recent = bb.get("recent_tasks", [])
            if not recent:
                print(router._dim("No recent tasks recorded in this project."))
            else:
                for i, t in enumerate(recent, 1):
                    files_str = ", ".join(t.get("touched_files", [])) or "none"
                    print(f"{router._dim(f'{i:>2}.')} {router._bold(str(t.get('prompt')))}")
                    print(f"    {router._cyan(f'{t.get('channel')}/{t.get('model')}')} "
                          f"{router._dim('·')} {t.get('status')} {router._dim('· touched:')} {files_str} "
                          f"{router._dim('· ' + str(t.get('task_id')))}")
            return False
        if cmd == "status":
            router.show_status(root, cfg, as_json=False)
            return False
        if cmd == "usage":
            router.show_usage(cfg, as_json=False)
            return False
        if cmd == "undo":
            router.undo(root, cfg, None, None, st.options())
            return False
        if cmd in ("help", "?"):
            rows = [("/status", "Show system status"), ("/usage", "Show usage and quotas"),
                    ("/history", "Show recent task history"), ("/new", "Start a fresh conversation"),
                    ("/undo", "Revert last checkpoint (esc esc)"), *((f"/{n}", d) for n, d in slash.COMMANDS.items()),
                    ("/exit", "Exit"), ("<task>", "Run a task, e.g. fix bug in +src/calc.py")]
            print(router._bold("Commands"))
            for name, desc in rows:
                print(f"  {router._cyan(f'{name:<9}')} {router._dim(desc)}")
            return False
        if slash.run(st, line):
            return False
        words = line.split()
        splat_err = detect_powershell_splatting(words)
        if splat_err:
            print(f"✖ PowerShell splatting detected ({splat_err}). Use '+path/file' or wrap in quotes '@path/file'",
                  file=sys.stderr)
            return False
        prompt, files = split_prompt(words, root)
        if not prompt:
            return False
        cont = bool(last_task_id)
        opts = router.TaskOptions(prompt=prompt, project_root=root, file_args=tuple(files), offline=st.offline,
                                  dry_run=st.dry_run, verbose=st.verbose, agent=False,
                                  continue_session=cont, task_id=last_task_id, channel=st.channel)
        before = (slash.last_task(root) or {}).get("task_id")
        router.run(opts, cfg)
        slash.show_changes(st, before)
        bb, _ = session.load_blackboard(root)
        last_task_id = (bb.get("system_identity", {}).get("last_session") or {}).get("task_id") or last_task_id
        return False

    if tui.enabled():
        return tui.Tui(root, st.channel, mode=st.label).run(step, lambda: router.LAST_ANSWERED_BY)
    print("CodingEngine REPL — Type a task or command (status, usage, history, undo, new, help, exit).")
    while True:
        try:
            line = input("ai > ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return ExitCode.OK
        if step(line):
            return ExitCode.OK


def main(argv: list[str] | None = None) -> int:
    os.environ["PYTHONUTF8"] = "1"  # W03: inherited by child Python processes
    os.environ["WSL_UTF8"] = "1"  # wsl.exe prints its own messages as UTF-16 otherwise
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # absent on redirected StringIO streams
            stream.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--statusline-capture"] and len(argv) == 2:  # hot path: called at every status render
        from ai_core.statusline import CLIS, capture
        return capture(argv[1]) if argv[1] in CLIS else ExitCode.USAGE
    # Everything after --ait / --fix is a command line and must reach it verbatim (`ai --fix pytest -x`).
    tracked: list[str] | None = None
    if argv[:1] == ["--ait"]:
        tracked, argv = argv[1:], []
    elif "--fix" in argv:
        i = j = argv.index("--fix") + 1
        while j < len(argv) and argv[j] in _FIX_OWN_FLAGS:  # `ai --fix --offline pytest -x`
            j += 1
        tracked, argv = argv[j:], argv[:i] + argv[i:j]
    args = build_parser().parse_args(argv)
    try:
        if args.dir:
            dir_path = canonical_root(Path(args.dir).resolve())  # resolve first: `--dir .` from a \\wsl$ cwd
            if not dir_path.is_dir():
                print(f"✖ --dir directory not found: {args.dir}", file=sys.stderr)
                return ExitCode.USAGE
            root = dir_path
        else:
            root = canonical_root(find_project_root(Path.cwd()))
    except WslError as exc:
        print(f"✖ {exc}", file=sys.stderr)
        return ExitCode.USAGE
    if os.path.normcase(os.path.normpath(root)) == os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows")):
        print("✖ cmd.exe cannot run in a UNC folder (it fell back to C:\\Windows): use ai.exe, "
              "or pass --dir \\\\wsl.localhost\\...", file=sys.stderr)
        return ExitCode.USAGE
    if args.verbose and (target := wsl_target(root)):
        print(f"→ WSL: {target.distro}:{target.linux_path}", file=sys.stderr)
    try:
        cfg = load_config(root)
    except ConfigError as exc:
        print(f"✖ {exc}", file=sys.stderr)
        return ExitCode.FAILED
    from ai_core.checkpoint import git_refusal
    # --recover must never wait on Git: it puts the user's secret files back. --status/--usage only report.
    if not (args.recover or args.status or args.usage) and (refusal := git_refusal(root)):
        print(f"✖ Git refuses this repository, so checkpoints, undo, --review and --commit would run without it.\n"
              f"{refusal}", file=sys.stderr)
        return ExitCode.FAILED

    from ai_core import router  # after config validation

    base = router.TaskOptions("", root, offline=args.offline, dry_run=args.dry_run, verbose=args.verbose,
                              agent=args.agent, task_id=args.task_id, channel=args.channel)
    if tracked is not None and not args.fix:  # ait
        from ai_core.runner import run_tracked
        if not tracked:
            print("usage: ait <command> [args...]", file=sys.stderr)
            return ExitCode.USAGE
        return run_tracked(root, tracked)[0]
    if args.fix:
        from ai_core.runner import fix
        return fix(root, cfg, tracked or [], base)
    if args.status:
        return router.show_status(root, cfg, as_json=args.agent)
    if args.benchmark:
        from ai_core.benchmark import run as run_benchmark
        from ai_core.router import OllamaError
        try:
            run_benchmark(cfg, log=lambda s: print(s, file=sys.stderr))
        except OllamaError as exc:
            print(f"✖ {exc}", file=sys.stderr)
            return ExitCode.OFFLINE_UNAVAILABLE
        return ExitCode.OK
    if args.install_mcp:
        from ai_core.mcp import install
        lines = install()
        print("\n".join(lines))
        return ExitCode.FAILED if any(ln.startswith("✖") for ln in lines) else ExitCode.OK
    if args.usage or args.sync_claude is not None or args.b_cooldown or args.refresh or args.install_hooks:
        try:
            cooldown = (args.b_cooldown[0], int(args.b_cooldown[1])) if args.b_cooldown else None
        except ValueError:
            print("ai: error: --b-cooldown MINUTES must be an integer", file=sys.stderr)
            return ExitCode.USAGE
        return router.show_usage(cfg, as_json=args.agent, sync_claude=args.sync_claude, b_cooldown=cooldown,
                                 refresh=args.refresh, install_hooks=args.install_hooks)
    if args.undo is not None:
        return router.undo(root, cfg, args.undo or None, args.task_id, base)
    if args.recover:
        from ai_core.vault import recover
        messages = recover(root)
        failed = any(m.startswith("⚠") for m in messages)
        if args.agent:
            print(json.dumps({"event": "result", "exit_code": int(failed), "messages": messages}, ensure_ascii=False))
        else:
            print("\n".join(messages) or "vault empty: nothing to recover")
        return ExitCode.FAILED if failed else ExitCode.OK
    if args.gc:
        return router.gc(root, cfg)
    if args.adr is not None:
        from ai_core.adr import run_cli
        return run_cli(root, args.adr, args.agent)
    if args.commit:
        from ai_core.commit import commit
        return commit(root, cfg, base)
    if args.review or args.cross or args.ultra or args.ponytail:
        from ai_core.review import review
        mode = "cross" if args.cross else "ultra" if args.ultra else "ponytail" if args.ponytail else "local"
        return review(root, cfg, mode, base, " ".join(args.words) or None)

    splat_err = detect_powershell_splatting(args.words)
    if splat_err:
        print(f"✖ PowerShell splatting detected ({splat_err}). Use '+path/file' or wrap in quotes '@path/file'",
              file=sys.stderr)
        return ExitCode.USAGE
    prompt, files = split_prompt(args.words, root)
    if args.file:
        files = list(dict.fromkeys([*files, *args.file]))
    if not prompt and sys.stdin.isatty() and not args.agent:
        return run_repl(root, cfg, args)
    if not prompt:
        build_parser().print_usage(sys.stderr)
        print("ai: error: a task description is required", file=sys.stderr)
        return ExitCode.USAGE
    opts = router.TaskOptions(prompt=prompt, project_root=root, file_args=tuple(files), offline=args.offline,
                              dry_run=args.dry_run, verbose=args.verbose, agent=args.agent,
                              continue_session=args.continue_session, task_id=args.task_id, channel=args.channel)
    return router.run(opts, cfg)


def ait_main() -> int:
    """`ait.exe` entry point (uv tool install): the whole command line is the tracked command."""
    return main(["--ait", *sys.argv[1:]])


if __name__ == "__main__":
    try:
        sys.exit(int(main()))
    except KeyboardInterrupt:
        print("\n✖ interrupted", file=sys.stderr)
        sys.exit(130)
