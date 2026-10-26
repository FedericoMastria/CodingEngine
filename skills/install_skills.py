"""Universal skill distribution (spec §12.2): ai-orchestrator + Ponytail + Superpowers for every agent.

Targets
  claude    ~/.claude/skills/ai-orchestrator (Ponytail/Superpowers are already Claude Code plugins)
  gemini    ~/.gemini/config/skills/  (Antigravity's documented global discovery root)
            ~/.gemini/antigravity/skills/
  hermes    ~/.hermes/skills/  (only if ~/.hermes exists)
  --target  any extra skills directory (e.g. Jarvis)
  --project AGENTS.md block (Cursor, Windsurf, Copilot, Antigravity) + .cursor/rules/ai-orchestrator.mdc

Safety: third-party skills are copied from the version-pinned, already-reviewed Claude Code plugin
cache (S06), never downloaded. A destination that this installer did not create (no marker file)
is never overwritten; in AGENTS.md only our delimited block is replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MARKER = ".codingengine-skill.json"
PLUGINS = ("ponytail@ponytail", "superpowers@superpowers-dev")
BEGIN, END = "<!-- codingengine:ai-orchestrator:begin -->", "<!-- codingengine:ai-orchestrator:end -->"
CURSOR_RULE = ".cursor/rules/ai-orchestrator.mdc"
MANAGED_LINE = "<!-- managed by CodingEngine install_skills.py -->"


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    source: Path
    origin: str  # "codingengine" | "<plugin>@<version>"


def ai_cmd() -> str:
    """Prefer the native `ai.exe` (uv tool install): agents spawning a .cmd go through `cmd.exe /c`,
    which re-parses > | & in arguments (BatBadBut class). The .cmd is only a fallback."""
    exe = Path.home() / ".local" / "bin" / "ai.exe"
    return str(exe) if exe.is_file() else shutil.which("ai.exe") or str(REPO / "ai.cmd")


def plugin_skills(home: Path) -> tuple[list[Skill], list[str]]:
    """Skills of the pinned Claude Code plugins, plus messages about missing plugins."""
    try:
        installed = json.loads((home / ".claude" / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], ["Claude Code plugin registry not found: Ponytail/Superpowers not distributed"]
    skills: list[Skill] = []
    notes: list[str] = []
    for plugin in PLUGINS:
        entries = installed.get("plugins", {}).get(plugin) or []
        if not entries:
            notes.append(f"{plugin} not installed in Claude Code: `claude plugin install {plugin}` (after reviewing it)")
            continue
        entry = entries[0]
        root = Path(entry["installPath"]) / "skills"
        skills += [Skill(d.name, d, f"{plugin} v{entry.get('version', '?')}")
                   for d in sorted(root.iterdir()) if d.is_dir() and (d / "SKILL.md").is_file()] if root.is_dir() else []
    return skills, notes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_skill(skill: Skill, dest_root: Path, dry_run: bool) -> str:
    dst = dest_root / skill.name
    if dst.exists() and not (dst / MARKER).is_file():
        return f"skip   {dst} (exists and is not managed by CodingEngine)"
    if dry_run:
        return f"would  {dst} <- {skill.origin}"
    dest_root.mkdir(parents=True, exist_ok=True)
    tmp = dest_root / f".{skill.name}.tmp-{secrets.token_hex(4)}"
    try:
        shutil.copytree(skill.source, tmp, ignore=shutil.ignore_patterns(MARKER, "__pycache__", ".git"))
        if skill.origin == "codingengine":
            md = tmp / "SKILL.md"
            md.write_text(md.read_text(encoding="utf-8").replace("{{AI_CMD}}", ai_cmd()), encoding="utf-8")
        (tmp / MARKER).write_text(json.dumps({
            "origin": skill.origin, "skill_sha256": _sha(tmp / "SKILL.md"),
            "installed_at": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
        if dst.exists():
            shutil.rmtree(dst)  # ours (marker checked above)
        os.replace(tmp, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"ok     {dst} <- {skill.origin}"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{secrets.token_hex(4)}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def agents_block() -> str:
    cmd = ai_cmd()
    return f"""{BEGIN}
## CodingEngine orchestrator

Delegate code changes in this repository to the local-first router instead of editing blindly:
`{cmd} --agent "<task>" +path/to/file` (mark files with `+path`, never `@path` in PowerShell).
It checkpoints first, routes to local Qwen / Antigravity / Claude Code, and verifies the result.
The last stdout line is `{{"event": "result", ...}}`; exit codes: 0 done, 1 failed, 2 usage,
10 needs clarification (answer with `{cmd} -c --task-id <id> --agent "<answer>"`), 11 offline unavailable,
12 quota exhausted, 13 secret blocked. Revert with `{cmd} --undo`. Never put secrets in a task.
{END}
"""


def install_agents_md(project: Path, dry_run: bool) -> str:
    path = project / "AGENTS.md"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if BEGIN in current and END in current:
        head, _, rest = current.partition(BEGIN)
        new = head + agents_block().rstrip("\n") + rest.partition(END)[2]
    else:
        new = current + ("\n" if current and not current.endswith("\n") else "") + ("\n" if current else "") + agents_block()
    if new == current:
        return f"same   {path}"
    if not dry_run:
        _write_text_atomic(path, new)
    return f"{'would ' if dry_run else 'ok    '} {path}"


def install_cursor_rule(project: Path, dry_run: bool) -> str:
    path = project / CURSOR_RULE
    if path.exists() and MANAGED_LINE not in path.read_text(encoding="utf-8", errors="replace"):
        return f"skip   {path} (exists and is not managed by CodingEngine)"
    body = agents_block().replace(BEGIN + "\n", "").replace(END + "\n", "")
    text = ("---\ndescription: Delegate code changes to the CodingEngine router (ai --agent)\n"
            f"alwaysApply: false\n---\n{MANAGED_LINE}\n{body}")
    if not dry_run:
        _write_text_atomic(path, text)
    return f"{'would ' if dry_run else 'ok    '} {path}"


def plan_targets(home: Path, extra: list[Path]) -> list[tuple[str, Path, bool]]:
    """(label, skills dir, only_orchestrator). Hermes only when it is actually installed."""
    targets = [("claude", home / ".claude" / "skills", True),
               ("gemini", home / ".gemini" / "config" / "skills", False),
               ("antigravity", home / ".gemini" / "antigravity" / "skills", False)]
    if (home / ".hermes").is_dir():
        targets.append(("hermes", home / ".hermes" / "skills", False))
    targets += [("extra", p, False) for p in extra]
    return targets


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Distribute ai-orchestrator, Ponytail and Superpowers skills.")
    p.add_argument("--project", action="append", type=Path, default=[], help="write AGENTS.md + Cursor rule here")
    p.add_argument("--target", action="append", type=Path, default=[], help="extra skills directory (e.g. Jarvis)")
    p.add_argument("--home", type=Path, default=Path.home(), help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true", help="show what would be written")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # absent on redirected StringIO streams
            stream.reconfigure(encoding="utf-8", errors="replace")

    orchestrator = Skill("ai-orchestrator", HERE / "ai-orchestrator", "codingengine")
    third_party, notes = plugin_skills(args.home)
    failures = 0
    for label, dest, only_ours in plan_targets(args.home, args.target):
        print(f"[{label}] {dest}")
        for skill in [orchestrator] if only_ours else [orchestrator, *third_party]:
            try:
                print("  " + install_skill(skill, dest, args.dry_run))
            except OSError as exc:  # WinError 5/32: AV scanner or an open editor on the folder
                failures += 1
                print(f"  fail   {dest / skill.name}: {exc}")
    for project in args.project:
        print(f"[project] {project}")
        if not project.is_dir():
            failures += 1
            print(f"  fail   {project} is not a directory")
            continue
        for step in (install_agents_md, install_cursor_rule):
            try:
                print("  " + step(project, args.dry_run))
            except OSError as exc:
                failures += 1
                print(f"  fail   {project}: {exc}")
    for note in notes:
        print(f"note: {note}")
    if not (args.home / ".hermes").is_dir():
        print("note: Hermes not found (~/.hermes); use --target <dir> for Jarvis/Hermes skill folders")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
