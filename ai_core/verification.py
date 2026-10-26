"""Fast verification with pre-task baseline (spec §5.C.1, R04).

L1 syntax of touched files (in-process: compile / json / tomllib), L2 optional project
typecheck/build command, L3 targeted tests for the touched Python modules. Each external
level has its own timeout; a timeout is `verification_timeout`, never `tests_failed`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ai_core import TaskStatus, kill_process_tree
from ai_core.config import CONTEXT_DIR_NAME, Config
from ai_core.security import sanitize_subprocess_env
from ai_core.wsl import project_argv, wsl_python, wsl_target

PASSED, FAILED, TIMEOUT, SKIPPED = "passed", "failed", "timeout", "skipped"
_OUTPUT_TAIL = 4000
_WIN_COMMAND_NOT_FOUND = 9009  # cmd.exe / Microsoft Store `python` alias when not installed
_POSIX_COMMAND_NOT_FOUND = 127  # sh inside a WSL distro
# pytest "FAILED tests/x.py::t - ...", unittest "FAIL: test_x (mod.Cls.test_x)".
# Ids starting with "(" are summary counters ("FAILED (failures=2)"), not tests.
_TEST_ID = re.compile(r"^(?:FAILED|ERROR|FAIL)[: ]\s*([^\s(]\S*(?: \([\w.]+\))?)", re.M)
# tsc "src/a.ts(12,5): error TS2322: ..." -> "src/a.ts:TS2322" (no line: edits above it must not make it "new")
_TSC_ERROR = re.compile(r"^(\S.*?)\(\d+,\d+\): error (TS\d+):", re.M)


@dataclass(frozen=True, slots=True)
class LevelResult:
    level: int
    status: str
    failures: frozenset[str] = frozenset()  # comparable failure ids; empty if unparseable
    output: str = ""


@dataclass(slots=True)
class Baseline:
    levels: dict[int, LevelResult] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    status: TaskStatus
    verification_level: int  # highest level that passed (0 = none)
    levels: list[LevelResult]
    new_failures: list[str] = field(default_factory=list)


def _run(cmd: Sequence[str], cwd: Path, timeout_s: int, level: int) -> LevelResult:
    try:
        proc = subprocess.Popen(
            project_argv(cwd, cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            encoding="utf-8", errors="replace", env=sanitize_subprocess_env(os.environ, warn=False),
        )
    except (FileNotFoundError, PermissionError) as exc:  # missing tool, WinError 5 on the executable
        return LevelResult(level, SKIPPED, output=f"command unavailable: {exc}")
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        output, _ = proc.communicate()
        return LevelResult(level, TIMEOUT, output=f"timed out after {timeout_s}s\n{(output or '')[-_OUTPUT_TAIL:]}")
    output = (output or "")[-_OUTPUT_TAIL:]
    if proc.returncode == 0:
        return LevelResult(level, PASSED, output=output)
    if proc.returncode in ((_WIN_COMMAND_NOT_FOUND, _POSIX_COMMAND_NOT_FOUND) if wsl_target(cwd) else (_WIN_COMMAND_NOT_FOUND,)):
        return LevelResult(level, SKIPPED, output=f"command unavailable (exit {proc.returncode}): {cmd[0]}")
    ids = set(_TEST_ID.findall(output)) | {f"{p.replace(chr(92), '/')}:{code}" for p, code in _TSC_ERROR.findall(output)}
    return LevelResult(level, FAILED, frozenset(ids), output)


def check_syntax(files: Iterable[Path], root: Path) -> LevelResult:
    failures: set[str] = set()
    messages: list[str] = []
    checked = 0
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in {".py", ".json", ".toml"} or not path.is_file():
            continue
        rel = Path(os.path.relpath(path, root)).as_posix()
        try:
            data = path.read_bytes()
            if suffix == ".py":
                compile(data, str(path), "exec", dont_inherit=True)  # bytes: honours PEP 263 cookies
            elif suffix == ".json":
                json.loads(data)
            else:
                tomllib.loads(data.decode("utf-8"))
            checked += 1
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:  # JSON/TOML errors subclass ValueError
            checked += 1
            failures.add(f"syntax:{rel}")
            messages.append(f"{rel}: {exc}")
        except OSError as exc:  # WinError 32 while an editor saves: report, don't guess
            messages.append(f"{rel}: unreadable ({exc})")
    if failures:
        return LevelResult(1, FAILED, frozenset(failures), "\n".join(messages))
    return LevelResult(1, PASSED if checked else SKIPPED, output="\n".join(messages))


def find_targeted_tests(files: Iterable[Path], root: Path, cfg: Config) -> list[str]:
    """Test modules for the given Python files: the file itself if it is a test, else test_<stem>.py / <stem>_test.py."""
    wanted: set[str] = set()
    tests: set[str] = set()
    for path in files:
        if path.suffix.lower() != ".py":
            continue
        name = path.name.lower()
        if name.startswith("test_") or name.endswith("_test.py"):
            if path.is_file():
                tests.add(Path(os.path.relpath(path, root)).as_posix())
        else:
            stem = path.stem.lower()
            wanted |= {f"test_{stem}.py", f"{stem}_test.py"}
    if wanted:
        pruned = {d.lower() for d in cfg.checkpoint_exclude_dirs} | {".git", CONTEXT_DIR_NAME}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in pruned]
            for fn in filenames:
                if fn.lower() in wanted:
                    tests.add(Path(os.path.relpath(Path(dirpath) / fn, root)).as_posix())
    return sorted(tests)


def default_l2_command(root: Path) -> list[str]:
    """`tsc --noEmit` for TypeScript projects with a local TypeScript install; [] otherwise (never npx: no downloads)."""
    tsc = root / "node_modules" / "typescript" / "bin" / "tsc"
    if wsl_target(root):  # Linux node inside the distro; Windows `node` and absolute UNC paths would not work there
        return ["node", "node_modules/typescript/bin/tsc", "--noEmit"] if tsc.is_file() and (root / "tsconfig.json").is_file() else []
    node = shutil.which("node")
    return [node, str(tsc), "--noEmit"] if node and tsc.is_file() and (root / "tsconfig.json").is_file() else []


def _run_levels(root: Path, cfg: Config, files: list[Path]) -> dict[int, LevelResult]:
    levels = {1: check_syntax(files, root)}
    l2 = cfg.verify_l2_command or default_l2_command(root)
    levels[2] = (_run(l2, root, cfg.verify_l2_timeout_s, 2) if l2
                 else LevelResult(2, SKIPPED, output="verify_l2_command not configured"))
    tests = find_targeted_tests(files, root, cfg)
    l3 = cfg.verify_l3_command
    if wsl_target(root) and l3[:1] == ["python"]:
        l3 = [wsl_python(root), *l3[1:]]
    levels[3] = (_run([*l3, *tests], root, cfg.verify_l3_timeout_s, 3) if tests
                 else LevelResult(3, SKIPPED, output="no targeted tests found"))
    return levels


def compute_baseline(root: Path, cfg: Config, target_files: Iterable[Path]) -> Baseline:
    """Run the same levels on the pre-task tree so already-red tests aren't blamed on the AI."""
    return Baseline(_run_levels(root, cfg, list(target_files)))


def verify(root: Path, cfg: Config, touched_files: Iterable[Path], baseline: Baseline | None = None) -> VerificationResult:
    levels = _run_levels(root, cfg, list(touched_files))
    ordered = [levels[i] for i in sorted(levels)]
    if any(lv.status == TIMEOUT for lv in ordered):
        return VerificationResult(TaskStatus.VERIFICATION_TIMEOUT, _passed_level(ordered), ordered)
    new_failures: list[str] = []
    preexisting = False
    for lv in ordered:
        if lv.status != FAILED:
            continue
        base = baseline.levels.get(lv.level) if baseline else None
        # Only comparable (parsed) failure ids can be excused; opaque failures always count as new.
        if base and base.status == FAILED and lv.failures and lv.failures <= base.failures:
            preexisting = True
        else:
            new_failures.extend(sorted(lv.failures - (base.failures if base else frozenset())) or [f"L{lv.level}"])
    if new_failures:
        return VerificationResult(TaskStatus.TESTS_FAILED, _passed_level(ordered), ordered, new_failures)
    if preexisting:
        return VerificationResult(TaskStatus.COMPLETED_WITH_PREEXISTING_FAILURES, _passed_level(ordered), ordered)
    level = _passed_level(ordered)
    status = {0: TaskStatus.COMPLETED_UNVERIFIED, 1: TaskStatus.COMPLETED_SYNTAX_ONLY}.get(level, TaskStatus.COMPLETED_VERIFIED)
    return VerificationResult(status, level, ordered)


def _passed_level(levels: list[LevelResult]) -> int:
    return max((lv.level for lv in levels if lv.status == PASSED), default=0)
