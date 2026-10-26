"""WSL mode: projects under `\\\\wsl.localhost\\<distro>\\...` keep running on Windows, but every *project* command
(git, verification, `ait`) is rewritten to run inside the distro via `wsl.exe` (spec 2026-10-26-wsl-mode-design).

`project_argv` is the only place that decides Windows vs WSL. Claude, agy and Ollama are Windows tools: never wrapped.
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_UNC = re.compile(r"^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)(?:\\(.*))?$", re.IGNORECASE)
_MNT = re.compile(r"^/mnt/([a-z])(/.*)?$")
_TIMEOUT_S = 30


class WslError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WslTarget:
    distro: str
    linux_path: str


def wsl_target(path: str | os.PathLike[str]) -> WslTarget | None:
    """Parse a WSL UNC path (either prefix, any case); None for every other path."""
    m = _UNC.match(os.fspath(path).replace("/", "\\").rstrip("\\"))
    return WslTarget(m[1], "/" + (m[2] or "").replace("\\", "/")) if m else None


def _unc(distro: str, linux_path: str) -> Path:
    return Path(rf"\\wsl.localhost\{distro}" + linux_path.replace("/", "\\").rstrip("\\"))


def to_windows(target: WslTarget, linux_path: str) -> Path:
    """Windows path of a Linux path git printed: /mnt/<d>/x -> D:\\x, anything else -> the distro's UNC share."""
    if m := _MNT.match(linux_path):
        return Path(f"{m[1].upper()}:" + (m[2] or "/").replace("/", "\\"))
    return _unc(target.distro, linux_path)


def _wsl_env() -> dict[str, str]:
    from ai_core.security import sanitize_subprocess_env  # security imports this module: a local import breaks the cycle
    # WSL_UTF8: without it wsl.exe's own messages are UTF-16. Billing vars must not ride WSLENV into the distro.
    return {**sanitize_subprocess_env(os.environ, warn=False), "WSL_UTF8": "1"}


def _wsl_out(*args: str) -> str:
    try:
        cp = subprocess.run(["wsl.exe", *args], capture_output=True, encoding="utf-8", errors="replace",
                            env=_wsl_env(), timeout=_TIMEOUT_S)
    except FileNotFoundError as exc:
        raise WslError("wsl.exe not found: WSL is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise WslError(f"wsl.exe {' '.join(args)} timed out after {_TIMEOUT_S}s") from exc
    if cp.returncode != 0:
        raise WslError(f"wsl.exe {' '.join(args)} failed: {(cp.stderr or cp.stdout).strip()}")
    return cp.stdout


@functools.cache
def distros() -> list[str]:
    return [line.strip().strip("\x00") for line in _wsl_out("-l", "-q").splitlines() if line.strip().strip("\x00")]


@functools.cache
def linux_home(distro: str) -> str:
    return _wsl_out("-d", distro, "-e", "sh", "-c", "echo $HOME").strip()


def canonical_root(path: Path) -> Path:
    """One spelling per project (`\\\\wsl$\\ubuntu` and `\\\\wsl.localhost\\Ubuntu` are the same), before any hashing."""
    t = wsl_target(path)
    if t is None:
        return path
    installed = distros()
    for name in installed:
        if name.lower() == t.distro.lower():
            return _unc(name, t.linux_path)
    raise WslError(f"WSL distro '{t.distro}' not found (installed: {', '.join(installed) or 'none'})")


def project_argv(cwd: Path, argv: Sequence[str]) -> list[str]:
    """Windows project: argv unchanged. WSL project: run it inside the distro, in the project's Linux directory.

    Arguments that are absolute paths in the same distro are translated to their Linux form (safety net).
    """
    t = wsl_target(cwd)
    if t is None:
        return list(argv)
    args = []
    for arg in argv:
        a = wsl_target(arg)
        args.append(a.linux_path if a and a.distro.lower() == t.distro.lower() else arg)
    return ["wsl.exe", "-d", t.distro, "--cd", t.linux_path, "-e", *args]


def wsl_python(root: Path) -> str:
    """The project's own interpreter (`.venv/bin/python`) if it has one, else the distro's `python3` (`python` is absent)."""
    # asked inside the distro: the venv's python (or the whole .venv) is a symlink, dangling from Windows' point of view
    try:
        found = subprocess.run(project_argv(root, ["test", "-x", ".venv/bin/python"]), cwd=root, capture_output=True,
                               env=_wsl_env(), timeout=_TIMEOUT_S).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        found = False
    return ".venv/bin/python" if found else "python3"


def project_key(root: Path, rel: str) -> str:
    """Comparison key for a project-relative path: exact in a WSL project, case-folded on Windows."""
    rel = rel.replace("\\", "/")
    return rel if wsl_target(root) else rel.lower()


@functools.cache
def linux_claude(distro: str) -> str:
    """Absolute path of the distro's `claude`. `wsl.exe -e` has a bare PATH (no ~/.local/bin): ask a login shell."""
    try:
        lines = _wsl_out("-d", distro, "-e", "sh", "-lc", "command -v claude").splitlines()
    except WslError as exc:
        raise WslError(f"claude is not installed in WSL distro '{distro}': install it there and log in") from exc
    paths = [line.strip() for line in lines if line.strip().startswith("/")]  # profiles may print banners first
    if not paths:
        raise WslError(f"claude is not installed in WSL distro '{distro}': install it there and log in")
    return paths[-1]
