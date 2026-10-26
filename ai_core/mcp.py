"""Context7 MCP with an exact, pre-installed version (spec §3.B.2, §12.2.4, S06).

Never `npx <pkg>` at runtime: the package is installed once into ~/.ai_mcp/context7 with an exact
version (package-lock.json freezes the transitive tree, `--ignore-scripts` blocks install hooks) and
each CLI runs that local copy with `node`. Disabled per call when the project holds secret files
(claude: `--strict-mcp-config`; agy runs under the Vault, so those files are not even present).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ai_core import config as cfgmod

# >= 2 weeks old when pinned (published 2026-10-09): newer releases get no supply-chain soak time.
CONTEXT7_PACKAGE = "@upstash/context7-mcp"
CONTEXT7_VERSION = "4.3.0"
NAME = "context7"


def install_dir() -> Path:
    return cfgmod.HOME / ".ai_mcp" / NAME


def entry_point() -> Path:
    return install_dir() / "node_modules" / "@upstash" / "context7-mcp" / "dist" / "index.js"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=300, check=False)


def install(clis: tuple[str, ...] = ("claude", "agy")) -> list[str]:
    """Install the pinned package locally and register it in each CLI. Returns report lines."""
    npm, node = shutil.which("npm"), shutil.which("node")
    if not npm or not node:
        return ["✖ node/npm not found on PATH"]
    report: list[str] = []
    installed = _run([npm, "ls", "--prefix", str(install_dir()), "--depth=0", "--json"]).stdout
    if f'"version": "{CONTEXT7_VERSION}"' not in installed:
        install_dir().mkdir(parents=True, exist_ok=True)
        proc = _run([npm, "install", "--prefix", str(install_dir()), "--save-exact", "--ignore-scripts",
                     "--no-fund", "--no-audit", f"{CONTEXT7_PACKAGE}@{CONTEXT7_VERSION}"])
        if proc.returncode != 0:
            return [f"✖ npm install failed: {(proc.stderr or proc.stdout).strip()[-400:]}"]
    if not entry_point().is_file():
        return [f"✖ {entry_point()} missing after install"]
    report.append(f"ok     {CONTEXT7_PACKAGE}@{CONTEXT7_VERSION} in {install_dir()}")
    for cli in clis:
        exe = shutil.which(cli)
        if not exe:
            report.append(f"skip   {cli}: not on PATH")
            continue
        _run([exe, "mcp", "remove", NAME] + (["--scope", "user"] if cli == "claude" else []))  # idempotent
        add = ([exe, "mcp", "add", "--scope", "user", NAME, "--", node, str(entry_point())] if cli == "claude"
               else [exe, "mcp", "add", NAME, "--", node, str(entry_point())])
        proc = _run(add)
        report.append(f"{'ok    ' if proc.returncode == 0 else '✖     '} {cli}: "
                      f"{(proc.stdout or proc.stderr).strip().splitlines()[-1:] or ['registered']}")
    return report
