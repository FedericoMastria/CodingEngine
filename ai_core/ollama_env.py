"""Effective Ollama server settings (spec §2.A, L02): read from the server's own startup log.

The variables belong to the Ollama server process, not to ours, so checking `os.environ` would lie
(the server may have started before a `setx`, or from another account). On Windows the server logs
`msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true ...]"` at every start: the last such line
in %LOCALAPPDATA%\\Ollama\\server.log is what it is really running with.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# variable -> (required value as `setx` takes it, predicate on the logged value)
REQUIRED: dict[str, tuple[str, object]] = {
    "OLLAMA_FLASH_ATTENTION": ("1", lambda v: v.lower() in ("true", "1")),
    "OLLAMA_KV_CACHE_TYPE": ("q8_0", lambda v: v.lower() == "q8_0"),
    # -1 is logged as Go's max duration (2562047h47m16.854775807s); any negative also means forever
    "OLLAMA_KEEP_ALIVE": ("-1", lambda v: v.startswith("-") or v.startswith("2562047h")),
    "OLLAMA_NUM_PARALLEL": ("1", lambda v: v == "1"),
}
_PAIR_RE = re.compile(r"\b(OLLAMA_[A-Z_]+):(\S*)")


def server_log() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Ollama" / "server.log"


def server_env(log: Path | None = None) -> dict[str, str] | None:
    """OLLAMA_* values of the last server start, or None when the log is missing/unreadable."""
    try:
        text = (log or server_log()).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if 'msg="server config"' in ln]
    if not lines:
        return None
    return {k: v.rstrip("]") for k, v in _PAIR_RE.findall(lines[-1])}


def issues(env: dict[str, str] | None) -> list[str]:
    """Human-readable mismatches; empty when compliant. Unknown state is reported, not assumed OK."""
    if env is None:
        return [f"server config not found in {server_log()} (cannot verify OLLAMA_* settings)"]
    return [f"{name}={env.get(name) or '(unset)'} (required {want})"
            for name, (want, ok) in REQUIRED.items() if not ok(env.get(name, ""))]  # type: ignore[operator]


def fix_hint() -> str:
    cmds = " ; ".join(f"setx {name} {want}" for name, (want, _) in REQUIRED.items())
    return f"{cmds}  — then quit Ollama from the tray and start it again"
