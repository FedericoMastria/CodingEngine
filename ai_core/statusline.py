"""Status-line hook: the official, free way both CLIs hand out real plan usage.

`ai.exe --statusline-capture claude|agy` is registered as the status-line command. The CLI pipes
its status JSON on stdin at each render; we save it atomically to ~/.ai_usage/<cli>.json and print
a short line for the CLI to display. Any failure degrades to printing nothing: the user's
status line must never break because of us.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_core import config as cfgmod

CLIS = ("claude", "agy")
USAGE_DIR: Path | None = None  # None = ~/.ai_usage; tests point it elsewhere
_STDIN_TIMEOUT_S = 2.0


def usage_dir() -> Path:
    return USAGE_DIR or cfgmod.HOME / ".ai_usage"


def capture_path(cli: str) -> Path:
    return usage_dir() / f"{cli}.json"


def _read_stdin(timeout_s: float) -> str:
    """stdin with a deadline: a CLI that never closes the pipe must not hang its own render."""
    box: list[str] = []
    t = threading.Thread(target=lambda: box.append(sys.stdin.read()), daemon=True)
    t.start()
    t.join(timeout_s)
    return box[0] if box else ""


def summarize(cli: str, data: dict[str, Any]) -> str:
    """Short text for the status line itself."""
    if cli == "claude":
        rl = data.get("rate_limits") or {}
        parts = [f"{label} {rl[key]['used_percentage']:.0f}%" for key, label in (("five_hour", "5h"), ("seven_day", "7d"))
                 if isinstance(rl.get(key), dict) and isinstance(rl[key].get("used_percentage"), (int, float))]
        model = (data.get("model") or {}).get("display_name") if isinstance(data.get("model"), dict) else None
        return " · ".join([p for p in [model, *parts] if p])
    windows = agy_windows({"data": data})
    return " · ".join(f"{name} {100 - w['remaining_pct']:.0f}%" for name, w in windows.items())


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def agy_windows(record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """{window: {remaining_pct, resets_at}} from agy's status JSON (verified on 1.4.7):
    "quota": {"gemini-5h": {"remaining_fraction": 0.90, "reset_time": "2026-10-25T22:29:34Z"}, ...}."""
    data = (record or {}).get("data")
    quota = data.get("quota") if isinstance(data, dict) else None
    out: dict[str, dict[str, Any]] = {}
    for name, w in (quota or {}).items():
        frac = _num(w.get("remaining_fraction")) if isinstance(w, dict) else None
        if frac is None:
            continue
        try:
            resets = datetime.fromisoformat(str(w.get("reset_time")).replace("Z", "+00:00"))
        except ValueError:
            resets = None
        out[str(name)] = {"remaining_pct": frac * 100, "resets_at": resets}  # full precision; round only to display
    return out


def claude_windows(record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """{five_hour|seven_day: {used_pct, resets_at}} from Claude Code's documented status JSON."""
    data = (record or {}).get("data")
    rl = data.get("rate_limits") if isinstance(data, dict) else None
    out: dict[str, dict[str, Any]] = {}
    for name, w in (rl or {}).items():
        used = _num(w.get("used_percentage")) if isinstance(w, dict) else None
        if used is not None:
            at = _num(w.get("resets_at"))
            out[str(name)] = {"used_pct": used, "resets_at": datetime.fromtimestamp(at, UTC) if at else None}
    return out


def settings_path(cli: str) -> Path:
    if cli == "claude":
        return cfgmod.HOME / ".claude" / "settings.json"
    return cfgmod.HOME / ".gemini" / "antigravity-cli" / "settings.json"


def hook_command(exe: Path, cli: str) -> str:
    return f"{exe} --statusline-capture {cli}"


def install_hook(cli: str, exe: Path) -> str:
    """Register our status-line command in the CLI's settings.json (same schema in claude and agy).
    Never replaces a status line the user configured; keeps a one-time backup of the original file."""
    path = settings_path(cli)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as exc:
        return f"skip   {path}: unreadable ({exc})"
    if not isinstance(data, dict):
        return f"skip   {path}: not a JSON object"
    wanted = {"type": "command", "command": hook_command(exe, cli)}
    current = data.get("statusLine")
    if current == wanted:
        return f"same   {path}"
    if current and "--statusline-capture" not in json.dumps(current):
        return f"skip   {path}: a status line is already configured ({json.dumps(current)[:80]})"
    backup = path.with_name(path.name + ".before-codingengine")
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())
    data["statusLine"] = wanted
    from ai_core.checkpoint import write_bytes_atomic
    write_bytes_atomic(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return f"ok     {path}"


def refresh_agy(cwd: Path, timeout_s: float = 45.0) -> bool:
    """Open interactive agy in a pseudo-console WITHOUT any prompt until its status line renders
    (verified: 0 input / 0 output tokens), then close it. True when a fresh capture arrived."""
    import time

    from ai_core.conpty import run_in_conpty
    from ai_core.security import sanitize_subprocess_env

    start = datetime.now(UTC)
    t0 = time.monotonic()

    def fresh(_out: str) -> bool:
        rec = load("agy")
        try:
            return bool(rec) and datetime.fromisoformat(rec["captured_at"]) >= start.replace(microsecond=0) \
                and bool(agy_windows(rec))
        except (KeyError, ValueError):
            return False

    run_in_conpty(["agy"], cwd=str(cwd), env=sanitize_subprocess_env(os.environ, warn=False, cloud=True),
                  timeout_s=timeout_s, done=lambda o: time.monotonic() - t0 > 2 and fresh(o))
    return fresh("")


def capture(cli: str) -> int:
    try:
        raw = _read_stdin(_STDIN_TIMEOUT_S)
        try:
            data: Any = json.loads(raw) if raw.strip() else None
        except ValueError:
            data = None
        record = {"captured_at": datetime.now(UTC).isoformat(timespec="seconds"), "cli": cli,
                  "data": data, "raw": None if data is not None else raw[:20000],
                  "argv": sys.argv[1:], "env": {k: v for k, v in os.environ.items()
                                                 if k.upper().startswith(("AGY_", "JETSKI_", "STATUSLINE"))}}
        from ai_core.checkpoint import write_json_atomic  # lazy: keep the hot path import-light
        usage_dir().mkdir(parents=True, exist_ok=True)
        write_json_atomic(capture_path(cli), record)
        line = summarize(cli, data) if isinstance(data, dict) else ""
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:  # a status line must never crash its host CLI
        pass
    return 0


def load(cli: str) -> dict[str, Any] | None:
    try:
        record = json.loads(capture_path(cli).read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else None
    except (OSError, ValueError):
        return None


def check_and_spawn_background_usage_refresh(cfg: Any, home: Path | None = None) -> bool:
    """Spawn background refresh of agy quota if older than usage_max_age_min (Q04, §7.A)."""
    import threading
    h = HOME if home is None else home
    agy_file = h / ".ai_usage" / "agy.json"
    stale = True
    if agy_file.is_file():
        try:
            mtime = agy_file.stat().st_mtime
            max_age_s = getattr(cfg, "usage_max_age_min", 15) * 60
            if (time.time() - mtime) < max_age_s:
                stale = False
        except OSError:
            pass
    if stale:
        t = threading.Thread(target=refresh_agy, args=(h,), daemon=True)
        t.start()
        return True
    return False
