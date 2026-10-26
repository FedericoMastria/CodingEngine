"""Project memory and sessions (spec §3.B.3, §6.2, §10): blackboard I/O, pending sessions with an
immutable origin checkpoint (R05), and the single-task-per-worktree lock with PID-reuse safety (R07)."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from ai_core.checkpoint import retry_win_lock, write_json_atomic
from ai_core.config import CONTEXT_DIR_NAME
from ai_core.security import redact_secrets

BLACKBOARD = "blackboard.json"
LOCK_FILE = ".lock"
SESSIONS_KEEP = 20
WORKING_SET_CAP = 20
RECENT_TASKS_CAP = 3
INITIAL_BLACKBOARD: dict[str, Any] = {
    "schema_version": 1,
    "system_identity": {"last_session": None},
    "pending_sessions": {},
    "architectural_decisions": [],
    "active_working_set": {"files": [], "pending_errors": []},
}


class SessionError(RuntimeError):
    pass


class ProjectBusyError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- blackboard


def load_blackboard(root: Path) -> tuple[dict[str, Any], bool]:
    """Return (blackboard, writable). A corrupt file is never overwritten."""
    path = root / CONTEXT_DIR_NAME / BLACKBOARD
    try:
        data = json.loads(retry_win_lock(path.read_text, "utf-8"))
        if isinstance(data, dict):
            return data, True
    except FileNotFoundError:
        return json.loads(json.dumps(INITIAL_BLACKBOARD)), True
    except (OSError, ValueError):
        pass
    print(f"⚠ {path} is unreadable or corrupt: project memory is read-only for this run", file=sys.stderr)
    return json.loads(json.dumps(INITIAL_BLACKBOARD)), False


def _redacted(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: _redacted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    return value


def save_blackboard(root: Path, bb: dict[str, Any]) -> None:
    """Secrets typed into a prompt are never persisted: on disk they would also be injected into every
    later dispatch, where the S01 scrubber would block every cloud task."""
    write_json_atomic(root / CONTEXT_DIR_NAME / BLACKBOARD, _redacted(bb))


def blackboard_context(bb: dict[str, Any]) -> str:
    """Data-only injection (M01): active ADRs + working set, inside <project_memory_data>."""
    adrs = [{k: a.get(k) for k in ("id", "title", "rationale")}
            for a in bb.get("architectural_decisions", []) if isinstance(a, dict) and a.get("status") == "active"]
    data = {
        "active_adrs": adrs,
        "active_working_set": bb.get("active_working_set", {}),
        "recent_tasks": bb.get("recent_tasks", []),
    }
    body = json.dumps(data, ensure_ascii=False).replace("</project_memory_data", "<\\/project_memory_data")
    return f"<project_memory_data>\n{body}\n</project_memory_data>"




def record_recent_task(
    bb: dict[str, Any], task_id: str, *, prompt: str, channel: str, model: str,
    status: str, touched_files: list[str], checkpoint_id: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Maintain rolling task history in blackboard.json for inter-prompt continuity."""
    tasks = bb.setdefault("recent_tasks", [])
    entry = {
        "task_id": task_id,
        "timestamp": _now(),
        "prompt": prompt[:400],
        "channel": channel,
        "model": model,
        "status": status,
        "touched_files": list(dict.fromkeys(touched_files))[:8],
        "checkpoint_id": checkpoint_id,
        "summary": summary[:200] if summary else "",
    }
    tasks[:] = [t for t in tasks if t.get("task_id") != task_id]
    tasks.insert(0, entry)
    tasks[:] = tasks[:RECENT_TASKS_CAP]
    return entry

def update_working_set(bb: dict[str, Any], touched: list[str], pending_errors: list[str]) -> None:
    ws = bb.setdefault("active_working_set", {})
    ws["files"] = list(dict.fromkeys(touched + ws.get("files", [])))[:WORKING_SET_CAP]
    ws["pending_errors"] = pending_errors


# ---------------------------------------------------------------- sessions


def record_turn(
    bb: dict[str, Any], task_id: str, *, channel: str, model: str, session_id: str | None, status: str,
    question: str | None, checkpoint_id: str | None, prompt: str, target_files: list[str],
    cost_usd: float | None = None, tokens: int = 0,
) -> dict[str, Any]:
    """Create/advance `pending_sessions[task_id]`. `origin_checkpoint_id` is written once, never overwritten."""
    sessions = bb.setdefault("pending_sessions", {})
    s = sessions.setdefault(task_id, {
        "created_at": _now(), "origin_checkpoint_id": None, "checkpoint_ids": [], "turn_count": 0,
        "cumulative_cost_usd": 0.0, "cumulative_tokens": 0, "prompt": prompt, "target_files": target_files,
    })
    if s["origin_checkpoint_id"] is None:
        s["origin_checkpoint_id"] = checkpoint_id
    if checkpoint_id and checkpoint_id not in s["checkpoint_ids"]:
        s["checkpoint_ids"].append(checkpoint_id)
    s.update({
        "channel": channel, "model": model, "session_or_conversation_id": session_id, "status": status,
        "pending_question": question, "latest_checkpoint_id": checkpoint_id or s.get("latest_checkpoint_id"),
        "turn_count": s["turn_count"] + 1, "updated_at": _now(),
        "cumulative_cost_usd": round(s["cumulative_cost_usd"] + (cost_usd or 0.0), 6),
        "cumulative_tokens": s["cumulative_tokens"] + tokens,
    })
    for old in sorted(sessions, key=lambda k: sessions[k].get("updated_at", ""))[:-SESSIONS_KEEP]:
        del sessions[old]
    bb.setdefault("system_identity", {})["last_session"] = {
        "task_id": task_id, "channel": channel, "model": model, "session_id": session_id,
        "checkpoint_id": checkpoint_id, "status": status, "pending_question": question, "at": _now(),
    }
    return s


def find_session(bb: dict[str, Any], task_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """`-c [--task-id X]`: the named session, else the last one."""
    tid = task_id or (bb.get("system_identity", {}).get("last_session") or {}).get("task_id")
    session = bb.get("pending_sessions", {}).get(tid) if tid else None
    if session is None:
        raise SessionError(f"no session {'for task ' + task_id if task_id else 'to continue'}")
    return tid, session


# ---------------------------------------------------------------- single-task lock (R06, R07)


def process_create_time(pid: int) -> int | None:
    """Creation timestamp of a *running* process, or None. Defeats Windows PID reuse."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetProcessTimes.argtypes = (wintypes.HANDLE, *(ctypes.POINTER(wintypes.FILETIME),) * 4)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            code = wintypes.DWORD()
            if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)) or \
                    not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return 0


@dataclass
class ProjectLock:
    """`.ai_context/.lock` with {pid, process_create_time, task_id}; stale locks are reclaimed."""

    root: Path
    task_id: str
    wait_s: float = 10.0

    @property
    def path(self) -> Path:
        return self.root / CONTEXT_DIR_NAME / LOCK_FILE

    def _holder(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _is_stale(self) -> bool:
        holder = self._holder()
        if holder is None:  # half-written by a racing process, or garbage: stale once it is old
            try:
                return time.time() - self.path.stat().st_mtime > 5
            except FileNotFoundError:
                return True
        return process_create_time(int(holder.get("pid", -1))) != holder.get("process_create_time")

    def __enter__(self) -> ProjectLock:
        record = json.dumps({"pid": os.getpid(), "process_create_time": process_create_time(os.getpid()),
                             "task_id": self.task_id, "at": _now()})
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(record)
                return self
            except (FileExistsError, PermissionError):
                if self._is_stale():
                    self.path.unlink(missing_ok=True)
                    continue
                if time.monotonic() > deadline:
                    holder = self._holder() or {}
                    raise ProjectBusyError(f"project busy: task {holder.get('task_id')} (pid {holder.get('pid')})") from None
                time.sleep(0.2)

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        if (self._holder() or {}).get("pid") == os.getpid():
            try:
                retry_win_lock(self.path.unlink)
            except FileNotFoundError:
                pass
