"""CodingEngine MVP core: local-first router for Claude Code CLI and Ollama (spec v1.0)."""
from __future__ import annotations

import subprocess
import sys
from enum import IntEnum, StrEnum

__version__ = "1.0.0"


class ExitCode(IntEnum):
    """Disambiguated exit codes (spec §6.3, W02). 2 stays reserved for argparse usage errors."""

    OK = 0
    FAILED = 1
    USAGE = 2
    NEEDS_CLARIFICATION = 10
    OFFLINE_UNAVAILABLE = 11
    QUOTA_EXHAUSTED = 12
    SECRET_BLOCKED = 13


class TaskStatus(StrEnum):
    """Task state machine (§5.C.1). Process exit 0 never implies `completed_verified` (Invariant #4)."""

    COMPLETED_VERIFIED = "completed_verified"
    COMPLETED_SYNTAX_ONLY = "completed_syntax_only"
    COMPLETED_WITH_PREEXISTING_FAILURES = "completed_with_preexisting_failures"
    COMPLETED_UNVERIFIED = "completed_unverified"
    TESTS_FAILED = "tests_failed"
    VERIFICATION_TIMEOUT = "verification_timeout"
    PATCH_FAILED = "patch_failed"
    PERMISSION_DENIED = "permission_denied"
    NEEDS_CLARIFICATION = "needs_clarification"
    PROVIDER_ERROR = "provider_error"
    QUOTA_EXHAUSTED = "quota_exhausted"
    SECRET_DETECTED_ABORTED = "secret_detected_aborted"
    LOCAL_MODEL_UNAVAILABLE = "local_model_unavailable"
    CONTEXT_EXCEEDED_OFFLINE = "context_exceeded_offline"
    ROLLED_BACK = "rolled_back"
    PARTIALLY_APPLIED = "partially_applied"
    OUTCOME_UNKNOWN = "outcome_unknown"
    PROJECT_BUSY = "project_busy"


EXIT_FOR_STATUS: dict[TaskStatus, ExitCode] = {
    TaskStatus.COMPLETED_VERIFIED: ExitCode.OK,
    TaskStatus.COMPLETED_SYNTAX_ONLY: ExitCode.OK,
    TaskStatus.COMPLETED_WITH_PREEXISTING_FAILURES: ExitCode.OK,
    TaskStatus.COMPLETED_UNVERIFIED: ExitCode.OK,
    TaskStatus.NEEDS_CLARIFICATION: ExitCode.NEEDS_CLARIFICATION,
    TaskStatus.LOCAL_MODEL_UNAVAILABLE: ExitCode.OFFLINE_UNAVAILABLE,
    TaskStatus.CONTEXT_EXCEEDED_OFFLINE: ExitCode.OFFLINE_UNAVAILABLE,
    TaskStatus.QUOTA_EXHAUSTED: ExitCode.QUOTA_EXHAUSTED,
    TaskStatus.SECRET_DETECTED_ABORTED: ExitCode.SECRET_BLOCKED,
}  # everything else -> ExitCode.FAILED


def estimate_tokens(text: str) -> int:
    """Conservative offline estimate (~3 chars/token for code); errs toward the cloud, never toward truncation."""
    return len(text) // 3 + 1


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a child and all its descendants (pytest workers, node helpers...).

    Popen.kill() on Windows only ends the direct child, leaving grandchildren holding
    pipes open; `taskkill /T` walks the whole tree natively.
    """
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, encoding="utf-8", errors="replace", check=False,
        )
    if proc.poll() is None:
        proc.kill()


ACTIVE_PROCS: set[subprocess.Popen] = set()


def track(proc: subprocess.Popen) -> subprocess.Popen:
    """Register a long-running child so `cancel_active` (TUI interrupt) can end it."""
    ACTIVE_PROCS.difference_update([p for p in ACTIVE_PROCS if p.poll() is not None])
    ACTIVE_PROCS.add(proc)
    return proc


def cancel_active() -> int:
    """Kill every tracked child tree (agent CLIs, verification, tracked commands). Returns how many were alive."""
    alive = [p for p in list(ACTIVE_PROCS) if p.poll() is None]
    for p in alive:
        kill_process_tree(p)
    return len(alive)
