"""Same-volume NTFS vault for secret files (spec §3.D.2, S05).

Secret files are *renamed* (os.replace, atomic, zero bytes copied) into
`<drive>:\\.ai_vault_<user_sid>\\<random_token>\\` on the project's own volume, with an ACL
restricted to the current user. A write-ahead journal makes every move recoverable: the vault
is restored in `finally`, by `ai --recover`, and automatically at every `ai` start. A file the
AI created in the meantime is kept as `<name>.ai_generated`, never overwritten.
"""
from __future__ import annotations

import csv
import functools
import getpass
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType

from ai_core.checkpoint import retry_win_lock, write_json_atomic
from ai_core.config import CONTEXT_DIR_NAME, Config
from ai_core.security import is_secret_file

VAULT_PARENT: Path | None = None  # None = root of the project's volume; tests point it elsewhere
JOURNAL = "journal.json"
MOVED, PENDING, RESTORED = "moved", "pending", "restored"


class VaultError(RuntimeError):
    pass


@functools.cache
def user_sid() -> str:
    """Current Windows user SID (stable, path-safe); a name hash elsewhere."""
    if sys.platform == "win32":
        try:
            out = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True,
                                 encoding="utf-8", errors="replace", timeout=10, check=True).stdout
            sid = next(csv.reader([out.strip()]))[1]
            if sid.startswith("S-1-"):
                return sid
        except (OSError, subprocess.SubprocessError, IndexError, StopIteration):
            pass
    return "u" + hashlib.sha256(getpass.getuser().encode()).hexdigest()[:16]


def vault_root(project_root: Path) -> Path:
    return (VAULT_PARENT or Path(project_root.resolve().anchor)) / f".ai_vault_{user_sid()}"


def find_secret_files(root: Path, cfg: Config) -> list[Path]:
    pruned = {d.lower() for d in cfg.checkpoint_exclude_dirs} | {".git", CONTEXT_DIR_NAME}
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in pruned]
        found += [Path(dirpath) / f for f in filenames
                  if is_secret_file(f, cfg) and not (Path(dirpath) / f).is_symlink()]
    return sorted(found)


def _restrict_acl(path: Path) -> None:
    """Owner-only ACL. Without it a folder under C:\\ would inherit 'Authenticated Users: Modify'."""
    if sys.platform != "win32":
        os.chmod(path, 0o700)
        return
    cp = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"*{user_sid()}:(OI)(CI)F"],
                        capture_output=True, encoding="utf-8", errors="replace")
    if cp.returncode != 0:
        raise VaultError(f"cannot restrict ACL on {path}: {(cp.stderr or cp.stdout).strip()}")


def _free_name(path: Path) -> Path:
    candidate = path.with_name(path.name + ".ai_generated")
    n = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.ai_generated.{n}")
        n += 1
    return candidate


def _restore_entries(journal_path: Path, entries: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """Put files back; returns (messages, failures). Updates the journal after every step."""
    messages: list[str] = []
    failures: list[str] = []
    for entry in entries:
        if entry["state"] == RESTORED:
            continue
        original, stored = Path(entry["original"]), Path(entry["stored"])
        try:
            if not stored.exists():  # crash before the move happened: the file never left
                entry["state"] = RESTORED
                continue
            if original.exists():
                kept = _free_name(original)
                retry_win_lock(os.replace, original, kept)
                messages.append(f"{original.name} created during the task kept as {kept.name}")
            original.parent.mkdir(parents=True, exist_ok=True)
            retry_win_lock(os.replace, stored, original)
            entry["state"] = RESTORED
            messages.append(f"restored {original}")
        except OSError as exc:
            failures.append(f"{original}: {exc}")
        write_json_atomic(journal_path, entries)
    if not failures:
        retry_win_lock(journal_path.unlink)
        shutil.rmtree(journal_path.parent, ignore_errors=True)
    return messages, failures


class NTFSVault:
    """Context manager: isolate secret files for the duration of a Channel B dispatch."""

    def __init__(self, project_root: Path, files: Iterable[Path]) -> None:
        self.project_root = project_root.resolve()
        self.files = [Path(f) for f in files]
        self.session_dir: Path | None = None
        self.entries: list[dict[str, str]] = []
        self.skipped: list[tuple[str, str]] = []  # (path, reason) left in place, e.g. WinError 32

    @property
    def journal(self) -> Path:
        assert self.session_dir is not None
        return self.session_dir / JOURNAL

    def __enter__(self) -> NTFSVault:
        if not self.files:
            return self
        root = vault_root(self.project_root)
        try:
            root.mkdir(exist_ok=True)
            _restrict_acl(root)
        except OSError as exc:  # WinError 5 at the drive root, read-only volume, ...
            raise VaultError(f"cannot create vault {root}: {exc}") from exc
        if os.stat(root).st_dev != os.stat(self.project_root).st_dev:
            raise VaultError(f"vault {root} is not on the project's volume: refusing cross-volume moves")
        self.session_dir = root / secrets.token_urlsafe(16)  # unpredictable name
        self.session_dir.mkdir()
        try:
            for i, src in enumerate(self.files):
                entry = {"original": str(src), "stored": str(self.session_dir / f"{i:04d}.vault"),
                         "project": str(self.project_root), "state": PENDING}
                self.entries.append(entry)
                write_json_atomic(self.journal, self.entries)  # write-ahead: journal before the move
                try:
                    retry_win_lock(os.replace, src, entry["stored"])
                    entry["state"] = MOVED
                except PermissionError as exc:  # held open by an IDE / test runner: never force it
                    entry["state"] = RESTORED
                    self.skipped.append((str(src), f"locked ({exc.strerror})"))
                except FileNotFoundError:
                    entry["state"] = RESTORED
                write_json_atomic(self.journal, self.entries)
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self) -> list[str]:
        if self.session_dir is None or not self.journal.exists():
            return []
        messages, failures = _restore_entries(self.journal, self.entries)
        if failures:
            raise VaultError("vault restore incomplete, run `ai --recover`: " + "; ".join(failures))
        return messages

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.restore()


def recover(project_root: Path) -> list[str]:
    """Restore anything left in the vault of this volume (crash, power loss, killed process)."""
    root = vault_root(project_root)
    if not root.is_dir():
        return []
    messages: list[str] = []
    for journal in sorted(root.glob(f"*/{JOURNAL}")):
        try:
            entries = json.loads(journal.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            messages.append(f"⚠ unreadable journal {journal}: {exc} (files left in {journal.parent})")
            continue
        done, failures = _restore_entries(journal, entries)
        messages += done + [f"⚠ {f}" for f in failures]
    return messages
