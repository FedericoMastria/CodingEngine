"""External transactional checkpoints and selective safe undo (spec §5.A: S01, R01-R03).

Layout: ~/.ai_checkpoints/<project_hash>/<checkpoint_id>/{manifest.json, files/<rel>}.
Nothing is ever written inside the workspace; secret files are hashed, never copied.
"""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from ai_core import config as cfgmod
from ai_core.config import Config
from ai_core.security import is_secret_file, is_within, path_key, sanitize_subprocess_env

MANIFEST = "manifest.json"
REF_PREFIX = "refs/ai/checkpoints/"
_ID_RE = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{6}$")
_WIN_LOCK_ERRORS = {5, 32}  # ERROR_ACCESS_DENIED (AV scanners, indexers), ERROR_SHARING_VIOLATION
_GIT_OPS_IN_PROGRESS = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply")


class CheckpointError(RuntimeError):
    pass


@dataclass(slots=True)
class FileRecord:
    pre_exists: bool
    sha256: str | None
    size: int | None
    mtime_ns: int | None
    stored: bool  # content copied under files/
    secret: bool = False


@dataclass(slots=True)
class Checkpoint:
    checkpoint_id: str
    project_root: str
    base: str  # git toplevel, or project root outside Git; all relative paths are against it
    created_at: str
    is_git: bool
    head_sha: str | None
    staged_tree_sha: str | None
    worktree_stash_sha: str | None
    untracked_snapshot: bool
    files: dict[str, FileRecord] = field(default_factory=dict)
    touched_files: dict[str, str | None] = field(default_factory=dict)  # rel -> post_task_sha256
    finalized: bool = False
    undone_at: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def worktree_source(self) -> str | None:
        """Git object holding the pre-task worktree content of tracked files."""
        return self.worktree_stash_sha or self.head_sha or self.staged_tree_sha

    @classmethod
    def from_json(cls, data: dict) -> Checkpoint:
        data = dict(data)
        data["files"] = {k: FileRecord(**v) for k, v in data.get("files", {}).items()}
        return cls(**data)


@dataclass(slots=True)
class UndoReport:
    checkpoint_id: str
    head_moved: bool = False
    restored: list[str] = field(default_factory=list)
    saved_copies: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------- helpers


def retry_win_lock[T](fn: Callable[..., T], *args: object, attempts: int = 5, delay_s: float = 0.2) -> T:
    """Retry transient Windows locks (WinError 32 file in use, WinError 5 from AV/indexer)."""
    for attempt in range(attempts):
        try:
            return fn(*args)
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in _WIN_LOCK_ERRORS or attempt == attempts - 1:
                raise
            time.sleep(delay_s * (attempt + 1))
    raise AssertionError("unreachable")


def file_sha256(path: Path) -> str | None:
    """SHA-256 of a file, or None when it does not exist."""
    def digest() -> str:
        with path.open("rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()
    try:
        return retry_win_lock(digest)
    except FileNotFoundError:
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git(base: Path, *args: str, check: bool = True, binary: bool = False) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args], cwd=base, capture_output=True,
        env=sanitize_subprocess_env(os.environ, warn=False),
        **({} if binary else {"encoding": "utf-8", "errors": "replace"}),
    )
    if check and cp.returncode != 0:
        err = cp.stderr if isinstance(cp.stderr, str) else cp.stderr.decode("utf-8", "replace")
        raise CheckpointError(f"git {' '.join(args)} failed: {err.strip()}")
    return cp


def _git_paths(base: Path, *args: str) -> set[str]:
    return {p for p in _git(base, *args).stdout.split("\0") if p}


def git_toplevel(project_root: Path) -> Path | None:
    try:
        cp = _git(project_root, "rev-parse", "--show-toplevel", check=False)
    except FileNotFoundError:  # git not installed
        return None
    return Path(os.path.normpath(cp.stdout.strip())) if cp.returncode == 0 else None


def git_operation_in_progress(base: Path) -> str | None:
    """R02: a merge/rebase/cherry-pick leaves conflicts in the index; never checkpoint over it."""
    git_dir = Path(_git(base, "rev-parse", "--absolute-git-dir").stdout.strip())
    return next((op for op in _GIT_OPS_IN_PROGRESS if (git_dir / op).exists()), None)


def _head_sha(base: Path) -> str | None:
    cp = _git(base, "rev-parse", "--verify", "-q", "HEAD", check=False)
    return cp.stdout.strip() or None


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Temp sibling + os.replace: readers never observe a half-written file (Invariant #2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.ai_tmp_{secrets.token_hex(4)}")
    try:
        tmp.write_bytes(data)
        retry_win_lock(os.replace, tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, data: object) -> None:
    write_bytes_atomic(path, json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))


def _copy_atomic(src: Path, dst: Path) -> None:
    """Like write_bytes_atomic, but streams the copy and keeps metadata (copy2)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.ai_tmp_{secrets.token_hex(4)}")
    try:
        retry_win_lock(shutil.copy2, src, tmp)
        retry_win_lock(os.replace, tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _rmtree_onexc(func: Callable[..., object], path: str, exc: BaseException) -> None:
    # copy2 preserves the read-only attribute, which makes Windows deletion fail with WinError 5.
    if isinstance(exc, PermissionError):
        os.chmod(path, stat.S_IWRITE)
        retry_win_lock(func, path)
    elif not isinstance(exc, FileNotFoundError):
        raise exc


def _excluded(rel: str, cfg: Config) -> bool:
    excluded = {d.lower() for d in cfg.checkpoint_exclude_dirs}
    return any(part.lower() in excluded for part in PurePosixPath(rel).parts[:-1])


# ---------------------------------------------------------------- locations


def project_hash(project_root: Path) -> str:
    return hashlib.sha256(path_key(project_root.resolve()).encode("utf-8")).hexdigest()[:16]


def checkpoints_dir(project_root: Path) -> Path:
    path = cfgmod.CHECKPOINTS_ROOT / project_hash(project_root)
    if is_within(path, project_root.resolve()):
        raise CheckpointError(f"checkpoint store {path} would be inside the workspace; refusing (S01)")
    return path


def _checkpoint_path(project_root: Path, checkpoint_id: str) -> Path:
    if not _ID_RE.fullmatch(checkpoint_id):
        raise CheckpointError(f"invalid checkpoint id {checkpoint_id!r}")
    return checkpoints_dir(project_root) / checkpoint_id


def save_checkpoint(project_root: Path, cp: Checkpoint) -> None:
    write_json_atomic(_checkpoint_path(project_root, cp.checkpoint_id) / MANIFEST, dataclasses.asdict(cp))


def list_checkpoints(project_root: Path) -> list[str]:
    store = checkpoints_dir(project_root)
    if not store.is_dir():
        return []
    return sorted(p.name for p in store.iterdir() if _ID_RE.fullmatch(p.name) and (p / MANIFEST).is_file())


def load_checkpoint(project_root: Path, checkpoint_id: str | None = None) -> Checkpoint:
    """Load a checkpoint; by default the newest finalized one that has not been undone and changed something."""
    if checkpoint_id is None:
        for cid in reversed(list_checkpoints(project_root)):
            cp = load_checkpoint(project_root, cid)
            if cp.finalized and cp.undone_at is None and cp.touched_files:  # no-op tasks have nothing to undo
                return cp
        raise CheckpointError("no undoable checkpoint for this project")
    path = _checkpoint_path(project_root, checkpoint_id) / MANIFEST
    try:
        return Checkpoint.from_json(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise CheckpointError(f"checkpoint {checkpoint_id} not found") from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise CheckpointError(f"checkpoint {checkpoint_id} manifest is corrupt: {exc}") from exc


# ---------------------------------------------------------------- create / finalize


def create_checkpoint(
    project_root: Path, cfg: Config, target_files: Iterable[Path], *, include_untracked: bool = True,
) -> Checkpoint:
    """Snapshot pre-task state before any dispatch. Raises CheckpointError if rollback can't be guaranteed."""
    project_root = project_root.resolve()
    top = git_toplevel(project_root)
    base = top or project_root
    cid = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{secrets.token_hex(3)}"
    ckpt_dir = _checkpoint_path(project_root, cid)
    cp = Checkpoint(
        checkpoint_id=cid, project_root=str(project_root), base=str(base), created_at=_now(),
        is_git=top is not None, head_sha=None, staged_tree_sha=None, worktree_stash_sha=None,
        untracked_snapshot=include_untracked and top is not None,
    )
    if cp.is_git and (op := git_operation_in_progress(base)):
        raise CheckpointError(f"Git {op} in progress: complete or abort it before running ai")

    (ckpt_dir / "files").mkdir(parents=True, exist_ok=False)
    try:
        if cp.is_git:
            cp.head_sha = _head_sha(base)
            cp.staged_tree_sha = _git(base, "write-tree").stdout.strip()
            if cp.head_sha:  # `stash create` needs an initial commit; it returns "" on a clean tree
                cp.worktree_stash_sha = _git(base, "stash", "create").stdout.strip() or None
            # Pin the objects so `git gc` can't prune them while the checkpoint is alive.
            _git(base, "update-ref", REF_PREFIX + cid, cp.worktree_stash_sha or cp.staged_tree_sha)
        for target in target_files:
            rel = Path(os.path.relpath(target.resolve(), base)).as_posix()
            cp.files[rel] = _snapshot(base / rel, ckpt_dir / "files" / rel, cfg, store=True, required=True)
        if cp.untracked_snapshot:
            budget = cfg.checkpoint_max_total_mb * 1024 * 1024
            for rel in sorted(_git_paths(base, "ls-files", "--others", "--exclude-standard", "-z")):
                if rel in cp.files or _excluded(rel, cfg):
                    continue
                try:
                    size = (base / rel).stat().st_size
                except OSError:
                    size = 0  # _snapshot records it as unreadable
                fits = size <= budget
                rec = _snapshot(base / rel, ckpt_dir / "files" / rel, cfg, store=fits, required=False)
                if rec.stored:
                    budget -= size
                elif not rec.secret:
                    cp.warnings.append(f"{rel}: not copied ({'unreadable' if fits else 'over size cap'}), hash only")
                cp.files[rel] = rec
        save_checkpoint(project_root, cp)
    except BaseException:
        shutil.rmtree(ckpt_dir, onexc=_rmtree_onexc)
        if cp.is_git:
            _git(base, "update-ref", "-d", REF_PREFIX + cid, check=False)
        raise
    gc_checkpoints(project_root, cfg)
    return cp


def _snapshot(src: Path, dst: Path, cfg: Config, *, store: bool, required: bool) -> FileRecord:
    if not src.exists():
        return FileRecord(pre_exists=False, sha256=None, size=None, mtime_ns=None, stored=False)
    secret = is_secret_file(src, cfg)
    try:
        st = src.stat()
        if store and not secret:
            _copy_atomic(src, dst)
            return FileRecord(True, file_sha256(dst), st.st_size, st.st_mtime_ns, stored=True)
        return FileRecord(True, file_sha256(src), st.st_size, st.st_mtime_ns, stored=False, secret=secret)
    except OSError as exc:  # WinError 32 persisting after retries, ACL denial, ...
        if required:
            raise CheckpointError(f"cannot snapshot {src}: {exc}") from exc
        return FileRecord(True, None, None, None, stored=False, secret=secret)


def finalize_checkpoint(
    project_root: Path, cfg: Config, cp: Checkpoint, extra_paths: Iterable[Path] = (),
) -> Checkpoint:
    """Record `touched_files` with their post-task SHA-256 once the task has finished."""
    base = Path(cp.base)
    changed_tracked: set[str] = set()
    current_untracked: set[str] = set()
    if cp.is_git and cp.worktree_source:
        changed_tracked = _git_paths(base, "diff", "--name-only", "--no-renames", "-z", cp.worktree_source)
        if cp.untracked_snapshot:
            current_untracked = {
                p for p in _git_paths(base, "ls-files", "--others", "--exclude-standard", "-z")
                if not _excluded(p, cfg)
            }
    extras: set[str] = set()
    for p in extra_paths:
        resolved = (base / p).resolve()
        if is_within(resolved, base):
            extras.add(Path(os.path.relpath(resolved, base)).as_posix())

    for rel in sorted(set(cp.files) | changed_tracked | current_untracked | extras):
        try:
            now = file_sha256(base / rel)
        except OSError as exc:  # still locked after retries: can't prove what undo would overwrite
            cp.warnings.append(f"{rel}: unreadable after task ({exc}); excluded from undo")
            continue
        rec = cp.files.get(rel)
        if rec is not None:
            changed = now != rec.sha256 or (now is not None) != rec.pre_exists
        elif rel in changed_tracked or rel in current_untracked:
            changed = True  # tracked-and-modified, or untracked that did not exist pre-task
        else:
            cp.warnings.append(f"{rel}: modified but pre-task state unknown (ignored file); cannot undo")
            continue
        if changed:
            cp.touched_files[rel] = now
    cp.finalized = True
    save_checkpoint(project_root, cp)
    return cp


# ---------------------------------------------------------------- undo / gc


def safe_undo(
    project_root: Path, cfg: Config, checkpoint_id: str | None = None,
    confirm: Callable[[str], bool] = lambda _msg: False,
) -> UndoReport:
    """Selective, non-destructive rollback of `touched_files` only (R01). Never `git read-tree`.

    - HEAD moved since the task: Git state is left alone; pre-task versions go to `<file>.ai_undo`.
    - A file whose current hash differs from `post_task_sha256` was edited by the user after the
      task and is skipped unless `confirm` approves the overwrite.
    """
    project_root = project_root.resolve()
    cp = load_checkpoint(project_root, checkpoint_id)
    if not cp.finalized:
        raise CheckpointError(f"checkpoint {cp.checkpoint_id} was never finalized (task still running?)")
    if cp.undone_at:
        raise CheckpointError(f"checkpoint {cp.checkpoint_id} was already undone at {cp.undone_at}")
    base = Path(cp.base)
    store = _checkpoint_path(project_root, cp.checkpoint_id) / "files"
    report = UndoReport(cp.checkpoint_id)
    pre_tracked: set[str] = set()
    pre_index: set[str] = set()
    current_index: set[str] = set()
    if cp.is_git:
        if op := git_operation_in_progress(base):
            raise CheckpointError(f"Git {op} in progress: complete or abort it before undo")
        report.head_moved = _head_sha(base) != cp.head_sha
        if cp.worktree_source:
            pre_tracked = _git_paths(base, "ls-tree", "-r", "-z", "--name-only", cp.worktree_source)
        if cp.staged_tree_sha:
            pre_index = _git_paths(base, "ls-tree", "-r", "-z", "--name-only", cp.staged_tree_sha)
        current_index = _git_paths(base, "ls-files", "-z")

    for rel, post_sha in sorted(cp.touched_files.items()):
        path = base / rel
        rec = cp.files.get(rel)
        try:
            if file_sha256(path) != post_sha and not confirm(f"{rel} was modified after the task. Overwrite it?"):
                report.skipped.append((rel, "modified after the task; left untouched"))
                continue
            if report.head_moved:
                _save_undo_copy(base, cp, rel, rec, store, pre_tracked, report)
                continue
            if rec is not None and rec.pre_exists:
                if not rec.stored:
                    report.skipped.append((rel, "pre-task content not snapshotted (secret/size cap)"))
                    continue
                _copy_atomic(store / rel, path)
                if file_sha256(path) != rec.sha256:
                    raise CheckpointError(f"{rel}: restored content does not match pre-task SHA-256")
            elif rel in pre_tracked:
                _git(base, "restore", f"--source={cp.worktree_source}", "--worktree", "--", rel)
            elif path.exists():
                retry_win_lock(path.unlink)  # created by the task
            if rel in pre_index or rel in current_index:
                _git(base, "restore", f"--source={cp.staged_tree_sha}", "--staged", "--", rel)
            report.restored.append(rel)
        except (OSError, CheckpointError) as exc:
            report.skipped.append((rel, f"error: {exc}"))

    if not report.head_moved:
        cp.undone_at = _now()
        save_checkpoint(project_root, cp)
    return report


def _save_undo_copy(
    base: Path, cp: Checkpoint, rel: str, rec: FileRecord | None, store: Path,
    pre_tracked: set[str], report: UndoReport,
) -> None:
    dst = base / f"{rel}.ai_undo"
    if dst.exists():
        report.skipped.append((rel, f"{dst.name} already exists"))
    elif rec is not None and rec.stored:
        _copy_atomic(store / rel, dst)
        report.saved_copies.append(f"{rel}.ai_undo")
    elif rel in pre_tracked:
        # --filters applies smudge/EOL conversion, giving the bytes Git would check out.
        write_bytes_atomic(dst, _git(base, "cat-file", "--filters", f"{cp.worktree_source}:{rel}", binary=True).stdout)
        report.saved_copies.append(f"{rel}.ai_undo")
    else:
        report.skipped.append((rel, "did not exist before the task; nothing to save"))


def diff_since(project_root: Path, checkpoint_id: str, cfg: Config, max_chars: int = 20000) -> str:
    """Unified diff of a finalized checkpoint's touched files, pre-task -> now (secrets never shown)."""
    cp = load_checkpoint(project_root, checkpoint_id)
    base = Path(cp.base)
    store = _checkpoint_path(project_root, cp.checkpoint_id) / "files"
    pre_tracked = (_git_paths(base, "ls-tree", "-r", "-z", "--name-only", cp.worktree_source)
                   if cp.is_git and cp.worktree_source else set())
    chunks: list[str] = []
    for rel in sorted(cp.touched_files):
        if is_secret_file(rel, cfg):
            chunks.append(f"--- {rel}: secret file changed (content withheld)\n")
            continue
        rec = cp.files.get(rel)
        if rec is not None and rec.pre_exists and not rec.stored:
            chunks.append(f"--- {rel}: changed (pre-task content over the snapshot size cap)\n")
            continue
        if rec is not None and rec.stored:
            before = (store / rel).read_bytes()
        elif rec is None and rel in pre_tracked:
            before = _git(base, "cat-file", "--filters", f"{cp.worktree_source}:{rel}", binary=True).stdout
        else:
            before = b""
        try:
            after = (base / rel).read_bytes()
        except FileNotFoundError:
            after = b""
        chunks += difflib.unified_diff(before.decode("utf-8", "replace").splitlines(keepends=True),
                                       after.decode("utf-8", "replace").splitlines(keepends=True),
                                       fromfile=f"a/{rel}", tofile=f"b/{rel}")
    text = "".join(chunks)
    return text if len(text) <= max_chars else text[:max_chars] + "\n... (diff truncated)\n"


def gc_checkpoints(project_root: Path, cfg: Config) -> list[str]:
    """R03 retention: keep newest `checkpoint_max_keep` and drop checkpoints > 7 days (spec §5.A, §9.2)."""
    from datetime import datetime
    import time
    ids = list_checkpoints(project_root)
    removed_set = set(ids[: max(0, len(ids) - cfg.checkpoint_max_keep)])
    cutoff = time.time() - (cfg.checkpoint_retention_days * 86400)
    for cid in ids:
        if cid in removed_set:
            continue
        try:
            cp = load_checkpoint(project_root, cid)
            if cp.created_at:
                ts = datetime.fromisoformat(cp.created_at).timestamp()
                if ts < cutoff:
                    removed_set.add(cid)
        except Exception:
            pass
    removed = [cid for cid in ids if cid in removed_set]
    for cid in removed:
        try:
            cp = load_checkpoint(project_root, cid)
            if cp.is_git:
                _git(Path(cp.base), "update-ref", "-d", REF_PREFIX + cid, check=False)
        except CheckpointError:
            pass  # corrupt manifest: still remove the directory
        shutil.rmtree(_checkpoint_path(project_root, cid), onexc=_rmtree_onexc)
    return removed
