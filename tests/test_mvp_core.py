"""Self-contained MVP Core tests (stdlib unittest only).

Run: uv run --no-project --python ">=3.12" python -m unittest discover -s tests -v
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core import ExitCode, TaskStatus, security
from ai_core import config as cfgmod
from ai_core.adapters.base import AdapterError, TaskRequest
from ai_core.adapters.claude_adapter import ClaudeAdapter, StreamState
from ai_core.checkpoint import (
    REF_PREFIX, CheckpointError, create_checkpoint, file_sha256, finalize_checkpoint, gc_checkpoints,
    list_checkpoints, safe_undo,
)
from ai_core.config import Config, ConfigError, load_config, validate_config
from ai_core.patcher import EditBlock, PatchError, apply_blocks, parse_blocks
from ai_core.security import (
    REDACTION, OfflineViolationError, PathValidationError, SecretPathError, is_secret_file, redact_secrets,
    require_loopback_url, sanitize_subprocess_env, scan_secrets, validate_target_files,
)
from ai_core.verification import compute_baseline, verify

REPO = Path(__file__).resolve().parents[1]
CFG = Config()
# Built by concatenation so this file itself never contains a scrubber-matching literal.
FAKE_ANTHROPIC = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_GH = "ghp_" + "a" * 18 + "B" * 18
FAKE_KEY_BLOCK = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          encoding="utf-8", errors="replace").stdout


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


class TempWorkspace(unittest.TestCase):
    """Workspace + external checkpoint store, both in a temp dir (store is NOT inside the workspace)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        tmp = Path(self._tmp.name).resolve()
        self.ws = tmp / "ws"
        self.ws.mkdir()
        self.store = tmp / "store"
        # Hermetic: checkpoint store, credit state and vault never touch the real home/drive root.
        from ai_core import statusline, vault
        for target, attr, value in ((cfgmod, "CHECKPOINTS_ROOT", self.store),
                                    (cfgmod, "STATE_PATH", tmp / "ai_state.json"),
                                    (vault, "VAULT_PARENT", tmp),
                                    (statusline, "USAGE_DIR", tmp / "ai_usage")):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def init_git(self) -> None:
        git(self.ws, "init", "-q")
        for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("core.autocrlf", "false")):
            git(self.ws, "config", key, value)
        write(self.ws / "a.py", "A = 1\n")
        write(self.ws / "b.py", "B = 1\n")
        write(self.ws / "other.py", "O = 1\n")
        write(self.ws / ".gitignore", "ignored/\n")
        git(self.ws, "add", ".")
        git(self.ws, "commit", "-qm", "init")

    def workspace_listing(self) -> set[str]:
        return {p.relative_to(self.ws).as_posix() for p in self.ws.rglob("*") if ".git" not in p.parts}


# ------------------------------------------------------------------ Invariant #5


class TestEnvSanitization(unittest.TestCase):
    REQUIRED = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

    def test_strips_every_required_key_without_mutating_input(self) -> None:
        env = {k: "secret-value" for k in self.REQUIRED} | {"PATH": r"C:\Windows", "USERPROFILE": r"C:\Users\x"}
        snapshot = dict(env)
        with redirect_stderr(io.StringIO()):
            clean = sanitize_subprocess_env(env)
        self.assertEqual(clean, {"PATH": r"C:\Windows", "USERPROFILE": r"C:\Users\x"})
        self.assertEqual(env, snapshot)

    def test_case_insensitive_names(self) -> None:
        with redirect_stderr(io.StringIO()):
            clean = sanitize_subprocess_env({"anthropic_api_key": "x", "Gemini_Api_Key": "y", "Path": "p"})
        self.assertEqual(clean, {"Path": "p"})

    def test_strips_metered_backend_switches_but_keeps_oauth(self) -> None:
        env = {"ANTHROPIC_BASE_URL": "https://proxy", "CLAUDE_CODE_USE_BEDROCK": "1", "OPENAI_API_KEY": "x",
               "GOOGLE_APPLICATION_CREDENTIALS": "c.json", "CLAUDE_CODE_OAUTH_TOKEN": "subscription"}
        clean = sanitize_subprocess_env(env, warn=False)
        self.assertEqual(clean, {"CLAUDE_CODE_OAUTH_TOKEN": "subscription"})

    def test_warning_names_keys_never_values(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            sanitize_subprocess_env({"ANTHROPIC_API_KEY": "sk-very-secret"})
        self.assertIn("ANTHROPIC_API_KEY", err.getvalue())
        self.assertNotIn("sk-very-secret", err.getvalue())


# ------------------------------------------------------------------ scrubber


class TestSecretScrubber(unittest.TestCase):
    def test_detects_known_secrets_with_line_numbers(self) -> None:
        text = "\n".join([
            "ok line", f"key = '{FAKE_ANTHROPIC}'", f"aws={FAKE_AWS}", FAKE_KEY_BLOCK,
            f"token: {FAKE_GH}", "DATABASE_URL=postgres://admin:Sup3rS3cret@db:5432/app",
            "Server=db;User Id=sa;Password=Hunter2!;",
        ])
        findings = scan_secrets(text, "cfg.txt")
        kinds = {(f.line, f.kind) for f in findings}
        self.assertIn((2, "anthropic_key"), kinds)
        self.assertIn((3, "aws_access_key"), kinds)
        self.assertIn((4, "private_key"), kinds)
        self.assertIn((7, "github_token"), kinds)
        self.assertIn((8, "connection_string_password"), kinds)
        self.assertIn((9, "connection_string_password"), kinds)
        for f in findings:  # a finding never carries the secret value
            self.assertTrue(str(f).startswith("cfg.txt:"))
            for secret in (FAKE_ANTHROPIC, FAKE_AWS, "Sup3rS3cret", "Hunter2!"):
                self.assertNotIn(secret, str(f))

    def test_redaction_masks_only_the_secret(self) -> None:
        text = f"url=postgres://admin:Sup3rS3cret@db/app key={FAKE_ANTHROPIC}"
        red = redact_secrets(text)
        self.assertEqual(red, f"url=postgres://admin:{REDACTION}@db/app key={REDACTION}")
        self.assertEqual(scan_secrets(red), [])

    def test_private_key_block_fully_redacted(self) -> None:
        red = redact_secrets(f"before\n{FAKE_KEY_BLOCK}\nafter")
        self.assertEqual(red, f"before\n{REDACTION}\nafter")

    def test_no_false_positives_on_ordinary_code(self) -> None:
        benign = "\n".join([
            "def connect(password=pw, user=None):", "url = 'postgres://user:${DB_PASS}@host/db'",
            "url2 = 'redis://user:<password>@host'", "import secrets  # secrets.py module",
            "x = 'sk-short'", "token = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        ])
        self.assertEqual(scan_secrets(benign), [])

    def test_high_entropy_assignment(self) -> None:
        findings = scan_secrets('client_secret = "Zx8fQ2pLm9Vt4Rk7Wn3Yb6Hj1Cd5Ge0S"')
        self.assertEqual([f.kind for f in findings], ["high_entropy_assignment"])


# ------------------------------------------------------------------ paths


class TestSecretFilesAndPaths(TempWorkspace):
    def test_secret_file_patterns_case_insensitive(self) -> None:
        for name in (".env", ".ENV", ".env.local", ".Env.Production", "prod.PEM", "server.Key", "Id_Rsa",
                     "credentials.json", "service-account-prod.json", "vars.TFVARS", ".npmrc"):
            self.assertTrue(is_secret_file(name, CFG), name)
        for name in (".env.example", ".ENV.Example", ".env.sample", ".env.template", "secrets.py",
                     "secrets.ts", "keyboard.py", "environment.py"):
            self.assertFalse(is_secret_file(name, CFG), name)

    def test_target_validation(self) -> None:
        write(self.ws / "src" / "calc.py", "x = 1\n")
        write(self.ws / ".env", "K=V\n")
        write(self.ws / ".ai_context" / "config.toml", "")
        outside = self.ws.parent / "outside.py"
        write(outside, "")
        ok = validate_target_files(["src/calc.py", r"src\calc.py"], self.ws, CFG)
        self.assertEqual(ok, [(self.ws / "src" / "calc.py").resolve()])  # de-duplicated
        for bad in (r"..\outside.py", "../outside.py", str(outside), "src/../../outside.py"):
            with self.assertRaises(PathValidationError, msg=bad):
                validate_target_files([bad], self.ws, CFG)
        for protected in (".ai_context/config.toml", ".AI_CONTEXT/config.toml", ".git/config"):
            with self.assertRaises(PathValidationError, msg=protected):
                validate_target_files([protected], self.ws, CFG)
        with self.assertRaises(SecretPathError):
            validate_target_files([".ENV"], self.ws, CFG)
        with self.assertRaises(PathValidationError):
            validate_target_files(["src"], self.ws, CFG)  # directory

    def test_new_files_only_when_requested_and_parent_exists(self) -> None:
        (self.ws / "src").mkdir()
        with self.assertRaises(PathValidationError):
            validate_target_files(["src/new.py"], self.ws, CFG)  # hallucinated
        self.assertEqual(len(validate_target_files(["src/new.py"], self.ws, CFG, creates_new_files=True)), 1)
        with self.assertRaises(PathValidationError):
            validate_target_files(["nodir/new.py"], self.ws, CFG, creates_new_files=True)

    @unittest.skipUnless(sys.platform == "win32", "Windows case-insensitive filesystem")
    def test_windows_case_insensitive_dedup(self) -> None:
        write(self.ws / "Calc.py", "")
        self.assertEqual(len(validate_target_files(["Calc.py", "calc.PY", "CALC.py"], self.ws, CFG)), 1)

    def test_symlink_escaping_root_rejected(self) -> None:
        target = self.ws.parent / "real.py"
        write(target, "")
        try:
            os.symlink(target, self.ws / "link.py")
        except OSError:  # WinError 1314: symlinks need Developer Mode / admin on Windows
            self.skipTest("symlink creation not permitted")
        with self.assertRaises(PathValidationError):
            validate_target_files(["link.py"], self.ws, CFG)


# ------------------------------------------------------------------ checkpoints & safe undo


class TestCheckpoint(TempWorkspace):
    def setUp(self) -> None:
        super().setUp()
        self.init_git()
        write(self.ws / "notes.txt", "pre-existing untracked\n")
        write(self.ws / ".env", "API_TOKEN=do-not-copy\n")  # untracked and NOT gitignored
        write(self.ws / "ignored" / "big.bin", "ignored content\n")

    def test_checkpoint_is_external_and_never_copies_secrets(self) -> None:
        before = self.workspace_listing()
        status_before = git(self.ws, "status", "--porcelain")
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        ckpt = next(self.store.glob(f"*/{cp.checkpoint_id}"))
        self.assertFalse(str(ckpt).lower().startswith(str(self.ws).lower()))
        self.assertTrue((ckpt / "manifest.json").is_file())
        self.assertTrue((ckpt / "files" / "a.py").is_file())
        self.assertTrue((ckpt / "files" / "notes.txt").is_file())
        self.assertFalse((ckpt / "files" / ".env").exists(), "secret content must never be copied")
        self.assertFalse((ckpt / "files" / "ignored").exists())
        env_rec = cp.files[".env"]
        self.assertTrue(env_rec.secret)
        self.assertFalse(env_rec.stored)
        self.assertEqual(env_rec.sha256, file_sha256(self.ws / ".env"))
        manifest = (ckpt / "manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("do-not-copy", manifest)
        self.assertIn(REF_PREFIX + cp.checkpoint_id, git(self.ws, "for-each-ref", REF_PREFIX))
        self.assertTrue(cp.staged_tree_sha and cp.head_sha)
        # Nothing was written inside the workspace and the user's Git state is unchanged.
        self.assertEqual(self.workspace_listing(), before)
        self.assertEqual(git(self.ws, "status", "--porcelain"), status_before)

    def test_safe_undo_is_selective_and_preserves_user_stage(self) -> None:
        write(self.ws / "a.py", "A = 'pre-task worktree'\n")      # unstaged user change
        write(self.ws / "b.py", "B = 'staged by user'\n")
        git(self.ws, "add", "b.py")                                # staged user change on untouched file
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        # --- the AI task
        write(self.ws / "a.py", "A = 'ai edit'\n")
        write(self.ws / "new_module.py", "N = 1\n")
        write(self.ws / "notes.txt", "ai rewrote notes\n")
        finalize_checkpoint(self.ws, CFG, cp)
        self.assertEqual(set(cp.touched_files), {"a.py", "new_module.py", "notes.txt"})
        # --- the user keeps working on other files after the task
        write(self.ws / "other.py", "O = 'user after task'\n")
        git(self.ws, "add", "other.py")

        report = safe_undo(self.ws, CFG)
        self.assertEqual(report.skipped, [])
        self.assertEqual(read(self.ws / "a.py"), "A = 'pre-task worktree'\n")
        self.assertFalse((self.ws / "new_module.py").exists())
        self.assertEqual(read(self.ws / "notes.txt"), "pre-existing untracked\n")
        staged = git(self.ws, "diff", "--cached", "--name-only").split()
        self.assertEqual(sorted(staged), ["b.py", "other.py"], "no global read-tree: user stage preserved")
        self.assertEqual(git(self.ws, "show", ":b.py"), "B = 'staged by user'\n")
        self.assertEqual(read(self.ws / "other.py"), "O = 'user after task'\n")
        self.assertEqual(read(self.ws / ".env"), "API_TOKEN=do-not-copy\n")

    def test_default_undo_skips_no_op_checkpoints(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        write(self.ws / "a.py", "A = 'ai edit'\n")
        finalize_checkpoint(self.ws, CFG, cp)
        empty = create_checkpoint(self.ws, CFG, [])  # a later task that changed nothing
        finalize_checkpoint(self.ws, CFG, empty)
        self.assertEqual(safe_undo(self.ws, CFG).checkpoint_id, cp.checkpoint_id)

    def test_undo_restores_index_and_worktree_of_touched_file_separately(self) -> None:
        write(self.ws / "a.py", "A = 'staged'\n")
        git(self.ws, "add", "a.py")
        write(self.ws / "a.py", "A = 'worktree'\n")
        cp = create_checkpoint(self.ws, CFG, [])
        write(self.ws / "a.py", "A = 'ai'\n")
        git(self.ws, "add", "a.py")  # the agent staged its own edit
        finalize_checkpoint(self.ws, CFG, cp)
        safe_undo(self.ws, CFG)
        self.assertEqual(read(self.ws / "a.py"), "A = 'worktree'\n")
        self.assertEqual(git(self.ws, "show", ":a.py"), "A = 'staged'\n")

    def test_undo_restores_tracked_file_deleted_by_task_and_unstages_new_file(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [])
        (self.ws / "b.py").unlink()
        write(self.ws / "added.py", "x = 1\n")
        git(self.ws, "add", "added.py")
        finalize_checkpoint(self.ws, CFG, cp)
        self.assertIsNone(cp.touched_files["b.py"])
        safe_undo(self.ws, CFG)
        self.assertEqual(read(self.ws / "b.py"), "B = 1\n")
        self.assertFalse((self.ws / "added.py").exists())
        self.assertNotIn("added.py", git(self.ws, "ls-files"))

    def test_undo_never_overwrites_user_changes_made_after_task(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        write(self.ws / "a.py", "A = 'ai'\n")
        finalize_checkpoint(self.ws, CFG, cp)
        write(self.ws / "a.py", "A = 'user fixed it by hand'\n")
        report = safe_undo(self.ws, CFG)  # default confirm -> False
        self.assertEqual([r for r, _ in report.skipped], ["a.py"])
        self.assertEqual(read(self.ws / "a.py"), "A = 'user fixed it by hand'\n")

    def test_undo_with_moved_head_leaves_git_alone_and_saves_ai_undo_copies(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [])
        write(self.ws / "a.py", "A = 'ai'\n")
        finalize_checkpoint(self.ws, CFG, cp)
        git(self.ws, "commit", "-qam", "user commits the AI change")
        head = git(self.ws, "rev-parse", "HEAD")
        report = safe_undo(self.ws, CFG)
        self.assertTrue(report.head_moved)
        self.assertEqual(git(self.ws, "rev-parse", "HEAD"), head)
        self.assertEqual(git(self.ws, "status", "--porcelain", "--untracked-files=no"), "")
        self.assertEqual(read(self.ws / "a.py"), "A = 'ai'\n")
        self.assertEqual(read(self.ws / "a.py.ai_undo"), "A = 1\n")

    def test_double_undo_refused(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        write(self.ws / "a.py", "A = 'ai'\n")
        finalize_checkpoint(self.ws, CFG, cp)
        safe_undo(self.ws, CFG)
        with self.assertRaises(CheckpointError):
            safe_undo(self.ws, CFG, cp.checkpoint_id)

    def test_unfinalized_checkpoint_and_bad_ids_rejected(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [])
        with self.assertRaises(CheckpointError):
            safe_undo(self.ws, CFG, cp.checkpoint_id)
        for bad in (r"..\..\Windows", "../x", "20260101T000000000000Z-zzzzzz"):
            with self.assertRaises(CheckpointError):
                safe_undo(self.ws, CFG, bad)

    def test_retention_removes_old_dirs_and_refs(self) -> None:
        cfg = dataclasses.replace(CFG, checkpoint_max_keep=2)
        ids = [create_checkpoint(self.ws, cfg, []).checkpoint_id for _ in range(3)]
        self.assertEqual(list_checkpoints(self.ws), ids[1:])
        refs = git(self.ws, "for-each-ref", "--format=%(refname)", REF_PREFIX).split()
        self.assertEqual(sorted(refs), sorted(REF_PREFIX + i for i in ids[1:]))
        self.assertEqual(gc_checkpoints(self.ws, cfg), [])

    def test_refuses_store_inside_workspace(self) -> None:
        with mock.patch.object(cfgmod, "CHECKPOINTS_ROOT", self.ws / ".ai_checkpoints"):
            with self.assertRaises(CheckpointError):
                create_checkpoint(self.ws, CFG, [])
        self.assertFalse((self.ws / ".ai_checkpoints").exists())

    def test_blocks_during_merge_or_rebase(self) -> None:
        (self.ws / ".git" / "MERGE_HEAD").write_text("0" * 40, encoding="utf-8")
        with self.assertRaises(CheckpointError):
            create_checkpoint(self.ws, CFG, [])
        self.assertEqual(git(self.ws, "for-each-ref", REF_PREFIX), "")


class TestCheckpointNonGit(TempWorkspace):
    def test_snapshot_and_undo_without_git(self) -> None:
        write(self.ws / "calc.py", "def f():\r\n    return 1\r\n")
        cp = create_checkpoint(self.ws, CFG, [self.ws / "calc.py", self.ws / "fresh.py"])
        self.assertFalse(cp.is_git)
        write(self.ws / "calc.py", "def f():\r\n    return 2\r\n")
        write(self.ws / "fresh.py", "x = 1\n")
        finalize_checkpoint(self.ws, CFG, cp)
        report = safe_undo(self.ws, CFG)
        self.assertEqual(sorted(report.restored), ["calc.py", "fresh.py"])
        self.assertEqual((self.ws / "calc.py").read_bytes(), b"def f():\r\n    return 1\r\n")  # CRLF exact
        self.assertFalse((self.ws / "fresh.py").exists())


# ------------------------------------------------------------------ patcher (L08)


class TestPatcher(unittest.TestCase):
    def test_exact_and_crlf_preserved(self) -> None:
        out = apply_blocks({"a.py": "x = 1\r\ny = 2\r\n"}, [EditBlock("a.py", "y = 2", "y = 3")])
        self.assertEqual(out, {"a.py": "x = 1\r\ny = 3\r\n"})

    def test_line_trimmed_match_reapplies_native_indentation(self) -> None:
        src = "class C:\n    def f(self):   \n        return 1\n"
        block = EditBlock("a.py", "def f(self):\n    return 1", "def f(self):\n    return 2")
        self.assertEqual(apply_blocks({"a.py": src}, [block])["a.py"],
                         "class C:\n    def f(self):\n        return 2\n")

    def test_ambiguous_and_missing_rejected(self) -> None:
        with self.assertRaises(PatchError):
            apply_blocks({"a.py": "x\nx\n"}, [EditBlock("a.py", "x", "y")])
        with self.assertRaises(PatchError):
            apply_blocks({"a.py": "x\n"}, [EditBlock("a.py", "nope", "y")])

    def test_all_or_nothing(self) -> None:
        originals = {"a.py": "a\n", "b.py": "b\n"}
        with self.assertRaises(PatchError):
            apply_blocks(originals, [EditBlock("a.py", "a", "A"), EditBlock("b.py", "missing", "B")])
        self.assertEqual(originals, {"a.py": "a\n", "b.py": "b\n"})

    def test_only_allowed_paths_with_windows_normalization(self) -> None:
        self.assertIn("src/a.py", apply_blocks({"src/a.py": "a\n"}, [EditBlock(r"SRC\A.py", "a", "b")]))
        for bad in ("../src/a.py", "other.py"):
            with self.assertRaises(PatchError):
                apply_blocks({"src/a.py": "a\n"}, [EditBlock(bad, "a", "b")])

    def test_parse_blocks_and_new_file(self) -> None:
        text = "```python\nsrc/new.py\n<<<<<<< SEARCH\n=======\nprint('hi')\n>>>>>>> REPLACE\n```\n"
        blocks = parse_blocks(text)
        self.assertEqual(blocks, [EditBlock("src/new.py", "", "print('hi')")])
        self.assertEqual(apply_blocks({"src/new.py": None}, blocks), {"src/new.py": "print('hi')\n"})
        with self.assertRaises(PatchError):
            parse_blocks("a.py\n<<<<<<< SEARCH\nx\n=======\ny\n")


# ------------------------------------------------------------------ verification (R04)


class TestVerification(TempWorkspace):
    SCRIPT = "import pathlib,sys; t=pathlib.Path('out.txt').read_text(); print(t); sys.exit(1 if 'FAIL' in t else 0)"

    def setUp(self) -> None:
        super().setUp()
        write(self.ws / "m.py", "def f():\n    return 1\n")
        write(self.ws / "tests" / "test_m.py", "")
        self.cfg = dataclasses.replace(CFG, verify_l3_command=[sys.executable, "-c", self.SCRIPT],
                                       verify_l3_timeout_s=30)

    def test_syntax_error_is_tests_failed(self) -> None:
        write(self.ws / "broken.py", "def f(:\n")
        res = verify(self.ws, CFG, [self.ws / "broken.py"])
        self.assertEqual(res.status, TaskStatus.TESTS_FAILED)
        self.assertEqual(res.new_failures, ["syntax:broken.py"])

    def test_syntax_only_when_no_tests(self) -> None:
        write(self.ws / "lone.py", "x = 1\n")
        write(self.ws / "data.json", '{"a": 1}')
        res = verify(self.ws, CFG, [self.ws / "lone.py", self.ws / "data.json"])
        self.assertEqual((res.status, res.verification_level), (TaskStatus.COMPLETED_SYNTAX_ONLY, 1))

    def test_targeted_tests_pass_is_verified(self) -> None:
        write(self.ws / "out.txt", "ok")
        res = verify(self.ws, self.cfg, [self.ws / "m.py"])
        self.assertEqual((res.status, res.verification_level), (TaskStatus.COMPLETED_VERIFIED, 3))

    def test_preexisting_failures_are_not_blamed_on_the_task(self) -> None:
        write(self.ws / "out.txt", "FAILED tests/test_m.py::test_old - boom")
        baseline = compute_baseline(self.ws, self.cfg, [self.ws / "m.py"])
        res = verify(self.ws, self.cfg, [self.ws / "m.py"], baseline)
        self.assertEqual(res.status, TaskStatus.COMPLETED_WITH_PREEXISTING_FAILURES)
        write(self.ws / "out.txt", "FAILED tests/test_m.py::test_old - boom\nFAILED tests/test_m.py::test_new - x")
        res = verify(self.ws, self.cfg, [self.ws / "m.py"], baseline)
        self.assertEqual(res.status, TaskStatus.TESTS_FAILED)
        self.assertEqual(res.new_failures, ["tests/test_m.py::test_new"])

    def test_unparseable_failures_are_never_excused(self) -> None:
        write(self.ws / "out.txt", "FAIL somewhere\nFAILED (failures=1)")
        cfg = dataclasses.replace(self.cfg, verify_l3_command=[sys.executable, "-c", "import sys; print('boom'); sys.exit(1)"])
        baseline = compute_baseline(self.ws, cfg, [self.ws / "m.py"])
        self.assertEqual(verify(self.ws, cfg, [self.ws / "m.py"], baseline).status, TaskStatus.TESTS_FAILED)

    def test_timeout_is_distinct_state(self) -> None:
        cfg = dataclasses.replace(self.cfg, verify_l3_command=[sys.executable, "-c", "import time; time.sleep(30)"],
                                  verify_l3_timeout_s=1)
        self.assertEqual(verify(self.ws, cfg, [self.ws / "m.py"]).status, TaskStatus.VERIFICATION_TIMEOUT)

    def test_missing_tool_is_skipped_not_failed(self) -> None:
        cfg = dataclasses.replace(self.cfg, verify_l3_command=["definitely-not-a-real-binary-xyz"])
        self.assertEqual(verify(self.ws, cfg, [self.ws / "m.py"]).status, TaskStatus.COMPLETED_SYNTAX_ONLY)


# ------------------------------------------------------------------ Claude adapter


class _Sink:
    def __init__(self) -> None:
        self.data, self.closed = "", False

    def write(self, s: str) -> int:
        self.data += s
        return len(s)

    def close(self) -> None:
        self.closed = True


class FakeProc:
    pid = 4242

    def __init__(self, lines: list[str], rc: int = 0) -> None:
        self.stdin, self.stdout, self.stderr = _Sink(), iter(lines), iter([])
        self.returncode: int | None = None
        self._rc = rc

    def wait(self) -> int:
        self.returncode = self._rc
        return self._rc

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def stream(*events: dict) -> list[str]:
    return [json.dumps(e) + "\n" for e in events]


INIT = {"type": "system", "subtype": "init", "session_id": "11111111-2222-3333-4444-555555555555", "model": "claude-sonnet-5-5"}
EDIT = {"type": "assistant", "session_id": INIT["session_id"], "message": {"content": [
    {"type": "text", "text": "Fixing."},
    {"type": "tool_use", "name": "Edit", "input": {"file_path": r"C:\proj\a.py", "old_string": "a", "new_string": "b"}}]}}
RESULT = {"type": "result", "subtype": "success", "is_error": False, "result": "Done.", "session_id": INIT["session_id"],
          "total_cost_usd": 0.0123, "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 10, "outputTokens": 5}}}


class TestClaudeAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ClaudeAdapter(CFG, binary="claude.exe")
        self.adapter.version = "2.1.310 (Claude Code)"  # skip the real --help probe

    def request(self) -> TaskRequest:
        return TaskRequest("t1", "fix the bug", REPO, "sonnet", context_block="<project_memory_data>{}</project_memory_data>",
                           system_append="PROFILE")

    def test_command_uses_verified_flags_and_mandatory_verbose(self) -> None:
        cmd = self.adapter.build_command(self.request())
        self.assertEqual(cmd[:2], ["claude.exe", "-p"])
        i = cmd.index("--output-format")
        self.assertEqual(cmd[i + 1], "stream-json")
        self.assertIn("--verbose", cmd, "stream-json with -p requires --verbose")
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "acceptEdits")
        self.assertEqual(cmd[cmd.index("--permission-prompts") + 1], "none")
        self.assertEqual(cmd[cmd.index("--max-budget-usd") + 1], "1.00")
        self.assertEqual(cmd[cmd.index("--model") + 1], "sonnet")
        deny = cmd[cmd.index("--disallowedTools") + 1].split(",")
        for rule in ("Read(**/.env)", "Edit(**/*.pem)", "Read(./.ai_context/**)", "Edit(~/.ai_checkpoints/**)",
                     "Read(~/.ssh/**)"):
            self.assertIn(rule, deny)
        self.assertNotIn("--fallback-model", cmd)
        self.assertNotIn("fix the bug", cmd, "prompt goes via stdin, never argv")

    def test_stream_parser_extracts_session_cost_usage_and_edits(self) -> None:
        state = StreamState()
        for line in stream(INIT, EDIT) + ["not json at all\n", "\n"] + stream(RESULT):
            state.feed(line)
        res = state.to_result(0, "")
        self.assertIsNone(res.status)
        self.assertEqual(res.session_id, INIT["session_id"])
        self.assertEqual(res.total_cost_usd, 0.0123)
        self.assertEqual(res.model_usage, RESULT["modelUsage"])
        self.assertEqual(res.edited_paths, [r"C:\proj\a.py"])
        self.assertEqual(state.malformed_lines, 1)

    def test_budget_clarification_and_provider_errors(self) -> None:
        cases = [
            ({**RESULT, "subtype": "error_max_budget_usd", "is_error": True}, TaskStatus.QUOTA_EXHAUSTED),
            ({**RESULT, "terminal_reason": "budget_exhausted"}, TaskStatus.QUOTA_EXHAUSTED),
            ({**RESULT, "is_error": True, "result": "Claude usage limit reached"}, TaskStatus.QUOTA_EXHAUSTED),
            ({**RESULT, "is_error": True, "result": "API Error: 500"}, TaskStatus.PROVIDER_ERROR),
            ({**RESULT, "result": "[NEEDS_CLARIFICATION]: which DB?"}, TaskStatus.NEEDS_CLARIFICATION),
        ]
        for event, expected in cases:
            state = StreamState()
            state.feed(json.dumps(event))
            self.assertEqual(state.to_result(0, "").status, expected, event)
        no_result = StreamState().to_result(1, "Error: --output-format=stream-json requires --verbose")
        self.assertEqual(no_result.status, TaskStatus.PROVIDER_ERROR)
        self.assertIn("requires --verbose", no_result.error or "")

    def test_run_task_sanitizes_env_and_sends_prompt_via_stdin(self) -> None:
        captured: dict = {}

        def fake_popen(cmd: list[str], **kwargs: object) -> FakeProc:
            captured.update(kwargs, cmd=cmd)
            captured["proc"] = FakeProc(stream(INIT, EDIT, RESULT))
            return captured["proc"]

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-should-never-leak", "GOOGLE_API_KEY": "g"}), \
                mock.patch("ai_core.adapters.claude_adapter.subprocess.Popen", side_effect=fake_popen), \
                redirect_stderr(io.StringIO()):
            res = self.adapter.run_task(self.request())
        self.assertNotIn("ANTHROPIC_API_KEY", captured["env"])
        self.assertNotIn("GOOGLE_API_KEY", captured["env"])
        self.assertEqual(captured["encoding"], "utf-8")
        self.assertEqual(captured["errors"], "replace")
        self.assertTrue(captured["proc"].stdin.data.endswith("fix the bug"))
        self.assertTrue(captured["proc"].stdin.closed)
        self.assertEqual((res.session_id, res.total_cost_usd), (INIT["session_id"], 0.0123))

    def test_usage_credits_flag_disables_channel_a(self) -> None:
        with self.assertRaises(AdapterError):
            ClaudeAdapter(dataclasses.replace(CFG, claude_allow_usage_credits=True))


# ------------------------------------------------------------------ Invariant #1


class TestOfflineInvariant(TempWorkspace):
    def tearDown(self) -> None:
        security._offline_latched = False  # the latch is one-way by design; reset only for tests

    def test_latch_blocks_cloud_adapter(self) -> None:
        adapter = ClaudeAdapter(CFG, binary="claude.exe")
        security.enforce_offline()
        with self.assertRaises(OfflineViolationError):
            ClaudeAdapter(CFG)
        with mock.patch("ai_core.adapters.claude_adapter.subprocess.Popen") as popen:
            with self.assertRaises(OfflineViolationError):
                adapter.run_task(TaskRequest("t", "p", self.ws, "haiku"))
        popen.assert_not_called()

    def test_offline_requires_loopback_ollama(self) -> None:
        require_loopback_url("http://127.0.0.1:11434")
        require_loopback_url("http://localhost:11434")
        with self.assertRaises(OfflineViolationError):
            require_loopback_url("http://192.168.1.50:11434")

    def test_offline_with_unreachable_ollama_exits_11_without_cloud(self) -> None:
        from ai_core import router
        write(self.ws / "a.py", "x = 1\n")
        for url in ("http://127.0.0.1:9", "http://gpu-box.lan:11434"):
            security._offline_latched = False
            cfg = dataclasses.replace(CFG, ollama_url=url)
            opts = router.TaskOptions("fix it", self.ws, ("a.py",), offline=True)
            with mock.patch.object(ClaudeAdapter, "run_task") as dispatch, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(router.run(opts, cfg), ExitCode.OFFLINE_UNAVAILABLE, url)
            dispatch.assert_not_called()
            self.assertFalse(list(self.store.glob("*/*")), "no checkpoint when nothing is dispatched")


# ------------------------------------------------------------------ CLI / router


class TestRouterAndCli(TempWorkspace):
    def tearDown(self) -> None:
        security._offline_latched = False

    def test_plus_file_syntax(self) -> None:
        from ai_router import split_prompt
        write(self.ws / "src" / "calc.py", "")
        prompt, files = split_prompt(["refactor", "+src/calc.py,", "and", "@b.py", "see", "src/calc.py"], self.ws)
        self.assertEqual(prompt, "refactor src/calc.py, and b.py see src/calc.py")
        self.assertEqual(files, ["src/calc.py", "b.py"])

    def test_secret_in_prompt_blocks_cloud_dispatch_with_exit_13(self) -> None:
        from ai_core import router
        cfg = dataclasses.replace(CFG, ollama_url="http://127.0.0.1:9")  # Ollama down -> default routing -> cloud
        opts = router.TaskOptions(f"use key {FAKE_ANTHROPIC} to call the API", self.ws)
        with mock.patch.object(ClaudeAdapter, "run_task") as dispatch, \
                mock.patch.object(sys.stdin, "isatty", return_value=False), \
                redirect_stdout(io.StringIO()), redirect_stderr(err := io.StringIO()):
            self.assertEqual(router.run(opts, cfg), ExitCode.SECRET_BLOCKED)
        dispatch.assert_not_called()
        self.assertFalse(list(self.store.glob("*/*")), "blocked before any checkpoint")
        self.assertNotIn(FAKE_ANTHROPIC, err.getvalue())

    def test_secret_file_target_exits_13(self) -> None:
        from ai_core import router
        write(self.ws / ".env.local", "K=V\n")
        opts = router.TaskOptions("read it", self.ws, (".env.local",))
        cfg = dataclasses.replace(CFG, ollama_url="http://127.0.0.1:9")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(router.run(opts, cfg), ExitCode.SECRET_BLOCKED)

    def test_routing_validation_drops_hallucinated_and_unsafe_paths(self) -> None:
        from ai_core import router
        write(self.ws / "real.py", "")
        data = {"files_scope": 1, "reasoning_depth": 2, "needs_repo_search": False, "creates_new_files": False,
                "task_type": "debug", "target_files": ["real.py", "ghost.py", "../x.py", ".env"], "routing_notes": None}
        routing = router.validate_routing(data, self.ws, CFG, [])
        self.assertEqual(routing.target_files, [(self.ws / "real.py").resolve()])
        with self.assertRaises(ValueError):
            router.validate_routing({**data, "reasoning_depth": True}, self.ws, CFG, [])
        with self.assertRaises(ValueError):
            router.validate_routing({**data, "task_type": "hack"}, self.ws, CFG, [])

    def test_decision_matrix(self) -> None:
        from ai_core import router
        all_up = router.Channels("available", frozenset({"gemini", "argon", "claude_gpt"}))
        one = [self.ws / "a.py"]
        local = router.Routing(1, 2, False, False, "debug", one, None, "ollama")

        def pick(routing: router.Routing, tokens: int = 1000, ch: router.Channels = all_up,
                 offline: bool = False) -> tuple[str, str]:
            d = router.decide(CFG, routing, tokens, offline, ch)
            return d.channel, d.model

        self.assertEqual(pick(local), ("local", CFG.ollama_model))
        # Sonnet/Opus default to self-refilling Channel B; Haiku lives only on Channel A (B01, Q01).
        self.assertEqual(pick(dataclasses.replace(local, reasoning_depth=3)), ("agy", CFG.agy_model_sonnet))
        self.assertEqual(pick(dataclasses.replace(local, reasoning_depth=4)), ("agy", CFG.agy_model_opus))
        # A small task with no files stays local; one that needs a repo search skips Tier-0 and goes to Haiku.
        self.assertEqual(pick(dataclasses.replace(local, target_files=[], reasoning_depth=1)),
                         ("local", CFG.ollama_model))
        search = dataclasses.replace(local, target_files=[], reasoning_depth=1, needs_repo_search=True)
        self.assertEqual(pick(search), ("claude", "haiku"))
        # Wide / multi-file / large context -> Gemini on Channel B.
        self.assertEqual(pick(local, 30000), ("agy", CFG.agy_model_argon_low))
        self.assertEqual(pick(dataclasses.replace(local, task_type="docs", reasoning_depth=3)),
                         ("agy", CFG.agy_model_argon_med))
        # Channel B bucket cooling down -> Channel A takes Sonnet/Opus (Q03).
        b_down = router.Channels("available", frozenset({"gemini", "argon"}))
        self.assertEqual(pick(dataclasses.replace(local, reasoning_depth=3), ch=b_down), ("claude", "sonnet"))
        # Channel A exhausted -> Haiku task falls to Gemini Flash; everything down -> forced Tier-0 or none.
        a_down = router.Channels("exhausted", frozenset({"gemini", "argon", "claude_gpt"}))
        self.assertEqual(pick(search, ch=a_down), ("agy", CFG.agy_model_flash))
        none = router.Channels("exhausted", frozenset())
        self.assertEqual(pick(dataclasses.replace(local, reasoning_depth=3), ch=none), ("local", CFG.ollama_model))
        # Nothing fits locally (too many tokens) and every cloud channel is down -> no decision.
        self.assertEqual(pick(dataclasses.replace(local, reasoning_depth=3), 99999, ch=none)[0], "none")
        # --offline wins over everything.
        self.assertEqual(pick(dataclasses.replace(local, reasoning_depth=4), 99999, offline=True)[0], "local")

    def test_cold_model_gets_load_allowance(self) -> None:
        # Regression: measured cold start = 48.7 s load + 19.9 s warm-up; a 25 s cap aborted the load forever.
        from ai_core import router
        reply = json.dumps({"files_scope": 1, "reasoning_depth": 1, "needs_repo_search": False,
                            "creates_new_files": False, "task_type": "debug", "target_files": [], "routing_notes": None})
        for cold, minimum in ((False, 0), (True, CFG.ollama_cold_load_timeout_s)):
            with mock.patch.object(router, "ollama_chat", return_value=(reply, "stop")) as chat, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(router.score_task(CFG, "t", self.ws, [], {}, 100, cold=cold).source, "ollama")
            timeout = chat.call_args.kwargs["timeout"]
            self.assertGreaterEqual(timeout, minimum)
            self.assertLessEqual(timeout, CFG.scoring_max_timeout_s + minimum)


# ------------------------------------------------------------------ config


class TestConfig(TempWorkspace):
    def test_repository_config_is_valid_and_calibrated(self) -> None:
        cfg = load_config(REPO)
        self.assertEqual((cfg.ollama_model, cfg.ollama_num_ctx, cfg.local_max_input_tokens),
                         ("qwen3.6:35b-a3b", 32768, 26000))
        self.assertFalse(cfg.claude_allow_usage_credits)
        self.assertEqual(cfg.claude_credit_reserve_pct, 15)
        self.assertIn(".env.*", cfg.secret_file_patterns)

    def test_missing_file_gives_defaults(self) -> None:
        self.assertEqual(load_config(self.ws), Config())

    def test_rejects_unknown_keys_wrong_types_and_enums(self) -> None:
        for bad in ({"olama_model": "x"}, {"secret_scanner_enabled": 1}, {"ollama_num_ctx": True},
                    {"ollama_num_ctx": "32768"}, {"ponytail_mode": "max"}, {"secret_file_patterns": [1]},
                    {"claude_credit_cycle_day": 32}, {"ollama_num_ctx": 0}):
            with self.assertRaises(ConfigError, msg=bad):
                validate_config(bad)
        self.assertEqual(validate_config({"claude_monthly_credit_usd": 20}).claude_monthly_credit_usd, 20.0)

    def test_repository_blackboard_is_valid_initial_structure(self) -> None:
        bb = json.loads((REPO / ".ai_context" / "blackboard.json").read_text(encoding="utf-8"))
        self.assertEqual(set(bb), {"schema_version", "system_identity", "pending_sessions",
                                   "architectural_decisions", "active_working_set", "chat_history",
                                   "recent_tasks"})


if __name__ == "__main__":
    unittest.main()
