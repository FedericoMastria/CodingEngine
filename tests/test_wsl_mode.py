"""WSL mode tests (stdlib unittest). Live classes are skipped when wsl.exe or the Ubuntu distro is missing.

Run: uv run --no-project --python ">=3.14" python -m unittest tests.test_wsl_mode -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import secrets
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared fixtures in test_mvp_core

import ai_router
from test_mvp_core import CFG, TempWorkspace, write

import dataclasses

from ai_core import ExitCode, TaskStatus, review, router, vault, verification, wsl
from ai_core.adapters import antigravity_adapter, claude_adapter
from ai_core.adapters.base import AdapterError, TaskRequest
from ai_core.checkpoint import (
    create_checkpoint, finalize_checkpoint, git_refusal, git_toplevel, safe_undo, write_bytes_atomic,
)
from ai_core.commit import _git as commit_git
from ai_core.config import load_config
from ai_core.patcher import EditBlock, OutsideTargetError, apply_blocks
from ai_core.repomap import list_project_files
from ai_core.review import collect_diff
from ai_core.runner import run_tracked
from ai_core.security import API_KEY_VARS, path_key
from ai_core.verification import PASSED, SKIPPED, TIMEOUT, LevelResult, default_l2_command, verify
from ai_core.wsl import WslError, WslTarget, canonical_root, project_argv, project_key, to_windows, wsl_target

UBUNTU = r"\\wsl.localhost\Ubuntu"
# Git for Windows ships a Windows sh.exe: a command must prove it ran in the distro, not just that it ran.
ON_WSL_KERNEL = "uname -r | grep -qi microsoft"


def _wsl_ready() -> bool:
    try:
        return "Ubuntu" in wsl.distros()
    except WslError:
        return False


LIVE = unittest.skipUnless(_wsl_ready(), "needs wsl.exe and the Ubuntu distro")


class TestWslPaths(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(wsl, "distros", return_value=["Ubuntu"])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_wsl_target_parses_both_prefixes(self) -> None:
        expected = WslTarget("Ubuntu", "/home/user/project")
        self.assertEqual(wsl_target(UBUNTU + r"\home\user\project"), expected)
        self.assertEqual(wsl_target(r"\\wsl$\Ubuntu\home\user\project"), expected)
        self.assertEqual(wsl_target(r"\\WSL.LOCALHOST\Ubuntu"), WslTarget("Ubuntu", "/"))

    def test_wsl_target_none_for_other_paths(self) -> None:
        self.assertIsNone(wsl_target(r"C:\x"))
        self.assertIsNone(wsl_target(r"\\server\share\x"))

    def test_to_windows_maps_linux_paths_back(self) -> None:
        t = WslTarget("Ubuntu", "/")
        self.assertEqual(to_windows(t, "/home/user/project"), Path(UBUNTU + r"\home\user\project"))
        self.assertEqual(to_windows(t, "/mnt/c/Users"), Path(r"C:\Users"))

    def test_canonical_root_dollar_and_case(self) -> None:
        self.assertEqual(canonical_root(Path(r"\\wsl$\ubuntu\home\f\P")), Path(UBUNTU + r"\home\f\P"))
        self.assertEqual(canonical_root(Path(r"A:\x")), Path(r"A:\x"))
        with self.assertRaises(WslError):
            canonical_root(Path(r"\\wsl.localhost\Nope\x"))

    def test_project_argv_passthrough_and_wsl(self) -> None:
        self.assertEqual(project_argv(Path(r"A:\p"), ["git", "status"]), ["git", "status"])
        root = Path(UBUNTU + r"\home\f\P")
        tsc = UBUNTU + r"\home\f\P\node_modules\typescript\bin\tsc"
        self.assertEqual(
            project_argv(root, ["node", tsc, "--noEmit"]),
            ["wsl.exe", "-d", "Ubuntu", "--cd", "/home/f/P", "-e", "node", "/home/f/P/node_modules/typescript/bin/tsc",
             "--noEmit"])


class TestWslEntry(unittest.TestCase):
    def test_refuses_windows_dir_fallback(self) -> None:
        err = io.StringIO()
        windir = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        with mock.patch.object(Path, "cwd", return_value=windir), redirect_stderr(err):
            self.assertEqual(ai_router.main(["--status"]), ExitCode.USAGE)
        self.assertIn("UNC", err.getvalue())
        self.assertIn("--dir", err.getvalue())

    def test_missing_distro_refused(self) -> None:
        err = io.StringIO()
        with mock.patch.object(wsl, "distros", return_value=["Ubuntu"]), redirect_stderr(err):
            self.assertEqual(ai_router.main(["--dir", r"\\wsl.localhost\Nope\x", "--status"]), ExitCode.USAGE)
        self.assertTrue(err.getvalue().startswith("✖ WSL distro 'Nope' not found"), err.getvalue())

    @LIVE
    def test_verbose_names_the_wsl_project(self) -> None:
        err = io.StringIO()
        with mock.patch("ai_core.checkpoint.git_refusal", return_value=None), \
                mock.patch.object(router, "show_status", return_value=ExitCode.OK) as status, redirect_stderr(err):
            code = ai_router.main(["--dir", r"\\wsl$\ubuntu\tmp", "--status", "-v"])
        self.assertEqual(code, ExitCode.OK)
        self.assertIn("→ WSL: Ubuntu:/tmp", err.getvalue())
        self.assertEqual(status.call_args.args[0], Path(UBUNTU + r"\tmp"))


class TestWslStatus(TempWorkspace):
    def _status(self, target: WslTarget | None) -> dict:
        out = io.StringIO()
        stub = mock.Mock(reachable=False, model_present=False, gpu_full=None, error="down")
        boom = mock.Mock(side_effect=router.AdapterError("no cli"))
        with mock.patch.object(router, "ollama_status", return_value=stub), \
                mock.patch.object(router, "_claude_adapter", boom), \
                mock.patch.object(router, "AntigravityAdapter", boom), \
                mock.patch.object(router, "wsl_target", return_value=target), redirect_stdout(out):
            router.show_status(self.ws, load_config(self.ws), as_json=True)
        return json.loads(out.getvalue().strip().splitlines()[-1])

    def test_status_reports_wsl_project(self) -> None:
        self.assertEqual(self._status(WslTarget("Ubuntu", "/home/f/P"))["wsl"], "Ubuntu:/home/f/P")
        self.assertNotIn("wsl", self._status(None))


@LIVE
class TestWslLive(unittest.TestCase):
    def test_distros_lists_ubuntu_as_utf8(self) -> None:
        self.assertIn("Ubuntu", wsl.distros())

    def test_linux_home_is_an_absolute_linux_path(self) -> None:
        home = wsl.linux_home("Ubuntu")
        self.assertTrue(home.startswith("/home/") and "\n" not in home and "\r" not in home, repr(home))


def _in_distro(script: str) -> None:
    subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "sh", "-c", script], check=True, capture_output=True,
                   env={**os.environ, "WSL_UTF8": "1"})


@LIVE
class WslRepo(TempWorkspace):
    """A throwaway Git repo inside the distro (~/ce_wsl_test_<hex>), seen from Windows through its UNC share.

    Committed: a.py, b.py, .gitignore. Then: b.py modified (the user's work in progress), notes.txt untracked,
    ignored/x.txt git-ignored.
    """

    def setUp(self) -> None:
        super().setUp()
        self.linux = f"{wsl.linux_home('Ubuntu')}/ce_wsl_test_{secrets.token_hex(4)}"
        _in_distro(
            f"set -e; mkdir {self.linux}; cd {self.linux}; git init -q; git config user.email t@example.com; "
            r"git config user.name t; printf 'A = 1\n' > a.py; printf 'B = 1\n' > b.py; printf 'ignored/\n' > .gitignore; "
            r"git add .; git commit -qm init; printf 'B = 2\n' > b.py; printf 'note\n' > notes.txt; "
            r"mkdir ignored; printf 'x\n' > ignored/x.txt")
        self.addCleanup(_in_distro, f"rm -rf {self.linux}")
        self.ws = to_windows(WslTarget("Ubuntu", "/"), self.linux)


class TestGitInDistro(WslRepo):
    def test_checkpoint_is_git_in_wsl(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        self.assertTrue(cp.is_git)
        self.assertEqual(git_toplevel(self.ws), self.ws)

    def test_checkpoint_undo_lf_exact_with_dirty_tree(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        self.assertTrue(cp.is_git)  # otherwise this only proves the non-Git fallback
        write(self.ws / "a.py", "A = 2\n")
        finalize_checkpoint(self.ws, CFG, cp)
        safe_undo(self.ws, CFG)
        self.assertEqual((self.ws / "a.py").read_bytes(), b"A = 1\n")  # LF, no CRLF from a Windows autocrlf
        self.assertEqual((self.ws / "b.py").read_bytes(), b"B = 2\n")  # the user's own change survives
        self.assertTrue((self.ws / "notes.txt").exists())

    def test_undo_across_spellings(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [self.ws / "a.py"])
        write(self.ws / "a.py", "A = 2\n")
        finalize_checkpoint(self.ws, CFG, cp)
        other = str(self.ws).replace(UBUNTU, r"\\wsl$\ubuntu")
        self.assertNotEqual(other, str(self.ws))
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(ai_router.main(["--dir", other, "--undo"]), ExitCode.OK)
        self.assertEqual((self.ws / "a.py").read_bytes(), b"A = 1\n")

    def test_git_refusal_none_in_wsl(self) -> None:
        self.assertIsNone(git_refusal(self.ws))

    def test_review_repomap_and_commit_use_linux_git(self) -> None:
        self.assertIn("b.py", [path for path, _ in collect_diff(self.ws, CFG)])
        files = list_project_files(self.ws, CFG)
        self.assertIn("a.py", files)
        self.assertIn("notes.txt", files)
        self.assertNotIn("ignored/x.txt", files)  # .gitignore honoured: Git is really being used
        self.assertIn("b.py", commit_git(self.ws, "diff", "--name-only").stdout)


LINUX_CLAUDE = "/home/f/.local/bin/claude"
WSL_ROOT = Path(UBUNTU + r"\home\f\P")
NOT_INSTALLED = "claude is not installed in WSL distro 'Ubuntu': install it there and log in"


class TestClaudeInDistro(unittest.TestCase):
    """Channel A on a WSL project runs the distro's own `claude`: Claude Code refuses every Edit on a UNC cwd."""

    def _adapter(self) -> claude_adapter.ClaudeAdapter:
        adapter = claude_adapter.ClaudeAdapter(CFG, binary="claude-windows.exe")
        healthy = "\n".join(claude_adapter.REQUIRED_FLAGS)
        patcher = mock.patch.object(adapter, "_run_quick", side_effect=lambda *a, **k: "2.1.310" if a == ("--version",) else healthy)
        patcher.start()
        self.addCleanup(patcher.stop)
        return adapter

    def _run(self, root: Path) -> dict:
        captured: dict = {}

        def fake_stream(cmd, cwd, payload, env, timeout_s, on_line):
            captured.update(cmd=list(cmd), cwd=cwd, payload=payload, env=env)
            return 0, "", False

        request = TaskRequest("t", "hello", root, "haiku", (), "ctx", "SYS\nTEM")
        with mock.patch.object(claude_adapter, "stream_process", fake_stream), \
                mock.patch.object(claude_adapter, "linux_claude", return_value=LINUX_CLAUDE):
            self._adapter().run_task(request)
        return captured

    def test_wsl_project_runs_claude_inside_the_distro(self) -> None:
        got = self._run(WSL_ROOT)
        strip = [arg for var in sorted(API_KEY_VARS) for arg in ("-u", var)]  # also removed on the Linux side
        head = ["wsl.exe", "-d", "Ubuntu", "--cd", "/home/f/P", "-e", "env", *strip, LINUX_CLAUDE, "-p"]
        self.assertEqual(got["cmd"][:len(head)], head)
        self.assertIn("SYS\nTEM", got["cmd"])  # multi-line system prompt arrives as one argument
        self.assertEqual((got["cwd"], got["payload"]), (WSL_ROOT, "ctx\n\nhello"))

    def test_windows_project_still_runs_the_windows_binary(self) -> None:
        got = self._run(Path(r"A:\p"))
        self.assertEqual(got["cmd"][:2], ["claude-windows.exe", "-p"])
        self.assertNotIn("wsl.exe", got["cmd"])

    def test_missing_claude_in_distro_is_a_clear_adapter_error(self) -> None:
        request = TaskRequest("t", "hello", WSL_ROOT, "haiku", (), "", "")
        with mock.patch.object(claude_adapter, "linux_claude", side_effect=WslError(NOT_INSTALLED)), \
                self.assertRaisesRegex(AdapterError, "not installed in WSL distro"):
            self._adapter().run_task(request)


class TestAgyOnWsl(unittest.TestCase):
    """agy with a UNC cwd asks for a login (probed 2026-10-26); from a Windows folder with --add-dir it edits fine."""

    def _run(self, root: Path) -> dict:
        captured: dict = {}

        def fake_stream(cmd, cwd, payload, env, timeout_s, on_line):
            captured.update(cmd=list(cmd), cwd=cwd)
            return 0, "", False

        adapter = antigravity_adapter.AntigravityAdapter(CFG, binary="agy.exe")
        adapter.version = "1.4.7"  # skip the real probe
        request = TaskRequest("t", "fix it", root, "gemini-flash", (), "repo-map", "SYS")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(antigravity_adapter, "stream_process", fake_stream),                 mock.patch.object(antigravity_adapter, "WSL_CWD", Path(tmp) / "agy_cwd"):
            adapter.run_task(request)
            captured["neutral"] = Path(tmp) / "agy_cwd"
            captured["neutral_created"] = captured["neutral"].is_dir()
        return captured

    def test_wsl_project_starts_in_a_windows_folder_with_the_project_added(self) -> None:
        got = self._run(WSL_ROOT)
        cmd = got["cmd"]
        self.assertEqual(got["cwd"], got["neutral"])
        self.assertTrue(got["neutral_created"])
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], str(WSL_ROOT))
        prompt = next(arg for arg in cmd if arg.startswith("--print="))
        self.assertIn(str(WSL_ROOT), prompt)  # relative paths in the task are resolved against the project root
        self.assertIn("repo-map", prompt)  # context is still sent

    def test_windows_project_is_unchanged(self) -> None:
        got = self._run(Path(r"A:\p"))
        self.assertEqual(got["cwd"], Path(r"A:\p"))
        self.assertNotIn("--add-dir", got["cmd"])


class TestChannelsOnWsl(unittest.TestCase):
    def _channels(self, root: Path, *, in_distro: bool, windows_cli: bool = False) -> tuple[router.Channels, str, mock.Mock]:
        err = io.StringIO()
        linux = mock.Mock(return_value=LINUX_CLAUDE) if in_distro else mock.Mock(side_effect=WslError(NOT_INSTALLED))
        which = (lambda name: rf"C:\tools\{name}.exe") if windows_cli else (lambda name: None)
        with mock.patch.object(router, "is_offline", return_value=False), mock.patch.object(router, "_warned", set()), \
                mock.patch.object(router, "linux_claude", linux), mock.patch.object(router.shutil, "which", side_effect=which), \
                redirect_stderr(err):
            channels = router.compute_channels(CFG, None, root)
            router.compute_channels(CFG, None, root)  # rotation asks again: no second warning
        return channels, err.getvalue(), linux

    def test_claude_in_the_distro_enables_channel_a_without_a_windows_claude(self) -> None:
        channels, warning, _ = self._channels(WSL_ROOT, in_distro=True)
        self.assertEqual((channels.a_state, channels.b_ready), (router.AVAILABLE, frozenset()))  # no agy installed here
        self.assertNotIn("agy", warning)

    def test_agy_is_available_on_wsl_projects(self) -> None:
        channels, warning, _ = self._channels(WSL_ROOT, in_distro=True, windows_cli=True)
        self.assertTrue(channels.b_ready)  # agy runs as the Windows binary, see TestAgyOnWsl
        self.assertNotIn("agy", warning)

    def test_missing_claude_in_the_distro_blocks_channel_a_with_a_hint(self) -> None:
        channels, warning, _ = self._channels(WSL_ROOT, in_distro=False)
        self.assertEqual(channels.a_state, "unavailable")
        self.assertIn("install it there", warning)

    def test_windows_projects_never_ask_the_distro(self) -> None:
        channels, _, linux = self._channels(Path(r"A:\p"), in_distro=False, windows_cli=True)
        self.assertEqual(channels.a_state, router.AVAILABLE)
        self.assertTrue(channels.b_ready)
        linux.assert_not_called()

    def _run_cloud(self, channel: str, **patches: object) -> router.Outcome:
        adapter = mock.Mock()
        ctx = router.TaskContext(router.default_routing([]), {}, "", "", "", 100)
        with mock.patch.object(router, "_adapter", return_value=adapter), redirect_stderr(io.StringIO()):
            out = router.run_cloud(router.TaskOptions("t", WSL_ROOT), CFG, ctx, router.Decision(channel, "m", "t"), "t1")
        adapter.run_task.assert_not_called()
        return out

    def test_forced_claude_without_claude_in_the_distro_says_how_to_fix_it(self) -> None:
        with mock.patch.object(router, "linux_claude", side_effect=WslError(NOT_INSTALLED)):
            out = self._run_cloud("claude")
        self.assertEqual(out.status, TaskStatus.PROVIDER_ERROR)
        self.assertIn("install it there", out.message)


class TestLinuxClaude(unittest.TestCase):
    def test_not_installed_is_a_clear_error(self) -> None:
        with mock.patch.object(wsl, "_wsl_out", side_effect=WslError("wsl.exe ... failed: ")), \
                self.assertRaisesRegex(WslError, "not installed in WSL distro 'Nowhere'.*log in"):
            wsl.linux_claude("Nowhere")

    @LIVE
    def test_resolves_the_login_shell_path(self) -> None:
        # `wsl.exe -e` has a bare PATH (no ~/.local/bin): only a login shell finds it
        try:
            path = wsl.linux_claude("Ubuntu")
        except WslError:
            self.skipTest("claude is not installed in the Ubuntu distro")
        self.assertTrue(path.startswith("/") and path.endswith("claude"), path)


class TestVaultInDistro(WslRepo):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(vault, "VAULT_PARENT", None)  # the real location, not TempWorkspace's override
        patcher.start()
        self.addCleanup(patcher.stop)
        self.vault_root = vault.vault_root(self.ws)
        self.addCleanup(_in_distro, f"rmdir --ignore-fail-on-non-empty {wsl_target(self.vault_root).linux_path} "
                                   "2>/dev/null || true")
        _in_distro(f"cd {self.linux}; " + r"printf 'K=v\n' > .env")

    def test_vault_root_in_linux_home(self) -> None:
        t = WslTarget("Ubuntu", "/")
        self.assertEqual(self.vault_root, to_windows(t, f"{wsl.linux_home('Ubuntu')}/.ai_vault_{vault.user_sid()}"))

    def test_vault_round_trip_wsl(self) -> None:
        env = self.ws / ".env"
        with vault.NTFSVault(self.ws, [env]):
            self.assertFalse(env.exists())
            mode = subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "stat", "-c", "%a",
                                   wsl_target(self.vault_root).linux_path], capture_output=True, text=True)
            self.assertEqual(mode.stdout.strip(), "700")
        self.assertEqual(env.read_bytes(), b"K=v\n")

    def test_vault_refuses_cross_filesystem(self) -> None:
        with mock.patch.object(vault, "_device", side_effect=["1", "2"]), \
                self.assertRaisesRegex(vault.VaultError, "not on the project's volume"):
            vault.NTFSVault(self.ws, [self.ws / ".env"]).__enter__()
        self.assertTrue((self.ws / ".env").exists())


class TestVaultRecoverWithoutWsl(unittest.TestCase):
    def test_recover_reports_instead_of_crashing(self) -> None:
        root = Path(UBUNTU + r"\home\f\P")
        with mock.patch.object(vault, "VAULT_PARENT", None), \
                mock.patch.object(vault, "linux_home", side_effect=WslError("wsl.exe not found")):
            messages = vault.recover(root)
        self.assertTrue(messages and messages[0].startswith("⚠"), messages)


class TestCaseSensitivity(unittest.TestCase):
    """Linux file names are case-sensitive: `Foo.py` and `foo.py` are two files in a WSL project."""

    def test_path_key_preserves_case_only_for_wsl_paths(self) -> None:
        self.assertNotEqual(path_key(UBUNTU + r"\home\f\Foo.py"), path_key(UBUNTU + r"\home\f\foo.py"))
        self.assertEqual(path_key(r"C:\A\Foo.py"), path_key(r"c:\a\foo.py"))

    def test_project_key_follows_the_project_kind(self) -> None:
        wsl_root, win_root = Path(UBUNTU + r"\home\f\P"), Path(r"A:\p")
        self.assertNotEqual(project_key(wsl_root, "Foo.py"), project_key(wsl_root, "foo.py"))
        self.assertEqual(project_key(win_root, "Src\\Foo.py"), project_key(win_root, "src/foo.py"))

    def test_patcher_exact_case_first(self) -> None:
        originals = {"foo.py": "y = 1\n", "Foo.py": "x = 1\n"}  # the case-folded lookup used to pick the last one
        self.assertEqual(apply_blocks(originals, [EditBlock("foo.py", "y = 1", "y = 2")]), {"foo.py": "y = 2\n"})

    def test_patcher_case_fallback_only_when_unambiguous(self) -> None:
        self.assertEqual(apply_blocks({"src/calc.py": "x = 1\n"}, [EditBlock("Src\\Calc.py", "x = 1", "x = 2")]),
                         {"src/calc.py": "x = 2\n"})
        with self.assertRaises(OutsideTargetError):
            apply_blocks({"foo.py": "a\n", "Foo.py": "b\n"}, [EditBlock("FOO.py", "a", "c")])

    def test_validate_routing_keeps_case_distinct_targets(self) -> None:
        root = Path(UBUNTU + r"\home\f\P")
        data = {"files_scope": 1, "reasoning_depth": 2, "needs_repo_search": False, "creates_new_files": False,
                "task_type": "debug", "target_files": []}
        routing = router.validate_routing(data, root, CFG, [root / "Foo.py", root / "foo.py"])
        self.assertEqual(len(routing.target_files), 2)


@contextlib.contextmanager
def _cwd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class TestReviewFixes(unittest.TestCase):
    """Final-review findings: each one failed before its fix."""

    @LIVE
    def test_relative_dir_from_a_wsl_cwd_is_canonicalised(self) -> None:
        err = io.StringIO()
        with _cwd(r"\\wsl$\ubuntu\tmp"), mock.patch("ai_core.checkpoint.git_refusal", return_value=None), \
                mock.patch.object(router, "show_status", return_value=ExitCode.OK) as status, redirect_stderr(err):
            self.assertEqual(ai_router.main(["--dir", ".", "--status", "-v"]), ExitCode.OK)
        self.assertEqual(status.call_args.args[0], Path(UBUNTU + r"\tmp"))  # one spelling, or checkpoints split
        self.assertIn("→ WSL: Ubuntu:/tmp", err.getvalue())

    def test_dir_dot_in_the_windows_folder_is_refused(self) -> None:
        err = io.StringIO()
        with _cwd(os.environ.get("SystemRoot", r"C:\Windows")), redirect_stderr(err):
            self.assertEqual(ai_router.main(["--dir", ".", "--status"]), ExitCode.USAGE)
        self.assertIn("UNC", err.getvalue())

    def test_wsl_launches_never_carry_billing_vars(self) -> None:
        env = {"ANTHROPIC_API_KEY": "x", "WSLENV": "ANTHROPIC_API_KEY"}
        root = Path(UBUNTU + r"\home\f\P")
        seen: list[dict] = []

        def fake_run(argv, **kwargs):
            seen.append(kwargs.get("env"))  # None = inherited os.environ, billing vars included
            if "rev-parse" in argv:
                raise RuntimeError("stop after review's first launch")
            return subprocess.CompletedProcess(argv, 0, stdout="1\n", stderr="")

        with mock.patch.dict(os.environ, env), mock.patch.object(vault.subprocess, "run", fake_run), \
                mock.patch.object(review.subprocess, "run", fake_run), \
                mock.patch.object(review, "git_toplevel", return_value=root):
            vault._device(root)
            vault._restrict_acl(root)
            with self.assertRaises(RuntimeError):
                review.collect_diff(root, CFG)
            self.assertFalse("ANTHROPIC_API_KEY" in wsl._wsl_env())  # not assertNotIn: it would print the whole environment
        self.assertEqual(len(seen), 3)
        for e in seen:
            self.assertIsNotNone(e)
            self.assertFalse("ANTHROPIC_API_KEY" in e)

class TestGitRefusalExemptions(TempWorkspace):
    def test_recover_and_status_do_not_need_git(self) -> None:
        self.init_git()  # before the env var: it would make `git init` itself fail
        with mock.patch.dict(os.environ, {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}), \
                mock.patch.object(router, "show_status", return_value=ExitCode.OK), \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(ai_router.main(["--dir", str(self.ws), "--recover"]), ExitCode.OK)
            self.assertEqual(ai_router.main(["--dir", str(self.ws), "--status"]), ExitCode.OK)


class TestModePreserved(WslRepo):
    """A rename over a file through the 9P share resets its mode to 644: scripts lose +x, 600 files open up."""

    def setUp(self) -> None:
        super().setUp()
        _in_distro(f"cd {self.linux}; " + r"printf '#!/bin/sh\necho 1\n' > run.sh; chmod 755 run.sh; "
                   r"printf 's\n' > priv; chmod 600 priv")

    def _mode(self, name: str) -> str:
        cp = subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "stat", "-c", "%a", f"{self.linux}/{name}"],
                            capture_output=True, text=True)
        return cp.stdout.strip()

    def test_atomic_write_keeps_the_mode(self) -> None:
        write_bytes_atomic(self.ws / "run.sh", b"#!/bin/sh\necho 2\n")
        write_bytes_atomic(self.ws / "priv", b"t\n")
        self.assertEqual((self._mode("run.sh"), self._mode("priv")), ("755", "600"))

    def test_checkpoint_edit_undo_keeps_the_mode(self) -> None:
        cp = create_checkpoint(self.ws, CFG, [self.ws / "run.sh"])
        write_bytes_atomic(self.ws / "run.sh", b"#!/bin/sh\necho 2\n")
        finalize_checkpoint(self.ws, CFG, cp)
        safe_undo(self.ws, CFG)
        self.assertEqual((self.ws / "run.sh").read_bytes(), b"#!/bin/sh\necho 1\n")
        self.assertEqual(self._mode("run.sh"), "755")


class TestExitCodes(unittest.TestCase):
    def test_exit_127_is_skipped_in_wsl_only(self) -> None:
        proc = mock.Mock(returncode=127)
        proc.communicate.return_value = ("sh: 1: pytest: not found", None)
        with mock.patch.object(verification.subprocess, "Popen", return_value=proc):
            self.assertEqual(verification._run(["pytest"], Path(UBUNTU + r"\home\f\P"), 5, 3).status, SKIPPED)
            self.assertEqual(verification._run(["bash", "-c", "x"], Path("."), 5, 3).status,
                             verification.FAILED)  # a Windows project's own 127 is a real failure


class TestVerifyInDistro(WslRepo):
    def setUp(self) -> None:
        super().setUp()
        _in_distro(f"cd {self.linux}; " + r"printf 'import unittest\nfrom a import A\nclass T(unittest.TestCase):\n"
                   r"    def test_a(self):\n        self.assertEqual(A, 1)\n' > test_a.py")

    def _l3_argv(self, cfg=CFG) -> list[str]:
        seen: dict[int, list[str]] = {}

        def fake(cmd, cwd, timeout_s, level):
            seen[level] = list(cmd)
            return LevelResult(level, PASSED)

        with mock.patch.object(verification, "_run", fake):
            verification._run_levels(self.ws, cfg, [self.ws / "a.py"])
        return seen[3]

    def test_l3_python_substitution(self) -> None:
        self.assertEqual(self._l3_argv()[:3], ["python3", "-m", "pytest"])
        _in_distro(f"cd {self.linux}; mkdir -p .venv/bin; ln -s /usr/bin/python3 .venv/bin/python")
        self.assertEqual(self._l3_argv()[0], ".venv/bin/python")
        own = dataclasses.replace(CFG, verify_l3_command=["uv", "run", "pytest"])
        self.assertEqual(self._l3_argv(own)[:3], ["uv", "run", "pytest"])  # a configured command is used as-is

    def test_l3_python_found_through_symlinked_venv(self) -> None:
        # a symlinked .venv directory is dangling from Windows' point of view: only the distro can tell it works
        self.addCleanup(_in_distro, f"rm -rf {self.linux}_venv")
        _in_distro(f"mkdir -p {self.linux}_venv/bin; ln -s /usr/bin/python3 {self.linux}_venv/bin/python; "
                   f"ln -s {self.linux}_venv {self.linux}/.venv")
        self.assertEqual(self._l3_argv()[0], ".venv/bin/python")

    def test_default_l2_uses_linux_node_and_relative_tsc(self) -> None:
        self.assertEqual(default_l2_command(self.ws), [])
        _in_distro(f"cd {self.linux}; mkdir -p node_modules/typescript/bin; touch node_modules/typescript/bin/tsc "
                   "tsconfig.json")
        self.assertEqual(default_l2_command(self.ws), ["node", "node_modules/typescript/bin/tsc", "--noEmit"])

    def test_verify_runs_in_distro(self) -> None:
        cfg = dataclasses.replace(CFG, verify_l3_command=["python3", "-m", "unittest"])
        self.assertEqual(verify(self.ws, cfg, [self.ws / "a.py"]).status, TaskStatus.COMPLETED_VERIFIED)
        write(self.ws / "a.py", "A = 22\n")  # another size: .pyc is validated by (mtime seconds, size), "A = 2" would hit it
        self.assertEqual(verify(self.ws, cfg, [self.ws / "a.py"]).status, TaskStatus.TESTS_FAILED)

    def test_timeout_kills_linux_child(self) -> None:
        self.assertEqual(verification._run(["sh", "-c", f"{ON_WSL_KERNEL} && sleep 6007"], self.ws, 2, 3).status, TIMEOUT)
        left = subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "pgrep", "-f", "sleep 6007"], capture_output=True)
        self.assertEqual(left.returncode, 1, "the Linux child outlived the killed wsl.exe")  # pgrep: 1 = no match

    def test_billing_vars_never_reach_distro(self) -> None:
        env = {"ANTHROPIC_API_KEY": "x", "WSLENV": "ANTHROPIC_API_KEY"}
        with mock.patch.dict(os.environ, env):
            result = verification._run(["sh", "-c", f"test -z $ANTHROPIC_API_KEY && {ON_WSL_KERNEL}"], self.ws, 10, 2)
        self.assertEqual(result.status, PASSED, result.output)

    def test_ait_runs_in_distro(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code, tail, _ = run_tracked(self.ws, ["pwd"])
        self.assertEqual(code, 0)
        self.assertEqual(tail.strip(), self.linux)


if __name__ == "__main__":
    unittest.main()
