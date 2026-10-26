"""REPL slash commands (stdlib unittest).

Run: uv run --no-project --python ">=3.14" python -m unittest tests.test_slash -v
"""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared fixtures in test_mvp_core

import ai_router
from test_mvp_core import CFG, TempWorkspace

from ai_core import ExitCode, TaskStatus, router, security, slash


class SlashTests(TempWorkspace):
    def setUp(self) -> None:
        super().setUp()
        self.st = slash.ReplState(self.ws, CFG)
        patcher = mock.patch.object(security, "_offline_latched", False)  # /offline latches for the process
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_line(self, line: str) -> tuple[bool, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            handled = slash.run(self.st, line)
        return handled, buf.getvalue()

    # ---------------------------------------------------------------- dispatch
    def test_plain_text_is_a_task_not_a_command(self) -> None:
        self.assertEqual(self.run_line("fix the bug in +a.py"), (False, ""))

    def test_unknown_command_is_reported_not_run_as_a_task(self) -> None:
        handled, out = self.run_line("/frobnicate now")
        self.assertTrue(handled)
        self.assertIn("unknown command /frobnicate", out)

    def test_handler_error_is_printed_and_the_repl_survives(self) -> None:
        with mock.patch.object(slash, "_gc", side_effect=RuntimeError("boom")), \
                mock.patch.dict(slash._HANDLERS, {"gc": mock.Mock(side_effect=RuntimeError("boom"))}):
            self.assertIn("✖ /gc: boom", self.run_line("/gc")[1])

    def test_keyboard_interrupt_is_not_swallowed(self) -> None:
        with mock.patch.dict(slash._HANDLERS, {"gc": mock.Mock(side_effect=KeyboardInterrupt)}):
            with self.assertRaises(KeyboardInterrupt):
                self.run_line("/gc")

    # ---------------------------------------------------------------- live modes
    def test_channel_switch_and_label(self) -> None:
        self.run_line("/channel agy")
        self.assertEqual((self.st.channel, self.st.options().channel), ("agy", "agy"))
        self.assertIn("channel: agy", self.st.label())
        self.run_line("/channel B")  # alias, case-insensitive
        self.assertEqual(self.st.channel, "agy")
        self.run_line("/channel a")
        self.assertEqual(self.st.channel, "anthropic")
        self.run_line("/channel auto")
        self.assertIsNone(self.st.channel)
        self.assertIn("channel: auto", self.st.label())

    def test_unknown_channel_keeps_the_current_one(self) -> None:
        self.run_line("/channel local")
        _, out = self.run_line("/channel gemini")
        self.assertIn("unknown channel", out)
        self.assertEqual(self.st.channel, "local")

    def test_channel_without_argument_only_reports(self) -> None:
        _, out = self.run_line("/channel")
        self.assertIn("auto", out)
        self.assertIsNone(self.st.channel)

    def test_offline_latches_the_process_and_shows_in_the_label(self) -> None:
        self.assertNotIn("OFFLINE", self.st.label())
        self.run_line("/offline")
        self.assertTrue(security.is_offline())
        self.assertTrue(self.st.offline)
        self.assertTrue(self.st.options().offline)
        self.assertIn("OFFLINE", self.st.label())
        with self.assertRaises(security.OfflineViolationError):
            security.require_cloud_allowed("claude")  # Invariant #1: nothing can undo it

    def test_offline_refuses_a_non_loopback_ollama_before_latching(self) -> None:
        import dataclasses
        self.st.cfg = dataclasses.replace(CFG, ollama_url="http://example.com:11434")
        _, out = self.run_line("/offline")
        self.assertIn("✖ /offline", out)
        self.assertFalse(security.is_offline())

    def test_dry_run_toggles_and_accepts_explicit_values(self) -> None:
        self.run_line("/dry-run")
        self.assertTrue(self.st.dry_run)
        self.assertIn("dry-run", self.st.label())
        self.run_line("/dry-run")
        self.assertFalse(self.st.dry_run)
        self.run_line("/dry-run on")
        self.run_line("/dry-run on")
        self.assertTrue(self.st.dry_run)
        self.run_line("/dry-run off")
        self.assertFalse(self.st.dry_run)

    # ---------------------------------------------------------------- wrappers of the CLI flags
    def test_review_parses_mode_and_target(self) -> None:
        with mock.patch("ai_core.review.review", return_value=ExitCode.OK) as rv:
            self.run_line("/review")
            self.run_line("/review cross")
            self.run_line("/review ultra v1.0..HEAD")
            self.run_line("/review src/app.py")  # not a mode: a target for the local review
        calls = [(c.args[2], c.args[4]) for c in rv.call_args_list]
        self.assertEqual(calls, [("local", None), ("cross", None), ("ultra", "v1.0..HEAD"), ("local", "src/app.py")])

    def test_review_uses_the_live_options(self) -> None:
        self.st.offline, self.st.channel = True, "local"
        with mock.patch("ai_core.review.review", return_value=ExitCode.OK) as rv:
            self.run_line("/review")
        opts = rv.call_args.args[3]
        self.assertEqual((opts.offline, opts.channel, opts.agent), (True, "local", False))

    def test_commit_fix_and_plan_forward_their_arguments(self) -> None:
        with mock.patch("ai_core.commit.commit", return_value=ExitCode.OK) as cm, \
                mock.patch("ai_core.runner.fix", return_value=ExitCode.OK) as fx, \
                mock.patch("ai_core.review.plan", return_value=ExitCode.OK) as pl:
            self.run_line("/commit")
            self.run_line("/fix pytest -x tests/")
            self.run_line("/fix")
            self.run_line("/plan add a --json flag to ai --status")
        self.assertEqual(cm.call_count, 1)
        self.assertEqual([c.args[2] for c in fx.call_args_list], [["pytest", "-x", "tests/"], []])
        self.assertEqual(pl.call_args.args[3], "add a --json flag to ai --status")

    def test_adr_list_add_and_quoted_title(self) -> None:
        self.assertIn("no ADRs", self.run_line("/adr")[1])
        self.assertIn("✔", self.run_line('/adr add "Use SQLite for state" "no server needed"')[1])
        out = self.run_line("/adr list")[1]
        self.assertIn("Use SQLite for state", out)

    def test_gc_runs(self) -> None:
        self.assertIn("removed 0", self.run_line("/gc")[1])

    # ---------------------------------------------------------------- diff
    def test_colorize_diff_marks_each_line_kind(self) -> None:
        diff = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n ctx"
        with mock.patch.object(router, "_use_color", return_value=True):
            lines = slash.colorize_diff(diff).splitlines()
        self.assertTrue(lines[0].startswith("\033[1m"))  # header bold
        self.assertTrue(lines[3].startswith("\033[36m"))  # hunk cyan
        self.assertTrue(lines[4].startswith("\033[31m"))  # removed red
        self.assertTrue(lines[5].startswith("\033[32m"))  # added green
        self.assertEqual(lines[6], " ctx")

    def test_colorize_diff_truncates_with_a_pointer_to_diff(self) -> None:
        out = slash.colorize_diff("\n".join(f"+l{i}" for i in range(100)), max_lines=10)
        self.assertEqual(len(out.splitlines()), 11)
        self.assertIn("90 more line(s)", out)
        self.assertIn("/diff", out)

    def task(self, task_id: str = "t1", touched: tuple[str, ...] = ("a.py",), cp: str | None = "cp1") -> dict:
        return {"task_id": task_id, "touched_files": list(touched), "checkpoint_id": cp}

    def test_diff_without_any_task(self) -> None:
        self.assertIn("no task with a checkpoint", self.run_line("/diff")[1])

    def test_diff_prints_the_last_checkpoint_diff_in_full(self) -> None:
        with mock.patch.object(slash, "last_task", return_value=self.task()), \
                mock.patch.object(slash, "diff_since", return_value="+" + "x\n+" * 200) as ds:
            _, out = self.run_line("/diff")
        self.assertEqual(ds.call_args.args[1], "cp1")
        self.assertGreater(len(out.splitlines()), 100)
        self.assertNotIn("more line(s)", out)

    def test_show_changes_prints_a_short_summary_for_a_new_task(self) -> None:
        buf = io.StringIO()
        with mock.patch.object(slash, "last_task", return_value=self.task("t2")), \
                mock.patch.object(slash, "diff_since", return_value="\n".join(f"+l{i}" for i in range(100))), \
                redirect_stdout(buf):
            slash.show_changes(self.st, before_task_id="t1")
        out = buf.getvalue()
        self.assertIn("changes: a.py", out)
        self.assertLessEqual(len(out.splitlines()), slash._CHANGES_LINES + 2)

    def test_show_changes_is_silent_when_nothing_new_or_nothing_changed(self) -> None:
        for task, before in ((self.task("t1"), "t1"),  # same task as before: the run produced no new one
                             (self.task("t2", touched=()), "t1"),  # nothing touched
                             (self.task("t2", cp=None), "t1"),  # no checkpoint to diff against
                             (None, None)):
            buf = io.StringIO()
            with mock.patch.object(slash, "last_task", return_value=task), \
                    mock.patch.object(slash, "diff_since", return_value="+x") as ds, redirect_stdout(buf):
                slash.show_changes(self.st, before_task_id=before)
            self.assertEqual((buf.getvalue(), ds.call_count), ("", 0), task)

    def test_show_changes_survives_a_diff_error(self) -> None:
        with mock.patch.object(slash, "last_task", return_value=self.task("t2")), \
                mock.patch.object(slash, "diff_since", side_effect=RuntimeError("git gone")):
            slash.show_changes(self.st, before_task_id="t1")  # must not raise


class PlanTests(TempWorkspace):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(security, "_offline_latched", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.opts = router.TaskOptions("", self.ws)

    def run_plan(self, task: str, opts: router.TaskOptions | None = None) -> tuple[ExitCode, str, list[str]]:
        from ai_core import review
        channels: list[str] = []

        def fake_run_cloud(o, cfg, ctx, decision, task_id, **kw):  # noqa: ANN001, ANN003
            channels.append(decision.channel)
            self.assertTrue(kw["read_only"])  # a plan must never be able to edit files
            self.assertIn("Do NOT modify any file", kw["prompt"])
            return router.Outcome(TaskStatus.COMPLETED_UNVERIFIED, decision.channel, decision.model)

        buf = io.StringIO()
        with mock.patch.object(router, "run_cloud", fake_run_cloud), redirect_stdout(buf):
            code = review.plan(self.ws, CFG, opts or self.opts, task)
        return code, buf.getvalue(), channels

    def test_plan_needs_a_task(self) -> None:
        code, out, channels = self.run_plan("  ")
        self.assertEqual((code, channels), (ExitCode.USAGE, []))
        self.assertIn("usage: /plan", out)

    def test_plan_refuses_when_offline(self) -> None:
        security.enforce_offline()
        code, _, channels = self.run_plan("add a flag")
        self.assertEqual((code, channels), (ExitCode.OFFLINE_UNAVAILABLE, []))

    def test_plan_runs_read_only_on_channel_a_first(self) -> None:
        code, _, channels = self.run_plan("add a flag")
        self.assertEqual((code, channels), (ExitCode.OK, ["claude"]))

    def test_plan_respects_a_forced_channel(self) -> None:
        _, _, channels = self.run_plan("add a flag", router.TaskOptions("", self.ws, channel="agy"))
        self.assertEqual(channels, ["agy"])

    def test_plan_rotates_to_the_other_channel_when_the_first_is_unavailable(self) -> None:
        from ai_core import review
        seen: list[str] = []

        def fake(o, cfg, ctx, decision, task_id, **kw):  # noqa: ANN001, ANN003
            seen.append(decision.channel)
            if decision.channel == "claude":
                return router.Outcome(TaskStatus.QUOTA_EXHAUSTED, "claude", message="5h window full", rotate=True)
            return router.Outcome(TaskStatus.COMPLETED_UNVERIFIED, "agy")

        with mock.patch.object(router, "run_cloud", fake), redirect_stdout(io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()):
            code = review.plan(self.ws, CFG, self.opts, "add a flag")
        self.assertEqual((code, seen), (ExitCode.OK, ["claude", "agy"]))


class ReplWiringTests(TempWorkspace):
    """`ai >` (plain fallback): the new commands are reachable and /channel changes what tasks run with."""

    def repl(self, lines: list[str], **flags: object) -> str:
        import argparse
        args = argparse.Namespace(channel=None, offline=False, dry_run=False, verbose=False, **flags)
        buf = io.StringIO()
        with mock.patch.object(security, "_offline_latched", False), \
                mock.patch("ai_core.tui.enabled", return_value=False), \
                mock.patch("builtins.input", side_effect=[*lines, "/exit"]), redirect_stdout(buf):
            ai_router.run_repl(self.ws, CFG, args)
        return buf.getvalue()

    def test_help_lists_the_new_commands(self) -> None:
        out = self.repl(["/help"])
        for name in ("/review", "/commit", "/diff", "/fix", "/plan", "/adr", "/channel", "/offline", "/dry-run", "/gc"):
            self.assertIn(name, out)

    def test_channel_set_in_the_repl_reaches_the_next_task(self) -> None:
        seen: list[router.TaskOptions] = []
        with mock.patch.object(router, "run", lambda opts, cfg: seen.append(opts) or ExitCode.OK):
            self.repl(["/channel agy", "/dry-run on", "do something"])
        self.assertEqual([(o.channel, o.dry_run, o.prompt) for o in seen], [("agy", True, "do something")])

    def test_unknown_slash_command_never_becomes_a_task(self) -> None:
        with mock.patch.object(router, "run") as run:
            out = self.repl(["/frobnicate"])
        run.assert_not_called()
        self.assertIn("unknown command", out)

    def test_status_bar_label_follows_the_repl_state(self) -> None:
        labels: list[str] = []

        class FakeTui:
            def __init__(self, root, channel, mode=None):  # noqa: ANN001
                self.mode = mode

            def run(self, step, answered_by):  # noqa: ANN001
                labels.append(self.mode())
                step("/channel local")
                step("/dry-run on")
                labels.append(self.mode())
                return 0

        import argparse
        args = argparse.Namespace(channel=None, offline=False, dry_run=False, verbose=False)
        with mock.patch("ai_core.tui.enabled", return_value=True), mock.patch("ai_core.tui.Tui", FakeTui), \
                redirect_stdout(io.StringIO()):
            ai_router.run_repl(self.ws, CFG, args)
        self.assertEqual(labels, ["channel: auto", "channel: local · dry-run"])


if __name__ == "__main__":
    unittest.main()
