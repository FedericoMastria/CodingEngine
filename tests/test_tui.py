"""TUI input-box tests (stdlib unittest, headless via a prompt_toolkit pipe).

Run: uv run --no-project --python ">=3.14" python -m unittest tests.test_tui -v
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core import tui

if tui.AVAILABLE:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

PAUSE = "|pause|"
ESC, CTRL_C, CTRL_D, CTRL_L = "\x1b", "\x03", "\x04", "\x0c"


async def _drive(keys: str, root: Path, wait: float = 0.4, before: str = "", setup=None) -> tuple[tui.Tui, int]:  # noqa: ANN001
    """Type `keys`, let the app render, return the Tui and the input box's rendered height."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        t = tui.Tui(root, None)
        t.loop, t.jobs = asyncio.get_running_loop(), asyncio.Queue()
        if setup:
            setup(t)
        t.app = t._build(lambda line: False)
        run = asyncio.create_task(t.app.run_async())
        await asyncio.sleep(0.2)
        if before:
            t._append(before)
        for chunk in keys.split(PAUSE):  # PAUSE lets async work (history load, completion) settle
            pipe.send_text(chunk)
            await asyncio.sleep(0.2)
        await asyncio.sleep(wait)
        height = t.input.window.render_info.window_height
        t.app.exit()
        await run
    return t, height


@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class InputBoxTests(unittest.TestCase):
    def run_keys(self, keys: str, wait: float = 0.4, before: str = "", setup=None) -> tuple[tui.Tui, int]:  # noqa: ANN001
        with tempfile.TemporaryDirectory() as d:
            return asyncio.run(_drive(keys, Path(d), wait, before, setup))

    def test_long_text_wraps_instead_of_scrolling_off(self) -> None:
        text = "parola " * 60  # 420 chars on an 80-col screen
        t, height = self.run_keys(text)
        self.assertEqual(t.input.text, text)
        self.assertGreater(height, 1)
        self.assertEqual(t.input.window.render_info.vertical_scroll, 0)  # first line still in view

    def test_height_is_capped_and_scrolls(self) -> None:
        t, height = self.run_keys("x" * 5000)
        self.assertEqual(height, tui._INPUT_ROWS)
        self.assertEqual(len(t.input.text), 5000)

    def test_enter_submits_whole_long_text_and_clears(self) -> None:
        text = "a" * 400
        t, _ = self.run_keys(text + "\r")
        self.assertEqual(t.input.text, "")
        self.assertEqual(t.jobs.get_nowait(), text)

    def test_alt_enter_and_ctrl_j_insert_newline_without_submitting(self) -> None:
        t, height = self.run_keys("uno" + ESC + "\rdue\ntre")  # alt+enter, ctrl+j
        self.assertEqual(t.input.text, "uno\ndue\ntre")
        self.assertEqual(height, 3)
        self.assertTrue(t.jobs.empty())

    def test_bracketed_multiline_paste_stays_in_box(self) -> None:
        t, height = self.run_keys(ESC + "[200~riga1\nriga2\r\nriga3" + ESC + "[201~")
        self.assertEqual(t.input.text.count("\n"), 2)
        self.assertEqual(height, 3)
        self.assertTrue(t.jobs.empty())

    def test_empty_enter_queues_nothing(self) -> None:
        t, _ = self.run_keys("   \r")
        self.assertTrue(t.jobs.empty())

    def test_slash_completion_still_offered(self) -> None:
        got = [c.text for c in tui._Slash().get_completions(Document("/st"), None)]
        self.assertEqual(got, ["/status"])

    def test_ctrl_c_clears_draft_then_needs_second_press_to_quit(self) -> None:
        t, _ = self.run_keys("bozza" + CTRL_C)
        self.assertEqual(t.input.text, "")
        self.assertEqual(t.quit_armed, 0.0)  # clearing a draft must not arm quitting
        t, _ = self.run_keys(CTRL_C)
        self.assertGreater(t.quit_armed, 0.0)
        self.assertIn("ctrl+c again", t._status()[0][1])

    def test_ctrl_d_with_a_draft_does_not_quit(self) -> None:
        t, _ = self.run_keys("abc" + CTRL_D)  # delete-char at end of text: a no-op, the app stays alive
        self.assertEqual(t.input.text, "abc")

    def test_escape_clears_draft(self) -> None:
        t, _ = self.run_keys("bozza" + ESC, wait=1.0)
        self.assertEqual(t.input.text, "")

    def test_end_key_moves_cursor_when_typing(self) -> None:
        t, _ = self.run_keys("abc" + ESC + "[H" + ESC + "[F")  # home, end
        self.assertEqual(t.input.buffer.cursor_position, 3)

    def test_ctrl_l_clears_output_pane(self) -> None:
        t, _ = self.run_keys(CTRL_L, before="hello\n")
        self.assertEqual(t.raw, "")

    def test_up_arrow_recalls_history(self) -> None:
        t, _ = self.run_keys("abc\r" + PAUSE + ESC + "[A")
        self.assertEqual(t.input.text, "abc")

    def test_enter_on_partial_slash_picks_suggestion_instead_of_running_it(self) -> None:
        t, _ = self.run_keys("/sta\r")
        self.assertTrue(t.jobs.empty())
        self.assertEqual(t.input.text, "/status")
        t, _ = self.run_keys("/status\r")  # a full command still submits
        self.assertEqual(t.jobs.get_nowait(), "/status")

    def test_file_completion_lists_project_files_and_skips_noise(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src" / "calc.py").write_text("x")
            (root / ".git").mkdir()
            (root / "Readme.md").write_text("x")
            c = tui._Slash(root)

            def got(text: str) -> list[str]:
                return [x.text for x in c.get_completions(Document(text), None)]

            self.assertEqual(got("fix +"), ["+src/", "+Readme.md"])  # dirs first, .git hidden
            self.assertEqual(got("fix +src/c"), ["+src/calc.py"])
            self.assertEqual(got("fix +src\\c"), ["+src/calc.py"])  # backslash typed on Windows
            self.assertEqual(got("fix +nope/"), [])
            self.assertEqual(got("plain text"), [])


@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class MouseSelectionTests(unittest.TestCase):
    """Drive Tui._mouse directly: events carry (line, col) of the output text, as Window hands them over."""

    RAW = "alpha beta\ngamma \033[31mred\033[0m delta\nlast\n"

    def setUp(self) -> None:
        from types import SimpleNamespace
        from unittest import mock
        self.t = tui.Tui(Path("."), None)
        self.t.app = SimpleNamespace(invalidate=lambda: None)
        self.t.raw = self.RAW
        self.t._formatted()
        self.copied: list[str] = []
        patcher = mock.patch.object(tui, "_copy", lambda text: self.copied.append(text) or True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ev(self, kind: str, row: int, col: int, left: bool = True) -> None:
        from types import SimpleNamespace
        from prompt_toolkit.mouse_events import MouseButton, MouseEventType
        self.t._mouse(SimpleNamespace(event_type=MouseEventType[kind], position=SimpleNamespace(y=row, x=col),
                                      button=MouseButton.LEFT if left else MouseButton.NONE))

    def drag(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        self.ev("MOUSE_DOWN", *a)
        self.ev("MOUSE_MOVE", *b)
        self.ev("MOUSE_UP", *b)

    def test_drag_across_lines_copies_exact_text(self) -> None:
        self.drag((0, 6), (1, 4))
        self.assertEqual(self.copied, ["beta\ngamma"])
        self.assertIn("copied", self.t._status()[0][1])

    def test_backwards_drag_is_the_same_selection(self) -> None:
        self.drag((1, 4), (0, 6))
        self.assertEqual(self.copied, ["beta\ngamma"])

    def test_ansi_colours_are_not_copied(self) -> None:
        self.drag((1, 0), (1, 8))
        self.assertEqual(self.copied, ["gamma red"])

    def test_highlight_reverses_only_the_selected_text(self) -> None:
        self.drag((0, 0), (0, 4))
        frags = self.t._formatted()
        self.assertEqual("".join(f[1] for f in frags), self.RAW.replace("\033[31m", "").replace("\033[0m", ""))
        self.assertEqual([f[1] for f in frags if "reverse" in f[0]], ["alpha"])

    def test_plain_click_clears_selection_and_copies_nothing(self) -> None:
        self.drag((0, 0), (0, 4))
        self.copied.clear()
        self.ev("MOUSE_DOWN", 2, 1)
        self.ev("MOUSE_UP", 2, 1)
        self.assertIsNone(self.t.sel)
        self.assertEqual(self.copied, [])

    def test_hover_without_button_does_not_select(self) -> None:
        self.ev("MOUSE_MOVE", 0, 3, left=False)
        self.assertIsNone(self.t.sel)



@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class KeyFeatureTests(InputBoxTests):
    """Claude-Code-style keys: interrupt, esc-esc undo, ctrl+r search, collapsed pastes, queue echo, status."""

    def test_esc_interrupts_a_running_task_instead_of_clearing(self) -> None:
        calls: list[int] = []

        def setup(t: tui.Tui) -> None:
            t.busy = True
            t._cancel = lambda: calls.append(1) or True  # type: ignore[method-assign]

        t, _ = self.run_keys(ESC, wait=1.0, setup=setup)
        self.assertEqual(calls, [1])
        self.assertTrue(t.jobs.empty())  # and esc must not have armed the undo gesture
        self.assertEqual(t.esc_armed, 0.0)

    def test_ctrl_c_interrupts_a_running_task_and_does_not_arm_quit(self) -> None:
        calls: list[int] = []

        def setup(t: tui.Tui) -> None:
            t._cancel = lambda: calls.append(1) or True  # type: ignore[method-assign]

        t, _ = self.run_keys("bozza" + CTRL_C + CTRL_C, setup=setup)
        self.assertEqual(calls, [1, 1])  # busy or not, _cancel decides; with a draft+busy both presses interrupt
        self.assertEqual(t.quit_armed, 0.0)

    def test_single_esc_when_idle_only_arms_the_undo_gesture(self) -> None:
        t, _ = self.run_keys(ESC, wait=1.0)
        self.assertGreater(t.esc_armed, 0.0)
        self.assertTrue(t.jobs.empty())

    def test_double_esc_when_idle_queues_undo_which_asks_before_acting(self) -> None:
        t, _ = self.run_keys(ESC + PAUSE + ESC, wait=1.0)
        self.assertEqual(t.jobs.get_nowait(), "/undo")  # router.undo then asks y/N before restoring anything
        self.assertIn("/undo", t.raw)

    def test_esc_with_a_draft_still_only_clears_the_draft(self) -> None:
        t, _ = self.run_keys("bozza" + ESC, wait=1.0)
        self.assertEqual((t.input.text, t.esc_armed), ("", 0.0))

    def test_ctrl_r_walks_back_through_matching_history(self) -> None:
        keys = ("alpha uno\r" + PAUSE + "beta due\r" + PAUSE + "alpha tre\r" + PAUSE + "alpha" + "\x12" + PAUSE)
        t, _ = self.run_keys(keys)
        self.assertEqual(t.input.text, "alpha tre")
        t, _ = self.run_keys(keys + "\x12" + PAUSE)
        self.assertEqual(t.input.text, "alpha uno")  # second press: the next older match (beta is skipped)
        t, _ = self.run_keys(keys + "\x12" + PAUSE + "\x12" + PAUSE)
        self.assertEqual(t.input.text, "alpha tre")  # wraps around

    def test_ctrl_r_without_a_match_keeps_the_text_and_says_so(self) -> None:
        t, _ = self.run_keys("zzz\x12")
        self.assertEqual(t.input.text, "zzz")
        self.assertIn("no history entry", t.notice[0])

    def test_big_paste_collapses_to_a_tag_and_is_sent_in_full(self) -> None:
        body = "\n".join(f"line {i}" for i in range(12))
        t, height = self.run_keys(ESC + "[200~" + body + ESC + "[201~", wait=0.6)
        self.assertEqual(t.input.text, "[Pasted text #1 +12 lines]")
        self.assertEqual(height, 1)
        t, _ = self.run_keys(ESC + "[200~" + body + ESC + "[201~" + " please review\r", wait=0.6)
        self.assertEqual(t.jobs.get_nowait(), body + " please review")  # the worker gets the real text
        self.assertIn("[Pasted text #1 +12 lines]", t.raw)  # the transcript keeps the short tag
        self.assertNotIn("line 7", t.raw)
        self.assertEqual(t.pastes, {})

    def test_long_single_line_paste_also_collapses(self) -> None:
        t, _ = self.run_keys(ESC + "[200~" + "x" * 1500 + ESC + "[201~")
        self.assertEqual(t.input.text, "[Pasted text #1 +1 lines]")

    def test_two_pastes_get_distinct_tags_and_both_expand(self) -> None:
        a, b = "\n".join("a" * 3 for _ in range(6)), "\n".join("b" * 3 for _ in range(7))
        keys = (ESC + "[200~" + a + ESC + "[201~ and " + ESC + "[200~" + b + ESC + "[201~\r")
        t, _ = self.run_keys(keys, wait=0.6)
        self.assertEqual(t.jobs.get_nowait(), f"{a} and {b}")

    def test_line_sent_while_busy_is_visibly_queued(self) -> None:
        t, _ = self.run_keys("next\r", setup=lambda t: setattr(t, "busy", True))
        self.assertIn("queued (#1)", t.raw)
        self.assertEqual((t.queued, t.jobs.get_nowait()), (1, "next"))

    def test_answering_a_question_is_not_queued(self) -> None:
        import queue

        def setup(t: tui.Tui) -> None:
            t.busy, t.pending = True, queue.Queue()

        t, _ = self.run_keys("y\r", setup=setup)
        self.assertEqual(t.pending.get_nowait(), "y")
        self.assertNotIn("queued", t.raw)
        self.assertTrue(t.jobs.empty())

    def test_status_bar_shows_the_live_mode_and_the_interrupt_hint(self) -> None:
        t = tui.Tui(Path("."), None, mode=lambda: "channel: agy · OFFLINE")
        self.assertIn("channel: agy · OFFLINE", t._status()[0][1])
        t.busy = True
        self.assertIn("esc to interrupt", t._status()[0][1])

    def test_new_slash_commands_complete(self) -> None:
        got = lambda text: [c.text for c in tui._Slash().get_completions(Document(text), None)]  # noqa: E731
        self.assertEqual(got("/ch"), ["/channel"])
        self.assertEqual(got("/re"), ["/review"])
        self.assertIn("/dry-run", got("/d"))
        self.assertIn("/diff", got("/d"))

    def test_enter_on_a_partial_new_command_picks_it(self) -> None:
        t, _ = self.run_keys("/chan\r")
        self.assertEqual(t.input.text, "/channel")
        self.assertTrue(t.jobs.empty())


@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class InterruptTests(unittest.TestCase):
    """_cancel for real: a thread running a step is stopped, with the child processes it started."""

    def setUp(self) -> None:
        import contextlib
        import io
        self.t = tui.Tui(Path("."), None)
        self.t.jobs = asyncio.Queue()
        self.out = io.StringIO()
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(self.out))
        self.addCleanup(stack.close)

    def start(self, step) -> "threading.Thread":  # noqa: ANN001
        import threading
        self.result: list[bool] = []
        th = threading.Thread(target=lambda: self.result.append(self.t._run_step(step, "x")), daemon=True)
        th.start()
        self.addCleanup(lambda: th.join(5))
        return th

    def wait_for(self, cond, timeout: float = 10.0) -> None:  # noqa: ANN001
        import time
        end = time.monotonic() + timeout
        while not cond():
            self.assertLess(time.monotonic(), end, "timed out")
            time.sleep(0.02)

    def test_cancel_with_nothing_running_is_a_noop(self) -> None:
        self.assertFalse(self.t._cancel())
        self.assertEqual(self.t.notice[0], "")

    def test_cancel_stops_a_python_step(self) -> None:
        import time

        def step(line: str) -> bool:
            while True:
                time.sleep(0.01)

        th = self.start(step)
        self.wait_for(lambda: self.t.tid is not None)
        self.assertTrue(self.t._cancel())
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertEqual(self.result, [False])
        self.assertIn("interrupted", self.out.getvalue())
        self.assertIn("/undo", self.out.getvalue())
        self.assertIsNone(self.t.tid)
        self.assertFalse(self.t._cancel())  # already over

    def test_cancel_kills_the_agent_process_and_returns_fast(self) -> None:
        import os
        import time

        from ai_core import ACTIVE_PROCS
        from ai_core.adapters.base import stream_process

        def step(line: str) -> bool:
            stream_process([sys.executable, "-c", "import time; time.sleep(120)"], Path("."), "", os.environ, 300,
                           lambda ln: None)
            return False

        th = self.start(step)
        self.wait_for(lambda: any(p.poll() is None for p in ACTIVE_PROCS) and self.t.tid is not None)
        procs = [p for p in ACTIVE_PROCS if p.poll() is None]
        started = time.monotonic()
        self.t._cancel()
        th.join(15)
        self.assertFalse(th.is_alive())
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(all(p.poll() is not None for p in procs), "the agent process survived the interrupt")
        self.assertIn("interrupted", self.out.getvalue())

    def test_cancel_declines_a_pending_question(self) -> None:
        def step(line: str) -> bool:
            return self.t._ask("Proceed? [y/N] ") == "y"

        th = self.start(step)
        self.wait_for(lambda: self.t.pending is not None)
        self.t._cancel()
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertEqual(self.result, [False])  # never an implicit yes
        self.assertIsNone(self.t.pending)

    def test_cancel_drops_the_queued_lines(self) -> None:
        import time

        self.t.queued = 2
        self.t.jobs.put_nowait("next one")
        self.t.jobs.put_nowait("and another")

        def step(line: str) -> bool:
            while True:
                time.sleep(0.01)

        th = self.start(step)
        self.wait_for(lambda: self.t.tid is not None)
        self.t._cancel()
        th.join(10)
        self.assertEqual((self.t.queued, self.t.jobs.qsize()), (0, 0))

    def test_a_step_that_finishes_normally_is_untouched(self) -> None:
        th = self.start(lambda line: True)
        th.join(5)
        self.assertEqual(self.result, [True])
        self.assertIsNone(self.t.tid)
        self.assertNotIn("interrupted", self.out.getvalue())

    def test_exceptions_other_than_the_interrupt_still_propagate_to_the_worker(self) -> None:
        def step(line: str) -> bool:
            raise ValueError("task crashed")

        with self.assertRaises(ValueError):
            self.t._run_step(step, "x")
        self.assertIsNone(self.t.tid)  # the thread id is released even when the step crashes


@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class WorkerTests(unittest.TestCase):
    """The real worker loop (`_main`): lines typed in the box reach `step`, long tasks ring the bell."""

    def run_main(self, keys: str, step, *, bell_after: float | None = None) -> tuple[tui.Tui, int]:  # noqa: ANN001
        from unittest import mock

        class CountingOutput(DummyOutput):
            bells = 0

            def bell(self) -> None:
                CountingOutput.bells += 1

        async def go() -> tui.Tui:
            with create_pipe_input() as pipe, create_app_session(input=pipe, output=CountingOutput()):
                t = tui.Tui(Path("."), None)
                t.app = t._build(step)
                stdout = mock.patch.object(sys, "stdout", tui._Out(t.append))  # as Tui.run does: print() feeds the pane
                stdout.start()
                self.addCleanup(stdout.stop)
                main = asyncio.create_task(t._main(step))
                await asyncio.sleep(0.3)
                for chunk in keys.split(PAUSE):
                    pipe.send_text(chunk)
                    await asyncio.sleep(0.8)  # the app and the worker thread need time between gestures
                try:
                    await asyncio.wait_for(main, 8)
                except TimeoutError:
                    t.app.exit()
                    await main
                return t

        patches = [mock.patch.object(tui, "_BELL_AFTER_S", bell_after)] if bell_after is not None else []
        for p in patches:
            p.start()
        try:
            return asyncio.run(go()), CountingOutput.bells
        finally:
            for p in patches:
                p.stop()

    def test_typed_lines_reach_step_in_order_and_exit_ends_the_app(self) -> None:
        seen: list[str] = []

        def step(line: str) -> bool:
            seen.append(line)
            return line == "/exit"

        t, _ = self.run_main("one\rtwo\r/exit\r", step)
        self.assertEqual(seen, ["one", "two", "/exit"])
        self.assertFalse(t.busy)

    def test_a_task_that_crashes_does_not_kill_the_repl(self) -> None:
        seen: list[str] = []

        def step(line: str) -> bool:
            seen.append(line)
            if line == "boom":
                raise RuntimeError("bad task")
            return line == "/exit"

        t, _ = self.run_main("boom\rnext\r/exit\r", step)
        self.assertEqual(seen, ["boom", "next", "/exit"])
        self.assertIn("✖ bad task", t.raw)

    def test_long_tasks_ring_the_bell_short_ones_do_not(self) -> None:
        _, bells = self.run_main("go\r/exit\r", lambda line: line == "/exit", bell_after=-1)
        self.assertEqual(bells, 2)  # every task counted as long
        _, bells = self.run_main("go\r/exit\r", lambda line: line == "/exit", bell_after=60)
        self.assertEqual(bells, 0)

    def test_esc_interrupts_the_running_task_and_the_repl_carries_on(self) -> None:
        import time
        seen: list[str] = []

        def step(line: str) -> bool:
            seen.append(line)
            if line == "spin":
                while True:  # an agent that never finishes by itself
                    time.sleep(0.01)
            return line == "/exit"

        t, _ = self.run_main("spin\r" + PAUSE + ESC + PAUSE + "next\r/exit\r", step)
        self.assertEqual(seen, ["spin", "next", "/exit"])
        self.assertIn("interrupted", t.raw)
        self.assertFalse(t.busy)

    def test_lines_typed_during_a_task_wait_and_esc_drops_them(self) -> None:
        import time
        seen: list[str] = []

        def step(line: str) -> bool:
            seen.append(line)
            if line == "spin":
                while True:
                    time.sleep(0.01)
            return line == "/exit"

        t, _ = self.run_main("spin\r" + PAUSE + "queued one\r" + PAUSE + ESC + PAUSE + "/exit\r", step)
        self.assertEqual(seen, ["spin", "/exit"])  # "queued one" was dropped by the interrupt
        self.assertIn("queued (#1)", t.raw)


@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class WordWrapTests(unittest.TestCase):
    """The window wraps by character; `_wrap` makes every line of output break at spaces instead."""

    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def wrap(self, line: str, width: int) -> list[str]:
        return tui._wrap(line, width)

    def widths(self, rows: list[str]) -> list[int]:
        from prompt_toolkit.utils import get_cwidth
        return [get_cwidth(self.ANSI_RE.sub("", r)) for r in rows]

    def test_short_line_is_untouched(self) -> None:
        self.assertEqual(self.wrap("   ? short", 80), ["   ? short"])
        self.assertEqual(self.wrap("", 80), [""])

    def test_breaks_at_spaces_never_inside_a_word(self) -> None:
        line = "Per i file nuovi, devo usare una sezione SEARCH/REPLACE vuota: conferma che vuoi creare il file " * 2
        rows = self.wrap(line, 40)
        self.assertTrue(all(w <= 40 for w in self.widths(rows)), self.widths(rows))
        self.assertEqual(" ".join(" ".join(r.split()) for r in rows), " ".join(line.split()))  # no word lost or split
        self.assertTrue(all("SEARCH/REPLACE" in r or "SEARCH/REPL" not in r for r in rows))

    def test_the_reported_case_keeps_search_replace_whole(self) -> None:
        line = ("   ? Il task richiede la modifica tramite SEARCH/REPLACE. Devo usare un blocco SEARCH/REPLACE con una "
                "sezione SEARCH vuota per creare il contenuto iniziale?")
        rows = self.wrap(line, 70)
        self.assertEqual(sum(r.count("SEARCH/REPLACE") for r in rows), 2)

    def test_continuation_rows_keep_the_leading_indent(self) -> None:
        rows = self.wrap("   ? " + "parola " * 30, 40)
        self.assertGreater(len(rows), 2)
        self.assertTrue(rows[0].startswith("   ? "))
        self.assertTrue(all(r.startswith("   ") and not r.startswith("    ") for r in rows[1:]), rows)

    def test_no_row_ends_with_a_space(self) -> None:
        for r in self.wrap("a " * 100, 30):
            self.assertEqual(r, r.rstrip(" "))

    def test_ansi_codes_are_free_and_stay_attached_to_their_words(self) -> None:
        line = "\x1b[1mgrassetto lungo\x1b[0m " + "normale " * 20 + "\x1b[36mcoda\x1b[0m"
        rows = self.wrap(line, 40)
        self.assertTrue(all(w <= 40 for w in self.widths(rows)))
        joined = "".join(rows)
        self.assertEqual(joined.count("\x1b[1m"), 1)
        self.assertEqual(joined.count("\x1b[0m"), 2)
        self.assertEqual(self.ANSI_RE.sub("", " ".join(rows)).split(), self.ANSI_RE.sub("", line).split())

    def test_a_word_longer_than_the_row_is_cut_to_fit(self) -> None:
        url = "https://example.com/" + "a" * 100
        rows = self.wrap("see " + url, 40)
        self.assertTrue(all(w <= 40 for w in self.widths(rows)), self.widths(rows))
        self.assertEqual("".join(rows).replace(" ", ""), ("see" + url))

    def test_a_long_word_with_colour_is_cut_without_losing_the_escape_codes(self) -> None:
        word = "\x1b[31m" + "x" * 90 + "\x1b[0m"
        rows = self.wrap(word, 30)
        self.assertTrue(all(w <= 30 for w in self.widths(rows)))
        self.assertEqual("".join(rows), word)

    def test_wide_characters_count_two_cells(self) -> None:
        rows = self.wrap("日本語 " * 20, 20)
        self.assertTrue(all(w <= 20 for w in self.widths(rows)), self.widths(rows))
        self.assertEqual("".join(rows).replace(" ", ""), "日本語" * 20)

    def test_very_narrow_window_does_not_wrap_at_all(self) -> None:
        self.assertEqual(self.wrap("word " * 20, 10), ["word " * 20])  # below 20 columns: leave it to the window

    def test_huge_indent_cannot_make_the_loop_stall(self) -> None:
        rows = self.wrap(" " * 60 + "word " * 20, 40)
        self.assertTrue(all(w <= 40 for w in self.widths(rows)), self.widths(rows))
        self.assertIn("word", " ".join(rows))

    def test_wrapped_lines_in_the_pane_follow_the_width(self) -> None:
        t = tui.Tui(Path("."), None)
        t._pane_width = lambda: 60  # type: ignore[method-assign]
        t._append("  ? " + "parola " * 40 + "\nnext line\n")
        rows = t._text().split("\n")
        self.assertGreater(len(rows), 4)
        self.assertTrue(all(len(r) <= 60 for r in rows))
        self.assertEqual(rows[-2], "next line")
        t._pane_width = lambda: 120  # type: ignore[method-assign]
        self.assertLess(len(t._text().split("\n")), len(rows))  # and it re-wraps on resize

    def test_scroll_math_counts_wrapped_rows(self) -> None:
        t = tui.Tui(Path("."), None)
        t._pane_width = lambda: 40  # type: ignore[method-assign]
        t._append("word " * 60 + "\n")
        self.assertEqual(t._cursor().y, t._text().count("\n"))
        self.assertGreater(t._cursor().y, 5)

    def test_selection_rows_are_the_wrapped_rows(self) -> None:
        copied: list[str] = []
        t = tui.Tui(Path("."), None)
        t._pane_width = lambda: 40  # type: ignore[method-assign]
        t.app = types.SimpleNamespace(invalidate=lambda: None)
        t._append("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda\n")
        t._formatted()
        with mock.patch.object(tui, "_copy", lambda text: copied.append(text) or True):
            t.sel = ((0, 0), (0, 4))
            t._copy_selection()
        self.assertEqual(copied, ["alpha"])
        self.assertEqual(t._text().split("\n")[0], "alpha beta gamma delta epsilon zeta eta")  # 39 cells, break at a space

    def test_cache_returns_the_same_rows_without_rework(self) -> None:
        tui._wrap_cached.cache_clear()
        tui._wrap_cached("word " * 40, 40)
        tui._wrap_cached("word " * 40, 40)
        info = tui._wrap_cached.cache_info()
        self.assertEqual((info.hits, info.misses), (1, 1))

    def test_a_clarification_card_wraps_by_word_in_the_pane(self) -> None:
        from ai_core import TaskStatus, router
        t = tui.Tui(Path("."), None)
        t._pane_width = lambda: 70  # type: ignore[method-assign]
        msg = ("Il task richiede di creare il file ciao2.txt. Per i file nuovi, devo usare una sezione SEARCH/REPLACE "
               "vuota. Confermi che vuoi semplicemente creare il file con il codice C all'interno?")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(router, "_use_color", return_value=False), \
                mock.patch.object(router, "IN_REPL", True):
            router._print_outcome(router.TaskOptions("q", Path(".")),
                                  router.Outcome(TaskStatus.NEEDS_CLARIFICATION, "local", "qwen", message=msg, task_id="t1"))
        t._append(buf.getvalue())
        text = t._text()
        self.assertIn("SEARCH/REPLACE", text)
        self.assertTrue(all(len(r) <= 70 for r in text.split("\n")))
        self.assertIn("type your answer below", text)
        self.assertNotIn("ai -c --task-id", text)


@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class ReflowTests(unittest.TestCase):
    """Replies are stored as markdown and laid out at display time, so a resize re-wraps them."""

    LONG = " ".join(f"parola{i}" for i in range(40))
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def setUp(self) -> None:
        from ai_core import router
        self.router = router
        self.t = tui.Tui(Path("."), None)
        self.width = 80
        self.t._pane_width = lambda: self.width  # type: ignore[method-assign]
        for patcher in (mock.patch.object(router, "MD_DEFER", self.t._defer_md),
                        mock.patch.object(router, "_safe_char", lambda char, fallback: char)):  # test console is cp1252
            patcher.start()
            self.addCleanup(patcher.stop)

    def reply(self, markdown: str) -> None:
        """What router._print_outcome prints for a reply: bullet + (deferred) body."""
        self.t._append(self.router._reply(self.router.render_markdown(markdown, True)) + "\n")

    def lines(self) -> list[str]:
        return [self.ANSI_RE.sub("", ln) for ln in self.t._text().split("\n")]

    def test_reply_is_stored_as_a_placeholder_not_as_laid_out_text(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.reply(self.LONG)
        self.assertIn("\x00MD1\x00", self.t.raw)
        self.assertNotIn("parola5", self.t.raw)
        self.assertEqual(self.t.md[1], self.LONG)

    def test_resize_rewraps_the_same_reply(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.reply(self.LONG)
        wide = self.lines()
        self.width = 50
        narrow = self.lines()
        self.assertLess(len(wide), len(narrow))
        self.assertTrue(all(len(ln) <= 50 for ln in narrow), [len(ln) for ln in narrow])
        self.width = 120
        self.assertLessEqual(len(self.lines()), len(wide))  # and it widens again
        self.assertEqual(" ".join(" ".join(ln.split()) for ln in narrow).replace("● ", "").strip(), self.LONG)  # no word lost

    def test_bullet_stays_on_the_first_line_and_the_rest_is_indented(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.reply("# Title\n\n" + self.LONG)
        body = [ln for ln in self.lines() if ln.strip()]
        self.assertTrue(body[0].startswith("● Title"))
        self.assertTrue(all(ln.startswith("  ") for ln in body[1:]))

    def test_text_around_the_reply_is_untouched(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.t._append("before\n")
            self.reply("**x**")
            self.t._append("after\n")
        self.assertEqual([ln for ln in self.lines() if ln], ["before", "● x", "after"])

    def test_two_replies_keep_their_own_source(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.reply("first reply")
            self.reply("second reply")
        self.assertEqual([ln for ln in self.lines() if ln], ["● first reply", "● second reply"])

    def test_layout_is_cached_per_width_not_redone_every_frame(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.reply(self.LONG)
        calls = []
        real = self.router.render_markdown_now
        with mock.patch.object(self.router, "render_markdown_now", lambda *a, **k: calls.append(a) or real(*a, **k)):
            for _ in range(5):
                self.t._text()
                self.t._append("more output\n")  # streaming output: raw changes, the reply must not be re-rendered
            self.width = 60
            self.t._text()
            self.t._text()
        self.assertEqual(len(calls), 2)  # once per width

    def test_scrollback_cut_never_leaves_half_a_placeholder_and_prunes_old_sources(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.reply("old reply")
        self.t._append("x" * 100 + "\n" * 5 + ("y" * 99 + "\n") * (tui._MAX_CHARS // 100))
        self.assertEqual(self.t.md, {})
        self.assertNotIn("\x00", self.t._text())
        self.assertLessEqual(len(self.t.raw), tui._MAX_CHARS)
        self.assertTrue(self.t.raw.startswith("y"))  # cut at a line boundary

    def test_a_kept_reply_survives_the_cut(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.t._append("z" * (tui._MAX_CHARS - 500) + "\n")
            self.reply("recent reply")
            self.t._append("q" * 600 + "\n")
        self.assertIn(1, self.t.md)
        self.assertIn("● recent reply", self.lines())

    def test_unknown_placeholder_is_dropped_quietly(self) -> None:
        self.t._append("a \x00MD99\x00 b\n")
        self.assertEqual(self.t._text().split("\n")[0], "a  b")

    def test_scroll_math_uses_the_laid_out_line_count(self) -> None:
        with mock.patch.object(self.router, "_use_color", return_value=True):
            self.reply("# T\n\n" + self.LONG)
        self.assertEqual(self.t._cursor().y, self.t._text().count("\n"))
        self.assertGreater(self.t._text().count("\n"), self.t.raw.count("\n"))

    def test_selection_copies_the_rendered_text(self) -> None:
        copied: list[str] = []
        with mock.patch.object(self.router, "_use_color", return_value=True), \
                mock.patch.object(tui, "_copy", lambda text: copied.append(text) or True):
            self.t.app = types.SimpleNamespace(invalidate=lambda: None)
            self.reply("# Title\n\nsome **bold** text")
            self.t._formatted()
            rows = [i for i, ln in enumerate(self.lines()) if "bold" in ln]
            self.t.sel = ((rows[0], 0), (rows[0], 30))
            self.t._copy_selection()
        self.assertEqual(copied, ["  some bold text"])  # rendered, without markers or ANSI

    def test_without_the_hook_replies_are_laid_out_immediately(self) -> None:
        with mock.patch.object(self.router, "MD_DEFER", None), mock.patch.object(self.router, "_use_color", return_value=True):
            out = self.router.render_markdown("# T\n\nbody", True)
        self.assertNotIn("\x00", out)
        self.assertIn("body", self.ANSI_RE.sub("", out))

    def test_run_installs_and_removes_the_hook(self) -> None:
        seen: list[object] = []

        def fake_run(coro) -> None:  # noqa: ANN001
            seen.append(self.router.MD_DEFER)
            coro.close()

        with mock.patch.object(self.router, "MD_DEFER", None), create_app_session(output=DummyOutput(), input=create_pipe_input().__enter__()), \
                mock.patch("asyncio.run", fake_run), mock.patch.object(self.router, "INPUT_FN", input):
            t = tui.Tui(Path("."), None)
            with contextlib.redirect_stdout(io.StringIO()):
                t.run(lambda line: False, lambda: "")
            self.assertEqual(seen, [t._defer_md])
            self.assertIsNone(self.router.MD_DEFER)

    def test_a_task_printing_a_reply_through_the_worker_ends_up_laid_out(self) -> None:
        from ai_core import TaskStatus

        async def go() -> tui.Tui:
            with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
                t = tui.Tui(Path("."), None)

                def step(line: str) -> bool:
                    self.router._print_outcome(self.router.TaskOptions("q", Path(".")), self.router.Outcome(
                        TaskStatus.COMPLETED_UNVERIFIED, message="## Plan\n\n```py\nprint(1)\n```\n\n- **one**"))
                    return line == "/exit"

                t.app = t._build(step)
                stdout = mock.patch.object(sys, "stdout", tui._Out(t.append))
                stdout.start()
                self.addCleanup(stdout.stop)
                with mock.patch.object(self.router, "MD_DEFER", t._defer_md):
                    main = asyncio.create_task(t._main(step))
                    await asyncio.sleep(0.3)
                    pipe.send_text("go\r")
                    await asyncio.sleep(1.0)
                    pipe.send_text("/exit\r")
                    await asyncio.wait_for(main, 8)
                return t

        with mock.patch.object(self.router, "_use_color", return_value=True):
            t = asyncio.run(go())
        text = self.ANSI_RE.sub("", t._text())
        self.assertIn("print(1)", text)
        self.assertIn("• one", text)
        self.assertNotIn("```", text)
        self.assertNotIn("\x00", text)


@unittest.skipUnless(tui.AVAILABLE, "prompt_toolkit missing")
class ScrollTests(unittest.TestCase):
    """End to end through the pipe: the visible top line must move on the very first tick."""

    WHEEL_UP, WHEEL_DOWN = "\x1b[<64;10;10M", "\x1b[<65;10;10M"
    PAGE_UP = "\x1b[5~"

    def scrolls(self, *keys: str, lines: int = 100, wrapped: bool = False) -> list[int]:
        """Visible top line before any key, then after each key."""
        async def go() -> list[int]:
            with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
                t = tui.Tui(Path("."), None)
                t.loop, t.jobs = asyncio.get_running_loop(), asyncio.Queue()
                t.app = t._build(lambda line: False)
                run = asyncio.create_task(t.app.run_async())
                await asyncio.sleep(0.2)
                t._append("".join(("lungo " * 30 if wrapped and i % 2 else "riga") + f" {i}\n" for i in range(lines)))
                await asyncio.sleep(0.3)
                seen = [t.output.render_info.vertical_scroll]
                for k in keys:
                    pipe.send_text(k)
                    await asyncio.sleep(0.3)
                    seen.append(t.output.render_info.vertical_scroll)
                t.app.exit()
                await run
                self.top = t.top
                return seen
        return asyncio.run(go())

    def test_first_wheel_tick_moves_the_view(self) -> None:
        start, up1, up2 = self.scrolls(self.WHEEL_UP, self.WHEEL_UP)
        self.assertEqual((start - up1, up1 - up2), (3, 3))

    def test_wheel_down_returns_to_following_the_output(self) -> None:
        start, _, back = self.scrolls(self.WHEEL_UP, self.WHEEL_DOWN)
        self.assertEqual(back, start)
        self.assertIsNone(self.top)

    def test_scrolling_stops_at_the_first_line(self) -> None:
        seen = self.scrolls(*[self.WHEEL_UP] * 60)
        self.assertEqual(seen[-1], 0)

    def test_wheel_moves_immediately_with_wrapped_lines(self) -> None:
        start, up1 = self.scrolls(self.WHEEL_UP, wrapped=True)
        self.assertLess(up1, start)

    def test_page_up_moves_half_a_screen(self) -> None:
        start, up = self.scrolls(self.PAGE_UP)
        self.assertGreaterEqual(start - up, 3)


@unittest.skipUnless(sys.platform == "win32" and tui.AVAILABLE, "Windows clipboard")
class ClipboardTests(unittest.TestCase):
    def test_real_clipboard_roundtrip_with_unicode(self) -> None:
        import ctypes
        from ctypes import wintypes
        u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
        u32.GetClipboardData.restype, k32.GlobalLock.restype = wintypes.HANDLE, ctypes.c_void_p
        u32.OpenClipboard.argtypes, k32.GlobalLock.argtypes = [wintypes.HWND], [wintypes.HANDLE]

        def read() -> str | None:
            u32.OpenClipboard(None)
            try:
                h = u32.GetClipboardData(13)
                return ctypes.wstring_at(k32.GlobalLock(h)) if h else None
            finally:
                u32.CloseClipboard()

        saved = read()  # the test borrows the user's clipboard: put their text back afterwards
        self.addCleanup(lambda: saved is not None and tui._copy(saved))
        sample = "riga " + chr(0x280B) + " " + chr(0xE8) + " " + chr(0x2714) + chr(10) + "fine"
        self.assertTrue(tui._copy(sample))
        self.assertEqual(read(), sample)

    def test_empty_text_is_not_copied(self) -> None:
        self.assertFalse(tui._copy(""))


if __name__ == "__main__":
    unittest.main()
