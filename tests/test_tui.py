"""TUI input-box tests (stdlib unittest, headless via a prompt_toolkit pipe).

Run: uv run --no-project --python ">=3.14" python -m unittest tests.test_tui -v
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

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
