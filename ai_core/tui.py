"""Claude-Code-style full-screen REPL: header box pinned on top, input + status pinned at the bottom,
output scrolling in between (mouse wheel / PageUp / PageDown).

Interactive `ai >` only; --agent/--json never imports this. Falls back to plain input() if the
libraries are missing or stdin/stdout is not a TTY.

Tasks run in a worker thread; their print() output is captured into the middle pane, so you can keep
typing (lines are queued) and answer y/N questions from the same input box.
"""
from __future__ import annotations

import asyncio
import queue
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

SLASH_COMMANDS = {
    "/help": "show commands", "/status": "system status", "/usage": "usage and quotas",
    "/history": "recent tasks", "/new": "fresh conversation", "/undo": "revert last checkpoint",
    "/exit": "quit",
}
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_STYLE = {"→": "90", "↗": "33", "↻": "33", "…": "90", "⚠": "33", "⛔": "31", "✖": "31", "✔": "32"}
_MAX_CHARS = 80_000  # shortcut: scrollback is capped, older output is dropped; raise if you need more history

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import (CompletionsMenu, Float, FloatContainer, FormattedTextControl, HSplit,
                                       Layout, VSplit, Window)
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, TextArea
    AVAILABLE = True
except ImportError:  # pragma: no cover - deps declared in pyproject
    AVAILABLE = False


def enabled() -> bool:
    return AVAILABLE and sys.stdin.isatty() and sys.stdout.isatty()


class _Out:
    """File-like sink for print(): feeds the output pane. Line-buffers the colorization of stderr."""

    encoding = "utf-8"

    def __init__(self, sink: Callable[[str], None], style_lines: bool = False) -> None:
        self.sink, self.style_lines, self.buf = sink, style_lines, ""

    def write(self, s: str) -> int:
        s = s.replace("\r", "")
        if not self.style_lines:
            self.sink(s)
            return len(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            code = _STYLE.get(line.lstrip()[:1])
            self.sink(f"\033[{code}m  {line.strip()}\033[0m\n" if code else line + "\n")
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True  # keeps the router's ANSI colors on

    def reconfigure(self, **_: Any) -> None:
        pass


if AVAILABLE:
    class _Slash(Completer):
        def get_completions(self, document: Any, complete_event: Any):  # noqa: ANN201
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:
                return
            for cmd, desc in SLASH_COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=desc)


class Tui:
    def __init__(self, root: Path, channel: str | None) -> None:
        self.root = root
        self.channel = channel or "auto"
        self.busy = False
        self.queued = 0
        self.tick = 0
        self.last_model = ""
        self.raw = ""
        self.scroll_back = 0  # lines above the bottom; 0 = follow the newest output
        self._parsed: Any = None
        self._parsed_len = -1
        self.pending: queue.Queue[str] | None = None  # set while a task waits for a y/N answer
        self.answered_by: Callable[[], str] = lambda: ""
        self.loop: asyncio.AbstractEventLoop | None = None
        self.app: Any = None
        self.jobs: asyncio.Queue[str] | None = None

    # ------------------------------------------------------------ output pane
    def append(self, text: str) -> None:
        """Thread-safe: print() from the worker thread lands here."""
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._append, text)

    def _append(self, text: str) -> None:
        if self.scroll_back:  # reading history: keep the view where it is
            self.scroll_back += text.count("\n")
        self.raw = (self.raw + text)[-_MAX_CHARS:]
        if self.app:
            self.app.invalidate()

    def _formatted(self) -> Any:
        if self._parsed_len != len(self.raw) or self._parsed is None:
            self._parsed, self._parsed_len = ANSI(self.raw), len(self.raw)
        return self._parsed

    def _cursor(self) -> Point:
        last = self.raw.count("\n")
        return Point(0, max(0, last - self.scroll_back))

    def _scroll(self, delta: int) -> None:
        self.scroll_back = max(0, min(self.raw.count("\n"), self.scroll_back + delta))
        self.app.invalidate()

    def _mouse(self, event: Any) -> None:
        if event.event_type == MouseEventType.SCROLL_UP:
            self._scroll(3)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll(-3)
        else:
            return NotImplemented

    # ------------------------------------------------------------ fixed chrome
    def _status(self) -> list[tuple[str, str]]:
        if self.pending is not None:
            state = "waiting for your answer above"
        elif self.busy:
            self.tick += 1
            q = f" · {self.queued} queued" if self.queued else ""
            state = f"{_SPIN[self.tick % len(_SPIN)]} working…{q}"
        else:
            state = f"/ for commands · ctrl+d to exit{' · last: ' + self.last_model if self.last_model else ''}"
        scrolled = " · ↑ scrolled (PageDown / End to follow)" if self.scroll_back else ""
        return [("class:status", f" {self.root.name} · channel: {self.channel} · {state}{scrolled}")]

    def _header(self) -> list[tuple[str, str]]:
        return [("bold fg:ansicyan", " CodingEngine"), ("class:dim", f"  {self.root}\n"),
                ("", " Type a task, or / for commands. Use +path/file to attach files.")]

    def _build(self, step: Callable[[str], bool]) -> Any:
        def accept(buf: Any) -> bool:
            line = buf.text.strip()
            if line:
                self._append(f"\033[36m❯\033[0m \033[1m{line}\033[0m\n")
                self.scroll_back = 0
                if self.pending is not None:
                    self.pending.put(line)
                else:
                    self.queued += 1
                    assert self.jobs is not None
                    self.jobs.put_nowait(line)
            return False  # clear the input box

        hist = self.root / ".ai_context" / "repl_history"
        try:
            hist.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(hist))
        except OSError:
            history = None
        self.input = TextArea(height=1, multiline=False, prompt="", completer=_Slash(), complete_while_typing=True,
                              history=history, accept_handler=accept, style="class:input")
        pane = FormattedTextControl(self._formatted, get_cursor_position=self._cursor, show_cursor=False,
                                    focusable=False)
        pane.mouse_handler = self._mouse  # type: ignore[method-assign]
        output = Window(pane, wrap_lines=True, always_hide_cursor=True)
        def rule() -> Any:  # one Window per use: prompt_toolkit containers must not be shared in a layout
            return Window(height=1, char="─", style="class:rule")

        body = HSplit([
            Frame(Window(FormattedTextControl(self._header), height=2), style="class:box"),
            output,
            rule(),
            VSplit([Window(FormattedTextControl([("class:prompt", "❯ ")]), width=2, dont_extend_width=True),
                    self.input], height=1),
            rule(),
            Window(FormattedTextControl(self._status), height=1),
        ])
        kb = KeyBindings()

        @kb.add("c-d")
        @kb.add("c-c")
        def _quit(event: Any) -> None:
            event.app.exit()

        @kb.add("pageup")
        def _up(event: Any) -> None:
            self._scroll(max(3, event.app.output.get_size().rows // 2))

        @kb.add("pagedown")
        def _down(event: Any) -> None:
            self._scroll(-max(3, event.app.output.get_size().rows // 2))

        @kb.add("end")
        def _end(event: Any) -> None:
            self.scroll_back = 0

        style = Style.from_dict({"box": "fg:ansicyan", "rule": "fg:#555555", "status": "fg:#888888",
                                 "dim": "fg:#888888", "prompt": "bold fg:ansicyan"})
        layout = Layout(FloatContainer(body, [Float(xcursor=True, ycursor=True,
                                                    content=CompletionsMenu(max_height=8))]),
                        focused_element=self.input)
        return Application(layout=layout, key_bindings=kb, style=style, full_screen=True, mouse_support=True,
                           refresh_interval=0.12)

    # ------------------------------------------------------------ run
    def _ask(self, prompt: str) -> str:
        """Replaces router.INPUT_FN: print the question, take the next line typed in the box."""
        print(prompt, end="", flush=True)
        self.pending = queue.Queue()
        try:
            return self.pending.get()
        finally:
            self.pending = None

    def run(self, step: Callable[[str], bool], answered_by: Callable[[], str]) -> int:
        from ai_core import router
        self.answered_by = answered_by
        router.INPUT_FN = self._ask
        real_out, real_err = sys.stdout, sys.stderr
        self.app = self._build(step)  # before the swap: prompt_toolkit binds its output to the real stdout
        sys.stdout, sys.stderr = _Out(self.append), _Out(self.append, style_lines=True)
        try:
            asyncio.run(self._main(step))
        finally:
            sys.stdout, sys.stderr = real_out, real_err
            router.INPUT_FN = input
        print("bye")
        return 0

    async def _main(self, step: Callable[[str], bool]) -> None:
        self.loop = asyncio.get_running_loop()
        self.jobs = asyncio.Queue()
        self._append("\033[90mPageUp/PageDown or mouse wheel scroll the output.\033[0m\n")

        async def worker() -> None:
            assert self.jobs is not None
            while True:
                line = await self.jobs.get()
                self.queued -= 1
                self.busy = True
                try:
                    done = await self.loop.run_in_executor(None, step, line)  # type: ignore[union-attr]
                except Exception as exc:  # keep the REPL alive on a task crash
                    print(f"✖ {exc}")
                    done = False
                self.busy = False
                self.last_model = self.answered_by() or self.last_model
                if done:
                    self.app.exit()
                    return

        task = asyncio.create_task(worker())
        try:
            await self.app.run_async()
        finally:
            if self.pending is not None:
                self.pending.put("")
            task.cancel()
