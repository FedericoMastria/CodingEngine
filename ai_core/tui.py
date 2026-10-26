"""Claude-Code-style full-screen REPL: header box pinned on top, input + status pinned at the bottom,
output scrolling in between (mouse wheel / PageUp / PageDown).

Interactive `ai >` only; --agent/--json never imports this. Falls back to plain input() if the
libraries are missing or stdin/stdout is not a TTY.

Tasks run in a worker thread; their print() output is captured into the middle pane, so you can keep
typing (lines are queued) and answer y/N questions from the same input box.
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import queue
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_core import cancel_active, router, slash

SLASH_COMMANDS = {
    "/help": "show commands", "/status": "system status", "/usage": "usage and quotas",
    "/history": "recent tasks", "/new": "fresh conversation", "/clear": "fresh conversation, clear screen",
    "/undo": "revert last checkpoint", **{f"/{n}": d for n, d in slash.COMMANDS.items()},
    "/exit": "quit",
}
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_STYLE = {"→": "90", "↗": "33", "↻": "33", "…": "90", "⚠": "33", "⛔": "31", "✖": "31", "✔": "32"}
_INPUT_ROWS = 8
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".ai_context"}
_QUIT_WINDOW = 2.0  # seconds within which a second ctrl+c quits
_REWIND_WINDOW = 1.5  # seconds within which a second esc asks to undo the last task
_BELL_AFTER_S = 15  # ring the terminal bell when a task that long finishes
_PASTE_LINES, _PASTE_CHARS = 5, 1000  # bigger pastes collapse to a [Pasted text #n] tag
_MD_TOKEN = re.compile(r"\x00MD(\d+)\x00")  # placeholder for a markdown reply, expanded at display time
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MAX_CHARS = 80_000  # shortcut: scrollback is capped, older output is dropped; raise if you need more history

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text
    from prompt_toolkit.filters import Condition, has_focus
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import (CompletionsMenu, Float, FloatContainer, FormattedTextControl, HSplit,
                                       Layout, VSplit, Window)
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.mouse_events import MouseButton, MouseEventType
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, TextArea
    AVAILABLE = True
except ImportError:  # pragma: no cover - deps declared in pyproject
    AVAILABLE = False


def _copy(text: str) -> bool:
    """Put `text` on the Windows clipboard (CF_UNICODETEXT). False if it could not."""
    if sys.platform != "win32" or not text:
        return False
    import ctypes
    from ctypes import wintypes
    k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
    k32.GlobalAlloc.restype, k32.GlobalAlloc.argtypes = wintypes.HGLOBAL, [wintypes.UINT, ctypes.c_size_t]
    k32.GlobalLock.restype, k32.GlobalLock.argtypes = ctypes.c_void_p, [wintypes.HGLOBAL]
    k32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    k32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    u32.OpenClipboard.argtypes = [wintypes.HWND]
    u32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    data = (text + "\0").encode("utf-16-le")
    handle = k32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
    if not handle:
        return False
    ctypes.memmove(k32.GlobalLock(handle), data, len(data))
    k32.GlobalUnlock(handle)
    for _ in range(10):  # another process may hold the clipboard for a moment
        if u32.OpenClipboard(None):
            break
        time.sleep(0.02)
    else:
        k32.GlobalFree(handle)
        return False
    try:
        u32.EmptyClipboard()
        if u32.SetClipboardData(13, handle):  # CF_UNICODETEXT; the system owns the memory from here
            return True
        k32.GlobalFree(handle)
        return False
    finally:
        u32.CloseClipboard()


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
        """`/command` completion for the whole input, `+path` completion for the word under the cursor."""

        def __init__(self, root: Path | None = None) -> None:
            self.root = root

        def get_completions(self, document: Any, complete_event: Any):  # noqa: ANN201
            text = document.text_before_cursor
            if text.startswith("/") and " " not in text:
                for cmd, desc in SLASH_COMMANDS.items():
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text), display_meta=desc)
                return
            word = document.get_word_before_cursor(WORD=True)
            if self.root is None or not word.startswith("+"):
                return
            head, _, prefix = word[1:].replace("\\", "/").rpartition("/")
            base = self.root / head if head else self.root
            try:
                entries = sorted(os.scandir(base), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return
            shown = 0
            for e in entries:
                if e.name in _SKIP_DIRS or not e.name.lower().startswith(prefix.lower()):
                    continue
                isdir = e.is_dir()
                yield Completion(f"+{head + '/' if head else ''}{e.name}{'/' if isdir else ''}",
                                 start_position=-len(word), display_meta="dir" if isdir else "")
                shown += 1
                if shown >= 50:
                    return


class Tui:
    def __init__(self, root: Path, channel: str | None, mode: Callable[[], str] | None = None) -> None:
        self.root = root
        self.channel = channel or "auto"
        self.mode = mode or (lambda: f"channel: {self.channel}")  # status-bar label, re-read at every render
        self.busy = False
        self.queued = 0
        self.tick = 0
        self.last_model = ""
        self.raw = ""
        self.top: int | None = None  # first visible line while the user scrolled away; None = follow the newest output
        self.output: Any = None  # the output Window (its render_info says what is on screen)
        self._parsed: Any = None
        self._parsed_src: str | None = None  # the expanded text `_parsed` was built from
        self.md: dict[int, str] = {}  # placeholder number -> markdown source of a reply
        self._md_n = 0
        self._md_cache: dict[tuple[int, int], str] = {}  # (placeholder, width) -> rendered ANSI
        self._exp: tuple[str, int, str] | None = None  # (raw it was built from, width, expanded text)
        self.pending: queue.Queue[str] | None = None  # set while a task waits for a y/N answer
        self.answered_by: Callable[[], str] = lambda: ""
        self.loop: asyncio.AbstractEventLoop | None = None
        self.app: Any = None
        self.jobs: asyncio.Queue[str] | None = None
        self.busy_since = 0.0
        self.quit_armed = 0.0  # monotonic time of the last ctrl+c that only armed quitting
        self.anchor: tuple[int, int] | None = None  # (line, col) where the mouse button went down
        self.sel: tuple[tuple[int, int], tuple[int, int]] | None = None  # normalised, end inclusive
        self.notice = ("", 0.0)  # transient status text and its monotonic expiry
        self.lock = threading.Lock()
        self.tid: int | None = None  # thread running the current task (set only while a step runs)
        self.esc_armed = 0.0  # monotonic time of the first esc of a possible esc-esc (undo)
        self.pastes: dict[str, str] = {}  # collapsed paste tag -> its full text
        self.search: tuple[str, int, str] | None = None  # ctrl+r: (query, match index, text it put in the box)

    # ------------------------------------------------------------ output pane
    def append(self, text: str) -> None:
        """Thread-safe: print() from the worker thread lands here."""
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._append, text)

    def _append(self, text: str) -> None:
        raw = self.raw + text
        if len(raw) > _MAX_CHARS:
            raw = raw[-_MAX_CHARS:]
            raw = raw[raw.find("\n") + 1:]  # never keep half a line: a half placeholder would show as control characters
            self.md = {n: src for n, src in self.md.items() if f"\x00MD{n}\x00" in raw}
            self._md_cache = {k: v for k, v in self._md_cache.items() if k[0] in self.md}
        self.raw = raw
        if self.app:
            self.app.invalidate()

    def _defer_md(self, source: str) -> str:
        """router.MD_DEFER: keep the markdown and print a placeholder line; thread-safe (called by the worker)."""
        n = self._md_n = self._md_n + 1
        self.md[n] = source
        return f"\x00MD{n}\x00"

    def _pane_width(self) -> int:
        info = self.output.render_info if self.output else None
        return info.window_width if info else shutil.get_terminal_size((100, 24)).columns

    def _text(self) -> str:
        """The output as displayed: `raw` with every markdown placeholder laid out for the current pane width.
        What precedes the placeholder on its line (the reply bullet) becomes the indent of the continuation lines."""
        width = self._pane_width()
        if self._exp and self._exp[0] is self.raw and self._exp[1] == width:
            return self._exp[2]
        if "\x00" not in self.raw:
            text = self.raw
        else:
            out: list[str] = []
            for line in self.raw.split("\n"):
                m = _MD_TOKEN.search(line)
                if not m:
                    out.append(line)
                    continue
                prefix, n = line[:m.start()], int(m.group(1))
                indent = len(_ANSI.sub("", prefix))
                key = (n, width - indent)
                if n in self.md and key not in self._md_cache:
                    self._md_cache[key] = router.render_markdown_now(self.md[n], True, width - indent)
                rendered = self._md_cache.get(key, "").split("\n")
                out.append(prefix + rendered[0])
                out.extend(" " * indent + ln if ln else ln for ln in rendered[1:])
                out[-1] += line[m.end():]
            text = "\n".join(out)
        self._exp = (self.raw, width, text)
        return text

    def _formatted(self) -> Any:
        text = self._text()
        if self._parsed_src is not text or self._parsed is None:
            self._parsed, self._parsed_src = ANSI(text), text
        if not self.sel:
            return self._parsed
        (r0, c0), (r1, c1) = self.sel
        out: list[tuple[str, str]] = []
        row = col = 0
        for style, text, *_ in to_formatted_text(self._parsed):
            for ch in text:
                hit = (r0, c0) <= (row, col) <= (r1, c1) and ch != "\n"
                st = style + " reverse" if hit else style
                if out and out[-1][0] == st:
                    out[-1] = (st, out[-1][1] + ch)
                else:
                    out.append((st, ch))
                row, col = (row + 1, 0) if ch == "\n" else (row, col + 1)
        return out

    def _selected_text(self) -> str:
        if not self.sel:
            return ""
        (r0, c0), (r1, c1) = self.sel
        plain = "".join(f[1] for f in to_formatted_text(self._parsed)).split("\n")
        lines = []
        for row in range(r0, min(r1, len(plain) - 1) + 1):
            lo = c0 if row == r0 else 0
            hi = c1 + 1 if row == r1 else len(plain[row])
            lines.append(plain[row][lo:hi].rstrip())
        return "\n".join(lines)

    def _copy_selection(self) -> None:
        text = self._selected_text()
        if text:
            ok = _copy(text)
            self.notice = (f"copied {len(text)} chars" if ok else "copy failed", time.monotonic() + 2)

    def _mouse(self, event: Any) -> None:
        kind, pos = event.event_type, (event.position.y, event.position.x)
        if kind == MouseEventType.SCROLL_UP:
            self._scroll(3)
        elif kind == MouseEventType.SCROLL_DOWN:
            self._scroll(-3)
        elif kind == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
            self.anchor, self.sel = pos, None
        elif kind == MouseEventType.MOUSE_MOVE and event.button == MouseButton.LEFT and self.anchor:
            self.sel = (min(self.anchor, pos), max(self.anchor, pos))
        elif kind == MouseEventType.MOUSE_UP and self.anchor:
            if self.sel and self.sel[0] != self.sel[1]:
                self._copy_selection()
            else:
                self.sel = None  # a plain click only clears
            self.anchor = None
        else:
            return NotImplemented
        self.app.invalidate()

    def _cursor(self) -> Point:
        last = self._text().count("\n")
        return Point(0, last if self.top is None else min(self.top, last))

    def _scroll(self, delta: int) -> None:
        """Move the viewport `delta` lines up (negative = down). Sets the window's scroll directly: moving only the
        invisible cursor would not scroll until it left the screen, a whole page of wheel ticks later."""
        info = self.output.render_info if self.output else None
        lines = self._text().count("\n") + 1
        if delta < 0 and info and info.displayed_lines[-1] - delta >= lines - 1:  # this move reaches the bottom: follow again
            self.top = None
        else:
            now = self.top if self.top is not None else (info.vertical_scroll if info else max(0, lines - 20))
            self.top = max(0, min(lines - 1, now - delta))
            if self.output:  # set directly: a wrapping Window ignores get_vertical_scroll
                self.output.vertical_scroll, self.output.vertical_scroll_2 = self.top, 0
        self.app.invalidate()

    # ------------------------------------------------------------ interrupt
    def _run_step(self, step: Callable[[str], bool], line: str) -> bool:
        """Worker-thread body: records its thread id so `_cancel` can interrupt exactly this task."""
        try:
            with self.lock:
                self.tid = threading.get_ident()
            try:
                return step(line)
            finally:
                with self.lock:
                    self.tid = None
        except KeyboardInterrupt:  # raised by _cancel; also covers the instant between the step ending and tid clearing
            print("✖ interrupted — files already changed stay as they are; /undo restores the last checkpoint")
            return False

    def _cancel(self) -> bool:
        """Stop the running task like ctrl+c would: kill its child processes, raise KeyboardInterrupt in its thread,
        decline a pending y/N and drop the queued lines. False if nothing was running."""
        with self.lock:
            if self.tid is None:
                return False
            if self.pending is not None:
                self.pending.put("")
            while self.jobs is not None and not self.jobs.empty():
                self.jobs.get_nowait()
                self.queued -= 1
            cancel_active()  # unblocks a thread parked on a pipe read, so the exception below is delivered at once
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self.tid), ctypes.py_object(KeyboardInterrupt))
        self.notice = ("interrupting…", time.monotonic() + 3)
        return True

    # ------------------------------------------------------------ fixed chrome
    def _status(self) -> list[tuple[str, str]]:
        if time.monotonic() < self.notice[1]:
            state = self.notice[0]
        elif self.pending is not None:
            state = "waiting for your answer above"
        elif self.busy:
            self.tick += 1
            q = f" · {self.queued} queued" if self.queued else ""
            state = f"{_SPIN[self.tick % len(_SPIN)]} working… {int(time.monotonic() - self.busy_since)}s · esc to interrupt{q}"
        elif time.monotonic() - self.quit_armed < _QUIT_WINDOW:
            state = "press ctrl+c again to exit"
        else:
            last = f" · last: {self.last_model}" if self.last_model else ""
            state = f"/ commands · + files · alt+enter newline · esc esc undo · ctrl+d exit{last}"
        scrolled = " · ↑ scrolled (PageDown / End to follow)" if self.top is not None else ""
        return [("class:status", f" {self.root.name} · {self.mode()} · {state}{scrolled}")]

    def _header(self) -> list[tuple[str, str]]:
        return [("bold fg:ansicyan", " CodingEngine"), ("class:dim", f"  {self.root}\n"),
                ("", " Type a task, or / for commands. Use +path/file to attach files.")]

    def _input_height(self) -> Any:
        info = self.input.window.render_info if hasattr(self, "input") else None
        width = max(1, info.window_width if info else 78)
        rows = sum(max(1, -(-len(ln) // width)) for ln in self.input.text.split("\n"))
        rows = min(rows, _INPUT_ROWS)
        return Dimension(min=rows, max=rows)  # exact: HSplit would otherwise hand the box spare rows

    def _build(self, step: Callable[[str], bool]) -> Any:
        def submit(line: str) -> None:
            """Echo `line` (collapsed paste tags stay short) and hand it, tags expanded, to the worker or a y/N."""
            self.sel = None
            if line == "/clear":
                self.raw = ""
            self._append(f"\033[36m❯\033[0m \033[1m{line.replace(chr(10), chr(10) + '  ')}\033[0m\n")
            self.top = None
            for tag, text in self.pastes.items():
                line = line.replace(tag, text)
            self.pastes.clear()
            if self.pending is not None:
                self.pending.put(line)
            else:
                if self.busy:
                    self._append(f"\033[90m  ⏳ queued (#{self.queued + 1}), runs after the current task; esc drops the queue\033[0m\n")
                self.queued += 1
                assert self.jobs is not None
                self.jobs.put_nowait(line)

        self._submit = submit

        def accept(buf: Any) -> bool:
            if line := buf.text.strip():
                submit(line)
            return False  # clear the input box

        hist = self.root / ".ai_context" / "repl_history"
        try:
            hist.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(hist))
        except OSError:
            history = None
        # wrapped, grows up to _INPUT_ROWS rows then scrolls: long text must stay visible, not slide off one row
        self.input = TextArea(height=self._input_height, multiline=True, wrap_lines=True, prompt="",
                              completer=_Slash(self.root), complete_while_typing=True, history=history,
                              accept_handler=accept, style="class:input")
        pane = FormattedTextControl(self._formatted, get_cursor_position=self._cursor, show_cursor=False,
                                    focusable=False)
        pane.mouse_handler = self._mouse  # type: ignore[method-assign]
        output = self.output = Window(pane, wrap_lines=True, always_hide_cursor=True)
        def rule() -> Any:  # one Window per use: prompt_toolkit containers must not be shared in a layout
            return Window(height=1, char="─", style="class:rule")

        body = HSplit([
            Frame(Window(FormattedTextControl(self._header), height=2), style="class:box"),
            output,
            rule(),
            VSplit([Window(FormattedTextControl([("class:prompt", "❯ ")]), width=2, dont_extend_width=True),
                    self.input]),
            rule(),
            Window(FormattedTextControl(self._status), height=1),
        ])
        kb = KeyBindings()

        empty = Condition(lambda: not self.input.text)

        @kb.add("c-d", filter=empty)
        def _quit(event: Any) -> None:
            event.app.exit()

        @kb.add("c-c")
        def _interrupt(event: Any) -> None:
            if self.sel:  # selection finished outside the pane (no mouse-up reached us): copy it now
                self._copy_selection()
                self.sel = self.anchor = None
            elif self._cancel():
                pass
            elif self.input.text:  # like claude code: first ctrl+c clears the draft
                self.input.buffer.reset()
            elif time.monotonic() - self.quit_armed < _QUIT_WINDOW:
                event.app.exit()
            else:
                self.quit_armed = time.monotonic()

        @kb.add("escape", filter=has_focus(self.input) & ~empty)
        def _clear_draft(event: Any) -> None:
            event.current_buffer.reset()

        @kb.add("escape", filter=has_focus(self.input) & empty)
        def _esc(event: Any) -> None:
            now = time.monotonic()
            if self.sel:
                self.sel = self.anchor = None
            elif self._cancel():
                pass
            elif now - self.esc_armed < _REWIND_WINDOW:  # esc esc: undo the last task (asks y/N first)
                self.esc_armed = 0.0
                submit("/undo")
            else:
                self.esc_armed = now
                self.notice = ("press esc again to undo the last task", now + _REWIND_WINDOW)

        @kb.add("c-r", filter=has_focus(self.input))
        def _history_search(event: Any) -> None:
            """Older history entry containing the text in the box; press again for the next older one."""
            buf = event.current_buffer
            query, idx = (self.search[0], self.search[1] + 1) if self.search and buf.text == self.search[2] else (buf.text, 0)
            seen: list[str] = []
            for entry in reversed(list(buf.history.get_strings())):
                if query.lower() in entry.lower() and entry not in seen:
                    seen.append(entry)
            if not seen:
                self.notice = (f"no history entry contains {query!r}", time.monotonic() + 2)
                return
            idx %= len(seen)
            buf.text = seen[idx]
            buf.cursor_position = len(buf.text)
            self.search = (query, idx, seen[idx])
            self.notice = (f"history {idx + 1}/{len(seen)} for {query!r} · ctrl+r for older", time.monotonic() + 3)

        @kb.add(Keys.BracketedPaste, filter=has_focus(self.input))
        def _paste(event: Any) -> None:
            data = event.data.replace("\r\n", "\n").replace("\r", "\n")
            if data.count("\n") + 1 >= _PASTE_LINES or len(data) > _PASTE_CHARS:  # keep the box readable
                tag = f"[Pasted text #{len(self.pastes) + 1} +{data.count(chr(10)) + 1} lines]"
                self.pastes[tag] = data
                data = tag
            event.current_buffer.insert_text(data)

        @kb.add("c-l")
        def _clear_screen(event: Any) -> None:
            self.raw, self.top, self.sel = "", None, None

        @kb.add("enter", filter=has_focus(self.input))
        def _submit(event: Any) -> None:
            buf, text = event.current_buffer, event.current_buffer.text
            state = buf.complete_state
            hits = [c for c in SLASH_COMMANDS if c.startswith(text)] if text.startswith("/") and " " not in text else []
            if hits and text not in SLASH_COMMANDS:  # enter on a partial /command picks the suggestion, never runs it
                picked = state.current_completion.text if state and state.current_completion else hits[0]
                buf.text = picked
                buf.cursor_position = len(picked)
            else:
                buf.validate_and_handle()

        @kb.add("escape", "enter", filter=has_focus(self.input))  # alt+enter
        @kb.add("c-j", filter=has_focus(self.input))
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("pageup")
        def _up(event: Any) -> None:
            self._scroll(max(3, event.app.output.get_size().rows // 2))

        @kb.add("pagedown")
        def _down(event: Any) -> None:
            self._scroll(-max(3, event.app.output.get_size().rows // 2))

        @kb.add("end", filter=empty)  # with a draft, End must still move the text cursor
        def _end(event: Any) -> None:
            self.top = None

        style = Style.from_dict({"box": "fg:ansicyan", "rule": "fg:#555555", "status": "fg:#888888",
                                 "dim": "fg:#888888", "prompt": "bold fg:ansicyan"})
        layout = Layout(FloatContainer(body, [Float(xcursor=True, ycursor=True,
                                                    content=CompletionsMenu(max_height=8))]),
                        focused_element=self.input)
        app = Application(layout=layout, key_bindings=kb, style=style, full_screen=True, mouse_support=True,
                          refresh_interval=0.12)
        app.timeoutlen = 0.4  # esc alone waits this long for a possible alt+enter
        app.ttimeoutlen = 0.1  # terminals send an escape sequence in one burst: do not wait the default 0.5 s for more
        return app

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
        self.answered_by = answered_by
        router.INPUT_FN = self._ask
        router.MD_DEFER = self._defer_md
        real_out, real_err = sys.stdout, sys.stderr
        self.app = self._build(step)  # before the swap: prompt_toolkit binds its output to the real stdout
        sys.stdout, sys.stderr = _Out(self.append), _Out(self.append, style_lines=True)
        try:
            asyncio.run(self._main(step))
        finally:
            sys.stdout, sys.stderr = real_out, real_err
            router.INPUT_FN = input
            router.MD_DEFER = None
        print("bye")
        return 0

    async def _main(self, step: Callable[[str], bool]) -> None:
        self.loop = asyncio.get_running_loop()
        self.jobs = asyncio.Queue()
        self._append("\033[90mWheel / PageUp / PageDown scroll the output; drag with the mouse to select and copy.\033[0m\n")

        async def worker() -> None:
            assert self.jobs is not None
            while True:
                line = await self.jobs.get()
                self.queued -= 1
                self.busy, self.busy_since = True, time.monotonic()
                try:
                    done = await self.loop.run_in_executor(None, self._run_step, step, line)  # type: ignore[union-attr]
                except Exception as exc:  # keep the REPL alive on a task crash
                    print(f"✖ {exc}")
                    done = False
                self.busy = False
                if time.monotonic() - self.busy_since > _BELL_AFTER_S:  # long task: tell the user it is done
                    self.app.output.bell()
                    self.app.output.flush()
                self.last_model = self.answered_by() or self.last_model
                if done:
                    self.app.exit()
                    return

        task = asyncio.create_task(worker())
        try:
            await self.app.run_async()
        finally:
            self._cancel()  # quitting mid-task must not leave an agent editing files behind the user's back
            if self.pending is not None:
                self.pending.put("")
            task.cancel()
