"""Markdown -> ANSI text for the terminal reply, via rich (headings, tables, quotes, syntax-highlighted code).

Callers go through `router.render_markdown`, which falls back to the regex renderer when colour is off or rich is
missing: nothing here may be required for a reply to print.
"""
from __future__ import annotations

import io
import re
import shutil

from rich.console import Console
from rich.markdown import Heading, Markdown
from rich.text import Text
from rich.theme import Theme

_THEME = Theme({"markdown.code": "cyan", "markdown.h1": "bold underline", "markdown.h2": "bold",
                "markdown.h3": "bold", "markdown.block_quote": "dim italic"})
_MIN_WIDTH = 40
_TRAILING = re.compile(r"[ ]+((?:\x1b\[[0-9;]*m)*)$")  # rich pads every line to the width; padding breaks copy and re-wrap


class _Heading(Heading):
    """Left-aligned, no box: chat replies are not documents (rich centres headings and boxes h1)."""

    def __rich_console__(self, console, options):  # noqa: ANN001, ANN204
        text = self.text
        text.justify = "left"
        if self.tag in ("h1", "h2"):
            yield Text("")
        yield text


class _Markdown(Markdown):
    elements = {**Markdown.elements, "heading_open": _Heading}


def render(text: str, indent: int = 2, width: int | None = None) -> str:
    """ANSI string for `text`, wrapped to `width` (default: the terminal) minus `indent` columns."""
    width = max(_MIN_WIDTH, (width or shutil.get_terminal_size((100, 24)).columns) - indent)
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="standard", width=width, theme=_THEME,
                      highlight=False, emoji=False, legacy_windows=False)
    console.print(_Markdown(text, code_theme="ansi_dark", hyperlinks=False))  # OSC 8 links are not understood by the pane
    return "\n".join(_TRAILING.sub(r"\1", ln) for ln in buf.getvalue().strip("\n").split("\n"))
