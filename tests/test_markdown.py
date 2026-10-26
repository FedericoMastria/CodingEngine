"""Markdown reply rendering (stdlib unittest).

Run: uv run --no-project --python ">=3.14" python -m unittest tests.test_markdown -v
"""
from __future__ import annotations

import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_core import TaskStatus, mdrender, router

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return ANSI_RE.sub("", text)


class RenderTests(unittest.TestCase):
    def test_headings_are_left_aligned_unboxed_and_lose_their_hashes(self) -> None:
        out = mdrender.render("# Big\n\n## Small\n\n### Tiny\n\ntext")
        lines = [ln for ln in plain(out).split("\n") if ln.strip()]
        self.assertEqual(lines, ["Big", "Small", "Tiny", "text"])  # no '#', no box, no centring
        self.assertIn("\x1b[1;4mBig", out)  # h1 bold + underline
        self.assertIn("\x1b[1mSmall", out)

    def test_inline_styles(self) -> None:
        out = mdrender.render("**bold** *it* `code`")
        self.assertIn("\x1b[1mbold\x1b[0m", out)
        self.assertIn("\x1b[3mit\x1b[0m", out)
        self.assertIn("\x1b[36mcode\x1b[0m", out)
        self.assertEqual(plain(out), "bold it code")

    def test_code_block_is_highlighted_and_keeps_its_text_and_indentation(self) -> None:
        out = mdrender.render("```python\ndef f(x):\n    return x + 1  # ok\n```")
        self.assertNotIn("```", out)
        self.assertEqual([ln for ln in plain(out).split("\n") if ln.strip()],
                         [" def f(x):", "     return x + 1  # ok"])
        self.assertIn("\x1b[94mdef\x1b[0m", out)  # keyword coloured
        self.assertIn("\x1b[94mreturn\x1b[0m", out)

    def test_code_block_without_a_language_is_kept_verbatim(self) -> None:
        out = mdrender.render("```\n**not bold** `x`\n```")
        self.assertIn("**not bold** `x`", plain(out))

    def test_table(self) -> None:
        out = plain(mdrender.render("| name | n |\n|---|---|\n| alpha | 1 |\n| beta | 22 |"))
        rows = [ln.split() for ln in out.split("\n") if ln.strip() and not set(ln.strip()) <= {"─"}]
        self.assertEqual(rows, [["name", "n"], ["alpha", "1"], ["beta", "22"]])

    def test_blockquote_and_lists(self) -> None:
        out = plain(mdrender.render("> quoted\n\n- one\n- two\n  - nested\n\n1. first\n2. second"))
        lines = [ln.strip() for ln in out.split("\n") if ln.strip()]
        self.assertEqual(lines[0], "▌ quoted")
        self.assertEqual(lines[1:4], ["• one", "• two", "• nested"])
        self.assertEqual(lines[4:], ["1 first", "2 second"])  # rich numbers items without a dot

    def test_links_show_their_target_without_osc8(self) -> None:
        out = mdrender.render("[docs](https://example.com/x)")
        self.assertNotIn("\x1b]8", out)  # the TUI pane would print OSC 8 as garbage
        self.assertEqual(plain(out), "docs (https://example.com/x)")

    def test_square_brackets_are_not_rich_markup(self) -> None:
        self.assertEqual(plain(mdrender.render("[bold]x[/bold] and [red]y")), "[bold]x[/bold] and [red]y")

    def test_no_trailing_padding_on_any_line(self) -> None:
        out = mdrender.render("# T\n\npara text here\n\n```py\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n> q")
        for ln in plain(out).split("\n"):
            self.assertEqual(ln, ln.rstrip(), repr(ln))

    def test_output_has_no_leading_or_trailing_blank_lines(self) -> None:
        out = mdrender.render("# Title\n\nbody\n")
        self.assertFalse(out.startswith("\n") or out.endswith("\n"))

    def test_wraps_to_the_terminal_width_minus_the_reply_indent(self) -> None:
        import os
        text = " ".join(["parola"] * 60)
        with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((60, 24))):
            lines = plain(mdrender.render(text)).split("\n")
        self.assertGreater(len(lines), 3)
        self.assertTrue(all(len(ln) <= 58 for ln in lines), [len(ln) for ln in lines])

    def test_tiny_terminal_still_renders(self) -> None:
        import os
        with mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((10, 5))):
            self.assertIn("hello", plain(mdrender.render("hello")))

    def test_output_parses_cleanly_in_the_tui_pane(self) -> None:
        from prompt_toolkit.formatted_text import ANSI, to_formatted_text
        src = "# T\n\n**b** `c` [l](http://x.y)\n\n- a\n\n> q\n\n```py\ndef f(): pass\n```\n\n| a |\n|---|\n| 1 |"
        out = mdrender.render(src)
        pane = "".join(f[1] for f in to_formatted_text(ANSI(out)))
        self.assertEqual(pane, plain(out))  # no escape sequence leaks into the visible text

    def test_non_ascii_text_survives(self) -> None:
        self.assertEqual(plain(mdrender.render("Perché è così — ✔ 日本語")), "Perché è così — ✔ 日本語")


class RouterIntegrationTests(unittest.TestCase):
    def test_render_markdown_uses_rich_when_colour_is_on(self) -> None:
        out = router.render_markdown("# T\n\n```py\nx = 1\n```", True)
        self.assertNotIn("```", out)
        self.assertNotIn("# T", plain(out))

    def test_without_colour_it_is_the_old_plain_renderer(self) -> None:
        raw = "**b** and `c`\n```\n**keep**\n```"
        self.assertEqual(router.render_markdown(raw, False), router.render_terminal_markdown(raw, False))

    def test_empty_text(self) -> None:
        self.assertEqual(router.render_markdown("", True), "")

    def test_falls_back_when_rich_is_missing(self) -> None:
        raw = "**b** and `c`"
        with mock.patch.dict(sys.modules, {"ai_core.mdrender": None}):  # `from ai_core.mdrender import` -> ImportError
            self.assertEqual(router.render_markdown(raw, True), router.render_terminal_markdown(raw, True))

    def test_falls_back_when_rendering_raises(self) -> None:
        raw = "**b**"
        with mock.patch.object(mdrender, "render", side_effect=RuntimeError("rich bug")):
            self.assertEqual(router.render_markdown(raw, True), router.render_terminal_markdown(raw, True))

    def outcome_text(self, **kw: object) -> str:
        buf = io.StringIO()
        with mock.patch.object(router, "_use_color", return_value=True), redirect_stdout(buf):
            router._print_outcome(router.TaskOptions("q", Path(".")), router.Outcome(TaskStatus.COMPLETED_UNVERIFIED, **kw))
        return buf.getvalue()

    def test_task_reply_is_rendered(self) -> None:
        out = self.outcome_text(message="## Plan\n\n- step **one**\n\n```py\nprint(1)\n```")
        text = plain(out)
        self.assertNotIn("```", text)
        self.assertNotIn("##", text)
        self.assertNotIn("**", text)
        self.assertIn("print(1)", text)
        self.assertIn("• step one", text)

    def test_reply_block_keeps_the_bullet_and_indent_on_rendered_lines(self) -> None:
        text = plain(self.outcome_text(message="# Title\n\nsecond paragraph"))
        body = [ln for ln in text.split("\n") if ln.strip()]
        self.assertTrue(body[0].startswith("●"))
        self.assertTrue(all(ln.startswith("  ") for ln in body[1:] if "answered by" not in ln))

    def test_say_md_is_raw_in_agent_mode(self) -> None:
        err, out = io.StringIO(), io.StringIO()
        with mock.patch("sys.stderr", err), redirect_stdout(out), mock.patch.object(router, "_use_color", return_value=True):
            router.say_md(router.TaskOptions("", Path("."), agent=True), "# T\n**b**")
        self.assertEqual((out.getvalue(), err.getvalue()), ("", "# T\n**b**\n"))

    def test_say_md_renders_for_humans(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), mock.patch.object(router, "_use_color", return_value=True):
            router.say_md(router.TaskOptions("", Path(".")), "# T\n\n**b**")
        self.assertEqual(plain(out.getvalue()).split(), ["T", "b"])

    def test_say_md_plain_when_stdout_is_not_a_terminal(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), mock.patch.object(router, "_use_color", return_value=False):
            router.say_md(router.TaskOptions("", Path(".")), "**b** `c`")
        self.assertEqual(out.getvalue(), "b c\n")


class ReviewOutputTests(unittest.TestCase):
    def test_local_review_report_is_rendered(self) -> None:
        from ai_core import review
        from ai_core.config import Config
        buf = io.StringIO()
        with mock.patch.object(review, "collect_diff", return_value=[("a.py", "+x")]), \
                mock.patch.object(review, "_local_review", return_value="**[high]** a.py:3 — `x` is wrong\n- fix it"), \
                mock.patch.object(router, "_use_color", return_value=True), redirect_stdout(buf):
            review.review(Path("."), Config(), "local", router.TaskOptions("", Path(".")))
        shown = [ln.strip() for ln in plain(buf.getvalue()).split("\n") if ln.strip()]
        self.assertEqual(shown[:2], ["[high] a.py:3 — x is wrong", "• fix it"])


if __name__ == "__main__":
    unittest.main()
