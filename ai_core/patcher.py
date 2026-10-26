"""Atomic SEARCH/REPLACE application with 2-stage matching (spec §5.B, L08).

All blocks are applied in memory first; any failure rejects the whole patch set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SEARCH = re.compile(r"^<{5,9} SEARCH\s*$")
_DIVIDER = re.compile(r"^={5,9}\s*$")
_REPLACE = re.compile(r"^>{5,9} REPLACE\s*$")
_FILE_TAG = re.compile(r'<file\s+path="([^"]+)"[^>]*>(?:\s*</file>)?')  # models sometimes echo the input tag


class PatchError(ValueError):
    pass


class OutsideTargetError(PatchError):
    """The model wants to edit a file outside the task's targets: the change is wider than routed."""

    def __init__(self, path: str, allowed: list[str]) -> None:
        super().__init__(f"{path}: not an allowed target file (this task may edit only: {', '.join(allowed) or 'none'})")
        self.path = path


@dataclass(frozen=True, slots=True)
class EditBlock:
    path: str
    search: str
    replace: str


def parse_blocks(text: str) -> list[EditBlock]:
    """Parse `path\\n<<<<<<< SEARCH\\n...\\n=======\\n...\\n>>>>>>> REPLACE` blocks (fences tolerated)."""
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[EditBlock] = []
    i = 0
    while i < len(lines):
        if not _SEARCH.match(lines[i]):
            i += 1
            continue
        path = next((c.strip().strip("`*:").strip() for c in reversed(lines[:i])
                     if c.strip() and not c.strip().startswith("```")), "")
        if m := _FILE_TAG.fullmatch(path):
            path = m[1]  # still checked against the allow-list in apply_blocks
        if not path:
            raise PatchError(f"SEARCH block at line {i + 1} has no file path")
        try:
            mid = next(j for j in range(i + 1, len(lines)) if _DIVIDER.match(lines[j]))
            end = next(j for j in range(mid + 1, len(lines)) if _REPLACE.match(lines[j]))
        except StopIteration:
            raise PatchError(f"unterminated SEARCH/REPLACE block for {path}") from None
        blocks.append(EditBlock(path, "\n".join(lines[i + 1:mid]), "\n".join(lines[mid + 1:end])))
        i = end + 1
    return blocks


def _indent(lines: list[str]) -> str:
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return ""
    prefix = min((ln[: len(ln) - len(ln.lstrip())] for ln in non_empty), key=len)
    while prefix and not all(ln.startswith(prefix) for ln in non_empty):
        prefix = prefix[:-1]
    return prefix


def _dedent(lines: list[str], prefix: str) -> list[str]:
    return [ln[len(prefix):].rstrip() if ln.startswith(prefix) else ln.rstrip() for ln in lines]


def apply_block(text: str, block: EditBlock) -> str:
    """Apply one block to LF-normalized `text`. Requires exactly one match per stage."""
    search, replace = block.search, block.replace
    if not search.strip():
        if text.strip():
            raise PatchError(f"{block.path}: empty SEARCH on a non-empty file")
        return replace + ("\n" if replace and not replace.endswith("\n") else "")
    # Stage 1: exact match.
    if (count := text.count(search)) == 1:
        return text.replace(search, replace, 1)
    if count > 1:
        raise PatchError(f"{block.path}: SEARCH matches {count} times; it must be unique")
    # Stage 2: trailing-whitespace / common-indentation insensitive, still unique.
    file_lines = text.split("\n")
    s_lines = search.split("\n")
    s_indent = _indent(s_lines)
    wanted = _dedent(s_lines, s_indent)
    n = len(s_lines)
    hits = [k for k in range(len(file_lines) - n + 1)
            if _dedent(window := file_lines[k:k + n], _indent(window)) == wanted]
    if len(hits) != 1:
        raise PatchError(f"{block.path}: SEARCH {'not found' if not hits else f'matches {len(hits)} times'}")
    k = hits[0]
    f_indent = _indent(file_lines[k:k + n])
    new = [f_indent + ln if ln.strip() else "" for ln in _dedent(replace.split("\n"), s_indent)]
    return "\n".join(file_lines[:k] + new + file_lines[k + n:])


def apply_blocks(originals: dict[str, str | None], blocks: list[EditBlock]) -> dict[str, str]:
    """Return the new content of every edited file; `None` original = file does not exist yet.

    Line endings are compared LF-normalized and restored to each file's native style (CRLF on Windows).
    """
    if not blocks:
        raise PatchError("no SEARCH/REPLACE blocks in model output")
    # Windows: model may write `Src\Calc.py` for `src/calc.py`. Exact allow-list lookup, no traversal.
    # Exact spelling first (WSL files may differ only by case); case-folding only if it names ONE target.
    exact = {k.replace("\\", "/"): k for k in originals}
    folded: dict[str, list[str]] = {}
    for k in originals:
        folded.setdefault(k.replace("\\", "/").lower(), []).append(k)
    work: dict[str, str] = {}
    crlf: dict[str, bool] = {}
    for block in blocks:
        wanted = block.path.replace("\\", "/").removeprefix("./")
        matches = folded.get(wanted.lower(), [])
        path = exact.get(wanted) or (matches[0] if len(matches) == 1 else None)
        if path is None:
            raise OutsideTargetError(block.path, list(originals))
        if path not in work:
            original = originals[path] or ""
            crlf[path] = "\r\n" in original
            work[path] = original.replace("\r\n", "\n")
        work[path] = apply_block(work[path], block)
    return {p: t.replace("\n", "\r\n") if crlf[p] else t for p, t in work.items()}

_CODE_BLOCK = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)\n```", re.DOTALL)


def apply_small_file_fallback(path: str, original: str, model_output: str) -> str:
    """Fallback for small files (<= 150 lines) on 2nd attempt (L08, §5.B).

    Replaces whole function via AST (for Python) or full file if < 80 lines or model emitted full code.
    """
    import ast
    orig_lines_list = original.replace("\r\n", "\n").split("\n")
    n_lines = len(orig_lines_list)
    if n_lines > 150:
        raise PatchError(f"{path}: file has {n_lines} lines (> 150), fallback not permitted")

    blocks = _CODE_BLOCK.findall(model_output)

    # 1. For Python files, attempt AST function/class replacement first
    if path.endswith(".py"):
        try:
            orig_ast = ast.parse(original, filename=path)
            orig_funcs = {n.name: n for n in ast.iter_child_nodes(orig_ast)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            for blk in (blocks or [model_output]):
                try:
                    out_ast = ast.parse(blk, filename="<model_output>")
                except SyntaxError:
                    continue
                out_funcs = {node.name: node for node in ast.iter_child_nodes(out_ast)
                             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                # Only use AST replacement if output contains a subset of functions (not the whole file)
                shared = set(out_funcs) & set(orig_funcs)
                if shared and len(out_funcs) < len(orig_funcs):
                    blk_lines = blk.replace("\r\n", "\n").split("\n")
                    work_lines = list(orig_lines_list)
                    for name in shared:
                        out_node = out_funcs[name]
                        orig_node = orig_funcs[name]
                        if hasattr(orig_node, "lineno") and hasattr(orig_node, "end_lineno"):
                            rep_lines = blk_lines[out_node.lineno - 1: out_node.end_lineno]
                            work_lines = work_lines[:orig_node.lineno - 1] + rep_lines + work_lines[orig_node.end_lineno:]
                    return "\n".join(work_lines) + ("\n" if original.endswith("\n") else "")
        except SyntaxError:
            pass

    # 2. Whole-file replacement if < 80 lines and a valid code block exists
    if n_lines < 80 and blocks:
        candidate = blocks[0].strip()
        if path.endswith(".py"):
            try:
                compile(candidate, path, "exec")
                return candidate + ("\n" if not candidate.endswith("\n") else "")
            except SyntaxError:
                pass
        elif candidate:
            return candidate + ("\n" if not candidate.endswith("\n") else "")

    raise PatchError(f"{path}: small-file fallback could not find matching replacement")

