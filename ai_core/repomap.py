"""AST Repo-Map (spec §3.A): compact project skeleton ranked by an import-graph PageRank.

Python uses the stdlib `ast`; other languages use signature regexes. The index is cached in
`.ai_context/repo_map_index.json` and refreshed only for files whose (mtime_ns, size) changed,
or whose SHA-256 changed for files in the active working set. Honours .gitignore (via git),
skips secret files, `.ai_context/`, excluded dirs and symlinks leaving the project.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ai_core import estimate_tokens
from ai_core.checkpoint import file_sha256, git_toplevel, write_json_atomic
from ai_core.config import CONTEXT_DIR_NAME, Config
from ai_core.security import is_secret_file, is_within, sanitize_subprocess_env
from ai_core.wsl import project_argv

INDEX_FILE, MAP_FILE = "repo_map_index.json", "repo_map.txt"
_INDEX_VERSION = 1
_MAX_FILE_BYTES = 512 * 1024
_MAX_SIG_LEN = 160
_JS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")
_SIGNATURES: dict[tuple[str, ...], tuple[re.Pattern[str], ...]] = {
    _JS: tuple(re.compile(p, re.M) for p in (
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+\w+\s*\([^)]*\)",
        r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+\w+[^{\n]*",
        r"^\s*export\s+(?:const|let)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
        r"^\s*(?:export\s+)?(?:interface|type)\s+\w+[^={\n]*")),
    (".go",): tuple(re.compile(p, re.M) for p in (
        r"^func\s+(?:\([^)]*\)\s*)?\w+\s*\([^)]*\)[^{\n]*", r"^type\s+\w+\s+(?:struct|interface)")),
    (".rs",): tuple(re.compile(p, re.M) for p in (
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+[^{;\n]*", r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+\w+",
        r"^\s*impl\b[^{\n]*")),
    (".java", ".kt", ".cs"): tuple(re.compile(p, re.M) for p in (
        r"^\s*(?:public|protected|internal)\s+(?:[\w<>\[\],?]+\s+)*(?:class|interface|record|enum|struct)\s+\w+[^{\n]*",
        r"^\s*(?:public|protected|internal)\s+(?:static\s+|async\s+|override\s+|virtual\s+|final\s+|abstract\s+)*"
        r"[\w<>\[\],?]+\s+\w+\s*\([^)\n]*\)")),
}
_JS_IMPORT = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*|^\s*import\s+)['"]([^'"]+)['"]""", re.M)
_JS_RESOLVE = ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", "/index.ts", "/index.tsx", "/index.js")


# ---------------------------------------------------------------- extraction


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_SIG_LEN else text[: _MAX_SIG_LEN - 1] + "…"


def _py_def(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    doc = (ast.get_docstring(node) or "").strip().splitlines()
    return indent + _clip(f"{prefix} {node.name}({ast.unparse(node.args)}){ret}") + (f"  # {_clip(doc[0])}" if doc else "")


def outline_python(source: str, package: str) -> tuple[list[str], list[str]]:
    """(signatures, absolute imported module names) from Python source; `package` anchors relative imports."""
    tree = ast.parse(source)
    sigs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            sigs.append(_py_def(node, ""))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            doc = (ast.get_docstring(node) or "").strip().splitlines()
            sigs.append(_clip(f"class {node.name}({bases}):" if bases else f"class {node.name}:")
                        + (f"  # {_clip(doc[0])}" if doc else ""))
            sigs += [_py_def(m, "    ") for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and (not m.name.startswith("_") or m.name == "__init__")]
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against this module's package
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - node.level + 1] if node.level > 1 else parts
                base = ".".join(p for p in [*parts, node.module or ""] if p)
            else:
                base = node.module or ""
            imports += [base] + [f"{base}.{a.name}" for a in node.names if base]
    return sigs, imports


def outline_regex(source: str, suffix: str) -> tuple[list[str], list[str]]:
    patterns = next((p for exts, p in _SIGNATURES.items() if suffix in exts), ())
    found = sorted((m.start(), _clip(m.group(0).strip().rstrip("{").strip())) for p in patterns for m in p.finditer(source))
    imports = _JS_IMPORT.findall(source) if suffix in _JS else []
    return [s for _, s in found], imports


def _module_name(rel: str) -> str:
    parts = list(PurePosixPath(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# ---------------------------------------------------------------- graph & ranking


def pagerank(nodes: Sequence[str], edges: dict[str, set[str]], weights: dict[str, float],
             damping: float = 0.85, iterations: int = 30) -> dict[str, float]:
    """Personalized PageRank; rank flows from importer to imported module."""
    if not nodes:
        return {}
    total = sum(weights.get(n, 1.0) for n in nodes)
    pers = {n: weights.get(n, 1.0) / total for n in nodes}
    rank = dict(pers)
    for _ in range(iterations):
        new = {n: (1 - damping) * pers[n] for n in nodes}
        dangling = 0.0
        for n in nodes:
            outs = edges.get(n)
            if outs:
                share = damping * rank[n] / len(outs)
                for o in outs:
                    new[o] += share
            else:
                dangling += damping * rank[n]
        for n in nodes:
            new[n] += dangling * pers[n]
        rank = new
    return rank


def _resolve_edges(index: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    modules = {_module_name(rel): rel for rel in index if rel.endswith(".py")}
    for name, rel in list(modules.items()):  # src-layout: `src/pkg/x.py` is imported as `pkg.x`
        if name.startswith("src."):
            modules.setdefault(name[4:], rel)
    files = set(index)
    edges: dict[str, set[str]] = {}
    for rel, entry in index.items():
        targets: set[str] = set()
        for imp in entry.get("imports", []):
            if rel.endswith(".py"):
                if imp in modules:
                    targets.add(modules[imp])
            elif imp.startswith("."):
                base = os.path.normpath(os.path.join(os.path.dirname(rel), imp)).replace("\\", "/")
                targets.update(c for c in (base + ext for ext in _JS_RESOLVE) if c in files)
        targets.discard(rel)
        if targets:
            edges[rel] = targets
    return edges


# ---------------------------------------------------------------- listing & cache


def list_project_files(root: Path, cfg: Config) -> list[str]:
    """Project-relative POSIX paths, .gitignore-aware when in Git, minus protected/secret/excluded."""
    excluded = {d.lower() for d in cfg.checkpoint_exclude_dirs} | {".git", CONTEXT_DIR_NAME}
    rels: list[str] = []
    if git_toplevel(root) is not None:
        cp = subprocess.run(project_argv(root, ["git", "-c", "core.quotepath=off", "ls-files", "-z", "--cached",
                                                "--others", "--exclude-standard", "--", "."]),
                            cwd=root, capture_output=True, encoding="utf-8",
                            errors="replace", env=sanitize_subprocess_env(os.environ, warn=False))
        rels = [p for p in cp.stdout.split("\0") if p] if cp.returncode == 0 else []
    if not rels:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in excluded and not d.startswith(".")]
            rels += [Path(os.path.relpath(Path(dirpath) / f, root)).as_posix() for f in filenames]
    root_resolved = root.resolve()
    out = []
    for rel in sorted(set(rels)):
        parts = PurePosixPath(rel).parts
        if any(p.lower() in excluded for p in parts[:-1]) or is_secret_file(rel, cfg):
            continue
        path = root / rel
        if path.is_symlink() and not is_within(path.resolve(), root_resolved):
            continue
        out.append(rel)
    return out


def _index_entry(path: Path, rel: str, track_sha: bool) -> dict[str, Any] | None:
    try:
        st = path.stat()
        if st.st_size > _MAX_FILE_BYTES:
            return {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "sigs": [], "imports": []}
        source = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):  # locked (WinError 32), unreadable, binary
        return None
    suffix = path.suffix.lower()
    module = _module_name(rel)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    try:
        sigs, imports = outline_python(source, package) if suffix == ".py" else outline_regex(source, suffix)
    except (SyntaxError, ValueError, RecursionError):
        sigs, imports = ["# (unparseable)"], []
    entry: dict[str, Any] = {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "sigs": sigs, "imports": imports}
    if track_sha:
        entry["sha256"] = file_sha256(path)
    return entry


def build_index(root: Path, cfg: Config, working_set: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    ctx = root / CONTEXT_DIR_NAME
    cached: dict[str, Any] = {}
    try:
        data = json.loads((ctx / INDEX_FILE).read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == _INDEX_VERSION:
            cached = data.get("files", {})
    except (OSError, ValueError):
        pass
    ws = set(working_set)
    index: dict[str, dict[str, Any]] = {}
    changed = False
    for rel in list_project_files(root, cfg):
        path = root / rel
        old = cached.get(rel)
        try:
            st = path.stat()
        except OSError:
            continue
        fresh = isinstance(old, dict) and old.get("mtime_ns") == st.st_mtime_ns and old.get("size") == st.st_size
        if fresh and rel in ws:
            fresh = old.get("sha256") == file_sha256(path)
        if fresh:
            index[rel] = old
        elif (entry := _index_entry(path, rel, rel in ws)) is not None:
            index[rel], changed = entry, True
    if (changed or set(cached) != set(index)) and ctx.is_dir():
        write_json_atomic(ctx / INDEX_FILE, {"version": _INDEX_VERSION, "files": index})
    return index


# ---------------------------------------------------------------- rendering


def build_repo_map(root: Path, cfg: Config, focus: Sequence[Path] = (), working_set: Sequence[str] = (), *, only_focus: bool = False) -> str:
    """Ranked skeleton within budget: 2,048 tokens when files are cited, else up to 8,192 (§3.A)."""
    index = build_index(root, cfg, working_set)
    if not index:
        return ""
    focus_rels = [Path(os.path.relpath(p, root)).as_posix() for p in focus]
    focus_dirs = {str(PurePosixPath(r).parent) for r in focus_rels}
    weights = {}
    for rel in index:
        w = 10.0 if rel in focus_rels else 4.0 if rel in working_set else 1.0
        weights[rel] = w * (2.0 if str(PurePosixPath(rel).parent) in focus_dirs else 1.0)
    rank = pagerank(list(index), _resolve_edges(index), weights)
    if only_focus:
        order = [r for r in focus_rels if r in index]
    else:
        order = [r for r in focus_rels if r in index] + sorted(
            (r for r in index if r not in focus_rels), key=lambda r: (-rank[r], r))
    budget = cfg.repo_map_tokens_with_targets if focus_rels else cfg.repo_map_tokens_without_targets
    lines: list[str] = []
    used = 0
    for rel in order:
        block = "\n".join([f"{rel}:" if index[rel]["sigs"] else rel, *(f"  {s}" for s in index[rel]["sigs"])])
        cost = estimate_tokens(block + "\n")
        if used + cost > budget:
            if rel in focus_rels and used + estimate_tokens(rel) <= budget:
                lines.append(rel)  # cited files always appear, at least by path
                used += estimate_tokens(rel + "\n")
            continue
        lines.append(block)
        used += cost
    text = "\n".join(lines)
    ctx = root / CONTEXT_DIR_NAME
    if ctx.is_dir():
        try:
            (ctx / MAP_FILE).write_text(text + "\n", encoding="utf-8")
        except OSError:
            pass  # cache only: an editor holding the file (WinError 32) must not fail the task
    return text
