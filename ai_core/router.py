"""Local decision engine (spec §4) and task execution.

Phase 0 invariants (--offline latch, vault journal, Git state, Ollama) -> Phase 1 pre-filter
(targets, repo-map, token budget) -> Phase 2 Ollama constrained-JSON scoring -> Phase 3 routing
over Channel A (claude), Channel B (agy) and Tier-0 (local) with monthly-credit and cooldown
awareness -> dispatch behind an external checkpoint, the secret scrubber, the same-volume vault
(Channel B) and baseline-aware verification. Quota exhaustion rotates to the other channel (Q03).
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ai_core import EXIT_FOR_STATUS, ExitCode, TaskStatus, estimate_tokens
from ai_core import adr as adrmod
from ai_core.adapters.antigravity_adapter import AntigravityAdapter, bucket_for_model
from ai_core.adapters.base import (
    CLARIFICATION_DIRECTIVE, DATA_ONLY_DIRECTIVE, NEEDS_CLARIFICATION_RE, SCOPE_DIRECTIVE, SYSTEMATIC_DEBUGGING_DIRECTIVE,
    AdapterError, AdapterResult, BaseAdapter, TaskRequest, refuse_slash_command,
)
from ai_core.checkpoint import (
    CheckpointError, create_checkpoint, diff_since, file_sha256, finalize_checkpoint, gc_checkpoints,
    git_operation_in_progress, git_toplevel, list_checkpoints, load_checkpoint, retry_win_lock, safe_undo,
    write_bytes_atomic,
)
from ai_core.config import CONTEXT_DIR_NAME, HOME, PROFILE_PATH, Config, ensure_context_dir
from ai_core.credit import AVAILABLE, EXHAUSTED, LIMITED, CreditError, CreditManager
from ai_core.patcher import OutsideTargetError, PatchError, apply_blocks, parse_blocks, apply_small_file_fallback
from ai_core.repomap import build_repo_map
from ai_core.security import (
    OfflineViolationError, PathValidationError, SecretPathError, detected_billing_vars, enforce_offline,
    is_offline, path_key, redact_secrets, require_cloud_allowed, require_loopback_url, scan_secrets, validate_target_files,
)
from ai_core.session import (
    ProjectBusyError, ProjectLock, SessionError, blackboard_context, find_session, load_blackboard, record_turn, record_recent_task,
    save_blackboard, update_working_set,
)
from ai_core import ollama_env
from ai_core.vault import NTFSVault, VaultError, find_secret_files, recover
from ai_core.verification import VerificationResult, compute_baseline, verify
from ai_core.wsl import WslError, linux_claude, project_key, wsl_target

TASK_TYPES = ("algorithmic", "architecture", "debug", "refactor", "boilerplate", "docs")
ROUTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "files_scope": {"type": "integer", "enum": [1, 2, 3]},
        "reasoning_depth": {"type": "integer", "enum": [1, 2, 3, 4]},
        "needs_repo_search": {"type": "boolean"},
        "creates_new_files": {"type": "boolean"},
        "conversational": {"type": "boolean"},
        "task_type": {"type": "string", "enum": list(TASK_TYPES)},
        "target_files": {"type": "array", "items": {"type": "string"}},
        "routing_notes": {"type": ["string", "null"]},
    },
    "required": ["files_scope", "reasoning_depth", "needs_repo_search", "creates_new_files", "conversational",
                 "task_type", "target_files", "routing_notes"],
}
SCORING_SYSTEM = (
    "You classify a software-engineering task for a router. Reply with JSON only, matching the schema.\n"
    "files_scope: 1 single file, 2 a few related files, 3 cross-cutting/many files.\n"
    "reasoning_depth: 1 trivial/mechanical, 2 standard, 3 hard (subtle bugs, design), 4 deep architecture.\n"
    "needs_repo_search: true if relevant files are unknown and must be searched for.\n"
    "creates_new_files: true only if the task explicitly asks to create new files.\n"
    "conversational: true if the message is chat, a greeting, or a question to be answered in words (about capabilities, "
    "concepts, the project, 'can you X?') and asks for NO change to any file; false if it asks to actually edit, create, "
    "fix or run something, even politely.\n"
    "target_files: project-relative paths that must be edited or created. If creates_new_files is true, target_files "
    "MUST include the relative path of the new file(s) to create (if a specific filename is not given, infer a sensible filename from the prompt, e.g. 'ciao.txt', 'notes.txt', 'script.py').\n"
    "routing_notes: short technical notes useful to a cloud agent, or null."
)
LOCAL_SYSTEM = (
    "You are a precise coding agent. Edit ONLY the files provided. Output ONLY SEARCH/REPLACE blocks:\n\n"
    "{path}\n<<<<<<< SEARCH\nexact existing lines, verbatim, enough to be unique\n"
    "=======\nreplacement lines\n>>>>>>> REPLACE\n\n"
    "The first line of each block is the bare file path (like {path}), never the <file> tag. "
    "SEARCH must match the current file exactly once. A file tagged status=\"new\" does not exist yet: "
    "create it with an empty SEARCH section. For simple file creation tasks, proceed directly with creating the target file without requesting clarification. No prose outside the blocks.\n"
    "If the change also needs files that are not provided (e.g. callers of a changed signature), do not ask "
    "permission: emit a block for each of them with its project-relative path; the router hands the task to an "
    "agent that can edit them. Permission to edit more files is never a [NEEDS_CLARIFICATION]: that marker is "
    "only for requirements the task leaves undefined."
)
SCORING_OPTIONS = {"temperature": 0.0, "num_predict": 512}
WIDE_TASK_TYPES = frozenset({"refactor", "boilerplate", "docs"})
ADR_TASK_TYPES = frozenset({"architecture", "refactor"})
_PREVIEW_HEAD, _PREVIEW_TAIL = 40, 20
_ADR_TIMEOUT_S = 120


class OllamaError(RuntimeError):
    pass


# ---------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class TaskOptions:
    prompt: str
    project_root: Path
    file_args: tuple[str, ...] = ()
    offline: bool = False
    dry_run: bool = False
    verbose: bool = False
    agent: bool = False  # headless: JSON-lines events on stdout, never prompts (W02)
    continue_session: bool = False  # ai -c
    task_id: str | None = None
    extra_directives: str = ""  # e.g. systematic-debugging protocol for --fix
    channel: str | None = None  # forced channel ("anthropic", "agy", "local")
    strict_verification: bool = False  # --fix: the failure to repair is pre-existing by definition, never excuse it


@dataclass(slots=True)
class Routing:
    files_scope: int
    reasoning_depth: int
    needs_repo_search: bool
    creates_new_files: bool
    task_type: str
    target_files: list[Path]
    routing_notes: str | None
    source: str  # "ollama" | "default"
    conversational: bool = False  # the scorer judged it a question/chat, not a coding task


@dataclass(frozen=True, slots=True)
class Decision:
    channel: str  # "local" | "claude" | "agy" | "none"
    model: str
    reason: str


@dataclass(frozen=True, slots=True)
class Channels:
    a_state: str  # available | limited | exhausted | unavailable
    b_ready: frozenset[str]  # Channel B buckets that are installed, configured and not cooling down


@dataclass(slots=True)
class TaskContext:
    routing: Routing
    contents: dict[Path, str | None]
    profile: str
    memory: str
    repo_map: str
    tokens: int
    new_dir_answers: dict[str, bool] = field(default_factory=dict)  # one question per new directory per task


@dataclass(slots=True)
class Outcome:
    status: TaskStatus
    channel: str = ""
    model: str = ""
    message: str = ""
    checkpoint_id: str | None = None
    touched: list[str] = field(default_factory=list)
    verification: VerificationResult | None = None
    adapter: AdapterResult | None = None
    rotate: bool = False  # this channel can't take the task right now: try the next one (Q03)
    proposed_adrs: list[str] = field(default_factory=list)
    task_id: str = ""
    attempts: list[tuple[str, int]] = field(default_factory=list)  # (stats key, 1 ok / 0 failed) per dispatch try
    wanted_files: list[str] = field(default_factory=list)  # local model tried to edit these non-target files


@dataclass(slots=True)
class OllamaStatus:
    reachable: bool
    model_present: bool = False
    gpu_full: bool | None = None  # None: model not loaded right now
    error: str | None = None


# ---------------------------------------------------------------- I/O (human vs --agent)


def interactive(opts: TaskOptions) -> bool:
    return not opts.agent and sys.stdin.isatty()



LAST_ANSWERED_BY = ""  # "channel/model" of the latest human-mode outcome (REPL toolbar + footer)


INPUT_FN = input  # the REPL TUI swaps this so y/N questions are answered from its fixed input box


def _use_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _use_color() else text

def _green(t: str) -> str: return _c(t, "32")
def _red(t: str) -> str: return _c(t, "31")
def _yellow(t: str) -> str: return _c(t, "33")
def _cyan(t: str) -> str: return _c(t, "36")
def _dim(t: str) -> str: return _c(t, "90")
def _bold(t: str) -> str: return _c(t, "1")

def _safe_char(char: str, fallback: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    if "utf" not in enc.lower():
        return fallback
    try:
        char.encode(enc)
        return char
    except (UnicodeEncodeError, LookupError):
        return fallback

_BOLD_RE = re.compile(r"\*\*([^\*\s\n](?:[^\*\n]*?[^\*\s\n])?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^\*\s\n](?:[^\*\n]*?[^\*\s\n])?)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`\n]+)`")
_BULLET_RE = re.compile(r"^(\s*)[*]\s+", re.MULTILINE)

def render_terminal_markdown(text: str, use_color: bool = True) -> str:
    if not text:
        return text
    bullet = _safe_char("\u2022", "-")
    parts = re.split(r"(```[\s\S]*?```)", text)
    result: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            result.append(part)
        else:
            part = _BULLET_RE.sub(lambda m: f"{m.group(1)}{bullet} ", part)
            part = _CODE_RE.sub(
                lambda m: f"\033[36m{m.group(1)}\033[0m" if use_color else m.group(1),
                part
            )
            part = _BOLD_RE.sub(
                lambda m: f"\033[1m{m.group(1)}\033[0m" if use_color else m.group(1),
                part
            )
            part = _ITALIC_RE.sub(
                lambda m: f"\033[3m{m.group(1)}\033[0m" if use_color else m.group(1),
                part
            )
            part = re.sub(
                r"^(#{1,6})\s+(.+)$",
                lambda m: f"\033[1m{m.group(2).strip()}\033[0m" if use_color else m.group(2).strip(),
                part,
                flags=re.MULTILINE
            )
            result.append(part)
    return "".join(result)

def _reply(body: str) -> str:
    """Claude-Code-style reply block on a terminal: bullet on the first line, the rest indented."""
    if not _use_color():
        return body
    first, _, rest = body.partition("\n")
    return _cyan(_safe_char("●", "*")) + " " + first + ("\n" + "\n".join("  " + ln for ln in rest.split("\n")) if rest else "")


def _section(title: str) -> str:
    return "\n" + _bold(title)


def _kv(key: str, value: str, width: int = 14) -> str:
    return f"  {_dim(f'{key:<{width}}')} {value}"


def _bar(pct: float, width: int = 12) -> str:
    """Usage bar: green < 60 %, yellow < 85 %, red above."""
    filled = max(0, min(width, round(pct / 100 * width)))
    paint = _green if pct < 60 else _yellow if pct < 85 else _red
    return paint(_safe_char("\u2588", "#") * filled) + _dim(_safe_char("\u2591", ".") * (width - filled))


def emit(opts: TaskOptions, event: str, **data: Any) -> None:
    """--agent: one JSON object per line on stdout (the machine contract)."""
    if opts.agent:
        print(json.dumps({"event": event, **data}, ensure_ascii=False, default=str), flush=True)


def say(opts: TaskOptions, text: str) -> None:
    """Human-readable output: stdout normally, stderr in --agent mode (stdout stays pure JSON)."""
    print(text, file=sys.stderr if opts.agent else sys.stdout)


def _ask(opts: TaskOptions, question: str, choices: str) -> str | None:
    """Interactive single-letter choice; None when non-interactive (callers must default to safe)."""
    if not interactive(opts):
        return None
    while True:
        try:
            answer = INPUT_FN(f"{question} ").strip().lower()[:1]
        except EOFError:
            return None
        if answer in choices:
            return answer


def _confirm(opts: TaskOptions, question: str) -> bool:
    return _ask(opts, f"{question} [y/N]", "yn") == "y"


def new_dir_confirmer(opts: TaskOptions, answers: dict[str, bool]) -> Callable[[Path], bool]:
    """Approve a new file inside missing directories: asked on a terminal, automatic in --agent / --dry-run."""
    def confirm(path: Path) -> bool:
        rel = _rel(path.parent, opts.project_root)
        if rel not in answers:
            if opts.agent or opts.dry_run:  # headless callers asked for the file; nothing is written in dry-run
                answers[rel] = True
                emit(opts, "new_directory", path=rel)
            else:
                answers[rel] = _confirm(opts, f"Directory '{rel}' does not exist. Create it for {path.name}?")
        return answers[rel]
    return confirm


# ---------------------------------------------------------------- Ollama (localhost only, no proxies)


def _ollama(cfg: Config, path: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    url = cfg.ollama_url.rstrip("/") + path
    if is_offline():
        require_loopback_url(url)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    # Empty ProxyHandler: HTTP(S)_PROXY / WinINET settings must never route local traffic off-box.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise OllamaError(f"{url}: HTTP {exc.code} {detail}") from exc
    except (OSError, ValueError) as exc:  # URLError, timeouts, refused, bad JSON
        raise OllamaError(f"{url}: {exc}") from exc
    if not isinstance(data, dict):
        raise OllamaError(f"{url}: unexpected response")
    return data


def ollama_status(cfg: Config, timeout: float = 3.0) -> OllamaStatus:
    try:
        tags = _ollama(cfg, "/api/tags", timeout=timeout)
        names = {m.get("name") for m in tags.get("models", []) if isinstance(m, dict)}
        status = OllamaStatus(True, cfg.ollama_model in names or f"{cfg.ollama_model}:latest" in names)
        for m in _ollama(cfg, "/api/ps", timeout=timeout).get("models", []):
            if isinstance(m, dict) and m.get("name") == cfg.ollama_model:
                status.gpu_full = m.get("size_vram") == m.get("size")  # L03: any CPU spill = degraded
        return status
    except OllamaError as exc:
        return OllamaStatus(False, error=str(exc))


def start_ollama(cfg: Config, wait: float = 20.0) -> str | None:
    """Spawn `ollama serve` when the default local server is down. Returns an error string, or None once it answers."""
    parsed = urllib.parse.urlparse(cfg.ollama_url)
    if parsed.hostname not in ("127.0.0.1", "localhost") or parsed.port != 11434:
        return "not the default local server"  # custom URL/port: `ollama serve` would not answer there
    exe = shutil.which("ollama") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")
    try:
        subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
    except OSError as exc:
        return f"cannot start ollama: {exc}"
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if ollama_status(cfg, timeout=1.0).reachable:
            return None
        time.sleep(0.5)
    return "ollama serve did not answer in time"


def ollama_chat(
    cfg: Config, messages: list[dict[str, str]], *, think: bool, options: dict[str, Any],
    timeout: float, fmt: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Return (content, done_reason). `think`/`format` are top-level fields, never in options (L04)."""
    payload: dict[str, Any] = {
        "model": cfg.ollama_model, "messages": messages, "stream": False, "think": think,
        "options": {**options, "num_ctx": cfg.ollama_num_ctx},  # same num_ctx everywhere (L02)
    }
    if fmt is not None:
        payload["format"] = fmt
    data = _ollama(cfg, "/api/chat", payload, timeout)
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise OllamaError("chat response without message content")
    return content, data.get("done_reason")


def _exec_options(cfg: Config, thinking: bool) -> dict[str, Any]:
    if thinking:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_predict": cfg.ollama_num_predict_thinking}
    return {"temperature": 0.1, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05,
            "num_predict": cfg.ollama_num_predict_standard}


# ---------------------------------------------------------------- context


def _rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _read_text(path: Path) -> str | None:
    """Exact UTF-8 text (CRLF preserved), or None for missing / non-UTF-8 files."""
    try:
        return retry_win_lock(path.read_bytes).decode("utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None


def load_profile(cfg: Config) -> str:
    if cfg.ponytail_mode == "off":
        return ""
    try:
        base_profile = PROFILE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        base_profile = ""
    if cfg.ponytail_mode == "lite":
        directive = "\n\n[PONYTAIL MODE: LITE]\nFollow basic YAGNI principles. Prefer the simplest change that fully solves the task. Avoid unnecessary abstractions."
    elif cfg.ponytail_mode == "ultra":
        directive = "\n\n[PONYTAIL MODE: ULTRA]\nMaximum parsimony. Produce the absolute minimal diff. Zero new abstractions, zero refactoring outside the target bug/feature. Output a concise explanation in one read."
    else:  # "full"
        directive = "\n\n[PONYTAIL MODE: FULL]\nStrictly follow the 7-step Ponytail Decision Ladder. Reject speculative generality, unused dependencies, and boilerplate. Write minimum viable code without compromising security."
    return (base_profile + directive).strip()


def _repo_map(root: Path, cfg: Config, targets: list[Path], working_set: list[str], *, only_focus: bool = False) -> str:
    try:
        return build_repo_map(root, cfg, targets, working_set, only_focus=only_focus)
    except (OSError, ValueError) as exc:  # the map is an optimisation: never fail a task over it
        print(f"⚠ repo-map unavailable: {exc}", file=sys.stderr)
        return ""


def _context_tokens(ctx: TaskContext, prompt: str) -> int:
    return estimate_tokens(ctx.profile + ctx.repo_map + ctx.memory + prompt + "".join(c or "" for c in ctx.contents.values()))


# ---------------------------------------------------------------- Phase 2: scoring



_FOLLOWUP_WORDS = frozenset({
    "ora", "adesso", "testalo", "testala", "testali", "rifattorizza", "rifattorizzalo",
    "continua", "anche", "invece", "modifica", "aggiorna", "correggi", "now", "continue",
})
_FOLLOWUP_PHRASES = re.compile(r"\b(?:what about|fix it|test it|refactor it|update it)\b")


def _is_followup_prompt(prompt: str) -> bool:
    low = prompt.lower()
    return bool(set(re.findall(r"\b\w+\b", low)) & _FOLLOWUP_WORDS) or bool(_FOLLOWUP_PHRASES.search(low))

def default_routing(targets: list[Path], task_type: str = "debug") -> Routing:
    n = len(targets)
    return Routing(1 if n <= 1 else 2 if n <= 3 else 3, 2, n == 0, False, task_type, list(targets), None, "default")


def validate_routing(data: Any, root: Path, cfg: Config, user_targets: list[Path],
                     confirm_new_dir: Callable[[Path], bool] | None = None) -> Routing:
    """Strict post-generation validation (L06): types, enums, anti-traversal, anti-hallucination."""
    if not isinstance(data, dict):
        raise ValueError("routing is not an object")
    for key, typ in (("files_scope", int), ("reasoning_depth", int), ("needs_repo_search", bool),
                     ("creates_new_files", bool), ("task_type", str), ("target_files", list)):
        if not isinstance(data.get(key), typ) or (typ is int and isinstance(data.get(key), bool)):
            raise ValueError(f"routing.{key} has wrong type")
    if data["files_scope"] not in (1, 2, 3) or data["reasoning_depth"] not in (1, 2, 3, 4):
        raise ValueError("routing scope/depth out of range")
    if data["task_type"] not in TASK_TYPES:
        raise ValueError(f"routing.task_type {data['task_type']!r} invalid")
    notes = data.get("routing_notes")
    targets = list(user_targets)
    for raw in data["target_files"]:
        if not isinstance(raw, str):
            continue
        try:
            targets += validate_target_files([raw], root, cfg, creates_new_files=data["creates_new_files"],
                                             confirm_new_dir=confirm_new_dir)
        except PathValidationError:
            continue  # hallucinated / unsafe path: dropped, the cloud gets no forced path for it
    unique = list({path_key(p): p for p in targets}.values())
    return Routing(data["files_scope"], data["reasoning_depth"], data["needs_repo_search"],
                   data["creates_new_files"], data["task_type"], unique,
                   notes[:2000] if isinstance(notes, str) and notes.strip() else None, "ollama",
                   data.get("conversational") is True)


def _prompt_mentions(prompt: str, paths: list[Path], root: Path) -> bool:
    """True if a prompt word shares a 4-letter stem with a path segment ("finanziario" ~ "finance.service.ts")."""
    # shortcut: stems miss synonyms ("saldo" vs "balance") and send those to repo search; ask the scorer if that bites
    words = {w[:4] for w in re.findall(r"[^\W\d_]{4,}", prompt.lower())}
    parts = {s[:4] for p in paths for s in re.split(r"[\W_]+", _rel(p, root).lower()) if len(s) >= 4}
    return bool(words & parts)


def score_task(cfg: Config, prompt: str, root: Path, targets: list[Path], contents: dict[Path, str | None],
               tokens: int, *, cold: bool = False, repo_map: str = "",
               confirm_new_dir: Callable[[Path], bool] | None = None) -> Routing:
    full = tokens <= cfg.scoring_full_content_max_tokens  # L01 fast path: previews only above threshold
    views = []
    for p in targets:
        text = contents.get(p)
        if text is None:
            views.append(f'<file path="{_rel(p, root)}" status="new-or-unreadable"/>')
            continue
        lines = text.splitlines()
        if not full and len(lines) > _PREVIEW_HEAD + _PREVIEW_TAIL:
            text = "\n".join(lines[:_PREVIEW_HEAD] + ["..."] + lines[-_PREVIEW_TAIL:])
        views.append(f'<file path="{_rel(p, root)}">\n{text}\n</file>')
    user = (f"<task>\n{prompt}\n</task>\n<attached_files>\n{chr(10).join(views)}\n</attached_files>\n"
            f"<repo_map>\n{repo_map}\n</repo_map>")
    payload_tokens = estimate_tokens(SCORING_SYSTEM + user)
    timeout = min(cfg.scoring_base_timeout_s + payload_tokens / cfg.calibrated_prefill_tps, cfg.scoring_max_timeout_s)
    if cold:  # an aborted request also aborts the load, so a cold model would never warm up
        print(f"… loading {cfg.ollama_model} into VRAM (cold start, one-time with OLLAMA_KEEP_ALIVE=-1)",
              file=sys.stderr)
        timeout += cfg.ollama_cold_load_timeout_s
    try:
        content, done = ollama_chat(cfg, [{"role": "system", "content": SCORING_SYSTEM}, {"role": "user", "content": user}],
                                    think=False, options=SCORING_OPTIONS, timeout=timeout, fmt=ROUTING_SCHEMA)
        if done == "length":
            raise ValueError("scoring output truncated")
        routing = validate_routing(json.loads(content), root, cfg, targets, confirm_new_dir)
        if not targets and routing.target_files and not routing.creates_new_files \
                and not _prompt_mentions(prompt, routing.target_files, root):
            # files picked from the repo map alone for a vague prompt: let an agent search instead of guessing
            routing.target_files, routing.needs_repo_search = [], True
        return routing
    except (OllamaError, ValueError) as exc:
        print(f"⚠ scoring unavailable ({exc}); using default routing", file=sys.stderr)
        return default_routing(targets)


# ---------------------------------------------------------------- Phase 3: decision


def claude_model_for_depth(cfg: Config, depth: int) -> str:
    return {1: cfg.claude_model_haiku, 4: cfg.claude_model_opus}.get(depth, cfg.claude_model_sonnet)


def claude_tier(cfg: Config, model: str) -> str:
    return {cfg.claude_model_haiku: "haiku", cfg.claude_model_opus: "opus"}.get(model, "sonnet")


def agy_model_for_depth(cfg: Config, depth: int, *, claude_family: bool) -> str:
    if claude_family:  # "Claude and GPT models" bucket of the Gemini Pro plan
        return cfg.agy_model_opus if depth == 4 else cfg.agy_model_sonnet
    return {1: cfg.agy_model_flash, 2: cfg.agy_model_argon_low, 3: cfg.agy_model_argon_med}.get(
        depth, cfg.agy_model_argon_high)


_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:  # compute_channels runs again at every channel rotation
        _warned.add(key)
        print(f"⚠ {message}", file=sys.stderr)


def compute_channels(cfg: Config, credit: CreditManager | None, root: Path | None = None) -> Channels:
    if is_offline():
        return Channels("unavailable", frozenset())
    target = wsl_target(root) if root else None
    if target:  # WSL project: Claude runs as the distro's own install (a UNC cwd blocks its edits); agy is unverified
        try:
            has_claude = linux_claude(target.distro)
        except WslError as exc:
            _warn_once("claude-" + target.distro, str(exc))
            has_claude = None
    else:
        has_claude = shutil.which("claude") or (HOME / ".local" / "bin" / "claude.exe").exists()
    a_state = "unavailable"
    if has_claude and not cfg.claude_allow_usage_credits:
        try:
            a_state = credit.channel_a().state if credit else AVAILABLE
        except CreditError as exc:  # can't read the counter: can't exclude crossing the reserve
            print(f"⚠ credit state unavailable ({exc}): Channel A blocked", file=sys.stderr)
            a_state = EXHAUSTED
    buckets: set[str] = set()
    if shutil.which("agy"):
        models = (cfg.agy_model_flash, cfg.agy_model_argon_low, cfg.agy_model_argon_med, cfg.agy_model_argon_high,
                  cfg.agy_model_sonnet, cfg.agy_model_opus)
        for bucket in {bucket_for_model(m) for m in models if m}:
            try:
                if credit is None or credit.channel_b_ready(bucket):
                    buckets.add(bucket)
            except CreditError:
                continue
    return Channels(a_state, frozenset(buckets))


_STATS_WINDOW = 20  # last N scored attempts kept per model|task_type|depth
_STATS_MIN_SAMPLES = 5
_STATS_SKIP_BELOW = 0.4  # local success rate under which Tier-0 is skipped
_STATS_EXPLORE_RATE = 0.1  # still try local this often, so the record can recover (e.g. after a Qwen upgrade)
_SCORE_OK = frozenset({TaskStatus.COMPLETED_VERIFIED, TaskStatus.COMPLETED_SYNTAX_ONLY,
                       TaskStatus.COMPLETED_WITH_PREEXISTING_FAILURES})
_SCORE_FAILED = frozenset({TaskStatus.TESTS_FAILED, TaskStatus.ROLLED_BACK, TaskStatus.PARTIALLY_APPLIED})
_MODEL_FAILURES = ("patch rejected", "verification failed", "model output truncated")


def _stats_key(model: str, routing: Routing) -> str:
    return f"{model}|{routing.task_type}|{routing.reasoning_depth}"


def attempt_score(out: Outcome) -> int | None:
    """1/0 when the outcome says something about the model's ability; None for quota, infra, chat, questions."""
    # unverified counts only with real edits: in TS/JS projects without verify_l2_command L0 is all there is
    if out.status in _SCORE_OK or (out.status == TaskStatus.COMPLETED_UNVERIFIED and out.touched):
        return 1
    if out.status in _SCORE_FAILED or (out.status == TaskStatus.PATCH_FAILED and out.message.startswith(_MODEL_FAILURES)):
        return 0
    return None


def record_attempts(bb: dict[str, Any], attempts: list[tuple[str, int]]) -> None:
    """`attempts`: (stats key at attempt time, score); the depth can be raised between attempts."""
    stats = bb.get("routing_stats")
    if not isinstance(stats, dict):
        stats = bb["routing_stats"] = {}
    for key, score in attempts:
        window = stats.get(key)
        window = [s for s in window if s in (0, 1)] if isinstance(window, list) else []
        stats[key] = (window + [score])[-_STATS_WINDOW:]


def _local_track_record(cfg: Config, routing: Routing, stats: dict[str, Any] | None) -> tuple[int, int] | None:
    """(successes, samples) when Tier-0 has a poor record on this kind of task, else None."""
    window = (stats or {}).get(_stats_key(cfg.ollama_model, routing))
    if not isinstance(window, list) or len(window) < _STATS_MIN_SAMPLES:
        return None
    ok = sum(1 for s in window if s == 1)
    if ok / len(window) >= _STATS_SKIP_BELOW or random.random() < _STATS_EXPLORE_RATE:
        return None
    return ok, len(window)


def decide(cfg: Config, routing: Routing, tokens: int, offline: bool, channels: Channels | None = None,
           *, exclude: frozenset[str] = frozenset(), allow_local: bool = True,
           forced_channel: str | None = None, stats: dict[str, Any] | None = None) -> Decision:
    """Spec §4 Phase 3, rules in order. `channels=None` means every cloud channel is usable.

    `stats` (blackboard routing_stats) lets a poor Tier-0 record send the task straight to the cloud."""
    n, depth = len(routing.target_files), routing.reasoning_depth
    if offline:
        return Decision("local", cfg.ollama_model, "--offline: cloud channels physically disabled")
    if forced_channel:
        fc = forced_channel.lower()
        if fc in ("anthropic", "claude"):
            m = (cfg.claude_model_opus if depth == 4 else
                 cfg.claude_model_sonnet if depth >= 2 else cfg.claude_model_haiku)
            return Decision("claude", m, f"--channel {forced_channel}: forced Channel A")
        if fc in ("agy", "antigravity", "gemini"):
            m = cfg.agy_model_argon_med if depth >= 3 else cfg.agy_model_flash
            return Decision("agy", m, f"--channel {forced_channel}: forced Channel B")
        if fc == "local":
            return Decision("local", cfg.ollama_model, f"--channel {forced_channel}: forced Tier-0 local")
    local_fits = 0 <= n <= cfg.local_max_files and tokens <= cfg.local_max_input_tokens
    skipped = ""
    if allow_local and local_fits and not routing.needs_repo_search and depth <= 2:
        if (record := _local_track_record(cfg, routing, stats)) is None:
            return Decision("local", cfg.ollama_model, f"{n} file(s), depth {depth}, ~{tokens} tokens")
        skipped = f"; local skipped: {record[0]}/{record[1]} recent successes on {routing.task_type}/d{depth}"

    a_ok = channels is None or channels.a_state in (AVAILABLE, LIMITED)

    def usable(d: Decision) -> bool:
        if d.channel in exclude or not d.model:
            return False
        if d.channel == "claude":
            return a_ok
        return channels is None or bucket_for_model(d.model) in channels.b_ready

    gemini = Decision("agy", agy_model_for_depth(cfg, depth, claude_family=False), "Channel B (Gemini)")
    claude_b = Decision("agy", agy_model_for_depth(cfg, depth, claude_family=True), "Channel B (Claude bucket)")
    claude_a = Decision("claude", claude_model_for_depth(cfg, depth), "Channel A (Claude Pro credit)")
    flash = Decision("agy", cfg.agy_model_flash, "Channel B (Gemini Flash, Argon unavailable)")
    if routing.files_scope == 3 or n > 3 or tokens > cfg.local_max_input_tokens or routing.task_type in WIDE_TASK_TYPES:
        # volume / multi-file / wide context -> Gemini first; Flash before the scarce Channel A,
        # except for hard work (depth >= 3, e.g. redesigns), where Flash is too weak: Claude instead
        if depth >= 3:
            candidates = [gemini, claude_b, claude_a]
        else:
            candidates = [gemini, claude_a] if gemini.model == flash.model else [gemini, flash, claude_a]
    elif depth == 1:
        candidates = [claude_a, gemini]  # Haiku lives only on Channel A and costs almost nothing
    else:
        candidates = [claude_b, claude_a]  # Sonnet/Opus: self-refilling Channel B first, scarce A as reserve
    detail = f"depth {depth}, {n} file(s), ~{tokens} tokens, type {routing.task_type}"
    for cand in candidates:
        if usable(cand):
            fallback = "" if cand is candidates[0] else " (preferred channel unavailable)"
            return Decision(cand.channel, cand.model, f"{cand.reason}: {detail}{fallback}{skipped}")
    if allow_local and local_fits and "local" not in exclude:
        return Decision("local", cfg.ollama_model, f"cloud channels exhausted/cooling down: forced Tier-0 ({detail})")
    return Decision("none", "", "every allowed channel is exhausted or cooling down")


# ---------------------------------------------------------------- local branch



def is_conversational_prompt(prompt: str) -> bool:
    """Return True if prompt is a greeting, general question or chat requiring no file modifications."""
    clean = prompt.strip("'\" \t\r\n").lower().rstrip("!?.,;: ")
    greetings = {
        "ciao", "salve", "buongiorno", "buonasera", "buondì", "ehi", "ei",
        "hello", "hi", "hey", "hola", "greetings", "good morning", "good evening",
        "chi sei", "chi sei tu", "cosa fai", "cosa puoi fare", "cosa sai fare",
        "cosa sei", "che cosa sei", "cosa sei tu", "parlami di te", "raccontami di te",
        "presentati", "chi ti ha creato", "come ti chiami",
        "who are you", "what are you", "what can you do", "help", "aiuto",
        "grazie", "grazie mille", "prego", "ok grazie", "va bene", "perfetto"
    }
    if clean in greetings:
        return True

    question_prefixes = (
        "come ", "cosa ", "che ", "che cosa ", "cos'è ", "qual ", "qual è ", "quali ", "perché ", "chi ", "dove ", "quando ",
        "spiega ", "spiegami ", "dimmi ", "descrivi ",
        "ti ricordi ", "ricordi ", "raccontami ", "parlami ", "presentati",
        "how ", "what ", "what is ", "what are ", "why ", "where ", "when ", "explain ", "tell me ", "describe ",
        "do you remember "
    )
    edit_verbs = (
        "modifica", "cambia", "aggiungi", "crea", "scrivi", "elimina", "cancella",
        "refactor", "fix", "ripara", "correggi", "implementa", "risolvi", "aggiusta",
        "sistemare", "sistemalo", "sistemala", "sistemali",  # bare "sistema" is also the noun
        "edit", "change", "add", "create", "write", "delete", "remove", "implement"
    )
    # verb + clitic/infinitive/-ing ("correggilo", "modificare", "fixing"), not "cambiato"/"creato"
    suffixes = ("", "lo", "la", "li", "le", "ne", "mi", "ci", "re", "ing")
    if clean.startswith(question_prefixes):
        words = re.findall(r"\w+", clean)
        if not any(w.startswith(v) and w[len(v):] in suffixes for v in edit_verbs for w in words):
            return True

    return False


CONVERSATIONAL_SYSTEM = (
    "Sei l'assistente AI di CodingEngine, un orchestratore e router dual-channel progettato per lo sviluppo software nativo su Windows.\n"
    "Fai da 'cervello locale' dell'ambiente di lavoro, in esecuzione tramite Ollama sulla GPU locale dell'utente (Tier-0, modello Qwen), "
    "con la capacità di connetterti intelligentemente ai canali cloud quando necessario per problemi complessi.\n\n"
    "Chi sei e come funzioni:\n"
    "- Identità: Sei il cervello locale di CodingEngine. Operi prioritariamente in locale (Tier-0) per garantire massima privacy, zero costi API e latenza minima su domande, chat, spiegazioni, analisi del codice e modifiche circoscritte.\n"
    "- Connessione Cloud intelligente: Quando una richiesta supera i limiti locali (es. refactoring architetturali complessi, modifiche cross-cutting su molti file, ragionamento ad altissima profondità), il router instrada il task verso:\n"
    "  * Canale A (Claude Code / Anthropic): per task ad alta precisione, audit o ragionamento architetturale (Haiku, Sonnet, Opus).\n"
    "  * Canale B (Antigravity / Gemini): per task ad ampio contesto, grandi basi di codice o modifiche massive multi-file.\n"
    "- Funzionalità chiave di CodingEngine:\n"
    "  * Checkpoint & Safe Undo: snapshot esterni automatici prima di ogni task; comando `ai undo` per rollback sicuri che preservano sempre le modifiche dell'utente.\n"
    "  * Verifica a 3 livelli (L1/L2/L3): verifica automatica di sintassi, linter e test prima di dichiarare completato un task.\n"
    "  * Sicurezza & Privacy: Scrubber automatico che maschera chiavi e segreti; modalità `--offline` che disabilita fisicamente qualsiasi connessione cloud; NTFS Vault per proteggere file sensibili su Channel B.\n"
    "  * Blackboard & Memoria: memoria rolling dei task recenti, storico conversazionale continuo nel REPL (`ai`), e tracciamento delle decisioni architetturali (ADR).\n"
    "  * Tooling CLI: comandi dedicati come `ai status` (stato GPU, quote e checkpoint), `ai usage` (consumo quote), `ai --fix` / `ait` (riparazione comandi falliti), `ai --review` (code review locale, cross o ultra), `ai --commit` (commit con controllo segreti).\n\n"
    "Comportamento nelle risposte:\n"
    "- Rispondi in modo cordiale, chiaro, competente, diretto e professionale, nella lingua dell'utente.\n"
    "- Se l'utente ti chiede chi sei, cosa puoi fare, come funzioni o quali sono le tue funzionalità, spiega la tua identità di cervello locale/orchestratore ibrido e descrivi chiaramente le funzionalità sopra elencate.\n"
    "- Se ti vengono chieste informazioni sul progetto corrente o sul codice, usa il contesto fornito (<repo_map>, blackboard, ecc.).\n"
    "- NON generare blocchi SEARCH/REPLACE né modifiche a file a meno che non ti venga esplicitamente richiesto di modificare il codice."
)

def run_conversational_chat(opts: TaskOptions, cfg: Config, ctx: TaskContext, prompt: str | None = None) -> Outcome:
    """Direct response from Tier-0 Ollama (qwen3.6) for conversational, greetings and Q&A."""
    user_prompt = prompt or opts.prompt
    emit(opts, "attempt", channel="local", thinking=False)
    system = CONVERSATIONAL_SYSTEM
    if ctx.profile:
        system += "\n\n" + ctx.profile

    root = opts.project_root
    bb, bb_writable = load_blackboard(root)
    chat_history = bb.get("chat_history", [])

    user_content = user_prompt
    about_project = any(k in user_prompt.lower() for k in ("progetto", "project", "repo", "codice", "code", "file", "architett", "blackboard", "config"))
    if about_project and not ctx.repo_map:  # the conversational fast path skips the map: build it only when asked
        ctx.repo_map = _repo_map(root, cfg, [], [])
    if about_project and ctx.repo_map:
        user_content = f"<repo_map>\n{ctx.repo_map}\n</repo_map>\n\n<task>\n{user_prompt}\n</task>"

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if isinstance(chat_history, list):
        for msg in chat_history[-8:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
    messages.append({"role": "user", "content": user_content})

    try:
        reply, _ = ollama_chat(
            cfg, messages, think=False,
            options={"temperature": 0.5, "num_predict": 1024},
            timeout=cfg.local_timeout_s
        )
    except OllamaError as exc:
        return Outcome(
            TaskStatus.LOCAL_MODEL_UNAVAILABLE if is_offline() else TaskStatus.PROVIDER_ERROR,
            "local", cfg.ollama_model, str(exc)
        )

    reply = reply.strip()
    task_id = opts.task_id or uuid.uuid4().hex[:12]
    if bb_writable:
        if not isinstance(chat_history, list):
            chat_history = []
        chat_history.append({"role": "user", "content": user_prompt})
        chat_history.append({"role": "assistant", "content": reply})
        bb["chat_history"] = chat_history[-12:]
        record_turn(bb, task_id, channel="local", model=cfg.ollama_model, session_id=None,
                    status=TaskStatus.COMPLETED_UNVERIFIED.value, question=None, checkpoint_id=None,
                    prompt=user_prompt, target_files=[], cost_usd=0.0, tokens=estimate_tokens(user_prompt + reply))
        record_recent_task(bb, task_id, prompt=user_prompt, channel="local", model=cfg.ollama_model,
                           status=TaskStatus.COMPLETED_UNVERIFIED.value, touched_files=[],
                           checkpoint_id=None, summary=reply[:200])
        save_blackboard(root, bb)

    return Outcome(
        status=TaskStatus.COMPLETED_UNVERIFIED,
        channel="local",
        model=cfg.ollama_model,
        message=reply,
        task_id=task_id,
        adapter=AdapterResult(process_exit_code=0, status=None, result_text=reply)
    )


def _infer_new_file_target(prompt: str, root: Path, cfg: Config,
                           confirm_new_dir: Callable[[Path], bool] | None = None) -> Path | None:
    """Infer target filename when user asks to create a file without naming +file explicitly."""
    # keep the directory part: "src/lib/x.ts" must not become "x.ts" at the project root
    m = re.search(r'\b([\w\-./\\]+\.(?:txt|py|js|ts|json|md|html|css|yaml|yml|toml|rs|go|c|cpp|h))\b', prompt, re.IGNORECASE)
    if m:
        try:
            val = validate_target_files([m.group(1)], root, cfg, creates_new_files=True, confirm_new_dir=confirm_new_dir)
            if val:
                return val[0]
        except PathValidationError:
            return None  # the user named this path: never let the model relocate it

    messages = [
        {"role": "system", "content": "You extract or invent the target filename to create from a user request. Return ONLY the relative filename (e.g. ciao.txt, notes.txt, hello.py), with no markdown formatting and no other words."},
        {"role": "user", "content": prompt}
    ]
    try:
        reply, _ = ollama_chat(cfg, messages, think=False, options={"temperature": 0.0, "num_predict": 32}, timeout=5.0)
        candidate = reply.strip().strip("'\"` \t\r\n").splitlines()[0].split()[0]
        val = validate_target_files([candidate], root, cfg, creates_new_files=True)
        if val:
            return val[0]
    except Exception:
        pass
    return None


def run_local(opts: TaskOptions, cfg: Config, ctx: TaskContext, prompt: str | None = None, *, extra_context: str = "", **kwargs: Any) -> Outcome:
    root = opts.project_root
    routing = ctx.routing
    out = Outcome(TaskStatus.PATCH_FAILED, "local", cfg.ollama_model)
    inferred_hint = ""
    if not routing.target_files:
        if routing.creates_new_files:
            inferred = _infer_new_file_target(prompt or opts.prompt, root, cfg,
                                              new_dir_confirmer(opts, ctx.new_dir_answers))
            if inferred is not None:
                routing.target_files.append(inferred)
                ctx.contents[inferred] = None
                inferred_hint = f"\n[Nota: il file da creare è '{_rel(inferred, root)}'. Procedi direttamente con la creazione del file e del contenuto richiesto senza chiedere conferme sul nome.]\n"
        if not routing.target_files:
            declined = [d for d, ok in ctx.new_dir_answers.items() if not ok]
            # a "create" verb with no inferable filename is a question ("Puoi creare file?"), not a patch: answer locally
            if not routing.needs_repo_search and not declined:
                return run_conversational_chat(opts, cfg, ctx, prompt)
            out.message = (f"directory {', '.join(declined)} does not exist and creating it was not approved"
                           if declined else "no target files: name them with +path/to/file")
            return out
    rels = {_rel(p, root): p for p in routing.target_files}
    originals = {rel: ctx.contents.get(p) for rel, p in rels.items()}
    if unreadable := [r for r, p in rels.items() if p.exists() and originals[r] is None]:
        out.message = f"not UTF-8 text, cannot patch locally: {', '.join(unreadable)}"
        return out
    try:
        cp = create_checkpoint(root, cfg, routing.target_files, include_untracked=False)
    except CheckpointError as exc:
        out.message = f"checkpoint failed, nothing dispatched: {exc}"
        return out
    out.checkpoint_id = cp.checkpoint_id
    emit(opts, "checkpoint", checkpoint_id=cp.checkpoint_id)
    baseline = None if opts.strict_verification else compute_baseline(root, cfg, [p for p in routing.target_files if p.exists()])
    files_xml = "\n".join(f'<file path="{r}" status="new"></file>' if originals[r] is None
                          else f'<file path="{r}">\n{originals[r]}\n</file>' for r in rels)
    system = "\n\n".join(s for s in (LOCAL_SYSTEM.replace("{path}", next(iter(rels))), CLARIFICATION_DIRECTIVE, DATA_ONLY_DIRECTIVE,
                                     SCOPE_DIRECTIVE, _directives(opts, routing),
                                     ctx.profile) if s)
    diff_block = f"\n{extra_context}\n" if extra_context else ""
    user = f"<repo_map>\n{ctx.repo_map}\n</repo_map>\n{ctx.memory}{diff_block}{inferred_hint}\n<files>\n{files_xml}\n</files>\n" \
           f"<task>\n{prompt or opts.prompt}\n</task>"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ever_written: set[str] = set()
    try:
        for thinking in (False, True):  # attempt 1 fast, attempt 2 with thinking (§2.A profiles)
            emit(opts, "attempt", channel="local", thinking=thinking)
            try:
                text, done = ollama_chat(cfg, messages, think=thinking, options=_exec_options(cfg, thinking),
                                         timeout=cfg.local_timeout_s)
            except OllamaError as exc:
                out.status = TaskStatus.LOCAL_MODEL_UNAVAILABLE if is_offline() else TaskStatus.PATCH_FAILED
                out.message = str(exc)
                break
            if done == "length":  # L04: truncated output is never applied, even partially
                out.message = "model output truncated (num_predict)"
                messages.append({"role": "user", "content": "Your output was truncated. Emit ONLY the minimal blocks."})
                continue
            try:
                blocks = parse_blocks(text)
            except PatchError:
                blocks = []
            # A marker counts only if the model actually stopped: output with edit blocks is a patch attempt.
            if not blocks and (m := NEEDS_CLARIFICATION_RE.search(text)):
                out.status, out.message = TaskStatus.NEEDS_CLARIFICATION, m[1].strip()
                break
            try:
                new = {r: t for r, t in apply_blocks(originals, parse_blocks(text)).items() if t != originals[r]}
                if not new:  # Invariant #4: a no-op patch must never become "completed_verified"
                    raise PatchError("the blocks change nothing; edit the files the task asks for")
            except PatchError as exc:
                if isinstance(exc, OutsideTargetError) and exc.path not in out.wanted_files:
                    out.wanted_files.append(exc.path)
                if thinking:
                    recovered = {}
                    for rel, orig in originals.items():
                        if orig is not None and len(orig.splitlines()) <= 150:
                            try:
                                fb = apply_small_file_fallback(rel, orig, text)
                                if fb != orig:
                                    recovered[rel] = fb
                            except Exception:
                                pass
                    if recovered:
                        new = recovered
                    else:
                        out.message = f"patch rejected: {exc}"
                        messages += [{"role": "assistant", "content": text},
                                     {"role": "user", "content": f"Patch rejected: {exc}. Re-emit corrected blocks."}]
                        continue
                else:
                    out.message = f"patch rejected: {exc}"
                    messages += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": f"Patch rejected: {exc}. Re-emit corrected blocks."}]
                    continue
            try:
                written = _write_patch(rels, originals, new)
            except OSError as exc:  # WinError 32/5 persisted after retries; already reverted
                out.message = f"write failed, patch reverted: {exc}"
                break
            ever_written |= set(written)
            result = verify(root, cfg, [rels[r] for r in written], baseline)
            if result.status == TaskStatus.TESTS_FAILED:
                _revert_patch(rels, originals, written)
                out.message = f"verification failed: {', '.join(result.new_failures)}"
                for fid in result.new_failures:  # "a.ts:TS2345", "tests/x.py::test_y": the patch broke these files
                    path = fid.split("::")[0].rsplit(":", 1)[0] if ":" in fid else ""
                    if path and path not in rels and path not in out.wanted_files and not fid.startswith("syntax:"):
                        out.wanted_files.append(path)
                tails = "\n".join(lv.output[-1500:] for lv in result.levels if lv.status == "failed")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Your patch broke verification:\n{tails}\nFix it."}]
                continue
            out.status, out.verification, out.message = result.status, result, ""
            break
    finally:
        try:
            finalize_checkpoint(root, cfg, cp, [rels[r] for r in ever_written])
        except CheckpointError as exc:
            print(f"⚠ checkpoint finalize failed: {exc}", file=sys.stderr)
    out.touched = sorted(cp.touched_files)
    return out


def _write_patch(rels: dict[str, Path], originals: dict[str, str | None], new: dict[str, str]) -> dict[str, str]:
    """Write each file atomically; returns {rel: sha256 written}. On failure, reverts what was written."""
    written: dict[str, str] = {}
    try:
        for rel, text in new.items():
            write_bytes_atomic(rels[rel], text.encode("utf-8"))
            written[rel] = file_sha256(rels[rel]) or ""
    except OSError:
        _revert_patch(rels, originals, written)
        raise
    return written


def _revert_patch(rels: dict[str, Path], originals: dict[str, str | None], written: dict[str, str]) -> None:
    for rel, sha in written.items():
        path = rels[rel]
        if file_sha256(path) != sha:  # someone edited it meanwhile: never clobber (Invariant #2)
            print(f"⚠ {rel} changed during verification; not reverted", file=sys.stderr)
        elif originals[rel] is None:
            retry_win_lock(path.unlink)
        else:
            write_bytes_atomic(path, originals[rel].encode("utf-8"))


def _directives(opts: TaskOptions, routing: Routing) -> str:
    """Systematic debugging for debug tasks with depth >= 3, plus any caller directives (--fix)."""
    parts = [opts.extra_directives]
    if routing.task_type == "debug" and routing.reasoning_depth >= 3 and SYSTEMATIC_DEBUGGING_DIRECTIVE not in parts[0]:
        parts.append(SYSTEMATIC_DEBUGGING_DIRECTIVE)
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------- cloud branch


def _adapter(cfg: Config, channel: str) -> BaseAdapter:
    return _claude_adapter(cfg) if channel == "claude" else AntigravityAdapter(cfg)


def _claude_adapter(cfg: Config) -> BaseAdapter:
    from ai_core.adapters.claude_adapter import ClaudeAdapter  # imported only when cloud is allowed
    return ClaudeAdapter(cfg)


def _scrub(opts: TaskOptions, cfg: Config, ctx: TaskContext, prompt: str, context: str) -> tuple[str, str, str | None]:
    """Privacy firewall (S01). Returns (prompt, context, action) with action None | 'local' | 'abort'."""
    if not cfg.secret_scanner_enabled:
        return prompt, context, None
    root = opts.project_root
    text_hits = scan_secrets(prompt, "<prompt>") + scan_secrets(context, "<context>")
    file_hits = [f for p in ctx.routing.target_files for f in scan_secrets(ctx.contents.get(p) or "", _rel(p, root))]
    if not (text_hits or file_hits):
        return prompt, context, None
    print("⛔ Privacy firewall: potential secrets (values never shown):", file=sys.stderr)
    for finding in text_hits + file_hits:
        print(f"   {finding}", file=sys.stderr)
    emit(opts, "secret_detected", locations=[str(f) for f in text_hits + file_hits])
    if opts.agent:
        choice = {"local_only": "l", "redact": "r", "abort": "a"}[cfg.secret_action_agent]
        if choice == "r" and file_hits:
            choice = "l"  # file contents can't be masked: the CLI reads them from disk
    else:
        # File contents can't be masked (the CLI reads them from disk): only local or abort.
        choice = _ask(opts, "[r]edact, [l]ocal model, [a]bort?" if not file_hits else "[l]ocal model, [a]bort?",
                      "rla" if not file_hits else "la")
    if choice == "r":
        return redact_secrets(prompt), redact_secrets(context), None
    if choice == "l":
        return prompt, context, "local"
    return prompt, context, "abort"


def run_cloud(opts: TaskOptions, cfg: Config, ctx: TaskContext, decision: Decision, task_id: str, *,
              prompt: str | None = None, extra_context: str = "", resume_id: str | None = None,
              read_only: bool = False, credit: CreditManager | None = None) -> Outcome:
    require_cloud_allowed(decision.channel)
    root = opts.project_root
    routing = ctx.routing
    out = Outcome(TaskStatus.PROVIDER_ERROR, decision.channel, decision.model)
    if target := wsl_target(root):  # also stops --channel, which skips the channel matrix
        if decision.channel == "claude":
            try:
                linux_claude(target.distro)
            except WslError as exc:
                out.message = str(exc)
                return out
    notes = [f"target_files:{', '.join(_rel(p, root) for p in routing.target_files)}"] if routing.target_files else []
    if routing.routing_notes:
        notes.append(routing.routing_notes)
    context = "\n".join(s for s in (
        f"<repo_map>\n{ctx.repo_map}\n</repo_map>" if ctx.repo_map else "", ctx.memory,
        f"<routing_context_notes>\n{chr(10).join(notes)}\n</routing_context_notes>" if notes else "",
        extra_context) if s)
    prompt, context, action = _scrub(opts, cfg, ctx, prompt or opts.prompt, context)
    if action == "local" and read_only:  # a read-only run must never turn into a local patch
        out.status = TaskStatus.SECRET_DETECTED_ABORTED
        out.message = "secrets in the material to review: run the local review instead (ai --review)"
        return out
    if action == "local":
        enforce_offline()  # local-only for the rest of this process: no escalation possible
        return run_local(opts, cfg, ctx, prompt)
    if action == "abort":
        out.status, out.message = TaskStatus.SECRET_DETECTED_ABORTED, "dispatch blocked by secret scrubber"
        return out

    budget: float | None = None
    if decision.channel == "claude" and credit is not None:
        try:
            verdict = credit.check_claude(claude_tier(cfg, decision.model), interactive=interactive(opts))
        except CreditError as exc:
            out.status, out.message, out.rotate = TaskStatus.QUOTA_EXHAUSTED, str(exc), True
            return out
        if not verdict.allowed:
            exhausted = not verdict.needs_confirmation
            out.status = TaskStatus.QUOTA_EXHAUSTED if exhausted else TaskStatus.PERMISSION_DENIED
            out.message, out.rotate = verdict.reason, exhausted
            return out
        if verdict.needs_confirmation and not _confirm(
                opts, f"{verdict.reason}. Spend up to ${verdict.max_budget_usd:.2f} of the Claude Pro credit?"):
            out.status, out.message = TaskStatus.PERMISSION_DENIED, f"not confirmed: {verdict.reason}"
            return out
        budget = verdict.max_budget_usd

    project_secrets = find_secret_files(root, cfg)
    secret_files = project_secrets if decision.channel == "agy" else []
    if secret_files and cfg.vault_isolation_mode == "never" and not _confirm(
            opts, f"Channel B has no file-level deny and vault_isolation_mode=never: {len(secret_files)} secret "
                  "file(s) would be readable by agy. Proceed anyway?"):
        out.status, out.message, out.rotate = TaskStatus.PERMISSION_DENIED, "secret files present, vault disabled", True
        return out

    try:
        adapter = _adapter(cfg, decision.channel)
        cp = create_checkpoint(root, cfg, routing.target_files, include_untracked=True)
    except (AdapterError, CheckpointError) as exc:
        out.message = f"nothing dispatched: {exc}"
        out.rotate = isinstance(exc, AdapterError)
        return out
    out.checkpoint_id = cp.checkpoint_id
    emit(opts, "checkpoint", checkpoint_id=cp.checkpoint_id)
    baseline = None if read_only or opts.strict_verification else \
        compute_baseline(root, cfg, [p for p in routing.target_files if p.exists()])
    system_append = "\n\n".join(s for s in (ctx.profile, CLARIFICATION_DIRECTIVE, DATA_ONLY_DIRECTIVE,
                                            "" if read_only else SCOPE_DIRECTIVE, _directives(opts, routing)) if s)
    request = TaskRequest(task_id, prompt, root, decision.model, tuple(routing.target_files), context, system_append,
                          opts.verbose, read_only, resume_id, budget, disable_mcp=bool(project_secrets),
                          reasoning_depth=routing.reasoning_depth)
    guarded = _guard_state_files(root)
    before = _tree_fingerprint(root, cfg) if read_only else {}
    result: AdapterResult | None = None
    emit(opts, "dispatch", channel=decision.channel, model=decision.model, resume=bool(resume_id))
    try:
        with NTFSVault(root, secret_files if cfg.vault_isolation_mode == "auto" else []) as vault:
            if vault.skipped:  # S05: a locked .env stays in place -> agy must not run here
                out.message = "secret file(s) locked by another process: " + ", ".join(p for p, _ in vault.skipped)
                out.status, out.rotate = TaskStatus.PERMISSION_DENIED, True
            else:
                result = adapter.run_task(request)
    except AdapterError as exc:
        out.message, out.rotate = str(exc), True
    except VaultError as exc:
        out.status, out.message, out.rotate = TaskStatus.PERMISSION_DENIED, f"vault unavailable: {exc}", True
    finally:
        _restore_state_files(guarded)
        try:
            finalize_checkpoint(root, cfg, cp, [Path(p) for p in (result.edited_paths if result else [])])
        except CheckpointError as exc:
            print(f"⚠ checkpoint finalize failed: {exc}", file=sys.stderr)
    out.touched = sorted(cp.touched_files)
    if read_only:  # S02: Git diff misses ignored files and non-Git projects; compare the whole tree too
        after = _tree_fingerprint(root, cfg)
        out.touched = sorted(set(out.touched) | {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)})
    if result is None:
        return out
    out.adapter = result
    _account(cfg, credit, decision, result)
    if result.permission_denials:
        print(f"⚠ {decision.channel}: {result.permission_denials} tool call(s) denied by policy", file=sys.stderr)
    if read_only and out.touched:  # S02: a read-only run must leave the tree untouched
        out.status = TaskStatus.PERMISSION_DENIED
        out.message = f"read-only run modified {', '.join(out.touched)}: revert with ai --undo"
        return out
    if result.status is not None:
        out.status = result.status
        out.message = result.clarification or result.error or ""
        out.rotate = result.channel_exhausted
        return out
    if read_only:
        out.status = TaskStatus.COMPLETED_UNVERIFIED
        return out
    if not out.touched:  # Invariant #4: nothing changed, so nothing was verified (a whole-project L2 would pass)
        if result.permission_denials:  # e.g. headless agy denies every shell command: this channel can't do it
            out.status, out.rotate = TaskStatus.PERMISSION_DENIED, True
            out.message = f"the agent stopped after {result.permission_denials} denied tool call(s) without changing any file"
        else:
            out.status, out.message = TaskStatus.COMPLETED_UNVERIFIED, "the agent finished without changing any file"
        return out
    base = Path(cp.base)
    out.verification = verify(root, cfg, [base / r for r, sha in cp.touched_files.items() if sha is not None], baseline)
    out.status = out.verification.status
    if out.status == TaskStatus.TESTS_FAILED and not read_only:
        if cfg.cloud_auto_rollback_on_failure:
            try:
                undo_rep = safe_undo(root, cfg, cp.checkpoint_id, confirm=lambda _msg: True)
                if undo_rep.skipped:
                    out.status = TaskStatus.PARTIALLY_APPLIED
                    out.message = f"tests failed ({', '.join(out.verification.new_failures)}): partially rolled back ({len(undo_rep.restored)} restored, {len(undo_rep.skipped)} skipped)"
                else:
                    out.status = TaskStatus.ROLLED_BACK
                    out.message = f"tests failed ({', '.join(out.verification.new_failures)}): changes safely rolled back to checkpoint {cp.checkpoint_id}"
            except Exception as roll_exc:
                out.message = f"tests failed ({', '.join(out.verification.new_failures)}); rollback error: {roll_exc}"
    return out


def _account(cfg: Config, credit: CreditManager | None, decision: Decision, result: AdapterResult) -> None:
    if credit is None:
        return
    try:
        if decision.channel == "claude":
            # modelUsage keys are the real model IDs (aliases like "sonnet" resolve server-side)
            credit.record_claude_result(result.total_cost_usd, channel_exhausted=result.channel_exhausted,
                                        error=result.error or "", model=",".join(result.model_usage) or decision.model)
            if result.rate_limit:
                credit.record_claude_rate_limit(result.rate_limit)
        else:
            bucket = bucket_for_model(decision.model)
            if result.channel_exhausted:
                until = credit.mark_b_cooldown(bucket, result.error or "")
                print(f"⚠ Channel B bucket '{bucket}' cooling down until {until:%H:%M} UTC", file=sys.stderr)
            tokens = sum(int(u.get("total_tokens") or 0) for u in result.model_usage.values() if isinstance(u, dict))
            credit.record_b_usage(bucket, tokens, decision.model)
    except CreditError as exc:
        print(f"⚠ credit accounting failed: {exc}", file=sys.stderr)


def _tree_fingerprint(root: Path, cfg: Config) -> dict[str, tuple[int, int]]:
    """{rel: (size, mtime_ns)} of the working tree (pruned), to prove a read-only run changed nothing."""
    pruned = {d.lower() for d in cfg.checkpoint_exclude_dirs} | {".git", CONTEXT_DIR_NAME}
    out: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in pruned]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                continue
            out[_rel(path, root)] = (st.st_size, st.st_mtime_ns)
    return out


def _guard_state_files(root: Path) -> dict[Path, bytes]:
    """R06: cloud subprocesses may never alter config.toml / blackboard.json."""
    ctx = root / CONTEXT_DIR_NAME
    return {p: p.read_bytes() for p in (ctx / "config.toml", ctx / "blackboard.json") if p.is_file()}


def _restore_state_files(guarded: dict[Path, bytes]) -> None:
    for path, data in guarded.items():
        try:
            if not path.is_file() or path.read_bytes() != data:
                write_bytes_atomic(path, data)
                print(f"⚠ {path.name} was altered by the subprocess and has been restored", file=sys.stderr)
        except OSError as exc:
            print(f"⚠ could not verify/restore {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------- dispatch with rotation (Q03)


def dispatch(opts: TaskOptions, cfg: Config, ctx: TaskContext, decision: Decision, task_id: str,
             credit: CreditManager | None, **cloud_kwargs: Any) -> Outcome:
    attempts: list[tuple[str, int]] = []
    out = _dispatch_loop(opts, cfg, ctx, decision, task_id, credit, attempts, **cloud_kwargs)
    out.attempts = attempts
    return out


def _raise_depth(ctx: TaskContext) -> None:
    ctx.routing.reasoning_depth = min(4, ctx.routing.reasoning_depth + 1)


def _add_context(cloud_kwargs: dict[str, Any], tag: str, text: str) -> None:
    cloud_kwargs["extra_context"] = "\n".join(s for s in (cloud_kwargs.get("extra_context", ""),
                                                           f"<{tag}>\n{text}\n</{tag}>") if s)


def _roll_back(opts: TaskOptions, cfg: Config, out: Outcome) -> bool:
    """Undo a cloud attempt before a repair retry; False (and `out` explains) if it can't be fully undone."""
    if not out.checkpoint_id:
        return False
    try:
        report = safe_undo(opts.project_root, cfg, out.checkpoint_id, confirm=lambda _msg: True)
    except CheckpointError as exc:
        out.message += f"; not retried: rollback failed ({exc})"
        return False
    if report.skipped:
        out.status = TaskStatus.PARTIALLY_APPLIED
        out.message += f"; not retried: {len(report.skipped)} file(s) could not be rolled back"
        return False
    out.status, out.touched = TaskStatus.ROLLED_BACK, []
    return True


def _dispatch_loop(opts: TaskOptions, cfg: Config, ctx: TaskContext, decision: Decision, task_id: str,
                   credit: CreditManager | None, attempts: list[tuple[str, int]], **cloud_kwargs: Any) -> Outcome:
    tried: set[str] = set()
    discovered: set[str] = set()  # files the local attempt proved the change needs (base-relative, lower case)
    repaired = False
    base = git_toplevel(opts.project_root) or opts.project_root
    while True:
        emit(opts, "routing", channel=decision.channel, model=decision.model, reason=decision.reason)
        if opts.verbose:
            print(f"→ {decision.channel}/{decision.model or '-'}: {decision.reason} [routing: {ctx.routing.source}]",
                  file=sys.stderr)
        if decision.channel == "none":
            return Outcome(TaskStatus.QUOTA_EXHAUSTED, message=decision.reason)
        if decision.channel == "local":
            out = run_local(opts, cfg, ctx, cloud_kwargs.get("prompt"))
            if (score := attempt_score(out)) is not None:
                attempts.append((_stats_key(cfg.ollama_model, ctx.routing), score))
            if out.status != TaskStatus.PATCH_FAILED or is_offline():
                return out
            print(f"↗ local branch failed ({out.message}); escalating to the cloud", file=sys.stderr)
            tried.add("local")
            for raw in out.wanted_files:  # the local attempt found the change spans these files too
                try:
                    found = validate_target_files([raw], opts.project_root, cfg)
                except PathValidationError:
                    continue  # hallucinated, secret or outside the project: never forwarded
                for p in found:
                    discovered.add(project_key(base, _rel(p, base)))
                    if p not in ctx.routing.target_files:
                        ctx.routing.target_files.append(p)
                        ctx.contents[p] = _read_text(p)
            ctx.tokens = _context_tokens(ctx, cloud_kwargs.get("prompt") or opts.prompt)
            _raise_depth(ctx)  # the scorer underestimated a task the local model could not do
            if out.message:  # tell the cloud agent what the local attempt hit, so it doesn't repeat it
                _add_context(cloud_kwargs, "local_attempt_failed", out.message)
        else:
            out = run_cloud(opts, cfg, ctx, decision, task_id, credit=credit, **cloud_kwargs)
            score = attempt_score(out)
            finished = out.status in _SCORE_OK or out.status == TaskStatus.COMPLETED_UNVERIFIED
            missing = sorted(discovered - {project_key(base, t) for t in out.touched}) if finished else []
            if missing:  # verified, yet files the change needs were left alone (e.g. compat shims instead)
                score = 0
                out.message = (f"incomplete: {', '.join(missing)} not changed, although the task needs it"
                               + (f"; {out.message}" if out.message else ""))
            if score is not None:
                attempts.append((_stats_key(decision.model, ctx.routing), score))
            if (out.status == TaskStatus.ROLLED_BACK or missing) and not repaired \
                    and cfg.cloud_auto_repair_enabled and not is_offline():
                if missing and not _roll_back(opts, cfg, out):
                    return out
                repaired = True
                failed_model = decision.model
                _raise_depth(ctx)
                _add_context(cloud_kwargs, "previous_cloud_attempt_failed", f"{failed_model}: {out.message}")
                decision = decide(cfg, ctx.routing, ctx.tokens, False, compute_channels(cfg, credit, opts.project_root),
                                  exclude=frozenset(tried), allow_local=False)
                if decision.channel == "none":
                    return out
                print(f"↻ {failed_model} failed ({out.message}); one repair attempt on {decision.model}",
                      file=sys.stderr)
                continue
            if not out.rotate or is_offline():
                return out
            if out.touched:  # partial edits exist: never stack another agent on them silently
                out.message += " — partial changes kept; continue with `ai -c` (rotates channel) or `ai --undo`"
                return out
            print(f"↻ {decision.channel} unavailable ({out.message}); rotating channel", file=sys.stderr)
            tried.add(decision.channel)
        channels = compute_channels(cfg, credit, opts.project_root)
        decision = decide(cfg, ctx.routing, ctx.tokens, False, channels, exclude=frozenset(tried),
                          allow_local="local" not in tried)
        if decision.channel == "none":
            return out


# ---------------------------------------------------------------- entry points


def run(opts: TaskOptions, cfg: Config) -> ExitCode:
    try:
        return _run(opts, cfg)
    except OfflineViolationError as exc:
        _print_outcome(opts, Outcome(TaskStatus.LOCAL_MODEL_UNAVAILABLE, "local", message=str(exc)))
        return ExitCode.OFFLINE_UNAVAILABLE


def _converse(opts: TaskOptions, cfg: Config, ctx: TaskContext) -> ExitCode:
    """Tier-0 chat answer: no checkpoint, no cloud, no patch."""
    decision = Decision("local", cfg.ollama_model, "conversational / direct Q&A [Tier-0 locale]")
    if opts.dry_run:
        return _dry_run_report(opts, cfg, ctx, decision)
    emit(opts, "routing", channel="local", model=cfg.ollama_model, reason=decision.reason)
    if opts.verbose:
        print(f"→ local/{cfg.ollama_model}: {decision.reason}", file=sys.stderr)
    _print_outcome(opts, run_conversational_chat(opts, cfg, ctx))
    return ExitCode.OK


def _preflight(opts: TaskOptions, cfg: Config) -> tuple[OllamaStatus | None, ExitCode | None]:
    """Phase 0: offline latch, vault journal recovery, Git state, Ollama reachability."""
    root = opts.project_root
    if opts.offline:
        enforce_offline()
        require_loopback_url(cfg.ollama_url)
    try:
        for msg in recover(root):
            if opts.verbose:
                print(f"  vault │ {msg}", file=sys.stderr)
    except OSError as exc:
        print(f"⚠ vault journal check failed: {exc}", file=sys.stderr)
    top = git_toplevel(root)
    if top is not None and (op := git_operation_in_progress(top)):
        say(opts, f"✖ Git {op} in progress: complete or abort it before running ai")
        return None, ExitCode.FAILED
    ollama = ollama_status(cfg, timeout=3.0 if opts.offline else 1.5)
    if not ollama.reachable and start_ollama(cfg) is None:
        say(opts, "ℹ Ollama was down: started `ollama serve`")
        ollama = ollama_status(cfg, timeout=3.0)
    if opts.offline and not (ollama.reachable and ollama.model_present):
        why = ollama.error or f"model {cfg.ollama_model} not installed"
        _print_outcome(opts, Outcome(TaskStatus.LOCAL_MODEL_UNAVAILABLE, "local", cfg.ollama_model, why))
        return None, ExitCode.OFFLINE_UNAVAILABLE
    if ollama.gpu_full is False and opts.verbose:
        print("⚠ local_degraded: model partially spilled to CPU RAM (check `ollama ps`)", file=sys.stderr)
    if ollama.reachable and opts.verbose and (bad := ollama_env.issues(ollama_env.server_env())):
        print(f"⚠ Ollama server settings (§2.A): {'; '.join(bad)}", file=sys.stderr)
    return ollama, None


def _run(opts: TaskOptions, cfg: Config) -> ExitCode:
    root = opts.project_root
    try:
        refuse_slash_command(opts.prompt, "claude/agy")
    except AdapterError as exc:  # before any dispatch or rotation: nothing to route
        say(opts, f"✖ {exc}")
        return ExitCode.USAGE
    ollama, early = _preflight(opts, cfg)
    if early is not None:
        return early
    assert ollama is not None
    if ollama.gpu_full is False:
        cfg = dataclasses.replace(cfg, ollama_num_ctx=min(cfg.ollama_num_ctx, 16384))
        if opts.verbose:
            print(f"⚠ local_degraded: auto-reduced ollama_num_ctx to {cfg.ollama_num_ctx} (L03)", file=sys.stderr)
    # Conversational / Direct Q&A fast-path (Tier-0 locale, zero cloud costs)
    if not opts.file_args and is_conversational_prompt(opts.prompt):
        blackboard, _ = load_blackboard(root)
        ctx = TaskContext(default_routing([]), {}, load_profile(cfg),
                          blackboard_context(blackboard), "", 0)
        ctx.tokens = estimate_tokens(opts.prompt)
        return _converse(opts, cfg, ctx)

    if opts.continue_session:
        return _continue(opts, cfg, ollama)

    # Phase 1 - targets, repo-map and budget
    try:
        new_dirs: dict[str, bool] = {}
        targets = validate_target_files(opts.file_args, root, cfg, creates_new_files=True,
                                        confirm_new_dir=new_dir_confirmer(opts, new_dirs))
    except SecretPathError as exc:
        _print_outcome(opts, Outcome(TaskStatus.SECRET_DETECTED_ABORTED, message=str(exc)))
        return ExitCode.SECRET_BLOCKED
    except PathValidationError as exc:
        say(opts, f"✖ {exc}")
        return ExitCode.USAGE
    blackboard, bb_writable = load_blackboard(root)
    recent_diff = ""
    if not targets and not opts.file_args and blackboard.get("recent_tasks") and _is_followup_prompt(opts.prompt):
        last_t = blackboard["recent_tasks"][0]
        inherited = [root / f for f in last_t.get("touched_files", []) if (root / f).is_file()]
        if inherited:
            targets = inherited
            if opts.verbose:
                print(f"→ Inherited {len(targets)} target file(s) from recent task '{last_t.get('task_id')}': {', '.join(last_t.get('touched_files', []))}", file=sys.stderr)
        if last_t.get("checkpoint_id"):
            try:
                recent_diff = diff_since(root, last_t["checkpoint_id"], cfg, max_chars=1500)
            except Exception:
                pass
    if not opts.continue_session:
        pending = blackboard.get("pending_sessions", {})
        if isinstance(pending, dict):
            for sid, sess in pending.items():
                if isinstance(sess, dict) and sess.get("status") == TaskStatus.NEEDS_CLARIFICATION:
                    sess_targets = {project_key(root, p) for p in sess.get("target_files", [])}
                    curr_targets = {project_key(root, _rel(p, root)) for p in targets}
                    if overlap := sess_targets & curr_targets:
                        print(f"⚠ warning: task touches file(s) from suspended session '{sid}' "
                              f"(in needs_clarification): {', '.join(sorted(overlap))}. "
                              f"Modifying them may invalidate that session (§10.1).", file=sys.stderr)
    working_set = list(blackboard.get("active_working_set", {}).get("files", []))
    ctx = TaskContext(default_routing(targets), {p: _read_text(p) for p in targets}, load_profile(cfg),
                      blackboard_context(blackboard), _repo_map(root, cfg, targets, working_set), 0, new_dirs)
    ctx.tokens = _context_tokens(ctx, opts.prompt)

    # Phase 2 - scoring (local, free); default routing when Ollama is down and cloud is allowed
    if ollama.reachable and ollama.model_present:
        ctx.routing = score_task(cfg, opts.prompt, root, targets, ctx.contents, ctx.tokens,
                                 cold=ollama.gpu_full is None, repo_map=ctx.repo_map,
                                 confirm_new_dir=new_dir_confirmer(opts, ctx.new_dir_answers))
    if ctx.routing.conversational and not targets and not opts.extra_directives:
        return _converse(opts, cfg, ctx)  # a question the keyword fast path missed ("Puoi creare file?")
    if opts.extra_directives and ctx.routing.task_type != "debug":
        ctx.routing.task_type = "debug"  # --fix is debugging by definition
    for p in ctx.routing.target_files:
        ctx.contents.setdefault(p, _read_text(p))
    ctx.tokens = _context_tokens(ctx, opts.prompt)
    if opts.offline and ctx.tokens > cfg.local_max_input_tokens:
        if ctx.repo_map and targets:
            compact_map = _repo_map(root, cfg, targets, working_set, only_focus=True)
            if compact_map != ctx.repo_map:
                ctx.repo_map = compact_map
                ctx.tokens = _context_tokens(ctx, opts.prompt)
        if ctx.tokens > cfg.local_max_input_tokens:
            _print_outcome(opts, Outcome(TaskStatus.CONTEXT_EXCEEDED_OFFLINE, "local", cfg.ollama_model,
                                         f"~{ctx.tokens} tokens > local_max_input_tokens={cfg.local_max_input_tokens}"))
            return ExitCode.OFFLINE_UNAVAILABLE

    # Phase 3 - decision
    credit = None if is_offline() else CreditManager(cfg)
    decision = decide(cfg, ctx.routing, ctx.tokens, is_offline(), compute_channels(cfg, credit, opts.project_root), forced_channel=opts.channel,
                      stats=blackboard.get("routing_stats") if cfg.adaptive_routing else None)
    if opts.dry_run:
        return _dry_run_report(opts, cfg, ctx, decision)

    ensure_context_dir(root)
    task_id = uuid.uuid4().hex[:12]
    try:
        with ProjectLock(root, task_id, cfg.lock_wait_timeout_s):
            extra_ctx = f"<recent_task_diff>\n{recent_diff}\n</recent_task_diff>" if recent_diff else ""
            outcome = dispatch(opts, cfg, ctx, decision, task_id, credit, extra_context=extra_ctx)
            outcome.task_id = task_id
            if bb_writable:
                _persist(opts, cfg, ollama, ctx, outcome, task_id, opts.prompt)
    except ProjectBusyError as exc:
        outcome = Outcome(TaskStatus.PROJECT_BUSY, message=str(exc), task_id=task_id)
    _print_outcome(opts, outcome)
    try:
        from ai_core.statusline import check_and_spawn_background_usage_refresh
        check_and_spawn_background_usage_refresh(cfg)
    except Exception:
        pass
    return EXIT_FOR_STATUS.get(outcome.status, ExitCode.FAILED)


def _persist(opts: TaskOptions, cfg: Config, ollama: OllamaStatus, ctx: TaskContext, outcome: Outcome,
             task_id: str, prompt: str, answer_source: str | None = None) -> None:
    """Blackboard: session turn, working set, Qwen ADR proposals (M01). Re-read to merge, never clobber."""
    root = opts.project_root
    bb, writable = load_blackboard(root)
    if not writable:
        return
    session = record_turn(
        bb, task_id, channel=outcome.channel, model=outcome.model,
        session_id=outcome.adapter.session_id if outcome.adapter else None, status=str(outcome.status),
        question=outcome.message if outcome.status == TaskStatus.NEEDS_CLARIFICATION else None,
        checkpoint_id=outcome.checkpoint_id, prompt=prompt,
        target_files=[_rel(p, root) for p in ctx.routing.target_files],
        cost_usd=outcome.adapter.total_cost_usd if outcome.adapter else None)
    if answer_source:
        session.setdefault("answers", []).append({"text": opts.prompt[:2000], "source": answer_source})
    update_working_set(bb, outcome.touched, outcome.verification.new_failures if outcome.verification else [])
    if cfg.adaptive_routing and ctx.routing.source == "ollama":  # default routing (ai -c, scoring down) is a guess
        record_attempts(bb, outcome.attempts)
    record_recent_task(
        bb, task_id, prompt=prompt, channel=outcome.channel, model=outcome.model,
        status=str(outcome.status), touched_files=outcome.touched,
        checkpoint_id=outcome.checkpoint_id, summary=outcome.message,
    )
    ok = EXIT_FOR_STATUS.get(outcome.status, ExitCode.FAILED) == ExitCode.OK
    if ok and outcome.touched and outcome.checkpoint_id and ctx.routing.task_type in ADR_TASK_TYPES \
            and ollama.reachable and ollama.model_present:
        try:
            diff = diff_since(root, outcome.checkpoint_id, cfg)
            chat = lambda msgs, **kw: ollama_chat(cfg, msgs, timeout=_ADR_TIMEOUT_S, **kw)  # noqa: E731
            for prop in adrmod.extract_proposals(chat, diff, prompt):
                adr = adrmod.add_decision(bb, prop["title"], prop["rationale"], "qwen_suggestion", task_id,
                                          cfg.adr_require_approval)
                outcome.proposed_adrs.append(adr["id"])
        except (OllamaError, CheckpointError, ValueError) as exc:
            print(f"⚠ ADR extraction skipped: {exc}", file=sys.stderr)
    try:
        save_blackboard(root, bb)
    except OSError as exc:
        print(f"⚠ blackboard not updated: {exc}", file=sys.stderr)


# ---------------------------------------------------------------- ai -c (sessions, Q03 rotation)


def _channel_usable(cfg: Config, channel: str, model: str, channels: Channels) -> bool:
    if channel == "claude":
        return channels.a_state in (AVAILABLE, LIMITED)
    if channel == "agy":
        return bucket_for_model(model) in channels.b_ready
    return channel == "local"


def _continue(opts: TaskOptions, cfg: Config, ollama: OllamaStatus) -> ExitCode:
    root = opts.project_root
    bb, bb_writable = load_blackboard(root)
    try:
        task_id, s = find_session(bb, opts.task_id)
    except SessionError as exc:
        say(opts, f"✖ {exc}")
        return ExitCode.USAGE
    targets: list[Path] = []
    for raw in s.get("target_files", []):
        try:
            targets += validate_target_files([raw], root, cfg, creates_new_files=True)
        except PathValidationError:
            continue
    working_set = list(bb.get("active_working_set", {}).get("files", []))
    routing = default_routing(targets)
    ctx = TaskContext(routing, {p: _read_text(p) for p in targets}, load_profile(cfg), blackboard_context(bb),
                      _repo_map(root, cfg, targets, working_set), 0)
    ctx.tokens = _context_tokens(ctx, opts.prompt)
    credit = None if is_offline() else CreditManager(cfg)
    channels = compute_channels(cfg, credit, opts.project_root)
    channel, model, sid = s.get("channel", ""), s.get("model", ""), s.get("session_or_conversation_id")
    question = s.get("pending_question") or ""
    history = (f"<previous_turn>\n<original_task>\n{s.get('prompt', '')}\n</original_task>\n"
               f"<agent_question>\n{question}\n</agent_question>\n</previous_turn>")
    kwargs: dict[str, Any] = {}
    if is_offline() or channel == "local":
        decision = Decision("local", cfg.ollama_model, "continue on Tier-0")
        kwargs["prompt"] = f"{s.get('prompt', '')}\n\nClarification answer: {opts.prompt}"
    elif sid and _channel_usable(cfg, channel, model, channels):
        decision = Decision(channel, model, f"resume session {sid}")
        kwargs.update(resume_id=sid, extra_context="<context_delta>\n" + json.dumps(
            bb.get("active_working_set", {}), ensure_ascii=False) + "\n</context_delta>")
    else:
        rotate_ok = cfg.rotate_session_on_cooldown if opts.agent else _confirm(
            opts, f"Session channel '{channel}' is exhausted/cooling down. Rotate to the alternative channel?")
        if not rotate_ok:
            _print_outcome(opts, Outcome(TaskStatus.QUOTA_EXHAUSTED, channel, model,
                                         f"channel {channel} unavailable and rotation declined", task_id=task_id))
            return ExitCode.QUOTA_EXHAUSTED
        decision = decide(cfg, routing, ctx.tokens, False, channels, exclude=frozenset({channel}), allow_local=False)
        diffs = [diff_since(root, cid, cfg) for cid in dict.fromkeys(
            c for c in (s.get("origin_checkpoint_id"), s.get("latest_checkpoint_id")) if c)]
        kwargs["extra_context"] = history + "\n<changes_so_far>\n" + "\n".join(d for d in diffs if d) + \
            "\n</changes_so_far>\nA previous agent session was interrupted; continue the original task."
    try:
        with ProjectLock(root, task_id, cfg.lock_wait_timeout_s):
            outcome = dispatch(opts, cfg, ctx, decision, task_id, credit, **kwargs)
            outcome.task_id = task_id
            if bb_writable:
                _persist(opts, cfg, ollama, ctx, outcome, task_id, s.get("prompt", opts.prompt),
                         answer_source="agent_external" if opts.agent else "user_interactive")
    except ProjectBusyError as exc:
        outcome = Outcome(TaskStatus.PROJECT_BUSY, message=str(exc), task_id=task_id)
    _print_outcome(opts, outcome)
    return EXIT_FOR_STATUS.get(outcome.status, ExitCode.FAILED)


# ---------------------------------------------------------------- reporting


def _dry_run_report(opts: TaskOptions, cfg: Config, ctx: TaskContext, decision: Decision) -> ExitCode:
    root = opts.project_root
    routing = ctx.routing
    print(f"→ {decision.channel}/{decision.model or '-'}: {decision.reason} [routing: {routing.source}]", file=sys.stderr)
    say(opts, f"dry-run: no checkpoint, no dispatch\n  tokens≈{ctx.tokens}  depth={routing.reasoning_depth}  "
              f"scope={routing.files_scope}  type={routing.task_type}  repo-map≈{estimate_tokens(ctx.repo_map)} tokens")
    say(opts, f"  targets: {', '.join(_rel(p, root) for p in routing.target_files) or '(none)'}")
    hits = scan_secrets(opts.prompt, "<prompt>") + [
        f for p in routing.target_files for f in scan_secrets(ctx.contents.get(p) or "", _rel(p, root))]
    for finding in hits:
        say(opts, f"  ⛔ secret: {finding}")
    command: list[str] = []
    if decision.channel in ("claude", "agy"):
        try:
            adapter = _adapter(cfg, decision.channel)
        except AdapterError as exc:
            say(opts, f"  ✖ {decision.channel} unavailable: {exc}")
            return ExitCode.FAILED
        req = TaskRequest("dry-run", opts.prompt, root, decision.model, system_append="<static profile+directives>")
        command = adapter.build_command(req)
        say(opts, f"  command: {subprocess.list2cmdline(command)}")
        if decision.channel == "agy" and (secret_files := find_secret_files(root, cfg)):
            say(opts, f"  vault: {len(secret_files)} secret file(s) would be isolated on the same volume")
    emit(opts, "dry_run", channel=decision.channel, model=decision.model, reason=decision.reason, tokens=ctx.tokens,
         targets=[_rel(p, root) for p in routing.target_files], secrets=[str(h) for h in hits], command=command)
    return ExitCode.SECRET_BLOCKED if hits and decision.channel in ("claude", "agy") else ExitCode.OK


def _print_outcome(opts: TaskOptions, o: Outcome) -> None:
    code = EXIT_FOR_STATUS.get(o.status, ExitCode.FAILED)
    if opts.agent:
        payload: dict[str, Any] = {
            "task_id": o.task_id or None, "task_status": str(o.status), "exit_code": int(code),
            "process_exit_code": o.adapter.process_exit_code if o.adapter else None,
            "verification_status": str(o.verification.status) if o.verification else None,
            "verification_level": o.verification.verification_level if o.verification else 0,
            "channel": o.channel or None, "model": o.model or None,
            "session_id": o.adapter.session_id if o.adapter else None,
            "cost_usd": o.adapter.total_cost_usd if o.adapter else None,
            "checkpoint_id": o.checkpoint_id, "touched_files": o.touched, "message": o.message,
            "result_text": (o.adapter.result_text[:4000] if o.adapter else ""), "proposed_adrs": o.proposed_adrs,
        }
        if o.status == TaskStatus.NEEDS_CLARIFICATION:
            payload["clarification"] = {"question": o.message,
                                        "resume": f'ai -c --task-id {o.task_id} --agent "<answer>"'}
        emit(opts, "result", **payload)
        return

    # Extract body cleanly without duplication
    body = ""
    if o.adapter and o.adapter.result_text:
        body = o.adapter.result_text.strip()
    elif o.message:
        body = o.message.strip()

    body = render_terminal_markdown(body, _use_color())

    global LAST_ANSWERED_BY
    LAST_ANSWERED_BY = (f"{o.channel}/{o.model}" if o.model else o.channel) if o.channel else ""

    # Case 1: Pure conversational Q&A without file modifications
    if not o.touched and code == ExitCode.OK:
        if body:
            print(f"\n{_reply(body)}" + ("" if LAST_ANSWERED_BY else "\n"))
        if LAST_ANSWERED_BY:
            print(_dim(f"\n  {_safe_char(chr(0x2570), '`')} answered by ") + _cyan(LAST_ANSWERED_BY) + "\n")
        return

    # Case 2: File modifications or errors -> show reply body then clean summary card
    if body and o.status != TaskStatus.NEEDS_CLARIFICATION and body != o.message:
        print(f"\n{_reply(body)}\n")

    width = 62
    char = _safe_char("\u2500", "-")
    dot = _safe_char(" \u00b7 ", " | ")
    divider = _dim(char * width)
    print(divider)
    level_str = f" (L{o.verification.verification_level})" if o.verification else ""
    mark = _green(_safe_char("\u2714", "[OK]")) if code == ExitCode.OK else _red(_safe_char("\u2716", "[X]"))
    status_str = _bold(f"{mark} {o.status}{level_str}")
    ch_info = _cyan(f"{o.channel}/{o.model}") if o.channel else ""
    print(f" {status_str}" + (f"{dot}{ch_info}" if ch_info else ""))

    if o.status == TaskStatus.NEEDS_CLARIFICATION:
        print(f"   {_yellow('?')} {o.message}")
        print(f"   {_dim('answer with:')} ai -c --task-id {o.task_id} \"<your answer>\"")
    elif code != ExitCode.OK and o.message:
        print(f"   {_red('error:')} {o.message}")

    if o.touched:
        print(f"   {_bold('touched:')}    {', '.join(o.touched)}")
    if o.checkpoint_id and o.touched:
        undo_cmd = _dim(f"revert with: ai --undo")
        print(f"   {_bold('checkpoint:')} {_dim(o.checkpoint_id)} ({undo_cmd})")

    details = []
    if o.adapter and o.adapter.total_cost_usd is not None:
        details.append(f"cost ${_dim(f'{o.adapter.total_cost_usd:.4f}')}")
    if o.adapter and o.adapter.process_exit_code is not None and o.adapter.process_exit_code != 0:
        details.append(f"exit {o.adapter.process_exit_code}")
    if details:
        print(f"   {_bold('usage:')}      {dot.join(details)}")

    for adr_id in o.proposed_adrs:
        print(f"   {_yellow('proposed:')}   {adr_id} ({_dim('review: ai --adr approve ' + adr_id)})")

    print(divider)


def show_status(root: Path, cfg: Config, as_json: bool = False) -> ExitCode:
    st = ollama_status(cfg)
    info: dict[str, Any] = {
        "project": str(root),
        **({"wsl": f"{target.distro}:{target.linux_path}"} if (target := wsl_target(root)) else {}),
        "billing_env_vars": detected_billing_vars(os.environ),
        "ollama": {"reachable": st.reachable, "model": cfg.ollama_model, "installed": st.model_present,
                   "gpu": {True: "100% GPU", False: "local_degraded", None: "not loaded"}[st.gpu_full],
                   "error": st.error, "server_env_issues": ollama_env.issues(ollama_env.server_env())},
    }
    for name, factory in (("claude", _claude_adapter), ("agy", AntigravityAdapter)):
        try:
            info[name] = factory(cfg).probe()  # type: ignore[attr-defined]
        except AdapterError as exc:
            info[name] = f"unavailable: {exc}"
    try:
        info["credit"] = CreditManager(cfg).summary()
    except CreditError as exc:
        info["credit"] = f"unavailable: {exc}"
    ids: list[str] = []
    latest = None
    try:
        ids = list_checkpoints(root)
        latest = load_checkpoint(root) if ids else None
    except CheckpointError:
        pass  # no undoable checkpoint (or store unusable): reported as "none"
    bb, _ = load_blackboard(root)
    pending = {k: v.get("pending_question") for k, v in bb.get("pending_sessions", {}).items()
               if v.get("status") == str(TaskStatus.NEEDS_CLARIFICATION)}
    info.update(checkpoints=len(ids), undoable=latest.checkpoint_id if latest else None,
                last_session=bb.get("system_identity", {}).get("last_session"), pending_clarifications=pending,
                proposed_adrs=[a["id"] for a in adrmod.list_decisions(bb, adrmod.PROPOSED)])
    if as_json:
        print(json.dumps(info, ensure_ascii=False, default=str))
        return ExitCode.OK
    mark = lambda ok: _green(_safe_char("\u2714", "ok")) if ok else _red(_safe_char("\u2716", "!!"))  # noqa: E731
    print(_bold("CodingEngine status") + _dim(f"  {root}"))
    if info["billing_env_vars"]:
        print(_yellow(f"⚠ API/billing variables in environment (stripped from every subprocess): {', '.join(info['billing_env_vars'])}"))
    o = info["ollama"]
    print(_section("Local"))
    print(_kv("ollama", f"{mark(o['reachable'])} {'up' if o['reachable'] else 'DOWN'}  {_dim(cfg.ollama_url)}"))
    print(_kv("model", f"{_cyan(str(o['model']))}  {mark(o['installed'])} {'installed' if o['installed'] else 'MISSING'}  "
                       f"{_dim(str(o['gpu'] if o['reachable'] else o['error']))}"))
    if o["server_env_issues"]:
        print(_yellow(f"  ⚠ server settings: {'; '.join(o['server_env_issues'])}"))
        print(_dim(f"    fix: {ollama_env.fix_hint()}"))
    print(_section("Channels"))
    print(_kv("claude", f"{info['claude']} {_dim(f'(budget cap ${cfg.claude_max_budget_usd_per_task:.2f}/task)')}"))
    print(_kv("agy", str(info["agy"])))
    if isinstance(info["credit"], dict):
        a = info["credit"]["channel_a"]
        pct_used = 100 - float(a["remaining_pct"])
        print(_kv("channel A", f"{a['state']}  {_bar(pct_used)} {_dim(f'${a['spent_usd']:.4f} of ${a['limit_usd']:.2f}, {a['remaining_pct']}% left, resets {a['next_reset']}')}"))
        for bucket, b in info["credit"]["channel_b"].items():
            state = _green("ready") if b["ready"] else _yellow("cooldown until " + str(b.get("cooldown_until")))
            print(_kv(f"B/{bucket}", f"{state} {_dim(f'(~{b.get('tokens_estimated', 0)} tokens, estimated)')}"))
    else:
        print(_kv("credit", str(info["credit"])))
    print(_section("Project"))
    print(_kv("checkpoints", f"{len(ids)} stored {_dim('· undoable:')} {info['undoable'] or 'none'}"))
    for tid, q in pending.items():
        print(_kv("pending", f"{_yellow(str(tid))}: {q}  {_dim(f'→ ai -c --task-id {tid} \"<answer>\"')}"))
    if info["proposed_adrs"]:
        print(_kv("ADRs", f"{', '.join(info['proposed_adrs'])}  {_dim('→ ai --adr list')}"))
    ls = info["last_session"]
    print(_kv("last task", _dim(json.dumps(ls, ensure_ascii=False)) if ls else _dim("none")))
    return ExitCode.OK


def show_usage(cfg: Config, *, as_json: bool = False, sync_claude: float | None = None,
               b_cooldown: tuple[str, int] | None = None, refresh: bool = False,
               install_hooks: bool = False) -> ExitCode:
    """`ai --usage`: REAL plan windows first, then the orchestrator's own ledger.

    Real sources (all free, verified 2026-10-25): claude `rate_limit_event` in every `-p` stream and
    Claude Code's status-line JSON; agy's status-line JSON `quota` block. `/usage` itself is never sent
    headlessly (both CLIs forward it to the model and pay for it).
    """
    from ai_core import statusline

    if install_hooks:
        exe = HOME / ".local" / "bin" / "ai.exe"
        if not exe.is_file():
            print("✖ ai.exe not found: run `uv tool install --editable .` first", file=sys.stderr)
            return ExitCode.FAILED
        for cli in statusline.CLIS:
            print(f"  {statusline.install_hook(cli, exe)}", file=sys.stderr)
    if refresh:
        print("… opening agy without any prompt until its status line reports quotas", file=sys.stderr)
        try:
            ok = statusline.refresh_agy(HOME)
        except OSError as exc:
            ok = False
            print(f"⚠ agy refresh failed: {exc}", file=sys.stderr)
        print(f"{'✔' if ok else '⚠'} agy quotas {'refreshed' if ok else 'not refreshed (is the hook installed? ai --usage --install-hooks)'}"
              "\n  claude windows refresh by themselves on every dispatch and interactive Claude Code session",
              file=sys.stderr)
    credit = CreditManager(cfg)
    try:
        if sync_claude is not None:
            previous = credit.sync_claude_spent(sync_claude)
            print(f"✔ Channel A counter synced: ${previous:.4f} → ${sync_claude:.4f}", file=sys.stderr)
        if b_cooldown is not None:
            bucket, minutes = b_cooldown
            if bucket not in ("gemini", "argon", "claude_gpt"):
                raise CreditError(f"unknown bucket {bucket!r} (gemini | argon | claude_gpt)")
            until = credit.mark_b_cooldown(bucket, "set manually from agy /usage", minutes)
            print(f"✔ bucket {bucket}: " + (f"cooldown until {until:%Y-%m-%d %H:%M} UTC" if minutes else "cleared"),
                  file=sys.stderr)
        report = credit.usage_report()
    except CreditError as exc:
        print(f"✖ {exc}", file=sys.stderr)
        return ExitCode.USAGE
    if as_json:
        print(json.dumps(report, ensure_ascii=False))
        return ExitCode.OK
    def when(iso: str | None) -> str:
        return datetime.fromisoformat(iso).astimezone().strftime("%a %d/%m %H:%M") if iso else "?"

    def age(iso: str | None) -> str:
        if not iso:
            return "never captured"
        minutes = int((datetime.now(UTC) - datetime.fromisoformat(iso)).total_seconds() // 60)
        return f"captured {minutes} min ago" if minutes < 120 else f"captured {minutes // 60} h ago"

    real = report["real"]
    rc, ra = real["claude"], real["agy"]

    def windows(rows: list[tuple[str, float, str, str]]) -> None:
        for name, used, resets, flag in rows:
            print(f"    {name:<26} {_bar(used)} {used:5.1f}%  {_dim('resets ' + resets)}{flag}")

    rejecting = _red(" · REJECTING") if rc["status"] == "rejected" else ""
    print(_bold("Claude plan windows") + _dim(f"  {age(rc['captured_at'])}") + rejecting)
    windows([(n, w["used_pct"], when(w["resets_at"]), "") for n, w in rc["windows"].items()])
    if not rc["windows"]:
        print(_dim("    (none yet: filled by the next claude dispatch or interactive Claude Code session)"))
    plan = f", {ra['plan']}" if ra["plan"] else ""
    print(_section("Antigravity quotas") + _dim(f"  {age(ra['captured_at'])}{plan}"))
    windows([(n, 100 - w["remaining_pct"], when(w["resets_at"]),
              _red("  ⛔ blocks routing") if w["remaining_pct"] <= cfg.agy_min_remaining_pct else "")
             for n, w in sorted(ra["windows"].items())])
    if not ra["windows"]:
        print(_dim("    (none yet: ai --usage --refresh)"))
    a = report["channel_a"]
    print(_section("Channel A") + _dim(f"  {a['state']} — {a['reason']}"))
    print(_dim(f"    ledger ${a['spent_usd']:.4f} list-price equivalent since {a['cycle_start']} (usage credits off: not a charge)"))
    for model, m in sorted(a["by_model"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        label = "manual sync" if model == "sync" else model
        calls = "" if model == "sync" else f"{m['calls']:>4} calls "
        print(f"    {_cyan(f'{label:<34}')} {_dim(calls)}${m['cost_usd']:.4f}")
    print(_section("Channel B") + _dim("  routing buckets (agy's own usage field)"))
    for bucket, b in sorted(report["channel_b"].items()):
        state = _yellow(f"blocked: {b['blocked']}") if b.get("blocked") else _green("ready")
        print(f"    {_cyan(f'{bucket:<11}')} {state}")
        print(_dim(f"    {'':<11} 5h: {b['calls_5h']} calls / {b['tokens_5h']:,} tok   7d: {b['tokens_7d']:,} tok"))
    return ExitCode.OK


def undo(root: Path, cfg: Config, checkpoint_id: str | None, task_id: str | None = None,
         opts: TaskOptions | None = None) -> ExitCode:
    """`--undo [ID]` one checkpoint; `--undo --task-id X` every turn of a task, newest first (R05)."""
    opts = opts or TaskOptions("", root)
    ids: list[str | None] = [checkpoint_id]
    if task_id:
        bb, _ = load_blackboard(root)
        try:
            _, s = find_session(bb, task_id)
        except SessionError as exc:
            say(opts, f"✖ {exc}")
            return ExitCode.FAILED
        ids = list(reversed(s.get("checkpoint_ids", [])))
        if not ids:
            say(opts, f"✖ task {task_id} has no checkpoints")
            return ExitCode.FAILED
    skipped = 0
    for cid in ids:
        try:
            report = safe_undo(root, cfg, cid, confirm=lambda q: _confirm(opts, q))
        except CheckpointError as exc:
            say(opts, f"⚠ {exc}")
            skipped += 1
            continue
        if report.head_moved:
            say(opts, "⚠ HEAD moved since the task: Git index/history untouched; pre-task versions saved as *.ai_undo")
        for rel in report.restored:
            say(opts, f"  ↺ {rel}")
        for rel in report.saved_copies:
            say(opts, f"  ⎘ {rel}")
        for rel, why in report.skipped:
            say(opts, f"  ⚠ {rel}: {why}")
        skipped += len(report.skipped)
        say(opts, f"{'✔' if not report.skipped else '⚠'} undo of {report.checkpoint_id}: "
                  f"{len(report.restored)} restored, {len(report.skipped)} skipped")
        emit(opts, "undo", checkpoint_id=report.checkpoint_id, restored=report.restored,
             skipped=[r for r, _ in report.skipped], head_moved=report.head_moved)
    return ExitCode.OK if not skipped else ExitCode.FAILED


def gc(root: Path, cfg: Config) -> ExitCode:
    removed = gc_checkpoints(root, cfg)
    print(f"✔ removed {len(removed)} expired checkpoint(s)" + (f": {', '.join(removed)}" if removed else ""))
    return ExitCode.OK
