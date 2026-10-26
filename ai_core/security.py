"""Trust boundaries: API-key-free subprocess env (B01), offline latch (Invariant #1),
pre-dispatch secret scrubber (S01) and anti-traversal target validation (L06)."""
from __future__ import annotations

import fnmatch
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from ai_core.config import CONTEXT_DIR_NAME, Config

# ---------------------------------------------------------------- Invariant #5: zero API billing

API_KEY_VARS = frozenset({
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
})
# Any of these can silently switch a CLI to a metered backend (custom gateway, Bedrock,
# Vertex, Foundry), so they are stripped as well: if billing can't be excluded, block it.
_BILLING_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_", "AWS_BEARER_TOKEN_BEDROCK", "GOOGLE_GENAI_USE_VERTEXAI")
# Cloud CLIs (claude.exe / agy.exe) get no provider credentials or config at all: they must
# authenticate with their own OAuth login only. Local tools (git, the user's tests) keep AWS_* etc.
_CLOUD_CLI_PREFIXES = ("ANTHROPIC_", "CLAUDE_", "GOOGLE_", "GEMINI_", "AWS_")


def _is_billing_var(name: str) -> bool:
    upper = name.upper()  # Windows env names are case-insensitive
    return upper in API_KEY_VARS or upper.startswith(_BILLING_PREFIXES) or upper.endswith("_API_KEY")


def sanitize_subprocess_env(env: Mapping[str, str], *, warn: bool = True, cloud: bool = False) -> dict[str, str]:
    """Return a copy of `env` without API keys / metered-backend switches. Never mutates `env`.

    `cloud=True` (subprocesses that are cloud CLIs) additionally drops every ANTHROPIC_/CLAUDE_/
    GOOGLE_/GEMINI_/AWS_ variable. The warning names only real API-key/billing variables.
    """
    clean = {k: v for k, v in env.items()
             if not _is_billing_var(k) and not (cloud and k.upper().startswith(_CLOUD_CLI_PREFIXES))}
    removed = sorted(k for k in env if _is_billing_var(k))
    if removed and warn:
        print(
            "⚠ Chiave API rilevata nell'ambiente e rimossa per il sottoprocesso: uso esclusivo "
            f"dell'abbonamento OAuth ({', '.join(removed)})",
            file=sys.stderr,
        )
    return clean


def detected_billing_vars(env: Mapping[str, str]) -> list[str]:
    return sorted(k for k in env if _is_billing_var(k))


# ---------------------------------------------------------------- Invariant #1: --offline latch


class OfflineViolationError(RuntimeError):
    """A cloud channel was reached while --offline (or local-only secret mode) is latched."""


_offline_latched = False


def enforce_offline() -> None:
    """One-way latch: once set, no code path in this process can re-enable cloud channels."""
    global _offline_latched
    _offline_latched = True


def is_offline() -> bool:
    return _offline_latched


def require_cloud_allowed(channel: str) -> None:
    if _offline_latched:
        raise OfflineViolationError(f"cloud channel '{channel}' is physically disabled (--offline)")


def require_loopback_url(url: str) -> None:
    """Offline must not leave the machine: the Ollama endpoint has to be loopback."""
    host = (urlsplit(url).hostname or "").lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise OfflineViolationError(f"--offline requires a loopback Ollama URL, got host '{host}'")


# ---------------------------------------------------------------- Level 1: content scrubber

REDACTION = "<REDACTED_SECRET>"

# A named group `secret` limits redaction to that part (keeps `postgres://user:` readable).
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
        r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|\Z)")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("stripe_key", re.compile(r"\b[rs]k_live_[0-9A-Za-z]{20,}")),
    # scheme://user:password@host - placeholders like ${PASS} / <pass> / %s are not secrets
    ("connection_string_password", re.compile(
        r"\b[a-z][a-z0-9+.\-]*://[^\s:/@'\"]+:(?P<secret>[^\s@/'\"$<{%*][^\s@/'\"]*)@", re.I)),
    # ADO/ODBC style `...;Password=xyz;` (requires `;` context so `f(password=pw)` isn't flagged)
    ("connection_string_password", re.compile(
        r"(?:^|;)[ \t]*(?:password|pwd)[ \t]*=[ \t]*(?P<secret>[^;'\"\s$<{%]+)", re.I | re.M)),
)
_HIGH_ENTROPY_ASSIGNMENT = re.compile(
    r"(?i)[\w.\-]*(?:secret|token|api[_\-]?key|passwd|password|credential)[\w.\-]*['\"]?[ \t]*[:=][ \t]*"
    r"(?P<q>['\"])(?P<secret>[A-Za-z0-9+/=_\-]{20,})(?P=q)")
_MIN_ENTROPY_BITS = 3.5


def _shannon_entropy(value: str) -> float:
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in Counter(value).values())


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """Location of a suspected secret. Deliberately carries no secret value (never log it)."""

    source: str
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.source}:{self.line} [{self.kind}]"


def _secret_spans(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping (start, end, kind) spans to redact, ordered by position."""
    spans: list[tuple[int, int, str]] = []
    for kind, pattern in _SECRET_PATTERNS:
        for m in pattern.finditer(text):
            g = "secret" if "secret" in pattern.groupindex else 0
            spans.append((m.start(g), m.end(g), kind))
    for m in _HIGH_ENTROPY_ASSIGNMENT.finditer(text):
        if _shannon_entropy(m["secret"]) >= _MIN_ENTROPY_BITS:
            spans.append((m.start("secret"), m.end("secret"), "high_entropy_assignment"))
    spans.sort()
    merged: list[tuple[int, int, str]] = []
    for span in spans:
        if merged and span[0] < merged[-1][1]:
            continue
        merged.append(span)
    return merged


def scan_secrets(text: str, source: str = "<prompt>") -> list[SecretFinding]:
    return [SecretFinding(source, text.count("\n", 0, start) + 1, kind) for start, _, kind in _secret_spans(text)]


def redact_secrets(text: str) -> str:
    for start, end, _ in reversed(_secret_spans(text)):
        text = text[:start] + REDACTION + text[end:]
    return text


# ---------------------------------------------------------------- Level 2: secret files & paths


class PathValidationError(ValueError):
    """A target path is outside the project, protected, sensitive, or hallucinated."""


class SecretPathError(PathValidationError):
    """A target path matches secret_file_patterns (exit code 13, never 2)."""


def path_key(path: str | os.PathLike[str]) -> str:
    """Canonical comparison key: normalized separators, lower-case on Windows (normcase)."""
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def is_secret_file(path: str | os.PathLike[str], cfg: Config) -> bool:
    name = os.path.basename(os.path.normpath(os.fspath(path))).lower()
    if name in {e.lower() for e in cfg.safe_template_exceptions}:
        return False
    return any(fnmatch.fnmatchcase(name, p.lower()) for p in cfg.secret_file_patterns)


def is_within(path: Path, root: Path) -> bool:
    key, root_key = path_key(path), path_key(root)
    return key == root_key or key.startswith(root_key.rstrip(os.sep) + os.sep)


def validate_target_files(
    raw_paths: Iterable[str], project_root: Path, cfg: Config, *, creates_new_files: bool = False,
    confirm_new_dir: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """Resolve and vet target files (L06). Returns resolved absolute paths, de-duplicated.

    Rejects: traversal / symlinks escaping the root, `.ai_context/` and `.git/`, secret files,
    directories, and non-existent files unless a new file is explicitly requested inside an
    existing directory of the project, or inside new directories that `confirm_new_dir` approves.
    """
    root = project_root.resolve()
    result: list[Path] = []
    seen: set[str] = set()
    for raw in raw_paths:
        text = raw.strip()
        if not text or "\0" in text:
            raise PathValidationError(f"invalid target path {raw!r}")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:  # WinError 5 on ACL-protected parents, broken reparse points
            raise PathValidationError(f"cannot resolve {text!r}: {exc}") from exc
        if not is_within(resolved, root):
            raise PathValidationError(f"{text!r} resolves outside the project root")
        rel_parts = {p.lower() for p in Path(os.path.relpath(resolved, root)).parts}
        if rel_parts & {CONTEXT_DIR_NAME, ".git"}:
            raise PathValidationError(f"{text!r} is inside a protected directory")
        if is_secret_file(resolved, cfg) or is_secret_file(text, cfg):
            raise SecretPathError(f"{text!r} matches secret_file_patterns and is never dispatched")
        if resolved.is_dir():
            raise PathValidationError(f"{text!r} is a directory, not a file")
        if not resolved.exists() and not (creates_new_files and resolved.parent.is_dir()):
            if not creates_new_files:
                raise PathValidationError(f"{text!r} does not exist")
            ancestor = resolved.parent
            while not ancestor.exists():  # stops at the root at the latest (checked above)
                ancestor = ancestor.parent
            if not ancestor.is_dir():
                raise PathValidationError(f"{text!r}: {ancestor.name} is a file, not a directory")
            if confirm_new_dir is None or not confirm_new_dir(resolved):
                raise PathValidationError(f"{text!r}: directory {Path(os.path.relpath(resolved.parent, root)).as_posix()}"
                                          " does not exist and creating it was not approved")
        if (key := path_key(resolved)) not in seen:
            seen.add(key)
            result.append(resolved)
    return result
