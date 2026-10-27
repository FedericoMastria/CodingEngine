"""Typed configuration loaded from `.ai_context/config.toml` (spec §11) plus well-known paths (§9)."""
from __future__ import annotations

import dataclasses
import itertools
import os
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path

CONTEXT_DIR_NAME = ".ai_context"
HOME = Path.home()
PROFILE_PATH = HOME / ".ai_profile.md"
# Invariant #2/#3: checkpoints live outside every workspace, never under .ai_context/.
CHECKPOINTS_ROOT = HOME / ".ai_checkpoints"
STATE_PATH = HOME / ".ai_state.json"  # Channel A monthly credit + Channel B cooldowns (§7.A)

_BUILD_MANIFESTS = ("pyproject.toml", "package.json", "Cargo.toml", "CMakeLists.txt")


class ConfigError(ValueError):
    """config.toml is malformed or violates a constraint."""


@dataclass(frozen=True, slots=True)
class Config:
    # Ollama & context (L01-L05)
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.6:35b-a3b"
    ollama_num_ctx: int = 32768
    local_max_input_tokens: int = 26000
    local_max_files: int = 3
    # Calibrated by `ai --benchmark` (§12.3) on dedicated GPU, 2026-10-25, FA on + q8_0 KV, 100% GPU:
    # real prefill 686 tok/s @1.7k -> 325 tok/s @21k; generation 59 -> 26 tok/s; ~3.7 chars/token.
    scoring_full_content_max_tokens: int = 5500  # ~8 s of prefill (estimate_tokens units)
    ollama_num_predict_standard: int = 4096
    ollama_num_predict_thinking: int = 6144
    calibrated_prefill_tps: float = 260.0  # slowest measured point x 0.8

    # Channel A - Claude Pro (B01, Q02)
    claude_monthly_credit_usd: float = 20.0
    claude_credit_cycle_day: int = 1
    claude_credit_reserve_pct: int = 15
    claude_allow_usage_credits: bool = False
    claude_use_advisor_on_depth_3: bool = False
    claude_exclude_dynamic_system_prompt_sections: bool = False
    cloud_auto_repair_enabled: bool = True
    cloud_auto_rollback_on_failure: bool = True
    claude_model_haiku: str = "haiku"
    claude_model_sonnet: str = "sonnet"
    claude_model_opus: str = "opus"
    claude_max_budget_usd_per_task: float = 1.0
    cloud_timeout_s: int = 1800

    # Channel B - Antigravity (B04, B05); IDs from `agy models` (agy 1.4.7, 2026-10-25).
    # Effort is encoded in the model-ID suffix, so `--effort` is never passed (single form, B04).
    agy_model_flash: str = "gemini-3.8-flash-high"
    agy_model_argon_low: str = "gemini-4-argon-low"
    agy_model_argon_med: str = "gemini-4-argon-medium"
    agy_model_argon_high: str = "gemini-4-argon-high"
    agy_model_sonnet: str = "claude-sonnet-5-5-medium"
    agy_model_opus: str = "claude-opus-5-5-high"
    agy_enable_headless_slash_commands: bool = False
    agy_cooldown_default_min: int = 60  # used when a quota error carries no reset time
    agy_min_remaining_pct: float = 5.0  # a real agy quota window at/below this blocks its bucket until reset

    # Timeouts (§12.3: 26k-token local task ~65 s prefill + up to ~240 s generation; cold load ~55 s)
    scoring_base_timeout_s: int = 3
    scoring_max_timeout_s: int = 25
    ollama_cold_load_timeout_s: int = 120
    verify_l1_timeout_s: int = 15
    verify_l2_timeout_s: int = 120
    verify_l3_timeout_s: int = 300
    worker_timeout_s: int = 15
    local_timeout_s: int = 600
    verify_l2_command: list[str] = field(default_factory=list)
    verify_l3_command: list[str] = field(default_factory=lambda: ["python", "-m", "pytest", "-q"])

    # Security, secrets, checkpoints (S01-S05, R03)
    secret_scanner_enabled: bool = True
    secret_file_patterns: list[str] = field(default_factory=lambda: [
        ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "id_ed25519*",
        "credentials.json", "service-account*.json", ".npmrc", ".pypirc", ".netrc", "*.tfvars",
    ])
    safe_template_exceptions: list[str] = field(default_factory=lambda: [
        ".env.example", ".env.sample", ".env.template",
    ])
    vault_isolation_mode: str = "auto"
    checkpoint_max_keep: int = 10
    checkpoint_retention_days: int = 7
    checkpoint_max_total_mb: int = 50
    usage_max_age_min: int = 60  # agy quota capture older than this is refreshed (after tasks, and by --usage)
    checkpoint_exclude_dirs: list[str] = field(default_factory=lambda: [
        "node_modules", ".venv", "venv", "target", "dist", "build", ".next", ".nuxt", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".gradle", "bin", "obj", "coverage",
    ])

    secret_action_agent: str = "local_only"  # --agent mode has nobody to ask (S01)

    # Ponytail & skills (M01, S06)
    ponytail_mode: str = "full"
    adr_require_approval: bool = True
    rotate_session_on_cooldown: bool = True
    adaptive_routing: bool = True  # skip Tier-0 for task kinds it keeps failing in this project

    # Sessions, agent mode, repo-map, runner (§3.A, §5.C.2, §10)
    agent_auto_approve: bool = False  # never applies to Opus on Channel A (B01)
    lock_wait_timeout_s: int = 10
    repo_map_tokens_with_targets: int = 2048
    repo_map_tokens_without_targets: int = 8192
    safe_verify_allowlist: list[str] = field(default_factory=lambda: [
        "pytest", "python -m pytest", "python -m unittest", "uv run", "npm test", "npm run test", "npx tsc",
        "tsc", "cargo test", "cargo check", "cargo build", "go test", "go build", "ruff", "mypy", "pyright",
        "dotnet test", "dotnet build",
    ])


_ENUMS: dict[str, frozenset[str]] = {
    "vault_isolation_mode": frozenset({"auto", "never"}),
    "ponytail_mode": frozenset({"off", "lite", "full", "ultra"}),
    "secret_action_agent": frozenset({"local_only", "redact", "abort"}),
}
_HINTS = typing.get_type_hints(Config)


def _check_type(key: str, value: object) -> None:
    hint = _HINTS[key]
    if hint is bool:
        ok = isinstance(value, bool)
    elif hint is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif hint is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif hint is str:
        ok = isinstance(value, str)
    else:  # list[str]
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
    if not ok:
        raise ConfigError(f"config.toml: '{key}' must be {getattr(hint, '__name__', hint)}, got {value!r}")


def validate_config(values: dict[str, object]) -> Config:
    unknown = sorted(set(values) - set(_HINTS))
    if unknown:
        raise ConfigError(f"config.toml: unknown keys {unknown}")
    for key, value in values.items():
        _check_type(key, value)
        if key in _ENUMS and value not in _ENUMS[key]:
            raise ConfigError(f"config.toml: '{key}' must be one of {sorted(_ENUMS[key])}, got {value!r}")
    cfg = Config(**{k: float(v) if _HINTS[k] is float else v for k, v in values.items()})
    for f in dataclasses.fields(Config):
        v = getattr(cfg, f.name)
        if _HINTS[f.name] in (int, float) and v <= 0 and f.name != "claude_credit_reserve_pct":
            raise ConfigError(f"config.toml: '{f.name}' must be > 0, got {v!r}")
    if not 1 <= cfg.claude_credit_cycle_day <= 31:
        raise ConfigError("config.toml: 'claude_credit_cycle_day' must be in 1..31")
    if not 0 <= cfg.claude_credit_reserve_pct < 100:
        raise ConfigError("config.toml: 'claude_credit_reserve_pct' must be in 0..99")
    if not cfg.ollama_url.startswith(("http://", "https://")):
        raise ConfigError("config.toml: 'ollama_url' must be an http(s) URL")
    return cfg


def load_config(project_root: Path) -> Config:
    """Load `<root>/.ai_context/config.toml`; a missing file means calibrated defaults."""
    path = project_root / CONTEXT_DIR_NAME / "config.toml"
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return Config()
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    except PermissionError as exc:  # WinError 5 / 32: locked by an editor or AV scanner
        raise ConfigError(f"{path}: cannot read ({exc.strerror}); close the program holding it") from exc
    return validate_config(data)


def find_project_root(start: Path) -> Path:
    """Monorepo-safe discovery (§9.1): nearest `.git`/`.ai_context`, then a build manifest, then `start`."""
    start = Path(os.path.normpath(start.resolve()))
    # A marker in the home folder or a drive root (dotfiles repo, stray .ai_context) must not swallow every
    # project below it: repo-map and checkpoints would span the whole disk. Climbing stops below them.
    parents = list(itertools.takewhile(lambda p: p != HOME.resolve() and p.parent != p, start.parents))
    for candidate in (start, *parents):
        if (candidate / ".git").exists() or (candidate / CONTEXT_DIR_NAME).is_dir():
            return candidate
    for candidate in (start, *parents):
        if any((candidate / m).is_file() for m in _BUILD_MANIFESTS) or any(candidate.glob("*.sln")):
            return candidate
    return start


def ensure_context_dir(project_root: Path) -> Path:
    """Create `.ai_context/` and make it invisible to Git (§3.C.1)."""
    ctx = project_root / CONTEXT_DIR_NAME
    ctx.mkdir(exist_ok=True)
    gitignore = ctx / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    exclude = project_root / ".git" / "info" / "exclude"
    if exclude.parent.is_dir():
        current = exclude.read_text(encoding="utf-8", errors="replace") if exclude.exists() else ""
        if f"{CONTEXT_DIR_NAME}/" not in current.split():
            with exclude.open("a", encoding="utf-8") as fh:
                fh.write(("" if current.endswith("\n") or not current else "\n") + f"{CONTEXT_DIR_NAME}/\n")
    return ctx
