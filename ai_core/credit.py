"""Channel economics (spec §7.A): Claude Pro monthly credit (B01, Q02) and Channel B cooldowns (Q04).

State lives in ~/.ai_state.json (outside every workspace), guarded by ~/.ai_state.lock and written
atomically. Thresholds on the remaining monthly credit: > 35% available, 15-35% limited (Haiku
only without confirmation), <= reserve (15%) exhausted = hard stop before dispatch.
"""
from __future__ import annotations

import calendar
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_core import config as cfgmod
from ai_core.checkpoint import retry_win_lock, write_json_atomic
from ai_core.config import Config

AVAILABLE, LIMITED, EXHAUSTED, COOLDOWN = "available", "limited", "exhausted", "cooldown"
_SEVERITY = {AVAILABLE: 0, LIMITED: 1, EXHAUSTED: 2}
LIMITED_AT_OR_BELOW_PCT = 35.0
# Real agy quota windows (status-line JSON, agy 1.4.7) that gate each routing bucket.
AGY_BUCKET_WINDOWS: dict[str, tuple[str, ...]] = {
    "claude_gpt": ("3p-5h", "3p-weekly"),
    "gemini": ("gemini-5h", "gemini-weekly"),
    "argon": ("gemini-argon-wtu-weekly", "gemini-5h"),
}
_LOCK_WAIT_S = 10.0
_LOCK_STALE_S = 30.0  # every transaction is a few ms; an older lock belongs to a dead process
_RESET_RE = re.compile(r"(?i)reset\w*\s+(?:in\s+)?(?:(\d+)\s*h\w*)?\s*(?:(\d+)\s*m\w*)?")


class CreditError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChannelAStatus:
    state: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    remaining_pct: float
    spendable_usd: float  # remaining minus the untouchable reserve
    cycle_start: date
    next_reset: date
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Verdict:
    allowed: bool
    needs_confirmation: bool
    max_budget_usd: float
    reason: str


def cycle_bounds(today: date, cycle_day: int) -> tuple[date, date]:
    """Current billing cycle [start, next_reset); day 31 clamps to short months."""
    def on(year: int, month: int) -> date:
        return date(year, month, min(cycle_day, calendar.monthrange(year, month)[1]))

    def shift(year: int, month: int, delta: int) -> tuple[int, int]:
        idx = year * 12 + month - 1 + delta
        return idx // 12, idx % 12 + 1

    start = on(today.year, today.month)
    if start > today:
        start = on(*shift(today.year, today.month, -1))
    return start, on(*shift(start.year, start.month, 1))


LEDGER_KEEP = 1000


def _ledger(data: dict[str, Any], **entry: Any) -> None:
    """Append-only usage log inside the state file, capped to the newest LEDGER_KEEP entries."""
    log = data.setdefault("ledger", [])
    log.append({"at": datetime.now(UTC).isoformat(timespec="seconds"), **entry})
    del log[:-LEDGER_KEEP]


def parse_reset_minutes(text: str) -> int | None:
    m = _RESET_RE.search(text or "")
    if not m or not (m[1] or m[2]):
        return None
    return int(m[1] or 0) * 60 + int(m[2] or 0)


class CreditManager:
    def __init__(self, cfg: Config, path: Path | None = None) -> None:
        self.cfg = cfg
        self.path = path or cfgmod.STATE_PATH
        self.lock_path = self.path.with_suffix(".lock")

    # ------------------------------------------------------------ persistence

    @contextmanager
    def _locked(self) -> Iterator[None]:
        deadline = time.monotonic() + _LOCK_WAIT_S
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (FileExistsError, PermissionError):  # PermissionError: lock being deleted (WinError 5)
                try:
                    if time.time() - self.lock_path.stat().st_mtime > _LOCK_STALE_S:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    raise CreditError(f"{self.lock_path} held by another ai process") from None
                time.sleep(0.05)
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            try:
                retry_win_lock(self.lock_path.unlink)
            except FileNotFoundError:
                pass

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except FileNotFoundError:
            return {"schema_version": 1}
        except (OSError, ValueError):
            pass
        # Fail-safe: unknown spend can't exclude crossing the reserve, so the cycle counts as spent.
        backup = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
        try:
            os.replace(self.path, backup)
        except OSError:
            pass
        print(f"⚠ {self.path} unreadable (kept as {backup.name}): Channel A blocked for this cycle", file=sys.stderr)
        return {"schema_version": 1, "channel_a": {"spent_this_cycle_usd": self.cfg.claude_monthly_credit_usd,
                                                    "corrupt_recovered": True}}

    @contextmanager
    def _transaction(self) -> Iterator[dict[str, Any]]:
        with self._locked():
            data = self._load()
            yield data
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, data)

    def _channel_a_data(self, data: dict[str, Any], today: date) -> dict[str, Any]:
        start, _ = cycle_bounds(today, self.cfg.claude_credit_cycle_day)
        a = data.setdefault("channel_a", {})
        if a.get("corrupt_recovered") and "cycle_start" not in a:
            a["cycle_start"] = start.isoformat()  # blocked for the current cycle only
        elif a.get("cycle_start") != start.isoformat():
            a.clear()
            a.update({"cycle_start": start.isoformat(), "spent_this_cycle_usd": 0.0, "calls": 0,
                      "exhausted_until": None, "last_error": None})
        return a

    # ------------------------------------------------------------ Channel A (Claude Pro monthly credit)

    def channel_a(self, today: date | None = None) -> ChannelAStatus:
        today = today or datetime.now().date()
        with self._transaction() as data:
            a = self._channel_a_data(data, today)
        start, nxt = cycle_bounds(today, self.cfg.claude_credit_cycle_day)
        limit = self.cfg.claude_monthly_credit_usd
        spent = float(a.get("spent_this_cycle_usd") or 0.0)
        remaining = max(0.0, limit - spent)
        pct = remaining / limit * 100
        spendable = max(0.0, remaining - limit * self.cfg.claude_credit_reserve_pct / 100)
        if (until := a.get("exhausted_until")) and date.fromisoformat(until) > today:
            return ChannelAStatus(EXHAUSTED, limit, spent, remaining, pct, 0.0, start, nxt,
                                  f"provider reported limit reached ({a.get('last_error') or 'no detail'})")
        if pct <= self.cfg.claude_credit_reserve_pct or spendable < 0.01:
            state, reason = EXHAUSTED, f"remaining {pct:.1f}% <= reserve {self.cfg.claude_credit_reserve_pct}%"
        elif pct <= LIMITED_AT_OR_BELOW_PCT:
            state, reason = LIMITED, f"remaining {pct:.1f}% (saving mode)"
        else:
            state, reason = AVAILABLE, f"remaining {pct:.1f}%"
        # Real plan windows (5h / 7d) win when worse: they are what actually blocks claude.
        _, status, windows = self.claude_real_windows()
        if windows or status == "rejected":
            name, w = max(windows.items(), key=lambda kv: kv[1]["used_pct"], default=("plan", {"used_pct": 100.0}))
            used = w["used_pct"]
            when = f", resets {w['resets_at']:%Y-%m-%d %H:%M} UTC" if w.get("resets_at") else ""
            if status == "rejected" or used >= 100 - self.cfg.claude_credit_reserve_pct:
                real = EXHAUSTED
            elif used >= 100 - LIMITED_AT_OR_BELOW_PCT:
                real = LIMITED
            else:
                real = AVAILABLE
            if _SEVERITY[real] > _SEVERITY[state]:
                state, reason = real, f"{name} window {used:.0f}% used{when} (real, from claude)"
            elif state == AVAILABLE:
                reason += f"; {name} window {used:.0f}% used{when}"
        return ChannelAStatus(state, limit, spent, remaining, pct, spendable, start, nxt, reason)

    def record_claude_rate_limit(self, info: dict[str, Any]) -> None:
        """Persist `rate_limit_event.rate_limit_info` from a claude stream (utilization is a 0-1 fraction)."""
        windows: dict[str, dict[str, Any]] = {}
        for name, w in (info.get("unifiedWindows") or {}).items():
            if isinstance(w, dict) and isinstance(w.get("utilization"), (int, float)):
                windows[name] = {"used_pct": round(w["utilization"] * 100, 2), "resets_at": w.get("resetsAt")}
        if not windows and isinstance(info.get("utilization"), (int, float)) and info.get("rateLimitType"):
            windows[info["rateLimitType"]] = {"used_pct": round(info["utilization"] * 100, 2),
                                              "resets_at": info.get("resetsAt")}
        with self._transaction() as data:
            data["claude_rate_limit"] = {"captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
                                         "status": info.get("status"), "windows": windows}

    def claude_real_windows(self, now: datetime | None = None) -> tuple[str | None, str | None, dict[str, Any]]:
        """(captured_at, status, live windows) from the freshest of: last dispatch's rate_limit_event, or the
        interactive status-line capture. Windows whose reset already passed are dropped (stale)."""
        from ai_core import statusline

        now = now or datetime.now(UTC)
        with self._transaction() as data:
            stored = data.get("claude_rate_limit") or {}
        candidates: list[tuple[str, str | None, dict[str, Any]]] = []
        if stored.get("captured_at"):
            wins = {n: {"used_pct": w["used_pct"],
                        "resets_at": datetime.fromtimestamp(w["resets_at"], UTC) if w.get("resets_at") else None}
                    for n, w in stored.get("windows", {}).items()}
            candidates.append((stored["captured_at"], stored.get("status"), wins))
        if (rec := statusline.load("claude")) and rec.get("captured_at"):
            candidates.append((rec["captured_at"], None, statusline.claude_windows(rec)))
        if not candidates:
            return None, None, {}
        captured, status, wins = max(candidates, key=lambda c: c[0])
        live = {n: w for n, w in wins.items() if w.get("resets_at") is None or w["resets_at"] > now}
        if status == "rejected" and not live:
            status = None  # the rejection belonged to a window that has since reset
        return captured, status, live

    def check_claude(self, tier: str, *, interactive: bool) -> Verdict:
        """Pre-dispatch hard stop (B01). `tier` is haiku | sonnet | opus."""
        st = self.channel_a()
        if st.state == EXHAUSTED:
            return Verdict(False, False, 0.0, f"Channel A exhausted until {st.next_reset}: {st.reason}")
        # --max-budget-usd is checked by the CLI BETWEEN turns: one turn can overshoot it (measured
        # 2026-10-25: cap $0.01, spent $0.1273). The cap bounds the task; the reserve absorbs that overshoot.
        budget = math.floor(round(min(self.cfg.claude_max_budget_usd_per_task, st.spendable_usd) * 100, 6)) / 100
        if budget < 0.01:
            return Verdict(False, False, 0.0, "no spendable credit above the reserve")
        if tier == "opus":  # always, even in --agent mode: agent_auto_approve never applies (B01)
            return Verdict(interactive, True, budget,
                           "Opus on Channel A requires explicit confirmation" + ("" if interactive else " (none possible)"))
        if st.state == LIMITED and tier != "haiku":
            ok = interactive or self.cfg.agent_auto_approve
            return Verdict(ok, interactive, budget, f"Channel A limited ({st.reason}): {tier} needs confirmation")
        return Verdict(True, False, budget, st.reason)

    def record_claude_result(self, cost_usd: float | None, *, channel_exhausted: bool = False, error: str = "",
                             model: str = "") -> None:
        today = datetime.now().date()
        with self._transaction() as data:
            a = self._channel_a_data(data, today)
            if cost_usd:
                a["spent_this_cycle_usd"] = round(float(a.get("spent_this_cycle_usd") or 0.0) + cost_usd, 6)
            a["calls"] = int(a.get("calls") or 0) + 1
            if channel_exhausted:
                a["exhausted_until"] = cycle_bounds(today, self.cfg.claude_credit_cycle_day)[1].isoformat()
                a["last_error"] = error[:300]
            _ledger(data, channel="A", model=model, cost_usd=round(cost_usd or 0.0, 6))

    def sync_claude_spent(self, spent_usd: float) -> float:
        """Align the counter with the interactive `/usage` dialog (the only free source: `-p /usage` costs)."""
        if not 0 <= spent_usd <= self.cfg.claude_monthly_credit_usd * 10:
            raise CreditError(f"implausible spend ${spent_usd:.2f}")
        with self._transaction() as data:
            a = self._channel_a_data(data, datetime.now().date())
            previous = float(a.get("spent_this_cycle_usd") or 0.0)
            a["spent_this_cycle_usd"] = round(spent_usd, 6)
            _ledger(data, channel="A", model="sync", cost_usd=round(spent_usd - previous, 6))
        return previous

    # ------------------------------------------------------------ Channel B (Gemini Pro buckets)

    def channel_b_ready(self, bucket: str, now: datetime | None = None) -> bool:
        return self.channel_b_block(bucket, now) is None

    def channel_b_block(self, bucket: str, now: datetime | None = None) -> str | None:
        """Why a Channel B bucket can't take work now (local cooldown or a real agy window), or None."""
        from ai_core import statusline

        now = now or datetime.now(UTC)
        with self._transaction() as data:
            until = data.get("channel_b", {}).get(bucket, {}).get("cooldown_until")
        if until and datetime.fromisoformat(until) > now:
            return f"cooldown until {until}"
        windows = statusline.agy_windows(statusline.load("agy"))
        for name in AGY_BUCKET_WINDOWS.get(bucket, ()):
            w = windows.get(name)
            if w and w["resets_at"] and w["resets_at"] > now and w["remaining_pct"] <= self.cfg.agy_min_remaining_pct:
                return f"{name} {w['remaining_pct']:.1f}% left until {w['resets_at']:%Y-%m-%d %H:%M} UTC (real)"
        return None

    def mark_b_cooldown(self, bucket: str, error: str, minutes: int | None = None) -> datetime:
        """Cooldown from a quota error (reset parsed from its text), or set by hand; minutes=0 clears it."""
        minutes = minutes if minutes is not None else parse_reset_minutes(error) or self.cfg.agy_cooldown_default_min
        until = datetime.now(UTC) + timedelta(minutes=minutes)
        with self._transaction() as data:
            b = data.setdefault("channel_b", {}).setdefault(bucket, {})
            b.update({"cooldown_until": until.isoformat(timespec="seconds") if minutes else None,
                      "last_error": error[:300]})
        return until

    def record_b_usage(self, bucket: str, tokens: int, model: str = "") -> None:
        """Local count from agy's own `usage` field (Q04). `agy --print=/usage` is not a headless command:
        it is sent to the model and burns quota (measured 32,760 tokens), so it is never called."""
        with self._transaction() as data:
            b = data.setdefault("channel_b", {}).setdefault(bucket, {})
            b["tokens_estimated"] = int(b.get("tokens_estimated") or 0) + tokens
            b["confidence"] = "estimated"
            _ledger(data, channel="B", model=model, bucket=bucket, tokens=tokens)

    def usage_report(self, now: datetime | None = None) -> dict[str, Any]:
        """`ai --usage`: Channel A per billing cycle and model; Channel B per bucket over the plan's
        5-hour and 7-day windows. Only what went through this orchestrator (plus manual syncs)."""
        now = now or datetime.now(UTC)
        st = self.channel_a()
        with self._transaction() as data:
            ledger = list(data.get("ledger", []))
            buckets = dict(data.get("channel_b", {}))
        cycle_start = datetime.combine(st.cycle_start, datetime.min.time(), UTC)
        by_model: dict[str, dict[str, float]] = {}
        windows = {"5h": now - timedelta(hours=5), "7d": now - timedelta(days=7)}
        b_usage: dict[str, dict[str, int]] = {}
        for e in ledger:
            at = datetime.fromisoformat(e["at"])
            if e["channel"] == "A" and at >= cycle_start:
                m = by_model.setdefault(e.get("model") or "?", {"calls": 0, "cost_usd": 0.0})
                m["calls"] += e.get("model") != "sync"
                m["cost_usd"] = round(m["cost_usd"] + e.get("cost_usd", 0.0), 6)
            elif e["channel"] == "B":
                w = b_usage.setdefault(e.get("bucket") or "?", {"calls_5h": 0, "tokens_5h": 0, "tokens_7d": 0})
                if at >= windows["7d"]:
                    w["tokens_7d"] += int(e.get("tokens", 0))
                if at >= windows["5h"]:
                    w["calls_5h"] += 1
                    w["tokens_5h"] += int(e.get("tokens", 0))
        for name in {*buckets, *AGY_BUCKET_WINDOWS}:  # every routing bucket appears, used or not
            until = buckets.get(name, {}).get("cooldown_until")
            b_usage.setdefault(name, {"calls_5h": 0, "tokens_5h": 0, "tokens_7d": 0})["cooldown_until"] = \
                until if until and datetime.fromisoformat(until) > now else None
        from ai_core import statusline

        claude_at, claude_status, claude_wins = self.claude_real_windows(now)
        agy_rec = statusline.load("agy")
        real = {
            "claude": {"captured_at": claude_at, "status": claude_status, "windows": {
                n: {"used_pct": w["used_pct"], "resets_at": w["resets_at"].isoformat() if w["resets_at"] else None}
                for n, w in claude_wins.items()}},
            "agy": {"captured_at": (agy_rec or {}).get("captured_at"), "plan": ((agy_rec or {}).get("data") or {}).get(
                "plan_tier"), "windows": {
                n: {"remaining_pct": w["remaining_pct"], "resets_at": w["resets_at"].isoformat() if w["resets_at"] else None}
                for n, w in statusline.agy_windows(agy_rec).items() if not w["resets_at"] or w["resets_at"] > now}},
        }
        for name in b_usage:
            b_usage[name]["blocked"] = self.channel_b_block(name, now)
        return {
            "real": real,
            "channel_a": {"state": st.state, "reason": st.reason, "spent_usd": round(st.spent_usd, 4),
                          "limit_usd": st.limit_usd, "remaining_usd": round(st.remaining_usd, 4),
                          "spendable_usd": round(st.spendable_usd, 4), "cycle_start": st.cycle_start.isoformat(),
                          "next_reset": st.next_reset.isoformat(), "by_model": by_model},
            "channel_b": b_usage,
            "source": "real: claude rate_limit_event / status line, agy status line; ledger: orchestrator calls "
                      "(claude costs are list-price equivalents, not charges)",
        }

    def summary(self) -> dict[str, Any]:
        st = self.channel_a()
        with self._transaction() as data:
            b = data.get("channel_b", {})
        return {"channel_a": {"state": st.state, "spent_usd": round(st.spent_usd, 4), "limit_usd": st.limit_usd,
                              "remaining_pct": round(st.remaining_pct, 1), "next_reset": st.next_reset.isoformat()},
                "channel_b": {k: {"ready": self.channel_b_ready(k), **v} for k, v in b.items()}}
