"""Tier-0 benchmark (spec §12.3, L01-L05): real prefill/generation speed of the local model.

Uses Ollama's own counters (prompt_eval_count/_duration, eval_count/_duration), so the numbers
are the server's, not wall-clock guesses. Every prompt starts with a fresh nonce so the KV prefix
cache can't fake a warm prefill. Results go to ~/.ai_benchmark.json together with the config
values derived from them; nothing is written to config.toml automatically.
"""
from __future__ import annotations

import json
import math
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_core import config as cfgmod
from ai_core import ollama_env
from ai_core.config import Config

SIZES = (2000, 8000, 16000, 26000)
RESULT_PATH = cfgmod.HOME / ".ai_benchmark.json"
_SCORING_BUDGET_S = 8.0  # the original fixed scoring timeout: full content only while prefill fits in it


def _filler(chars: int) -> str:
    """Real source code (what the router actually sends), repeated to the requested size."""
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted((Path(__file__).parent).rglob("*.py")))
    return (src * (chars // len(src) + 1))[:chars]


def _call(cfg: Config, prompt: str, *, think: bool, num_predict: int, timeout: float) -> dict[str, Any]:
    from ai_core.router import _ollama
    t0 = time.monotonic()
    data = _ollama(cfg, "/api/chat", {
        "model": cfg.ollama_model, "stream": False, "think": think,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": cfg.ollama_num_ctx, "num_predict": num_predict,
                    "temperature": 0.6 if think else 0.1, "top_p": 0.95 if think else 0.8, "top_k": 20},
    }, timeout)
    ns = 1e9
    pe, pd = data.get("prompt_eval_count") or 0, (data.get("prompt_eval_duration") or 0) / ns
    ec, ed = data.get("eval_count") or 0, (data.get("eval_duration") or 0) / ns
    return {"prompt_tokens": pe, "prefill_s": round(pd, 3), "prefill_tps": round(pe / pd, 1) if pd else None,
            "gen_tokens": ec, "gen_s": round(ed, 3), "gen_tps": round(ec / ed, 1) if ed else None,
            "load_s": round((data.get("load_duration") or 0) / ns, 3), "wall_s": round(time.monotonic() - t0, 3)}


def _gpu(cfg: Config) -> dict[str, Any]:
    from ai_core.router import _ollama
    for m in _ollama(cfg, "/api/ps").get("models", []):
        if isinstance(m, dict) and m.get("name") == cfg.ollama_model:
            return {"size": m.get("size"), "size_vram": m.get("size_vram"),
                    "full_gpu": m.get("size_vram") == m.get("size"), "context_length": m.get("context_length")}
    return {"full_gpu": None}


def recommend(rows: list[dict[str, Any]], think: dict[str, Any], base_call: dict[str, Any]) -> dict[str, Any]:
    """Config values from measurements, conservative by construction (slowest point, margins)."""
    tps = [r["prefill_tps"] for r in rows if r.get("prefill_tps")]
    slow = min(tps)
    cpt = sum(r["chars"] for r in rows) / sum(r["prompt_tokens"] for r in rows)
    # real tokens prefilled within the budget (piecewise-linear on the measured points), then converted to
    # the router's own unit: estimate_tokens() = chars/3, which overcounts real tokens by cpt/3
    pts = [(0, 0.0)] + sorted((r["prompt_tokens"], r["prefill_s"]) for r in rows)
    real = pts[-1][0]
    for (t0, s0), (t1, s1) in zip(pts, pts[1:], strict=False):
        if s1 >= _SCORING_BUDGET_S:
            real = t0 + (t1 - t0) * (_SCORING_BUDGET_S - s0) / (s1 - s0)
            break
    full_max = max(1000, math.floor(real * cpt / 3 / 500) * 500)
    return {
        "calibrated_prefill_tps": math.floor(slow * 0.8),  # 20% margin for a busy GPU / Jarvis on the slot
        # fixed part of a scoring call = everything except prefill (load, ~JSON output), x1.5, >= 3 s
        "scoring_base_timeout_s": max(3, math.ceil((base_call["wall_s"] - base_call["prefill_s"]) * 1.5)),
        "scoring_full_content_max_tokens": full_max,
        "chars_per_token": round(cpt, 2),
        "thinking_gen_tps": think.get("gen_tps"),
        # quality of temperatures can't be measured by speed: keep the model card values (L05)
        "thinking_temperature": 0.6,
    }


def run(cfg: Config, log=print) -> dict[str, Any]:  # noqa: ANN001 - any callable(str)
    issues = ollama_env.issues(ollama_env.server_env())
    for i in issues:
        log(f"⚠ {i}")
    log(f"warm-up / load {cfg.ollama_model} (num_ctx={cfg.ollama_num_ctx}) …")
    warm = _call(cfg, "Reply with OK.", think=False, num_predict=4, timeout=cfg.ollama_cold_load_timeout_s + 60)
    log(f"  load {warm['load_s']} s, wall {warm['wall_s']} s")
    gpu = _gpu(cfg)
    log(f"  /api/ps: {gpu}")
    rows: list[dict[str, Any]] = []
    for target in SIZES:
        nonce = f"[benchmark nonce {secrets.token_hex(8)}]\n"
        text = nonce + _filler(target * 3) + "\n\nSummarize the code above in one sentence."
        r = _call(cfg, text, think=False, num_predict=128, timeout=600)
        r.update(target=target, chars=len(text))
        rows.append(r)
        log(f"  {target:>6} tok: prefill {r['prompt_tokens']} tok in {r['prefill_s']} s = {r['prefill_tps']} tok/s;"
            f" gen {r['gen_tps']} tok/s")
    log("thinking generation (think=true) …")
    think = _call(cfg, f"[nonce {secrets.token_hex(8)}] Is 1009 prime? Explain briefly.", think=True,
                  num_predict=1024, timeout=600)
    log(f"  gen {think['gen_tokens']} tok at {think['gen_tps']} tok/s")
    rec = recommend(rows, think, rows[0])
    result = {"measured_at": datetime.now(UTC).isoformat(timespec="seconds"), "model": cfg.ollama_model,
              "num_ctx": cfg.ollama_num_ctx, "server_env_issues": issues, "gpu": gpu, "warmup": warm,
              "prefill": rows, "thinking": think, "recommended": rec}
    from ai_core.checkpoint import write_json_atomic
    write_json_atomic(RESULT_PATH, result)
    log(f"recommended: {json.dumps(rec)}\nsaved {RESULT_PATH}")
    return result
