# MLX benchmark — 2026-09-22 (post-rebase onto v0.7.0)

Model: `Qwen3.5-9B-MLX-8bit` (local path, 8-bit MLX)
Host: Apple Silicon, machine under load avg ~6 (absolute numbers are
therefore conservative; all comparisons were run back-to-back under
comparable load, so the RATIOS are the meaningful result).
Method: `usage.completion_tokens / wall_latency`, non-streamed (the SSE
stream carries no `usage` block). 128 completion tokens per request.

## Headline: grace_seconds dominates MLX throughput

| Path | warm decode | per-request latency |
|-|-|-|
| Direct to `mlx_lm` sidecar (bypass Turbohaul) | **~43 tok/s** | ~3.0 s |
| Turbohaul, `grace_seconds: 30` (default config) | **3.85 tok/s** | 33.2 s |
| Turbohaul, `grace_seconds: 2` | **25.61 tok/s** | 5.0 s |
| oMLX (external baseline, earlier session) | ~49.85 tok/s | — |

Every Turbohaul-routed request took ~33.2 s against ~3.0 s direct, and the
excess tracked `grace_seconds` exactly. Dropping grace 30 → 2 moved warm
decode 3.85 → 25.61 tok/s (**6.7×**) with no code change. The residual gap to
the ~43 tok/s direct number is consistent with the remaining 2 s hold.

**The earlier "~2.6–3× slower than oMLX" conclusion was measuring the grace
window, not the engine.** MLX decode itself is competitive (~43 vs ~49.85);
what made Turbohaul look slow on this backend was a queue/lifecycle setting.

## Why grace hurts MLX specifically

`grace_seconds` holds the slot open after a completion so a same-thread
follow-up can reuse the warm process and its KV cache. On llama.cpp that hold
buys a real prefill saving (the `/slots` KV bin). **MLX has no KV bin** — the
MLX spawn path skips `_restore_slot_kv` entirely — so the hold costs latency
and returns nothing. Same root asymmetry as the Fix D warm-inherit bug.

Suggested follow-up (NOT in this PR): default `grace_seconds` low (or 0) when
`backend == "mlx"`, since the reuse it protects does not exist there.

## Correctness (Fix D) — verified end-to-end

Three different prompts, no `session_id` (the bare-curl shape that triggered
the bug), against a live 9B sidecar: **3/3 distinct, correct answers**, both
streamed and non-streamed. Pre-fix, request 2+ replayed request 1's answer.

## Measurement traps hit while collecting these numbers

1. **`mlx_lm server`'s `/v1/models` lists the HF CACHE, not the loaded model.**
   Its first entry here was a stale `Llama-3.2-1B-Instruct-4bit`; sending that
   id made the sidecar load *that* model and report 44.85 tok/s — a 1B number
   that looked plausible for a 9B. Always pass the sidecar's real `--model`
   arg (`--model-id=` on the probe).
2. **The SSE stream has no `usage` block** — use `--no-stream` for ground-truth
   tok/s. Never count SSE chunks (reasoning + keepalive chunks invalidate it).
3. **Reasoning models** return the answer under `message.reasoning`, leaving
   `content` absent.
4. **`DEVNULL` sidecar stdio hides engine failures.** Run with
   `TURBOHAUL_MLX_LOG_DIR=<dir>` to capture them.

## Reproduce

```bash
# terminal 1
TURBOHAUL_MLX_LOG_DIR=/tmp/mlxlogs PYTHONPATH=src \
  python -m turbohaul.__main__ --config ~/.turbohaul/turbohaul.yaml

# terminal 2 — through Turbohaul (correctness + tok/s)
python tests/manual_mlx_prompt_replay_check.py qwen3.5-9b-mlx-8bit --no-stream

# terminal 2 — direct to the sidecar, same model, for the overhead delta
python tests/manual_mlx_sidecar_probe.py 11500 --bench --model-id=<the --model arg>
```
