# Turbohaul MLX vs oMLX — benchmark (Qwen3.5-9B-MLX-8bit)

Date: 2026-07-13
Machine: macOS Apple Silicon (M-series, G13D GPU), 64 GB RAM
Model: Qwen3.5-9B-MLX-8bit (8-bit MLX, ~9.7 GB weights) — same artifact for both runners
Prompt: ~600-token maritime essay (fixed) unless noted; max_tokens=256

## oMLX (target — v0.5.1, /opt/homebrew/bin/oMLX serve, port 8011)
- Cold load from disk (usage.model_load_duration): 3.37 s
- Idle RSS (server process only): 136 MB  (model lives in separate engine pool)
- Model RSS at generation: ~10,838 MB (~10.6 GB)
- Warm prefill TTFT (streaming, first token): 0.014 s
- Warm decode tok/s (ground truth, usage.completion_tokens / latency): 49.85
- Warm non-streaming latency (256 tok): 5.14 s
- Does NOT cache: each request generates fresh.

## Turbohaul MLX (PR #6, manager :11401 -> mlx_lm sidecar)
- Cold first-request (load + 256 tok gen): 15.74 s  -> load ≈ 2-3 s, gen ≈ 13-15 s
- Manager idle RSS: 75-82 MB
- mlx_lm sidecar RSS (the model): ~9,970 MB (~9.7 GB)
- Real decode tok/s (first, uncached request): ~16-19 tok/s
  (15.74 s/256 = 16.3 ; 13.42 s/256 = 19.1 across two distinct real gens)
- Warm prefill TTFT: not cleanly measurable (see bug)

## Headline comparison
| Metric                | oMLX       | Turbohaul MLX | Ratio |
|-----------------------|------------|---------------|-------|
| Cold load from disk   | 3.37 s     | ~2-3 s*       | ~1x   |
| Decode tok/s (warm)   | 49.85      | ~16-19**      | 2.6-3x slower |
| Idle process RSS      | 136 MB     | 82 MB (mgr)   | -     |
| Model RSS             | ~10.6 GB   | ~9.7 GB       | ~1x   |

*Turbohaul embeds load in first-request latency (mlx_lm doesn't report load_duration).
**Turbohaul real decode only observable on the FIRST uncached request.

## CRITICAL BUG FOUND: completion cache/replay ignores prompt
After the first completion, Turbohaul serves the SAME cached `reasoning` body for
every subsequent request, regardless of prompt. Proven:
- Fresh restart, then two DIFFERENT prompts
  ("What is 12 times 13?" and "Name three primary colors")
  both returned identical `reasoning` head:
  'Thinking Process: 1. Analyze the Request: The user is asking for the produ...'
  with identical usage (prompt_tokens 26, cached_tokens 25).
- A sequence of 5 unique prompts all returned the verbatim same reasoning text.
- Consequence: warm requests return in ~0.003 s (instant cache hit) or ~35 s
  (stream-replay of 256 cached tokens), NEITHER is real inference.
- oMLX does NOT exhibit this — it generates fresh each time.

This blocks any warm-speed measurement on Turbohaul and is a correctness defect:
the model effectively answers only the first prompt it ever sees.

## Methodology notes / caveats
- Decode tok/s uses the server's own usage.completion_tokens / wall latency
  (ground truth). Counting SSE chunks as tokens is invalid because both runners
  emit reasoning_content + keepalive pseudo-chunks.
- oMLX exposes usage.model_load_duration; mlx_lm does not, so Turbohaul cold load
  is inferred from first-request latency minus observed warm-gen time.
- Because of the cache bug, Turbohaul's "warm" speed cannot be measured; the
  ~16-19 tok/s figure is from first (uncached) requests only.
- Runners benchmarked sequentially (one model resident at a time) for memory safety.
