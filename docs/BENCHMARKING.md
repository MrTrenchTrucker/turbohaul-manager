# Benchmarking Turbohaul — `bench_runner.py`

A small, dependency-light harness for head-to-head speed/resource comparisons
of inference runners **on the same model**. It was written to compare
Turbohaul's MLX path against native **oMLX** on Apple Silicon (same MLX model,
same quant), but it works against any OpenAI-compatible `/v1/chat/completions`
server.

Location: `bench_runner.py` (repo root). Output: a JSON file you can diff/plot.

## What it measures

| Metric | How | Source |
|---|---|---|
| Cold model load from disk | restart the server, then first request | `usage.model_load_duration` (oMLX) or first-request TTFT (mlx_lm, which doesn't report a load timer) |
| Warm prefill (TTFT) | streaming time-to-first-token | wall clock, stream start |
| Decode tok/s (ground truth) | non-streaming `usage.completion_tokens / latency` | server's own `usage` block |
| RSS (MB) | `ps -o rss` | idle + peak-during-generation of the server process |

**Why token counts come from `usage`, not SSE chunks:** both oMLX and
`mlx_lm server` emit `reasoning_content` + `keepalive` pseudo-chunks during
streaming. Counting SSE lines as tokens is wrong by a wide margin. The
non-streaming `usage.completion_tokens` is the authoritative count, so decode
tok/s is computed as `completion_tokens / wall_latency`. Streaming is used
**only** to time the first token (TTFT).

## Usage

```bash
# oMLX (restart the server first to guarantee a fresh cold load)
python bench_runner.py --base http://127.0.0.1:8011 --key omlx-api-key \
    --model Qwen3.5-9B-MLX-8bit --out /tmp/results_omlx.json --cold

# Turbohaul (restart the manager first; --model is the Turbohaul manifest tag)
python bench_runner.py --base http://127.0.0.1:11401 --key "" \
    --model qwen3.5-9b-mlx-8bit --out /tmp/results_turbohaul.json --cold
```

`--cold` means "the server was just restarted" — the harness assumes the first
request pays the disk-load cost. For oMLX that cost is reported directly via
`usage.model_load_duration`; for `mlx_lm` it is embedded in the first request's
latency (subtract a known warm-gen time to estimate it).

## Methodology rules (read before trusting numbers)

1. **Run runners sequentially, never two big models resident at once.** They
   share unified memory / GPU and will compete. Stop runner A fully (kill its
   process) before starting runner B.
2. **Use a unique prompt per request when measuring warm decode.** Turbohaul's
   completion proxy has an idempotent completion-cache (M5/WIN2) keyed on
   `messages`, and — more importantly — a warm-inherit path (see below) that can
   serve a *previous* turn's prompt. Reusing the same prompt, or sending no
   `session_id`, can make a "warm" request return near-instantly or answer the
   wrong prompt. Vary the prompt text each call to defeat both.
3. **mlx_lm does not report `model_load_duration`.** Cold load is inferred from
   first-request latency minus a separately measured warm-gen time.
4. **RSS is sampled on the *server* PID** (`find_pid(port)`), not the sidecar.
   For Turbohaul that is the manager process (~80 MB), **not** the `mlx_lm`
   sidecar (~9–10 GB for a 9B-8bit model). To capture the sidecar RSS, sample
   `pgrep -f "mlx_lm server"` separately. (This is a known limitation of the
   harness; fix by passing the sidecar PID or resolving it from the manager.)

## Known caveats / bugs discovered while using this tool

- **Turbohaul completion-cache replay (MLX, 2026-07-13).** On the Apple-Silicon
  MLX path, a request that arrives with **no `session_id`** inherits the idle
  holder's stale `client_meta` (including the *previous* request's `messages`)
  via the "idle-hot warm-inherit" branch in `manager._process_slot`. The result:
  the first request answers correctly, every later request (same or different
  prompt) replays the first prompt's answer. `mlx_lm server` itself is
  prompt-aware (verified by talking to it directly), so the corruption is in
  Turbohaul's slot reuse, not in MLX. Fix direction: do not inherit stale
  `client_meta["messages"]` for backends without a KV bin to restore (MLX). See
  `docs/MLX_BACKEND_PORT_SPEC.md` and the open issue notes. Until fixed, send a
  distinct `session_id` per logical conversation and a unique prompt per request
  to get real (if still ~3x slower than oMLX) numbers.
- **oMLX does not cache**: each request generates fresh, so its warm numbers are
  reliable as-is.

## Reference results (Apple Silicon, 64 GB, Qwen3.5-9B-MLX-8bit)

See `bench_results_2026-07-13.md` in the repo root. Headline: oMLX decode
≈ 49.9 tok/s, Turbohaul MLX first-request decode ≈ 16–19 tok/s (≈2.6–3x
slower) with comparable load time and model RSS; the gap is generation
throughput, not memory.
