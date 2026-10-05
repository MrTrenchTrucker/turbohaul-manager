# Multi-Agent GPU Sharing via Turbohaul

**Status:** Supported pattern. Multiple agents can share one GPU via Turbohaul-Manager; model swaps between different models are serialized through the same queue.

---

## What this is

Turbohaul-Manager lets **multiple AI agents target the same GPU (or GPUs) at the same time** through a single inference endpoint. Agents do not negotiate, queue manually, or coordinate — they submit requests to Turbohaul and Turbohaul handles slot ownership, model swapping, and eviction.

The pattern is **multiplexed serialization**, not parallel execution:

- Multiple agents may submit concurrently.
- Turbohaul holds requests in a queue: FIFO order, with bounded same-model affinity (`queue.max_consecutive_same_model`, `queue.max_other_model_wait_s`) and Fast Lane priority (see [FAST_LANE.md](./FAST_LANE.md)).
- By default (`queue.max_parallel_sidecars` = 1) one `llama-server` child process holds the GPU at any moment.
- Same-model follow-ups inherit the warm process (ACTIVE_MATCH cascade + IDLE_HOT warm-hold, `queue.idle_hot_load_seconds`, default 600 s).
- Different-model requests trigger clean teardown + spawn (model swap).
- VRAM, RAM, CPU, and IO-wait guardrails refuse spawn when the host is at risk.

By default this is sharing-by-time-slicing, not concurrent-tensor-parallelism. When a card fits only one production-sized model at a time, parallel execution of two large LLMs on the same card is not the operational goal.

## Example: two agents, two models

In an example setup, **two agents default their OpenAI-shape calls to Turbohaul**, and a third agent routes to Turbohaul on a per-task basis:

| Agent | Role | Default LLM backend | Model when routed to Turbohaul |
|---|---|---|---|
| Advisor A | advisor (27B reasoning) | Turbohaul `:11401/v1` | my-model-27b |
| Advisor B | advisor (35B reasoning) | Turbohaul `:11401/v1` | my-model-35b |
| Worker | tool-calling worker | hosted cloud API (default) | my-model-27b (per-task tool calls) |

The point of the example isn't "three agents call Turbohaul concurrently" — it is: **when traffic enters Turbohaul from multiple sources, the queue serializes it and the model-swap path between the 27B and the 35B model runs without the two colliding.**

Example sequence (model-swap serialization on a shared GPU):

1. The worker runs a multi-tool task that routes through Turbohaul for my-model-27b inference.
2. The same agent then calls Advisor B (which defaults to Turbohaul) for advice.
3. Turbohaul sees the new request for a different model → finalizes the 27b slot → spawns 35b.
4. Follow-up tools re-target 27b → Turbohaul finalizes 35b → re-spawns 27b.

What to look for:

- Slot cycle `27b → 35b → 27b`, with each model finalized before the next one spawns.
- `evictions.total_lifetime` in `/status` counts every resident unload (model-swap make-room, idle timeout, driver-death reap) as well as client-disconnect evictions, so it increments on each swap.
- `/status` (`loading`, `active` and `idle_hot` blocks) shows the transitions.
- Spawn flags can be checked on the live process: both spawns should carry the TurboQuant flag set (see [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md)) in `/proc/<pid>/cmdline`.

## Architecture summary

```
Agent A ─┐
Agent B ─┼──► Turbohaul ──► FIFO queue ──► single llama-server slot ──► GPU
Agent C ─┘     │
               ├─ ACTIVE_MATCH cascade (same-thread same-model follow-ups inherit warm process)
               ├─ IDLE_HOT warm-hold, default 600 s (different-thread same-model reuse)
               ├─ Clean teardown + spawn on different-model request
               ├─ Background sweeper (60s cadence) finalizes orphaned STAGED slots (older than 24 h by default)
               └─ Guardrails (VRAM / RAM / CPU / IO-wait) refuse spawn under load
```

Agents see a standard OpenAI-shape `POST /v1/chat/completions`. Turbohaul is transparent — no agent-side changes are needed.

## When this matters

- Shared single-GPU box with multiple agents.
- Mixed-model traffic (different agents need different models; some need 27B, some need 35B).
- Cost-sensitive deployment where one large GPU is preferred over multiple smaller cards.
- Multi-agent patterns where you want one upgrade path (one model registry, one queue, one observability surface).

## When this does not apply

- Concurrent decoding of several conversations inside one engine: that is the manifest's `llama_server_flags.parallel` (see [MODEL_CONFIG_REFERENCE.md](./MODEL_CONFIG_REFERENCE.md)), not something the queue provides by itself.
- Sub-100ms latency requirements (queue depth + model-swap cost dominates).
- Models large enough that two cannot coexist in VRAM (Turbohaul does not magic this — it serializes).

## See also

- [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md) — KV cache compression flag doctrine.
- [PERSISTENCE_CHECKLIST.md](./PERSISTENCE_CHECKLIST.md) — persistence and recovery checklist for a Turbohaul deployment.
- Repo root `README.md` for `/v1/chat/completions` API surface and quickstart.
