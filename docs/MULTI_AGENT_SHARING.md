# Multi-Agent GPU Sharing via Turbohaul

**Status:** Proven in production 2026-05-19. Multiple concurrent agents share one Blackwell GPU via Turbohaul-Manager. Smoke test exercised model-swap serialization across two different models routed from one worker.

---

## What this is

Turbohaul-Manager lets **multiple AI agents target the same GPU at the same time** through a single inference endpoint. Agents do not negotiate, queue manually, or coordinate — they submit requests to Turbohaul and Turbohaul handles slot ownership, model swapping, and eviction.

The pattern is **multiplexed serialization**, not parallel execution:

- Multiple agents may submit concurrently.
- Turbohaul holds requests in a FIFO queue.
- One `llama-server` child process holds the GPU at any moment.
- Same-model follow-ups inherit the warm process (ACTIVE_MATCH cascade + IDLE_HOT 5-min warm-hold).
- Different-model requests trigger clean teardown + spawn (model swap).
- VRAM, RAM, CPU, and IO-wait guardrails refuse spawn when the host is at risk.

This is sharing-by-time-slicing, not concurrent-tensor-parallelism. A 24GB Blackwell card fits one production-sized model at a time, so parallel execution of two large LLMs on the same card is not the operational goal.

## What was proven 2026-05-19

In the deployment, **two reasoning-advisor agents default their OpenAI-shape calls to Turbohaul**, and a third worker agent routes to Turbohaul on a per-task basis:

| Agent | Role | Default LLM backend | Model when routed to Turbohaul |
|---|---|---|---|
| Advisor (27B) | reasoning advisor | Turbohaul `:11401/v1` | qwen3.6-27b-dense |
| Advisor (35B) | reasoning advisor | Turbohaul `:11401/v1` | qwen3.6-35b-moe |
| Worker | tool-calling worker | cloud endpoint (default) | qwen3.6-27b-dense (per-task tool calls) |

The point of the smoke wasn't "three agents call Turbohaul concurrently" — it was: **when traffic enters Turbohaul from multiple sources, the queue serializes cleanly and the model-swap path between 27B-dense and 35B-MoE works without collision.**

Smoke test (model-swap serialization on shared GPU):

1. The worker agent ran a multi-tool task that routed through Turbohaul for qwen3.6-27b-dense inference.
2. The same agent then called the 35B reasoning advisor (which defaults to Turbohaul) for advisement.
3. Turbohaul saw the new request for a different model → finalized 27b slot → spawned 35b.
4. The advisor returned a substantive 3-bullet verdict at 85% confidence.
5. Follow-up tools re-targeted 27b → Turbohaul finalized 35b → re-spawned 27b.

Observed:

- Slot cycle `27b → 35b → 27b` clean.
- `evictions.total_lifetime` held at **0** (no force-eviction needed; natural finalization).
- `/status` transitions tracked the swap in real time.
- Both spawns ran with the full TurboQuant flag set (see [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md)) verified live via `/proc/<pid>/cmdline`.
- No collisions, no crashes, no advisor timeouts on the spawn boundary.

## Architecture summary

```
Agent A ─┐
Agent B ─┼──► Turbohaul ──► FIFO queue ──► single llama-server slot ──► GPU
Agent C ─┘     │
               ├─ ACTIVE_MATCH cascade (same-thread same-model follow-ups inherit warm process)
               ├─ IDLE_HOT 5-min warm-hold (different-thread same-model reuse)
               ├─ Clean teardown + spawn on different-model request
               ├─ Background sweeper (60s cadence) finalizes orphaned STAGED slots
               └─ Guardrails (VRAM / RAM / CPU / IO-wait) refuse spawn under load
```

Agents see a standard OpenAI-shape `POST /v1/chat/completions`. Turbohaul is transparent — no agent-side changes are needed.

## When this matters

- Shared single-GPU box with multiple agents.
- Mixed-model traffic (different agents need different models; some need 27B, some need 35B).
- Cost-sensitive deployment where one Blackwell beats running multiple smaller cards.
- Multi-agent patterns where you want one upgrade path (one model registry, one queue, one observability surface).

## When this does not apply

- True parallel-tensor inference (use `--parallel N` on a single `llama-server` directly, not Turbohaul).
- Sub-100ms latency requirements (queue depth + model-swap cost dominates).
- Models large enough that two cannot coexist in VRAM (Turbohaul does not magic this — it serializes).

## See also

- [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md) — KV cache compression flag doctrine that made this fit cleanly.
- [PERSISTENCE_CHECKLIST.md](./PERSISTENCE_CHECKLIST.md) — deployment persistence checklist.
- Repo root `README.md` for `/v1/chat/completions` API surface and quickstart.
