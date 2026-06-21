# Turbohaul-Manager — Architecture

**Repository:** [github.com/MrTrenchTrucker/turbohaul-manager](https://github.com/MrTrenchTrucker/turbohaul-manager)
**License:** MIT (see [LICENSE](../LICENSE) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md))

Turbohaul-Manager is a standalone HTTP inference server for running large local models on a **single GPU**. It is a FastAPI/asyncio **management plane** that supervises one [TurboQuant-enabled `llama.cpp`](https://github.com/TheTom/llama-cpp-turboquant) `llama-server` subprocess at a time, presents an **Ollama- and OpenAI-compatible** API surface, and time-slices a single card across any number of clients with no client-side coordination.

This document describes the system as it exists today: what the components are, how a request moves through them, and the invariants that keep one GPU healthy under concurrent load.

---

## 1. Mission and shape

Turbohaul-Manager is a **one-stop shop for local inference on a single GPU**: one endpoint, one model registry, one queue, one observability surface.

- It mimics the **Ollama API surface** (`/api/chat`, `/api/tags`, `/api/show`) and the **OpenAI surface** (`/v1/chat/completions`, `/v1/embeddings`), so any Ollama- or OpenAI-aware client can use it transparently.
- It uses a **TurboQuant-enabled fork of `llama.cpp`** as its inference backend, supervised as a `llama-server` subprocess per active model. Turbohaul-Manager embeds no model code; it owns process lifecycle, queueing, and HTTP shaping.
- It provides **bring-your-own-model (BYOM)** blob storage: pull a GGUF from a HuggingFace repo, an HTTPS URL (SSRF-guarded), or import a local file.
- It provides a **FIFO request queue with grace and idle hot-hold** that solves the cross-process race a naive "load on demand" manager exhibits when several clients hit one GPU at once.

The single load-bearing design choice is the **single-slot serialization model**: at most one `llama-server` child holds the GPU at any moment (`max_parallel_sidecars` defaults to `1`). A 24 GB-class card fits one production-sized model at a time, so Turbohaul-Manager does not run two large models side by side. Instead it **time-slices**:

- Concurrent requests from any number of clients land in a FIFO queue.
- The queue is drained one slot at a time onto the single warm process.
- Same-model, same-thread follow-ups inherit the warm process (no reload).
- A request for a *different* model triggers a clean teardown and respawn (a "model swap").

This is **multiplexed serialization, not tensor parallelism**. Multiple independent clients share one GPU, the warm model is reused aggressively, and swaps are clean and supervised. Serialization is *cross-model*, not within a model — see §5 for the per-model concurrent dispatch capability that lets a single model serve multiple requests at once on its one warm sidecar.

### Terminology

- **Sidecar** — one supervised `llama-server` subprocess running a single model from explicit manifest flags.
- **Slot** — a queue entry `{slot_id, model_tag, prompt, thread_id, state, …}`. Cold until activated.
- **Active slot** — the slot currently loaded in VRAM and serving a request.
- **Grace** — the short window after a request completes where the model stays loaded for `thread_id`-matched follow-ups.
- **Idle hot-hold** — the longer window after the queue drains where the last model stays warm for any same-model request.

### What it fits — and does not

This design fits a **shared single-GPU box with mixed-model traffic** and a single upgrade path and observability surface. It does **not** fit sub-100 ms latency targets (queue and swap cost dominate) or workloads that need two large models resident at once.

---

## 2. Components

```
                     ┌──────────────────────────────────────────────┐
   OpenAI client ───►│  FastAPI app  (api/)                          │
   Ollama client ───►│   /v1/chat/completions  /api/chat  /api/tags  │
   curl / SDK    ───►│   /api/manifests /api/config /status /ws/state │
                     │   /v1/embeddings  /ui/live/output/stream (SSE) │
                     └───────────────┬──────────────────────────────┘
                                     │ submit(slot)
                                     ▼
                     ┌──────────────────────────────────────────────┐
                     │  Manager  (manager.py)                        │
                     │   • worker loop (single consumer)             │
                     │   • request state machine (fsm.py)            │
                     │   • grace / idle hot-hold timers (queue.py)   │
                     │   • event bus → /ws/state (redacted)          │
                     └───┬───────────────┬───────────────┬──────────┘
                         │               │               │
                         ▼               ▼               ▼
                 ┌──────────────┐ ┌─────────────┐ ┌────────────────┐
                 │ Safety gates │ │ Subprocess  │ │ Singleton +    │
                 │ (safety.py)  │ │ supervision │ │ orphan reaper  │
                 │ VRAM/RAM/CPU │ │(subprocess_ │ │ (singleton.py) │
                 │ /IO + KV-fit │ │  mgr.py)    │ │ flock + GPU    │
                 └──────────────┘ └──────┬──────┘ └────────────────┘
                                         │ Popen (setsid, argv-only)
                                         ▼
                                 ┌────────────────┐      ┌─────────────┐
                                 │  llama-server  │◄────►│  GPU (VRAM) │
                                 │  (subprocess)  │      └─────────────┘
                                 └────────────────┘
            ┌───────────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────────┐
   Storage: │ blob store    │  │ manifests/ │  │ config yaml │  │ state.sqlite │
            │ (blob_store)  │  │ (manifest) │  │ (config.py) │  │ (state.py)   │
            │ sha256 CAS    │  │ per-model  │  │ boot+runtime│  │ slots+audit  │
            └───────────────┘  └────────────┘  └─────────────┘  └──────────────┘
```

| Module | Responsibility |
|---|---|
| `api/` | FastAPI routes — OpenAI + Ollama surfaces, embeddings (`/v1/embeddings`), manifests/config/status/logging/WS, and the live-output SSE stream (`api/live_stream.py` → `/ui/live/output/stream`). Request shaping, streaming (SSE), `response_format` validation. Tool-call recovery lives in `api/tool_call_recovery.py`. |
| `manager.py` | The single worker loop, the request state-machine driver, grace/idle timers, the event bus, per-model concurrent fan-out (`_fan_out_and_drain` / `_drain_nonstreaming_riders`), teardown + reconciliation orchestration, and the `status_snapshot` payload. |
| `fsm.py` | The legal-transition table and guards for the per-request state machine. Any illegal hop raises `InvalidTransition` rather than silently corrupting state. |
| `slot.py` | The `Slot` dataclass, the `SlotState` enum, and `derive_thread_id_prefix_hash` (the redacted thread-id prefix). |
| `queue.py` | Two-tier FIFO (acceptance buffer + bounded staging), grace timer, idle timer, model-affinity pop. |
| `safety.py` | Pre-spawn host guardrails: free RAM, free VRAM, CPU load-per-core, IO-wait, and a closed-form KV-cache fit check. |
| `singleton.py` | Singleton-per-GPU enforcement: state-file `flock`, foreign-GPU scan, orphan `llama-server` reaper. |
| `subprocess_mgr.py` | Spawn (argv-only, `setsid`), health polling, drained-SIGTERM teardown via `killpg`, VRAM-clear verification, binary integrity pin. |
| `manifest.py` | Per-model manifest schema, the closed `llama_server_flags` allowlist + denylist, atomic ETag/If-Match writes. |
| `config.py` | The split config model — frozen `BootConfig` (restart to change) vs mutable `RuntimeConfig` (PUT-able). |
| `blob_store.py` | Content-addressed (sha256) GGUF storage, atomic write + verify, immutability, re-verify-on-stage helper. |
| `ssrf_guard.py` | URL/IP validation for pulls — HTTPS-only, private-range blocklist, double-resolve. |
| `state.py` | `state.sqlite` schema (slots + audit + pull history), the audit connection pool, boot reconciliation. |
| `live_monitor.py` | The live inference monitor: a single ~1 Hz `LiveSlotsPoller` (pure observer reading the sidecar `/slots`) that computes tok/s + progress for the `generation` block on `/status`. |
| `frontend/` | React + Vite + TypeScript single-page UI (dashboard, live inference, models, queue, config editor, logs) served from the same port. |

---

## 3. Request flow

One sentence per stage:

1. A client POSTs a chat request; the route validates `response_format`, derives a `thread_id` if absent, and **submits a slot** to the manager.
2. The manager's **single worker loop** pops the next FIFO slot, runs the **safety gates**, and either **inherits a warm process** (same model still hot) or **spawns** a new `llama-server`.
3. The slot goes **ACTIVE**, the request streams or completes (or, for a `--parallel N` model, **fans out** to serve up to N same-model riders concurrently on the one sidecar, draining them all before advancing), then enters a **grace** window for same-thread follow-ups.
4. When grace expires the model is either **handed to the idle hot-hold** (stays warm) or **torn down** (SIGTERM → group kill → VRAM-clear verification).
5. State transitions are audited to `state.sqlite` and broadcast (redacted) on `/ws/state`. The `/status` snapshot exposes the full live picture, and high-volume output **text** is carried separately on the `GET /ui/live/output/stream` SSE endpoint.

---

## 4. Request lifecycle state machine

Every request becomes a **slot** that walks a finite state machine. The legal transitions are enforced by a table in `fsm.py`; an illegal hop raises `InvalidTransition` rather than silently corrupting state.

### 4.1 The states

There are **12** slot states (`SlotState`, a `str`-backed enum). Two useful classes cut across them:

- **Warm states** — the `llama-server` process is loaded in VRAM: `LOADING, ACTIVE, GRACE, GRACE_BUSY, ACTIVE_MATCH, IDLE_HOT`.
- **In-flight states** — a request is actively being processed: `ACTIVE, GRACE_BUSY, ACTIVE_MATCH`.

| State | Meaning |
|---|---|
| `RECEIVED` | Initial state when a slot is created. Transient — enqueued immediately. |
| `ACCEPT_BUFFER` | Parked in the (large) acceptance buffer because the bounded staging queue is full. |
| `STAGED` | In the bounded FIFO staging queue, awaiting activation. |
| `LOADING` | The sidecar is being spawned and health-waited — or a warm process is being inherited (fast path). |
| `LOADING_FAIL` | Spawn/health failure, a missing-manifest bail, or a safety-gate refusal. |
| `ACTIVE` | The request is being served against the warm sidecar (streams or completes here). |
| `GRACE` | Request done; sidecar stays warm watching for a same-`(thread_id, model_tag)` follow-up. |
| `GRACE_BUSY` | Defined in the table for completeness; the live worker promotes `STAGED → ACTIVE_MATCH → ACTIVE` directly and does not enter it. |
| `ACTIVE_MATCH` | A matched follow-up promoted onto the warm sidecar during the grace window. |
| `POPPED` | Slot lifecycle complete (finished or failed terminally). |
| `IDLE_HOT` | The idle hot-hold. On the happy path this is a manager-level handle hold, not a per-slot DB state (see §4.4). |
| `COLD` | Terminal. Subprocess gone, slot finalized in `state.sqlite`. |

### 4.2 The legal-transition table

```
RECEIVED      → ACCEPT_BUFFER, STAGED
ACCEPT_BUFFER → STAGED, COLD
STAGED        → LOADING, COLD, ACTIVE_MATCH
LOADING       → ACTIVE, LOADING_FAIL, COLD
LOADING_FAIL  → STAGED, POPPED
ACTIVE        → GRACE, ACTIVE_MATCH
GRACE         → GRACE_BUSY, POPPED, ACTIVE
GRACE_BUSY    → GRACE, POPPED
ACTIVE_MATCH  → ACTIVE
POPPED        → IDLE_HOT, STAGED, COLD
IDLE_HOT      → ACTIVE, POPPED, COLD
COLD          → ∅  (terminal)
```

`transition(old, new)` raises `InvalidTransition` if the hop is not in the table. Helpers exist for `can_transition`, `legal_targets`, and `is_terminal` (true only for `COLD`).

### 4.3 The happy path

```
        submit
          │
          ▼
      RECEIVED
          │ enqueue (staging has room)
          ▼
       STAGED  ──── staging full ────► ACCEPT_BUFFER ──┐ promoted when room frees
          │ worker pops next                           │
          ▼ ◄──────────────────────────────────────────┘
       LOADING
          │   ┌─ warm-inherit: idle process for the same model is still hot → skip spawn
          │   └─ spawn: new llama-server, poll /health until ready
          ▼
        ACTIVE  ── streams or completes ──┐
          │                               │
          ▼                               │
        GRACE  ◄─── same-thread follow-up ─┘  (ACTIVE_MATCH cascade, §5)
          │ grace window expires, no follow-up
          ▼
        POPPED
          │
          ├─ idle hot-hold enabled? → keep process warm (manager holds the handle)
          │                            slot row marked COLD, reason "grace-expired-held-idle"
          │
          └─ else → teardown (SIGTERM → group kill → VRAM-clear verify) → COLD
```

The worker loop:

1. `pop_next()` returns a `STAGED` slot. `STAGED → LOADING`.
2. **Warm-inherit** the idle process if it's the same model and not expired, **or spawn** a new `llama-server` and `wait_until_healthy`.
3. On health failure: `LOADING → LOADING_FAIL → POPPED`, teardown, fail the caller's future.
4. `LOADING → ACTIVE`; then one of three dispatch shapes:
   - **Non-streaming (default):** `await self._complete_fn(slot, handle)`, resolve the completion future.
   - **Streaming:** the slot stays `ACTIVE` for the *full stream lifetime*. The worker hands the `SidecarHandle` to the HTTP route after health-200; the route owns the `httpx.stream()` and signals when the stream closes (normal exhaustion, client disconnect, or error); the worker blocks on that event — with a hard 3600 s cap — before advancing. Keeping the slot `ACTIVE` the whole time is what enforces the single-slot invariant.
   - **Fan-out (`parallel:N` only):** `_fan_out_and_drain` (see §5).
5. `ACTIVE → GRACE`; start the grace timer.
6. Grace loop: each matched follow-up walks `STAGED → ACTIVE_MATCH → ACTIVE → GRACE` on the same warm process.
7. `GRACE → POPPED`; hand the process to the idle hold or tear it down.

### 4.4 As-built notes

- **`IDLE_HOT` is a manager-level handle hold on the happy path.** When a model stays warm after grace, the *slot row* is written `COLD` (reason `grace-expired-held-idle`) while the **manager** holds the live process handle (`_idle_handle` / `_idle_model_tag` / `_idle_expires_at`). The next same-model request inherits that handle. The `IDLE_HOT` enum member drives warm-state classification and the transition table.
- **The warm-hold duration is client-controllable** (§6): the grace→idle hold honors the latest request's `keep_alive`.
- **Boot recovery is a reconciliation flow, not an FSM state.** On startup the manager sweeps stale/orphaned rows straight to terminal `COLD` (see §9).

### 4.5 Failure handling

| Failure | Path | Result |
|---|---|---|
| Health-poll timeout (cold load never became ready) | `LOADING → LOADING_FAIL → POPPED` | Teardown; future fails with `loading-fail-health-timeout`. Timeout = `loading_health_timeout_s` (default **600 s**). |
| Manifest missing while a warm process for a *different* model is held | `LOADING → LOADING_FAIL → POPPED` | Fail fast **without** tearing down the good warm process — a bogus tag must not cost a healthy warm hold. |
| Safety gate refusal | `… → LOADING_FAIL → POPPED` | No spawn; future fails with `safety_gate_refused`. |
| Client disconnects before activation | queue eviction | Slot flagged evicted; future fails with `SlotEvictedError` → the route returns **HTTP 499**. |
| Matched follow-up whose state drifted | `_force_cold` | The worker terminal-parks the matched slot (`active_match_state_drift:<state>`) and keeps driving the anchor's grace window; the fan-out admit loop applies the same guard (`fanout_rider_drift:<state>`). |
| Uncaught worker exception | `_force_cold` | Future failed, active process torn down **before** force-cold (no PID leak), slot walked to `COLD`. |

A failed load surfaces to the client immediately; there is no automatic load-retry counter — the client retries.

### 4.6 Client-disconnect eviction and the background sweeper

When a client closes the connection before its slot activates, the slot is **evicted** from the queue and its future fails with a dedicated `SlotEvictedError` (deliberately *not* `CancelledError`, which would slip past `except Exception`) so the route can map it to **HTTP 499**. To keep the hot path off the SQLite fsync stall, the row's terminal finalize is **deferred** to a background sweeper that runs every `background_sweep_interval_s` (default **60 s**) and finalizes orphaned `STAGED`/`pid IS NULL` rows older than `background_sweep_min_age_s` (default **24 h**) to `COLD` in batches.

---

## 5. Queue, grace, idle hot-hold, and per-model concurrency

This is the heart of how Turbohaul-Manager keeps one GPU busy and warm without clients coordinating: a **two-tier FIFO queue**, a **grace window** with an **ACTIVE_MATCH cascade**, an **idle hot-hold**, and **per-model concurrent dispatch**.

### 5.1 The two-tier FIFO queue

```
   submit() ─► ┌──────────────────┐   full?   ┌────────────────────┐
               │  staging queue    │◄──────────│  acceptance buffer  │
               │  (bounded FIFO)   │  promote  │  (large overflow)   │
               │  default max 100  │           │  default max 10000  │
               └─────────┬─────────┘           └────────────────────┘
                         │ pop_next()
                         ▼
                   worker loop (single consumer)
```

- **Staging queue** — the bounded FIFO the worker drains, default depth **100** (`staging_queue_depth`). Slots here are `STAGED`.
- **Acceptance buffer** — a large overflow deque, default cap **10000** (`acceptance_buffer_max`). When staging is full, new slots land here as `ACCEPT_BUFFER` and are promoted into staging as room frees.
- Both are guarded by one async lock. Back-pressure is fail-fast: `QueueFull` (buffer at cap) and `QueueClosed` (shutting down) are exceptions, not blocking waits.

**Model-affinity pop (live by default).** `pop_next` takes an optional `warm_model_tag`, and the worker loop always passes the model currently warm. With the production defaults (`max_consecutive_same_model=3`, `max_other_model_wait_s=20.0`) the worker **prefers popping a same-model staged slot** so the warm sidecar is reused instead of swapped, rather than strictly draining the FIFO head. The FIFO head is forced only when:

- the FIFO-head other-model request has aged past `max_other_model_wait_s` (head starvation), **or**
- the worker has already popped `max_consecutive_same_model` of this model in a row (fairness batch cap).

The strict-FIFO path is preserved exactly as a no-op when `warm_model_tag` is `None` or the affinity-disabling defaults are set.

### 5.2 The grace window

When a request finishes, its slot goes `ACTIVE → GRACE` and a grace timer starts. During the window the `llama-server` process **stays loaded** so a quick follow-up on the same conversation gets a sub-second warm handoff instead of a cold reload.

- `grace_seconds` — the window length, default **30 s**.
- The match key is `(thread_id, model_tag)` and the timer must not be expired.
- `max_grace_extensions` — the **starvation cap**, default **5**. Each matched follow-up may *extend* the window, but after this many consecutive extensions the slot is forced to `POPPED` so one chatty conversation can't hold the GPU forever while other queued work waits.

**`thread_id` — the warm-reuse key.** `thread_id` is a client-supplied routing hint (not auth). Same-thread, same-model follow-ups within the grace window reuse the warm process.

- SDK and application clients pass an explicit `thread_id` (top-level body field, or via `extra_body` for OpenAI SDKs).
- **Naive clients that don't supply one** get an auto-derived `thread_id`: take the first 64 whitespace tokens of the prompt, hash `sha256(model_tag + "\0" + tokens)`, and use `"auto-" + hexdigest[:24]`. Same model + same leading prompt prefix → same thread → warm reuse, with no client awareness of thread semantics.

> **Practical rule:** for multi-turn agent loops, always pass the same `thread_id` on every turn. Without it, turns after the first re-derive (or, with distinct prompts, get a fresh thread) and pay a cold-load cost.

### 5.3 The ACTIVE_MATCH cascade

Within an open grace window the worker actively pulls matching follow-ups onto the warm process — the cascade that makes multi-turn loops fast. There are two entry points:

- **A fresh submission arrives during grace.** If it matches the open grace `(thread_id, model_tag)`, it is enqueued at the **FIFO head** (it jumps the line, since the process is already warm) and the grace timer is extended, subject to the cap.
- **The grace loop pulls a matching staged slot.** While the grace deadline hasn't passed, the worker atomically finds and removes the first matching `STAGED` slot, reuses the anchor's process (`STAGED → ACTIVE_MATCH → ACTIVE`, with a state-drift guard), runs the request, then `ACTIVE → GRACE` (resetting the deadline if the extension cap isn't hit) and `GRACE → POPPED` for the matched slot. The **anchor** slot remains the grace driver and the warm process is reused.

```
  Turn 1: cold load → ACTIVE → GRACE  ┐
  Turn 2 (same thread_id, within 30s):  │  warm process reused,
          STAGED → ACTIVE_MATCH → ACTIVE │  sub-second — no reload
          → GRACE                        │
  Turn 3 (same): … → ACTIVE_MATCH → …    ┘  (until grace expires or
                                            max_grace_extensions hit)
```

### 5.4 The idle hot-hold

When grace finally expires with no follow-up, the slot goes `GRACE → POPPED`, and the manager decides whether to keep the process **warm** so an unrelated request for the *same model* still skips the cold load. The hold duration follows the latest request's `keep_alive` (§6):

- **Hold:** if the computed idle duration is > 0 and a live process exists, the manager moves the active handle into the idle holder (`_idle_handle`, `_idle_model_tag`, `_idle_expires_at`), writes an `idle_hot_enter` audit event, and finalizes the *slot row* to `COLD` (reason `grace-expired-held-idle`). The process stays alive.
- **Teardown:** otherwise the process is torn down (drained-SIGTERM → VRAM-clear verify) and the idle timer is cleared.

When the next slot reaches `LOADING`:

- **Warm-inherit** if the idle holder exists, its model tag matches the new slot, and it hasn't expired → adopt the handle, clear the idle holder, **skip spawn + health-wait** entirely.
- **Model swap** if the new slot wants a *different* model → tear down the stale idle holder first, then spawn the new model.

When the queue is empty and the idle holder has passed its expiry, an identity-guarded tick (so a holder that was just refreshed isn't killed) tears down the idle process in the background — without blocking the worker loop on the SIGTERM grace.

### 5.5 Per-model concurrent dispatch (fan-out)

For the common **`parallel:1`** model the single-consumer, one-in-flight picture above holds exactly. When the active model's manifest declares **`parallel > 1`**, the worker serves multiple same-model requests **concurrently on the one warm sidecar** via `llama.cpp` continuous batching — per-model concurrency on top of the single-model-resident invariant.

The `ACTIVE` step does not serve a single request. Instead the worker calls `_fan_out_and_drain(slot, handle, n_parallel)` and admits up to `handle.parallel` same-model riders onto the **one** shared sidecar. The admission cap is `handle.parallel` directly — pinned on the handle from the actual `--parallel` argv — **not** `max_parallel_sidecars` (which is the separate multi-process knob that defaults to 1). The fan-out is **one-shot** (admit extra riders from the queue exactly once, never refilling, so it can never livelock a model swap):

- The anchor (rider 0) is already `ACTIVE`. Each extra rider walks `STAGED → LOADING → ACTIVE` on the shared handle.
- Riders are dispatched concurrently — non-streaming riders via concurrent completion POSTs (`_drain_nonstreaming_riders`), streaming riders via their own routes and stream-done barriers.
- As each rider finishes it is decoupled from the shared sidecar (`pid` set to `None` so teardown can never reap the shared process) and force-cold'd with reason `fanout_rider_drained`. Only the **anchor** continues the normal `GRACE` / teardown flow.
- A hard **drain-before-swap** guard (`assert not self._inflight`) protects `ACTIVE → GRACE`: `_fan_out_and_drain` returns only when every rider's route has genuinely closed, so the sidecar is never torn down or swapped while a rider's `httpx` stream may still be open. The worker loop remains the sole FSM mutator.

A useful pairing: a **main model** can be served strictly in series (`parallel:1`) while a **sub-agent model** is served with true parallel dispatch (`parallel:N`) whenever it is the active model — per-model parallelism layered cleanly over the single-model-loaded invariant. A `parallel > 1` manifest is cross-validated at load time (`kv_unified: true` required, `ctx_size` divisible by `parallel`, per-slot window ≥ 8192) and at the VRAM-fit gate (see §7).

### 5.6 Tunable constants (defaults)

All of these are runtime-mutable via `PUT /api/config` or the matching env var:

| Constant | Default | Env | Meaning |
|---|---|---|---|
| `max_parallel_sidecars` | `1` | `TURBOHAUL_MAX_PARALLEL` | Concurrent sidecars (single-slot invariant) |
| `staging_queue_depth` | `100` | `TURBOHAUL_STAGING_DEPTH` | Bounded FIFO depth |
| `acceptance_buffer_max` | `10000` | `TURBOHAUL_ACCEPT_MAX` | Overflow buffer cap |
| `grace_seconds` | `30` | `TURBOHAUL_GRACE_S` | Grace window length |
| `idle_hot_load_seconds` | `600` | `TURBOHAUL_IDLE_HOT_S` | Default idle hot-hold |
| `max_grace_extensions` | `5` | `TURBOHAUL_MAX_GRACE_EXT` | Starvation cap |
| `max_consecutive_same_model` | `3` | `TURBOHAUL_MAX_CONSECUTIVE_SAME_MODEL` | Affinity batch cap |
| `max_other_model_wait_s` | `20.0` | `TURBOHAUL_MAX_OTHER_MODEL_WAIT_S` | Starved other-model FIFO-head override age |
| `loading_health_timeout_s` | `600` | — | Cold-load health-wait ceiling |
| `KEEP_ALIVE_MAX_S` | `1800` | — | Hard ceiling on honored `keep_alive` |
| `background_sweep_interval_s` | `60` | — | Orphan-row sweeper cadence |
| `background_sweep_min_age_s` | `86400` | — | Min age before the sweeper finalizes a row |

---

## 6. `keep_alive` — client-controlled warm-hold

The grace→idle hold honors the *latest* request's `keep_alive` (Ollama "timer resets on request receipt" semantics — captured on `ACTIVE`, refreshed on each `ACTIVE_MATCH` promotion), capped at `KEEP_ALIVE_MAX_S = 1800 s`:

| `keep_alive` value | Idle hold |
|---|---|
| not supplied (`None`) | `idle_hot_load_seconds` (default **600 s** = 10 min) |
| a positive duration (`"10m"`, `1800`, …) | `min(value, KEEP_ALIVE_MAX_S)` — cap **1800 s** (30 min) |
| `-1` (Ollama "pin until cap") | the cap, **1800 s** |
| `0` | warm-hold disabled — tear the sidecar down immediately (reason `grace-expired`) |

`keep_alive: 0` is also the supported way to force a **cold-spawn** so a changed spawn-time manifest flag takes effect (see §8.4).

---

## 7. Safety model and process supervision

Three subsystems keep a single GPU healthy under load: **pre-spawn safety gates**, the **singleton-per-GPU invariant**, and **production-grade subprocess supervision**.

> **Gates degrade gracefully.** Every external probe (`nvidia-smi`, `/proc`) degrades to a pass / empty result if unavailable — deliberate dev-mode tolerance. Production hosts must have working `nvidia-smi` and `/proc`. **One deliberate exception:** the KV-cache-fit gate does **not** degrade open for `parallel > 1` configs — when `nvidia-smi` is unreadable and `parallel > 1`, the gate **refuses** the spawn (a blind parallel:N spawn with no VRAM probe is a guaranteed OOM risk).

### 7.1 The five safety gates

Before spawning a sidecar, the manager runs five gates (when `safety_enabled`, the default). They run **without short-circuiting** — all five always evaluate and any failing gate refuses the spawn (the slot goes `LOADING_FAIL → POPPED`).

| # | Gate | Measures | Threshold (default) |
|---|---|---|---|
| 1 | **Free RAM** | `MemAvailable` from `/proc/meminfo` | `safety_min_free_ram_mib` = **1024** |
| 2 | **Free VRAM** | GPU `memory.free` via `nvidia-smi` | `max(safety_min_free_vram_mib=512, manifest expected_vram_bytes)` |
| 3 | **KV-cache fit** | closed-form KV estimate vs free VRAM | overhead floor **1024 MiB** |
| 4 | **CPU load** | 1-min load avg ÷ CPU count | `safety_max_load_per_core` = **0.9** |
| 5 | **IO-wait** | Δ iowait jiffies from `/proc/stat` over a sample window | `safety_max_iowait_percent` = **30.0%** over **0.4 s** |

All four configurable thresholds are runtime-mutable via `PUT /api/config`. `nvidia-smi` is resolved once to an absolute path at import (PATH-injection resistant). Gate 5 blocks the spawn path synchronously for the ~0.4 s sample window — a deliberate, bounded cost.

### 7.2 The KV-cache fit gate (closed-form)

Gate 3 is the load-bearing check for user-programmable context sizes — a client that bumps `ctx_size` in a manifest is caught even if it didn't re-tune `expected_vram_bytes`. It is a calibrated closed form; no model load is required. See §8.5 for the full math. In short:

- **KV resident in VRAM (default):** require `gguf_size + kv_estimate + overhead ≤ free_vram`.
- **KV offloaded to host RAM (`no_kv_offload: true`):** the VRAM requirement drops the KV term and adds a context-linear scratch term, and a complementary check ensures the KV cache fits available host RAM.
- **Parallel:N accounting:** add a flat `(parallel − 1) × PER_SLOT_COMPUTE_FLOOR_MIB` term (`PER_SLOT_COMPUTE_FLOOR_MIB = 256`) for the per-slot compute/attention buffers. The KV term is **not** multiplied (one aggregate `ctx_size` window split across slots; a unified host-RAM KV pool is shared).

### 7.3 Singleton-per-GPU invariant

Turbohaul-Manager must be the **only writer to the GPU** on its host. Three layers enforce this:

1. **State-file `flock`** — `singleton.py` implements (and unit-tests) `acquire_state_lock`, a non-blocking exclusive `fcntl.flock` on the state file that raises `SingletonViolation` if a second instance is running. (This helper is not yet wired into boot in the shipped code; the live single-instance guarantee currently rests on layers 2–3.)
2. **Foreign-GPU compute scan** — `nvidia-smi --query-compute-apps` lists every process holding GPU memory; the manager compares against the PIDs it owns and annotates any foreign compute process. The refuse-to-start decision on a non-empty foreign list is a CLI choice.
3. **Orphan `llama-server` reaping** — if the manager crashed and left a child behind, that child re-parents to init (or the container sub-reaper). On boot the reaper scans `/proc` for `llama-server` processes whose parent is init/the sub-reaper, parses each `--port`, keeps only those in the manager's port range (`default_port_base` .. `+100`), and reaps the rest (start-time capture for PID-reuse safety, `SIGTERM`, poll up to 5 s, then `SIGKILL`).

A separate **intra-lifetime orphan scan** catches sidecars that lost their handle while the manager is still running (e.g. a cancelled unwind), walks `/proc`, matches the port range, and `SIGTERM`s any matching PID the manager isn't tracking — a safety net against handle-loss bugs.

### 7.4 Subprocess supervision

**Spawn.**

```
Popen(
  [binary, "--port", <port>, "--host", "127.0.0.1", "-m", <gguf_path>, *flag_argv],
  start_new_session=True,   # setsid → own process group → clean group teardown
  # stdout/stderr → DEVNULL
)
```

- **argv only, never a shell string** — no shell-metacharacter exposure. Flag values are validated/rejected upstream in the manifest layer (§8).
- **Host pinned to `127.0.0.1`** — the child `llama-server` is never directly reachable off-box.
- **`start_new_session=True` (`setsid`)** puts the child in its own process group so teardown can `killpg` the whole group.
- **`stdout`/`stderr` → `DEVNULL`.** Intentional: an undrained `PIPE` fills the OS pipe buffer once `llama-server` logs enough and blocks the child's `write()`, breaking the drained-teardown contract. Structured logs come from `llama-server`'s own log file.
- **Binary-pin / TOCTOU close:** if a verified file descriptor for the backend binary is held, the spawn execs via `/proc/self/fd/<fd>` so the exact hashed inode runs even if the path is swapped after verification.

**Health-poll.** Poll `GET http://127.0.0.1:<port>/health` every 2 s. **Ready** on HTTP 200 with a JSON body whose lowercased `status` is one of `ok / ready / healthy / loaded`. A 200 body that isn't a dict or is missing `status` raises `SchemaMismatch` (catches a backend health-contract change). The wait fast-fails if the child has already exited, but a slow-but-alive cold load is never killed. Overall ceiling: `loading_health_timeout_s` (default 600 s) — a 20 GB GGUF can legitimately cold-load for minutes.

**Drained teardown.** Signal the whole process group via `killpg(pgid, SIGTERM)` (works because of `setsid` at spawn), poll every 0.2 s for up to the configured window, then `killpg(pgid, SIGKILL)` on timeout. Every exit path does an explicit `waitpid` reap so no `<defunct>` zombie is left behind to defeat the orphan reaper's parent check.

**VRAM-clear verification.** After teardown the manager polls `nvidia-smi --query-gpu=memory.used` every 1 s for up to 30 s, treating a ≥ 90 % drop of the expected footprint as cleared — defending the "CUDA allocator stuck after kill" failure mode. If `nvidia-smi` is unavailable it trusts the kill.

**Graceful shutdown.** On manager `SIGTERM`: stop the worker + sweeper, drain the queue (failing waiting callers' futures so they don't hang to the request timeout), tear down the idle holder, and release the binary fd and the singleton lock.

---

## 8. Backend, manifests, and configuration

### 8.1 TurboQuant KV-cache compression

The KV cache grows linearly with context length and can rival the model weights at long contexts. The TurboQuant fork adds compressed KV-cache data types beyond the upstream set, selected per-model via `cache_type_k` and `cache_type_v`:

| Type | Class | Relative KV size factor (vs f16) |
|---|---|---|
| `f32` | uncompressed | 2.00 |
| `f16` / `bf16` | half precision (baseline) | 1.00 |
| `q8_0` | 8-bit | 0.50 |
| `q5_0` / `q5_1` | 5-bit | 0.32 |
| `q4_0` / `q4_1` / `iq4_nl` | 4-bit | 0.25 |
| `turbo2` | TurboQuant | 0.125 |
| `turbo3` | TurboQuant | 0.1875 |
| `turbo4` | TurboQuant | 0.25 |

`turbo3` is a balanced production default. The KV estimator keys off `cache_type_k` (falling back to `f16` if unset). Compressed KV rides the fused-attention path, so `flash_attn: true` is required.

### 8.2 The production flag doctrine

A small set of spawn-time flags should be on by default for production manifests, complementing the `cache_type_k/v` choice:

| Flag | Value | Why |
|---|---|---|
| `flash_attn` | `true` | Enables the fused-attention path required for compressed-KV and low-precision kernels. |
| `no_context_shift` | `true` | Avoids a context-shift loop that can stall long-context inference. |
| `cache_reuse` | `256` | Enables prefix-cache reuse across requests in the same warm slot. |
| `slot_prompt_similarity` | `0.5` | Lets the slot reuse the prefix cache even when the prompt isn't byte-identical. |
| `no_perf` | `true` | Suppresses per-request perf logging. |

### 8.3 Per-model manifests

One YAML per model at `<manifests_path>/<model_tag>.yaml`, defining how its `llama-server` is spawned. Edited via `PUT /api/manifests/{tag}` (with an `If-Match` ETag) or on disk; hot-reloaded on the next cold-spawn.

```yaml
model_tag: my-model                 # primary key + filename stem; strict regex
display_name: "My Model 27B Q4"     # free text
description: "A 27B-class dense chat model, Q4 quant, compressed KV."
gguf_blob_sha256: "1a2b…64hex"      # content-address of the blob in the store
gguf_size_bytes: 21000000000
context_size: 131072                # model context length (manifest-level)
expected_vram_bytes: 22500000000    # used by the VRAM-fit pre-check
revision: 1                         # the ETag; server-incremented on each write

llama_server_flags:                 # the CLOSED allowlist (§8.4)
  ctx_size: 131072
  n_gpu_layers: 999
  cache_type_k: turbo3
  cache_type_v: turbo3
  flash_attn: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true
  jinja: true                       # REQUIRED for tool calls + thinking-block preservation
  reasoning: auto
  reasoning_budget: 500

prompt_template:
  system_default: ""
  stop_tokens: ["<|im_end|>", "<|endoftext|>"]
```

`model_tag` must match `^[a-z0-9][a-z0-9._-]{0,63}$` (lowercase ASCII, 1–64 chars, no traversal), re-validated at every path resolution. `gguf_blob_sha256` must be lowercase 64-hex. `jinja: true` is load-bearing — tool calls and thinking-block preservation only work under the Jinja chat-template branch.

### 8.4 The closed flag allowlist

`llama_server_flags` is a **closed allowlist** (over 100 flags). Validation runs in order per flag: (1) deny check, (2) dangerous-suffix forward-defense, (3) allowlist membership, (4) type/enum/bounds validation. **Adding a new flag is a code change plus review, not a YAML edit — by design.**

Allowed categories include performance/memory (`ctx_size`, `n_gpu_layers`, `parallel`, `flash_attn`, …), KV cache (`cache_type_k/v` incl. `turbo2/3/4`, `kv_offload`, `no_kv_offload`, `kv_unified`), RoPE/YaRN context extension, MoE/multi-GPU (`cpu_moe`, `n_cpu_moe`, `split_mode`), sampling, chat/template (`chat_template` name-only, `jinja`), reasoning (`reasoning_format`, `reasoning`, `reasoning_budget`), speculative decode / MTP (below), server toggles, and debug.

**Denied flags (path / credential / RCE / SSRF class)** are explicitly rejected: any flag taking a filesystem path, a URL, a credential, or exposing remote-fetch or command execution — `model`, `host`, `port`, `lora*`, any `*_file`, `model_url` / `hf_repo*`, `path` / `models_dir`, `api_key*`, `ssl_*_file`, `chat_template_file`, `override_kv`, `tools`, and more. A **suffix forward-defense** also rejects any *future* flag whose name ends in `_file/_path/_dir/_url/_repo/_key` (or starts with `hf_/lora/ssl_/api_key/…`), so a new dangerous flag is blocked before it's ever enumerated. The `chat_template` value must not contain `{%` or `{{` (a template-injection guard) — custom Jinja bodies can only come via the (denied) file path. The argv builder re-checks the allowlist and denylist a second time before constructing the command line (defense in depth).

A cross-field guard enforces the `parallel > 1` rules at manifest-load time: `kv_unified: true` required, `ctx_size` evenly divisible by `parallel`, and per-slot window `ctx_size // parallel` ≥ `PER_SLOT_CTX_FLOOR` (8192).

### 8.5 Spawn-time vs request-time flags

| Layer | Examples | Reload behavior |
|---|---|---|
| **Spawn argv** (process fork) | `flash_attn`, `no_context_shift`, `cache_reuse`, `ctx_size`, `cache_type_k/v`, `n_gpu_layers`, `jinja`, the MTP `spec_*` flags | **Cold-spawn only.** A manifest `PUT` does not affect a running `llama-server`; the old command line persists until the process exits. |
| **Request body** (per-call) | `temperature`, `top_p`, `top_k`, `stop`, `max_tokens`, `reasoning_budget` | **Hot.** Applied per request through the forwarder. |

To apply a changed spawn flag to a running model, force a cold-spawn: send a request with `keep_alive: 0`, wait for the natural idle teardown, or restart the container.

#### KV-cache & VRAM sizing math

The fit gate (Gate 3, §7.2) estimates whether a model + KV + overhead fits free VRAM:

```
gguf_mib            = gguf_size_bytes / (1024 * 1024)
bytes_per_tok_f16   = (9 * gguf_mib) / 1024        # ≈ 9 KB/token per GiB of model body, at f16
bytes_per_tok       = bytes_per_tok_f16 * quant_factor   # quant_factor from the table in §8.1
kv_cache_mib        = (bytes_per_tok * ctx_size) / 1024
```

**KV resident in VRAM (default):** `refuse if gguf_mib + kv_cache_mib + overhead_mib + par_extra > free_vram_mib` (overhead floor 1024 MiB).

**KV offloaded to host RAM (`no_kv_offload: true`):** the VRAM need drops the KV term and carries a context-linear scratch term instead, and the KV must fit host RAM:

```
vram_need = gguf_mib + (overhead_mib + ctx_size / 128 + par_extra)
refuse if vram_need > free_vram_mib
also refuse if kv_cache_mib > free_host_ram_mib            # complementary host-RAM check
```

This is the mechanism behind **KV offload to host RAM** — model weights stay on the GPU, the KV cache lives in system RAM, and the sidecar still serves requests (including `parallel:N` fan-out) normally. `par_extra = (parallel − 1) × PER_SLOT_COMPUTE_FLOOR_MIB` (256 MiB per extra slot). Compressed KV turns "won't fit at all" into "fits with a tuned context" — use a Gate 3 refusal as the signal to lower `ctx_size`, pick a more aggressive `cache_type_*`, or enable `no_kv_offload`.

### 8.6 Multi-token prediction (MTP) / speculative decoding

The manifest allowlist includes a draft-MTP speculative-decoding family (`spec_type: draft-mtp`, `spec_draft_n_max/n_min`, `spec_draft_p_min/p_split`, `spec_draft_ngl`, `spec_draft_backend_sampling`). Speculative decoding runs a cheap draft to propose several tokens per step that the main model verifies in one pass — raising tokens/sec when the draft is accurate, with no quality loss (the main model still decides). These are spawn-argv flags and compose with the TurboQuant `cache_type_k/v` + `flash_attn` path. They help most on predictable, low-entropy continuations (structured output, code) and least on high-entropy creative text.

### 8.7 Top-level config — boot vs runtime

The top-level config lives at `/etc/turbohaul/turbohaul.yaml` (overridable via `--config` or `$TURBOHAUL_CONFIG_PATH`) and splits into two models with very different mutability:

- **`BootConfig`** — frozen, restart required: `server` (`host` default `127.0.0.1`, `port` default `11401`, `allow_public_bind`), `storage` (blob/manifests/import/state paths), `runtime` (`llama_server_binary`, `llama_server_binary_sha256`, `default_port_base` default `11500`), `ui` (`enabled`, `static_path`). A `PUT /api/config` touching any boot field returns **HTTP 403** — this is what prevents a config-driven binary-swap attack. `GET /api/config` redacts every storage/runtime/ui `Path` field to its basename so absolute on-disk write targets aren't disclosed.
- **`RuntimeConfig`** — mutable via `PUT /api/config` (only the `queue` and `pull` sections; unknown sections → 400, boot sections → 403): all the queue/grace/idle/affinity knobs, the five `safety_*` thresholds, and `pull` settings (`hf_api_key_env`, `hf_host_allowlist`, `pull_url_https_only`, `pull_concurrency`, `per_stream_max_bytes` default 100 GiB). A `monitor` section (`enabled`, `poll_interval_s`) wires the live inference monitor from YAML/env at boot.

A handful of `TURBOHAUL_*` env vars override the YAML at load (env wins) — host, port, and the queue tuning knobs.

### 8.8 Atomic manifest writes + ETag concurrency

Writes are crash-safe and concurrency-safe: serialize to YAML → same-directory tempfile → `flush` + `fsync(file)` → `chmod 0o600` → `os.replace` (atomic rename) → `fsync(parent dir)`; the tempfile is unlinked on any error. Optimistic concurrency uses ETag/If-Match: `GET` returns `ETag: "<revision>"`; **create** (file absent) takes no `If-Match`; **update** (file present) requires `If-Match` (missing or mismatched → `ConcurrencyError` → HTTP 412), and `revision` is incremented on success. Every manifest read, write, and delete passes through a realpath-containment + symlink-rejection check, and `list_manifests` independently re-validates each file stem so a malformed or symlinked file dropped on disk is never surfaced.

---

## 9. Boot reconciliation and persistence

There is no dedicated FSM state for recovery; on startup the manager reconciles persisted state and sweeps anything stale to `COLD`:

1. **Orphan reaper** — reap untracked `llama-server` processes left by a prior crash (§7.3).
2. **Foreign-GPU detect** — list GPU compute PIDs the manager doesn't own (informational).
3. **`state.sqlite` reconcile** — two passes: slots with a live-looking pid that is actually gone → `COLD` (`boot-reconcile-orphaned-pid`); never-spawned slots (null pid) not in a terminal/idle state → `COLD` (`boot-reconcile-pre-active-orphan`).
4. **Binary integrity pin** — if a backend binary hash is configured, verify it and hold an open fd so every spawn execs the exact verified inode.

After reconciliation the manager starts the `worker_loop`, the background sweeper, and the live-monitor poller, then begins accepting traffic.

**`state.sqlite` schema:** a `slots` table (one row per slot — `slot_id` PK, `model_tag`, `thread_id`, `state`, `port`, `pid`, timestamps, `end_reason`, `extension_count`), an append-only `audit_events` table (state transitions + lifecycle events, drives `/v1/logging`), and a `pull_history` table. The DB runs in WAL mode with `synchronous=NORMAL`, `foreign_keys=ON`, and a 5 s busy timeout; audit writes go through a long-lived pooled connection so async callers never block the event loop.

**Persistence is mandatory.** Everything mutable lives under one data directory (e.g. `/var/lib/turbohaul`). Without a bind-mount, `state.sqlite`, `manifests/*.yaml`, and the blob store live in the container's writable layer and are destroyed by `docker rm` or layer corruption. Because GGUFs are deterministic and content-addressed, a lost blob store can be **re-pulled** from source given the sha256 inventory, while `state.sqlite` and manifests are recovered from a config mirror.

---

## 10. The API surface

Turbohaul-Manager exposes an **Ollama-superset** API: both shapes are served by one listener (default `127.0.0.1:11401`) — OpenAI-shape under `/v1/*`, Ollama-shape under `/api/*`, plus management/observability extensions. No app-layer auth (§11) — any `api_key` string works.

| Method + Path | Shape | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | Chat completion (streaming + non-streaming) |
| `POST /api/chat` | Ollama | Chat (forwarded as OpenAI internally, re-shaped to Ollama on return) |
| `POST /v1/embeddings` | OpenAI | Embeddings (forwarded to the sidecar) |
| `GET /api/tags` | Ollama | List installed models |
| `GET /api/show?name=<tag>` | Ollama | Model details (sanitized manifest) |
| `GET /api/version` | ext | Version + backend identity |
| `GET /health` | ext | Liveness (`{status:"ok"}`) |
| `GET /status` | ext | Queue + active + grace + idle + eviction + live-generation snapshot |
| `GET /api/config` | ext | Read merged config (storage/runtime paths shown as basenames only) |
| `PUT /api/config` | ext | Apply runtime-mutable config (`queue`/`pull` only) |
| `GET /api/manifests/{tag}` | ext | Read a manifest (`ETag` header) |
| `PUT /api/manifests/{tag}` | ext | Write a manifest (`If-Match` required on update) |
| `DELETE /api/manifests/{tag}` | ext | Remove a manifest |
| `POST /api/pull-url` | ext | Pull a blob from an HTTPS URL (SSRF-guarded) |
| `POST /api/pull-hf` | ext | Pull from a HuggingFace repo (host-allowlisted) |
| `POST /api/import` | ext | Import a local GGUF (sandboxed to the import root) |
| `DELETE /api/delete` | Ollama | Delete a blob by sha256/digest |
| `GET /v1/logging` | ext | Paginated, redacted audit-event stream |
| `WS /ws/state` | ext | Redacted live state-event stream |
| `GET /ui/live/output/stream` | ext | SSE — anchor-follow live assistant-output text |
| `GET /ui`, `/ui/{path}` | static | Single-page UI (when `ui.enabled` and the static dir exists) |

> The Ollama registry `POST /api/pull` is a **501 stub** — the functional pull endpoints are `/api/pull-hf` and `/api/pull-url`. `/api/generate` and `/v1/completions` are not implemented; the completion endpoints are `/api/chat` and `/v1/chat/completions`. Child `llama-server` sidecars use port base `11500` (internal only, never a public listener).

### 10.1 Chat, `response_format`, and streaming

Both chat endpoints accept the same superset: standard OpenAI fields, `response_format` (`text` / `json_object` / `json_schema`), tool-call fields (`tools`, `tool_choice`, `parallel_tool_calls`, `function_call`, `functions`, forwarded verbatim), `keep_alive`, and `thread_id`. The Ollama path forwards to the same internal OpenAI pipeline and re-shapes on return (`created` → `created_at`, `finish_reason` → `done_reason`, string `arguments` → parsed object).

`response_format` is validated at handler entry on both streaming and non-streaming paths: `json_object` is accept-and-forward; `json_schema` is fully validated (bounded before compile — serialized ≤ 64 KiB, depth ≤ 16, ≤ 64 properties, `$ref` rejected, `additionalProperties: false` required — then compiled with a Draft 2020-12 validator; a bad schema → HTTP 422). For a `json_schema` request on a *thinking* model, the server validates the first response (stripping any `<think>…</think>` wrapper) and does exactly one retry with thinking disabled on failure.

Streaming (`stream: true`, `/v1/chat/completions` only) returns `text/event-stream` with a cold-load heartbeat (`: keep-alive` comments every 12 s while waiting up to 7200 s for the slot to reach ACTIVE) so clients with short read-timeouts don't disconnect during a cold load. Once ready, raw SSE chunks are piped through unmodified; reasoning-merge and tool-call recovery are intentionally skipped on the streaming path. Mid-stream errors stay HTTP 200 (a synthetic `data: {"error": …}` frame rather than a torn connection). Ollama `/api/chat` streaming combined with tools is rejected with HTTP 400.

### 10.2 Error codes

| Code | Meaning |
|---|---|
| **400** | Bad request (unsupported `response_format` type, Ollama stream+tools). |
| **422** | Input validation failed (e.g. `json_schema` invalid). |
| **499** | Client closed the request before the slot activated. |
| **502** | Sidecar returned an upstream error (often context overflow, or post-retry schema noncompliance). |
| **503** + `Retry-After` | The active sidecar disconnected or crashed mid-response (frequently KV-cache OOM) — back off and retry. |
| **504** | Sidecar timed out generating. |
| **500** | Other internal error — including a pre-spawn safety-gate refusal on the chat route. |

### 10.3 Tool-call recovery

Some chat-templated GGUFs, run through `llama.cpp`'s Jinja path, emit a tool call as **JSON text inside `message.content`** instead of populating the structured `message.tool_calls` field, and a client that only reads the structured field sees "no tool call." Turbohaul-Manager **transparently recovers** these. This is documented upstream in `llama.cpp` issues [#20809](https://github.com/ggml-org/llama.cpp/issues/20809), [#20837](https://github.com/ggml-org/llama.cpp/issues/20837), and [#20260](https://github.com/ggml-org/llama.cpp/issues/20260) (notably Qwen3-family models).

The recovery post-processor runs after the reasoning merge and before the route returns, on the **non-streaming path only** (per-chunk rewriting would break streaming UX), serving both `/v1/chat/completions` and `/api/chat`. It catches canonical `{"name":…,"arguments":…}` text JSON (via a brace-balancer that handles nested objects), `<tool_call>…</tool_call>` XML wrappers, and parallel calls. It deliberately does **not** catch calls inside `<think>…</think>` blocks (only content after the last `</think>` is scanned), calls to names not in the request's `tools` allowlist (anti-hallucination), or malformed JSON. When it fires it populates `tool_calls` (one synthetic-id entry per call), strips the matched spans from `content`, and flips `finish_reason` to `tool_calls`. It is **idempotent** — a no-op when upstream already populated `tool_calls` — so it is safe to leave enabled permanently. Prerequisites: `jinja: true` in the manifest, and the request must advertise `tools`.

### 10.4 Redaction & observability

Two observability surfaces, both redaction-disciplined:

- **`/v1/logging`** — cursor-paginated audit history with a ~20,000-token page envelope. Every payload is recursively walked and any key in the redaction denylist (`prompt`, `response`, `context`, `stderr`, `stdout`, `messages`) is dropped; an undecodable row becomes a `{_decode_error: true}` sentinel rather than 500ing.
- **`/ws/state`** — on connect the socket gets `{event:"connected", snapshot:<status>}` then streams state events from a per-connection bounded queue (a full queue drops events rather than blocking the worker). Broadcast events carry only `{event, slot_id, model_tag, state, thread_id_prefix}` (thread-id first 8 chars). **Never broadcast:** prompt/response text, stderr/stdout, full thread-ids, IPs, message bodies — enforced both by a publish-time key denylist and by emitter discipline (only the 8-char prefix is ever put on an event).

### 10.5 Live inference monitor

`GET /status` returns a redacted snapshot with `queue`, `active`, `loading`, `grace`, `idle_hot`, `evictions`, `background_sweeper`, `parallel_slots` (`used`/`max`), and a top-level **`generation`** block. The active-slot dict carries only `{slot_id, model_tag, state, thread_id_prefix, pid, port}`. Live tok/s and progress live in the separate `generation` block — `state`, `tok_s`, `tok_s_instant`, `n_decoded`, `max_tokens`, `n_remain`, `n_prompt_tokens`, `n_ctx`, `prompt_progress`, `pct`, `eta_s`, `stalled`, `streaming`, `riders`, `generation_id`, `measured_at_iso` — written await-free by the ~1 Hz `LiveSlotsPoller` (a pure observer reading the sidecar `/slots`; idle default when nothing is live).

High-volume live **output text** is carried on `GET /ui/live/output/stream`, an SSE stream that anchor-follows whatever generation is currently live (switching server-side the instant `generation_id` turns over) and streams only assistant-output deltas — never the prompt, IPs, or full thread-id. The two planes are joined by `generation_id` (`blake2b(pid:spawn_seq:slot_id)[:8]`, non-reversible): `generation` is the **metrics** plane, `/ui/live/output/stream` the **text** plane. The monitor is gated by `monitor.enabled` (an ops kill-switch) and `monitor.poll_interval_s` (single-poller cadence, default 1.0 s — one reader regardless of front-end client count).

---

## 11. Trust boundary

Turbohaul-Manager runs under a **network-perimeter trust model**: **no app-layer authentication** on any endpoint. The trust boundary is the **bind address**.

- `server.host` defaults to `127.0.0.1` (loopback). It can be bound to a specific private interface via config. A public bind (`0.0.0.0`) requires the explicit `--allow-public-bind` CLI flag (or `TURBOHAUL_ALLOW_PUBLIC_BIND=1`), which overrides the configured host at startup — never public by default. A literal `host: 0.0.0.0` in the YAML is rejected by the config validator.
- Every mutating surface (`PUT /api/config`, `PUT /api/manifests`, `/api/pull-*`, `/api/import`) is as reachable as the read surfaces, so adding auth on one endpoint while leaving the others open would be strictly worse than the uniform posture. The intended hardening path is an **external authenticating reverse proxy** (e.g. Caddy or nginx with a bearer token) applied uniformly to all paths, with the manager bound to loopback behind it.
- The load-bearing protection against accidental data exposure is **redaction discipline**, not auth: prompts, responses, stderr, full thread-ids, and IPs are never broadcast on `/ws/state` and are scrubbed from `/v1/logging` audit payloads (§10.4). The denylist is a tripwire; the primary defense is that emitters never put sensitive content on an event in the first place.

If you need to expose Turbohaul-Manager beyond a trusted perimeter, put an authenticating proxy in front of it and bind the manager to loopback behind that proxy.

### 11.1 Pull-path safety (SSRF guard)

`POST /api/pull-url` runs every URL through a strict guard before any connection:

- **HTTPS only.** `http`, `file`, `ftp`, `gopher`, `dict`, `ldap`, `data`, … and a missing hostname → HTTP 400.
- **Private/blocked IP ranges** — the resolved IP must not be in RFC1918, loopback, link-local (incl. cloud-metadata `169.254.0.0/16`), CGNAT/overlay (`100.64.0.0/10`), multicast/reserved, or the IPv6 equivalents including ULA, `64:ff9b::/96` (NAT64), `::/96` (IPv4-compatible), and `::ffff:0:0/96` (IPv4-mapped — defeats embedding a blocked v4 inside a v6 literal). Unparseable addresses are treated as blocked (fail-closed). IP-literal URLs are deny-checked directly.
- **DNS-rebind / round-robin defense** — `_double_resolve_check` re-runs the full SSRF validation twice back-to-back and refuses if the two resolutions diverge in host or IP (catching the public-at-validate, internal-at-connect rebind pattern and round-robin DNS).
- **Redirect handling** — the redirect-aware streamer re-runs the full SSRF check on every hop, enforces the HF host allowlist per hop, **strips the `Authorization` header on any cross-host hop** (defending against token exfil via a redirect to an attacker host), and caps hops at 5.

A configured HuggingFace token is attached **only** when the request host matches the `hf_host_allowlist` (exact host or a `.host` suffix) — it can't be exfiltrated to a non-HF host.

### 11.2 Import safety and the blob store

Models are GGUF blobs stored content-addressed by sha256 under `<blob_store_path>/sha256/<ab>/<full-64-hex>`. The write lifecycle is integrity-first: stream to a tempfile opened `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW` (no clobber, no symlink redirect), compute the sha256 incrementally, enforce the per-stream byte ceiling before each write (`per_stream_max_bytes`, default 100 GiB), verify the hash (against the caller's `expected_sha256` if supplied, else adopt the computed hash as the address), `os.replace` atomically into the sharded final path, `fsync` the parent dir, and `chmod 0o400` (read-only, tamper-evident). A `verify_blob_on_stage` re-verify helper provides defense-in-depth against TOCTOU blob swaps.

`POST /api/import` is sandboxed: the path must be absolute and resolve under the configured import root (anything outside → 400), the file is opened `O_NOFOLLOW` (symlink or symlink-escape rejected), an always-denied system-path denylist (`/proc/`, `/sys/`, `/dev/`, `/etc/`, `/root/`, `/var/run/`, `/var/lib/dpkg/`, `/boot/`) is enforced independent of the sandbox, and the first 4 bytes must be the `GGUF` magic. It then lands in the same content-addressed store via the same atomic-write + hash-verify + immutability lifecycle.

---

## 12. Front-end

The UI is a **React + Vite + TypeScript** single-page app served from the same port as the API (when `ui.enabled` and the static dir exists). It is a thin client over the API — it never writes the filesystem directly; all edits go through `PUT /api/config` and `PUT /api/manifests/{tag}` with `If-Match` ETag concurrency, and live state arrives over `/ws/state` and the live-output SSE stream.

| View | Purpose |
|---|---|
| **Dashboard** | Active sidecar, queue depth, and overall manager health at a glance. |
| **Live Inference** | The live inference monitor — tok/s, progress, and the live assistant-output stream for the current generation (the metrics `generation` block joined to the `/ui/live/output/stream` text plane). |
| **Models** | Installed models and sizes; pull (HuggingFace / URL) and import (path under the import root); delete. |
| **Queue** | The FIFO list with positions, model tags, and per-slot status. |
| **Config** | Top-level config editor + per-model manifest editor with ETag-aware UX (a 412 prompts a reload-and-re-apply) and restart-required field flagging. |
| **Logs** | Redacted audit history from `/v1/logging`. |

---

## 13. Licensing & attribution

- **Turbohaul-Manager** is **MIT** licensed (see [LICENSE](../LICENSE)).
- **Inference backend:** `llama-server` built from the TurboQuant fork of `llama.cpp` ([github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)) — Ollama-compatible HTTP API surface.
- Third-party dependency licenses (MIT / BSD-3-Clause / Apache-2.0, all permissive) are attributed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- "Ollama-compatible" is used only nominatively (API-shape compatibility), with no source vendored.

---

## 14. Compatibility notes

- **OpenAI-compatible clients** — point any OpenAI SDK at `http://<host>:11401/v1` with any `api_key` string and pass `thread_id` via `extra_body` for warm reuse.
- **Ollama-compatible clients** — the `/api/*` surface (`/api/chat`, `/api/tags`, `/api/show`) is served on the same port. Turbohaul does not bind the standard Ollama port `11434`; point clients at `11401`.
- **Open WebUI** — Turbohaul-Manager is compatible with Open WebUI today via the Ollama-compatible surface plus the auto-derived `thread_id` for clients that don't send one. The only current limitation is that not every Open-WebUI reporting/telemetry field is populated yet — a minor known gap, not a missing capability.

For end-to-end client recipes (Hermes, OpenAI SDK, langchain, llama-index, LiteLLM, Ollama clients, curl), see the project documentation and the README.
