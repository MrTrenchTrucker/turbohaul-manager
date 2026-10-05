# Turbohaul-Manager — Architecture (As Built)

**Version:** v0.8.0 · **Scope:** the system as built; every mechanism is named by source file and symbol, and quoted configuration values are the code defaults (§9).
**Set it up first:** [README.md](README.md) · [docs/AI_AGENT_SETUP.md](docs/AI_AGENT_SETUP.md)

---

**Turbohaul-Manager serves many AI agents from one inference host: multiple agents, their sub-agents, and all of their requests at the same time.**

- **A whole fleet on one host.** Requests wait in a two-tier queue (a staging queue of 100 and an acceptance buffer of 10,000). The parallel machinery is always on: the resident dispatcher serves at every `max_parallel_sidecars` value, and the default of 1 spawns one sidecar at a time (single series) and queues the rest. Raising it keeps several models resident at once (up to 32, double parallel), and a model's `parallel` setting serves several contexts at once on one engine (series parallel).
- **Agents keep their place.** Each (session, role) has one precomputed KV copy. It is saved when its model unloads and restored when the agent returns, so the engine prefills only what follows the saved prefix.
- **Drop-in for existing clients.** The API is Ollama-compatible and OpenAI-compatible, so clients point at it unchanged.
- **Self-contained.** The engine source and the Python wheels are vendored, and `Dockerfile.engine-src` builds it offline.

```
   agents  ·  sub-agents  ·  any app
              │   Ollama-compatible and OpenAI-compatible API  (§7)
              ▼
┌─ TURBOHAUL MANAGER  (§3, §6) ──────────────────────────────────────────────────────────┐
│ queue   staging queue 100  +  acceptance buffer 10,000                                 │
│ order   Fast Lane  >  main lane  >  compression  >  model affinity / FIFO              │
│ route   reuse a loaded engine, or reserve VRAM and start one; hold it warm after use   │
└────────────────────────────────────────────────────────────────────────────────────────┘
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 llama-server :11500     llama-server :11501     llama-server :11502     engines, one per loaded model (§5)
   Model A                 Model B                 Model C
        └───────────────────────┴───────────────────────┘
 KV cache per (session, role):   VRAM  ─▶  RAM tier  ─▶  SSD tier   (§4)          dashboard at /ui  (§11)
```

---

## 1. What Turbohaul-Manager is

**Turbohaul-Manager serves many AI agents from one inference host.** Requests enter a two-tier queue:

- a **staging queue** (`staging_queue_depth`, default 100), and
- an **acceptance buffer** (`acceptance_buffer_max`, default 10,000) that takes the overflow and refills staging as it drains.

Staging pops in FIFO order unless model affinity or Fast Lane picks another entry. Together with the grace window and idle-hot retention (§3), this decides which engine stays loaded.

**Fast Lane** changes how the queue orders work when it is enabled (`fastlane.enabled`, off by default). A request from a client named in a Fast Lane rule ranks by the rule's position first and by its traffic class (tag rank) second, and unlisted clients rank below every listed one (an unlisted waiter older than `max_normal_wait_s` is still served first). It also picks which idle model gives way when the server is full, and it never interrupts or cancels a running turn. See §3.1, §6.1, [docs/FAST_LANE.md](docs/FAST_LANE.md) and [docs/design/FAST_LANE_ARCHITECTURE.md](docs/design/FAST_LANE_ARCHITECTURE.md).

The manager serves the load in whichever **residency mode** the hardware allows:

- **Single series:** one model resident, one context at a time. Requests serialize.
- **Series parallel:** one model resident, **several context windows served at once** by the same engine (the engine's `parallel` slots).
- **Double parallel:** **several models resident at once** (`max_parallel_sidecars`, up to 32; §3.6), each of which can itself be series-parallel.

The mode is set by configuration alone (`max_parallel_sidecars`, manifest `llama_server_flags.parallel`); see [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md).

### 1.1 The three residency modes at a glance

**Single series:** `max_parallel_sidecars: 1` (the default) with `parallel` unset or 1 (an engine spawned without `--parallel` is treated as one slot wide). One model, one context at a time. Requests serialize.

```
 agents / clients                 TURBOHAUL MANAGER                           GPU
┌────────────┐
│ main agent │─┐   ┌─────────────────────────────────┐   ┌───────────────────────────┐
└────────────┘ │   │ staging queue (100)             │   │ llama-server :11500       │
┌────────────┐ ├─▶ │   ▲ overflow: acceptance buffer │   │ ┌───────────────────────┐ │
│ sub-agents │─┤   │     (10,000)                    │──▶│ │ slot 0: ONE active    │ │
└────────────┘ │   │ pop order: Fast Lane, main lane,│   │ │ context window        │ │
┌────────────┐ │   │ compression, affinity / FIFO    │   │ └───────────────────────┘ │
│ any app    │─┘   └─────────────────────────────────┘   │       Model A resident    │
└────────────┘                                           └───────────────────────────┘
```

By default a request for a different model waits until the resident is `IDLE_EVICTABLE`; the resident is then evicted and the new engine spawns. With Fast Lane, a higher-ranked claimant can designate a victim, torn down at its turn boundary or when its grace ends.

**Series parallel:** a manifest `llama_server_flags.parallel` of N (1 to 256) spawns the engine with `--parallel N`. Same-model requests fan out onto the running engine, up to N in flight (`_fan_out_on_resident`), and drain completely before the resident enters grace or is torn down.

```
 agents / clients                 TURBOHAUL MANAGER                           GPU
┌────────────┐                                              ┌───────────────────────────┐
│ main agent │─┐   ┌─────────────────────────────────┐      │ llama-server :11500       │
└────────────┘ │   │ queue (as above)                │      │ spawned --parallel N      │
┌────────────┐ ├─▶ │                                 │─────▶│ ┌────────┐ ┌────────┐     │
│ sub-agent  │─┤   │ same-model requests FAN OUT     │      │ │ slot 0 │ │ slot 1 │     │
└────────────┘ │   │ onto the ONE running engine     │      │ │ ctx A  │ │ ctx B  │     │
┌────────────┐ │   │ (up to N in flight, drained     │      │ └────────┘ └────────┘     │
│ sub-agent  │─┘   │ before grace or teardown)       │      │ ┌────────┐   Model A      │
└────────────┘     └─────────────────────────────────┘      │ │ slot 2 │   resident     │
                                                            │ │ ctx C  │                │
                                                            │ └────────┘                │
                                                            └───────────────────────────┘
```

**Double parallel:** `max_parallel_sidecars` of 2 or more keeps several models resident. Each resident has its own state machine (`ResidentState`: `RESERVED_LOADING`, `ACTIVE`, `IDLE_EVICTABLE`, `DEAD`), its own grace and idle timers (an open grace window shows as phase `GRACE` in status), and its own port (lowest free from `default_port_base`, 11500 by default). `_route_or_reserve` admits a resident under a VRAM budget.

```
 agents / clients              TURBOHAUL MANAGER                              GPU(s)
┌────────────┐      ┌──────────────────────────────────┐      ┌────────────────────────────┐
│ agent 1    │─┐    │ queue + resident dispatcher      │─────▶│ llama-server :11500        │
└────────────┘ │    │ (max_parallel_sidecars, up to 32)│      │ Model A  [slot 0 | slot 1] │
┌────────────┐ ├──▶ │                                  │      ├────────────────────────────┤
│ agent 2    │─┤    │  • VRAM-budget admission         │─────▶│ llama-server :11501        │
└────────────┘ │    │  • per-model Resident state      │      │ Model B  [slot 0]          │
┌────────────┐ │    │  • idle eviction (priority-aware)│      ├────────────────────────────┤
│ agent 3    │─┘    │  • per-resident grace / idle     │─────▶│ llama-server :11502        │
└────────────┘      └──────────────────────────────────┘      │ Model C  [slot 0 | slot 1] │
                                                              └────────────────────────────┘
```

When room is needed, an idle resident is evicted in priority order (`_lru_idle_unloadable`): unlisted before Fast Lane-listed, then the lowest-priority rule, then the numerically worse tag rank, then the least recently active. A model tag may also run one engine per card when its manifest sets `auto_place: true` and `llama_server_flags.split_mode: "none"`, within the same budget (`engine_budget/`).

**The KV-cache tiers (§4.5) use the same directories in every mode:** VRAM (native prefix reuse), the RAM tier (`SLOT_SAVE_DIR`, expected to be a tmpfs mount) and the SSD tier (`SLOT_PERSIST_DIR`), a cold copy written in the background when a different identity takes over an engine's slot. A restore hydrates RAM from SSD when a persisted copy exists.

- The unload-time save to SSD is work in progress; an eviction teardown saves to the RAM tier only.
- The clean-prefix anchor save is gated on observed single-series operation. `_probe_and_save_clean_kv` declines when the sidecar runs with `--parallel` above 1 or when another resident is `ACTIVE`, because the clean-bin invariant does not hold once concurrent contexts can displace the tip between probe, save and the real request.
- Each decline emits an event, and the observed count of other `ACTIVE` sidecars is logged for contexts of at least 40,000 characters or at the displacement seam.

**Classification and KV.** Each request is classified into one of five classes (main, user-message, sub-agent, curator, compression) from `is_*` labels or a role string in `client_meta`, or inferred from the prompt chain when no labels are present (`kv_classify.py`). The outgoing identity's KV state is saved to the RAM tier when its model is evicted and restored when the model is spawned again, so follow-up turns reuse the precomputed prefix instead of re-prefilling. A sub-agent's KV is never cross-restored.

**Trust model:** network-perimeter security. The bind address is the boundary and there is no application-layer authentication: the only middleware is `BodySizeLimitMiddleware` and bearer headers are only outbound (§8).

## 2. System map & repo layout

The manager is **one Python process** (uvicorn, default port **11401**); each loaded model is a separate supervised `llama-server` process. The front end is static-built and mounted at `/ui` by the same FastAPI app. In the default config durable state lives under `/var/lib/turbohaul` (§12).

The repository, folder by folder:

```
src/turbohaul/                    the manager (Python, FastAPI + asyncio)
  manager.py                      TurbohaulManager: dispatcher, residents, slot lifecycle, KV orchestration
  queue.py                        TurbohaulQueue (Fast Lane / main-lane / compression / affinity ladder),
                                  GraceTimer, IdleHotTimer
  slot.py                         Slot dataclass, SlotState enum, thread-id prefix-hash derivation
  fsm.py                          slot-state transition table, validators
  fastlane.py                     Fast Lane rule compile / match / rank / lint
  fastlane_resolve.py             FastLaneNameResolver (rule container_name -> addresses, fail closed),
                                  FastLaneClientNamer (names on the Discovered list, display only)
  kv_policy.py                    resolve_kv() decision chokepoint, prefix-hash chains, bin/meta filenames
  kv_classify.py                  role/label -> class resolution, POLICIES registry, event taxonomy
  safety.py                       pre-spawn gates (RAM, VRAM, KV-fit, MoE-RAM, CPU utilisation, iowait, tensor-split)
  singleton.py                    state-lock flock, orphan reapers, foreign-GPU-app detection
  subprocess_mgr.py               sidecar spawn/supervision, KV tier directory constants
  load_verify_log.py              LOAD_VERIFY proof emitter + /status ring
  spec_downgrade_log.py           SPEC_DOWNGRADE observability ring (§10)
  live_monitor.py                 live pollers (tok/s, prefill %, VRAM, residents)
  telemetry.py                    rotating flap-telemetry JSONL pipeline + ring buffer
  config.py                       TurbohaulConfig loader, env-override map, Boot/Runtime split (§9)
  __main__.py                     uvicorn entry point (config path, public-bind opt-in, log filters)
  state.py                        state.sqlite persistence + audit-event pool
  manifest.py                     model and plugin manifests (plugins: work in progress) behind a `kind`
                                  discriminator, tag validation, flag allowlist, ETag writes
  blob_store.py                   content-addressed GGUF blob store (atomic staged writes, stale-upload GC)
  ssrf_guard.py                   pull-URL validation (scheme, hosts, IP ranges, rebinding)
  plugin_invoke.py                `resolve_endpoint` (SSRF-guard chokepoint) and `invoke_plugin` for an operator-declared plugin container (§9.2; work in progress)
  plugin_health.py                PluginHealthMonitor: boot and periodic probe of the plugin registry (work in progress)
  _gguf_meta.py                   stdlib GGUF KV-header reader (attention dims for the KV-fit estimate)
  _gguf_tensor_offsets.py         stdlib tensor-info reader (MoE placement sizes)
  model_meta.py                   per-blob model description metadata, keyed by blob digest
  engine_budget/                  pure: how many engines one model tag may run (effective_engine_cap)
  engine_launch/                  pure: the environment an engine process starts with (launch_env)
  fastlane_client_names/          pure: which confirmed container name the Discovered list shows
  api/                            FastAPI routes and helpers: chat_completion, ollama, models, embeddings,
                                  pull, import_, manifests, config_put, config_schema, logging,
                                  telemetry, fastlane_census, live_stream, ws_state, tool_call_recovery, body_size_limit,
                                  plugins and exec_ws (work in progress), main (app factory + lifespan)
src/frontend/                     React + Vite + TypeScript front end, served from the same port (§11)
engine/llama-cpp-turboquant/      vendored engine source snapshot (see engine.lock, §13)
engine/ffshim/                    generic PATH shim that bridges an executable call to the manager's plugin
                                  exec WebSocket (§9.2; work in progress)
tools/, scripts/, smoke/, tests/   doc/registry checkers, engine build guards, smoke env, tests
vendor/pywheels/                  vendored Python wheels, installed offline by Dockerfile.engine-src
docker/, Dockerfile*              default config yaml; images: Dockerfile (manager only, engine binary mounted; ffshim as ffmpeg and ffprobe),
                                  Dockerfile.cuda-multi (compiles the engine for sm_75 to sm_120; same ffshim),
                                  Dockerfile.engine-src (self-contained build, §5.4; ffmpeg from apt, no shim); see §12
docs/                             operator guides and reference notes
```

## 3. Request lifecycle & slot state machine

**In short:** a request is admitted once, waits in the queue (when Fast Lane is enabled, its client ranking is checked first, §6.1), is routed to a resident engine (or a new engine is started for it), is served, and then its engine lingers briefly in case the same client sends a follow-up.

```
 request ─▶ admit ─▶ queue ─▶ route ─▶ serve ─▶ GRACE ─▶ idle window ─▶ unload
            (§3.1)   (§3.1)   (§3.3)   (§3.3)   (§3.4)    (§3.5)        (KV saved, §4.5)
                                                   │
                                      follow-up ◀──┘  same client, same model: served again on the warm engine
```

### 3.1 Admission

Routes call `submit_and_wait` (non-streaming; adds a completion cache and single-flight) and `submit_for_streaming` (returns the slot at once; the route attaches to the stream later). Both wrap **`submit()`**, the single admission chokepoint:

1. Refuses work after shutdown (`QueueClosed`).
2. Derives a missing `thread_id` from the prompt-prefix hash (§4.1).
3. Creates the `Slot` (`RECEIVED`) with `admission_ctx_len` and the admission prefix-hash chain.
4. Stamps `slot.admission_role = _bin_role(client_meta)`, which survives later `client_meta` identity restoration (§4.7).
5. Matches the Fast Lane table (`match_fastlane`); a matcher failure means "no match".
6. Enqueues FIFO: `TurbohaulQueue.enqueue` uses staging (`STAGED`) while it has room, else the acceptance buffer (`ACCEPT_BUFFER`).

**Queue** (`queue.py`): a **staging queue** (`staging_queue_depth`, default 100) and an **acceptance buffer** (`acceptance_buffer_max`, default 10,000) under one lock.

- Overflow raises `QueueFull` (HTTP 500).
- A disconnect watcher sets the slot's `disconnect_event`; a popped slot with that event set is flagged evicted and failed with `SlotEvictedError` (HTTP 499).

**Priority ladder** (`pop_next`), re-evaluated each pop:

1. **Fast Lane:** a guarded ranked pick runs first when a policy is active (§6.1).
2. **Main-lane reservation** (`main_lane_reserved`, default on; `main_lane_identity_keys`, default `["is_main"]`): a queued main request outranks FIFO and warm-model affinity. Admission-only: running work is not interrupted.
3. **Compression skip-ahead:** a compression-class request jumps queued sub-agent and other work.
4. **Model affinity / FIFO:** same-model staged work is preferred over the head, bounded by `max_other_model_wait_s` (default 20 s, judged on the oldest other-model waiter in staging or the acceptance buffer) and `max_consecutive_same_model` (default 3). The count cap never forces a swap.

### 3.2 Slot state machine

`SlotState` has 12 members; `fsm.py` (`LEGAL_TRANSITIONS`, `transition()`) lists and validates every legal edge. The live path:

```
anchor:     RECEIVED -> STAGED | ACCEPT_BUFFER -> STAGED -> LOADING -> ACTIVE -> GRACE -> POPPED
follow-up:  STAGED -> ACTIVE_MATCH -> ACTIVE -> GRACE -> POPPED     (warm engine; anchor waits in GRACE)
skip-grace: ACTIVE -> POPPED     (unload target, or outranked by a parked Fast Lane claim)
```

A spawn failure, health timeout or safety-gate refusal fails the slot's completion future or hands it to another engine of the same model (`_fail_or_reroute_refused_spawn`); the slot does not enter `LOADING_FAIL`. Exceptions: a VRAM-only gate refusal with a reclaim in flight first waits on the spawn reclaim barrier (`spawn_reclaim_wait_max_s`, default 10 s); an out-of-memory health failure for a connected client is re-queued (`_requeue_on_oom_load_failure`) unless the model can never fit.

Each loaded engine is a **Resident**. `ResidentState` has five members: `RESERVED_LOADING`, `ACTIVE`, `GRACE`, `IDLE_EVICTABLE`, `DEAD`. Stored state moves `RESERVED_LOADING` → `ACTIVE` ⇄ `IDLE_EVICTABLE`, ending in `DEAD`; it stays `ACTIVE` while `_resident_phase` reports `GRACE` (the window lives in the resident's `GraceTimer`).

### 3.3 Dispatch loop and resident serving path

`worker_loop` returns into **`_dispatch_loop`** at every `max_parallel_sidecars` value. Each iteration:

1. Calls `queue.pop_next` with the most recently active resident's model, the Fast Lane policy, whether an engine slot is free, and the `(thread_id, model_tag)` pairs inside live grace windows (`_grace_active_exclusions`), which the pop hides so the grace loop claims them.
2. With nothing to pop, waits on `_dispatch_wake` for at most `_DISPATCH_DEFER_BACKOFF_S` (50 ms).
3. Fails a disconnect-evicted slot with `SlotEvictedError`; otherwise calls `_route_or_reserve`, and a routing exception fails the slot's future.

`_route_or_reserve` decides inside one `_registry_lock` section:

```
 slot ─▶ _route_or_reserve
          ├─ hit ────────────────────▶ slot goes to the live same-model resident's inbox
          ├─ miss, room ─────────────▶ reserve the budget, start a new RESERVED_LOADING resident
          ├─ miss at the count cap ──▶ unload an idle victim (DEAD at once), reserve the model
          ├─ miss, VRAM over-commit ─▶ unload an idle victim, re-queue the slot
          └─ nothing evictable ──────▶ _defer_unroutable: re-queue at the head after a backoff
```

- **Hit:** a live same-model resident gets the slot on its `inbox`; an `IDLE_EVICTABLE` resident flips to `ACTIVE`.
- **Miss with room:** under the count cap with VRAM admitted (`_vram_admits_locked`), `_reserve_and_start_locked` claims the budget, creates a `RESERVED_LOADING` resident and starts its `_drive_resident` task.
- **Miss at the count cap, VRAM available:** `_lru_idle_unloadable` picks an idle resident with no running turn (else a designated victim whose grace ended), `_begin_unload_locked` marks it `DEAD` and deregisters it at once, and the model is reserved in the same locked section. Victim order is priority-aware with Fast Lane enabled (§6.1), else least-recently-active.
- **Miss with VRAM over-commit:** an idle victim is unloaded and the slot is re-queued, not reserved in-band, because the victim's VRAM frees only after teardown.
- **Nothing evictable:** `_defer_unroutable` re-queues the slot at the head after a backoff (50 ms when residents are busy, 1 s when waiting for VRAM), cut short when a resident parks idle, a grace window lapses or an eviction frees capacity. There is no retry cap: the slot waits until served or the client disconnects.

`_drive_resident` spawns the engine (`_spawn_for_resident`: safety gates §6, spawn §5, health wait bounded by `loading_health_timeout_s`, KV restore §4.6, LOAD_VERIFY §4.8), then loops: slot from inbox, `_serve_on_resident`, grace, idle. A streaming slot hands the engine handle to the route and stays `ACTIVE` until the stream closes (capped at one hour). At `--parallel 1` nothing else runs meanwhile; at higher `parallel`, `_fan_out_on_resident` keeps admitting same-model requests (§3.6).

### 3.4 Grace window & ACTIVE_MATCH follow-ups

After a serve the anchor enters `GRACE` for `grace_seconds` (default 30), owned by `(thread_id, model_tag)`. The grace loop polls every 50 ms and claims a follow-up two ways:

- `queue.pop_matched_thread`: same `thread_id` and `model_tag`; it yields to a better-ranked waiting request of the same client.
- `drain_inbox_and_staging_match_hash_chain`: a later request whose admission hash chain extends the anchor's, covering a `thread_id` that drifts as the conversation grows.

```
 serve ends ─▶ GRACE (grace_seconds, default 30)
                 │
                 ├─ follow-up matches ─▶ STAGED ─▶ ACTIVE_MATCH ─▶ ACTIVE on the warm engine (no reload)
                 │                        window re-armed, up to max_grace_extensions (default 5)
                 └─ no match ──────────▶ window ends ─▶ idle-hot retention (§3.5)
```

A match is promoted `STAGED` → `ACTIVE_MATCH` → `ACTIVE` on the warm engine with no reload (`_complete_fn` runs on the anchor's running handle while the anchor stays in `GRACE`), then goes `GRACE` → `POPPED`. `GraceTimer.pause` freezes only the reported countdown during the serve; the loop's exit test is a wall-clock deadline that `GraceTimer.restart_for_followup` re-arms up to `max_grace_extensions` (default 5) times: once the extensions are spent the loop can end right after a long serve.

The loop ends early when the resident becomes a designated unload target or a parked Fast Lane claim outranks it, or when an ordinary (not Fast Lane-listed) request for another model has waited past `max_other_model_wait_s`.

### 3.5 Idle-hot retention & keep_alive

When grace ends, `_idle_window_seconds` maps the resident's latest `keep_alive` (the anchor's, replaced by any follow-up that carries one) into the idle window:

| `keep_alive` | Idle window |
|---|---|
| absent | the per-model default |
| negative ("pin") | `KEEP_ALIVE_MAX_S` (1800 s) |
| any other value | `min(keep_alive, 1800)` |

The per-model default is manifest `sleep_idle_seconds` when positive, 1800 s when -1, else `idle_hot_load_seconds` (default 600 s).

At a window of 0, `_same_model_residency_floor` holds the engine `_SAME_MODEL_QUEUED_HOLD_S` (30 s) if the queue head is the same model and no strictly higher-priority client waits; otherwise the resident unloads at once (KV save: §4.5).

For a positive window the resident is set `IDLE_EVICTABLE` first, the dispatcher and make-room waiters are woken, and only then is `idle_expires_at` set. The timer does not gate evictability: `_lru_idle_unloadable` tests the state, so a waiting request can reclaim the resident at any point, unless it holds an unexpired Fast Lane staleness grant. The window ends by:

```
 idle window ─┬─ reuse      a same-model request flips the resident back to ACTIVE
              ├─ expiry     the driver's inbox wait times out and it unloads
              └─ eviction   a make-room decision
```

### 3.6 Multi-slot mode

`max_parallel_sidecars` (1 to 32, default 1; env `TURBOHAUL_MAX_PARALLEL`) is the engine budget. At 1 it runs one engine at a time and queues the rest: a second model unloads an idle resident and is reserved or re-queued as in §3.3, and waits while none is idle.

Residents are keyed by `resident_key` (the model tag, `tag#N` for extra instances). A model gets several engines, one per card, only when its manifest sets `auto_place` and `split_mode: none` (`_wants_one_card_per_engine`), bounded by the budget and the cards it fits; a request goes to its session's pinned instance, else the least-loaded ready one. A manifest's `llama_server_flags.parallel` sets the engine's `--parallel`; `_fan_out_on_resident` serves that many same-model requests at once and drains them before grace.

## 4. KV-cache orchestration

The core design is **one precomputed KV copy per (session, role), reused whenever the saved prefix is valid.** Between tool calls the main agent's context stays in VRAM (`_kv_vram_anchor`). At an unload seam it is saved to the RAM-backed KV tier, and it is copied to the SSD persist tier when another identity takes over the engine slot (§4.5); on return the saved prefix is restored and the engine prefills only what follows it. Disposable roles (sub-agents, curator, compression) do not save by default and use their own bins.

```
 the main agent's KV copy, through its life

 between tool calls          ──▶ stays in VRAM (_kv_vram_anchor)
 its model unloads (a seam)  ──▶ saved to the RAM-backed KV tier
 another identity takes the  ──▶ copied to the SSD persist tier
   engine slot
 the agent returns           ──▶ saved prefix restored; the engine prefills only what follows it
```

### 4.1 Identity derivation ladder (thread_id)

Every request resolves to a `thread_id` by a three-step ladder (`chat_completion.py`, mirrored on `/api/chat`):

1. **Explicit** `thread_id` in the payload.
2. **IP + first-message fingerprint**, only when `max_parallel_sidecars` is 1: `agent-ip-<ip>-auto-<hash>` (bare `agent-ip-<ip>` when no usable first message exists).
3. **Manager fallback** (`slot.py` `derive_thread_id_prefix_hash`, called from `TurbohaulManager.submit`): `auto-` plus the first 24 hex characters of `sha256(model_tag, first N words of the whitespace-normalized prompt)`. Conversation *extensions* (same prefix, more tokens) map to the **same** `thread_id`, so the grace-window follow-up lookup (`_serve_on_resident` calling `queue.pop_matched_thread`) works for clients that send no identity. `N` is `prefix_token_count` (default 256, `TURBOHAUL_PREFIX_TOKEN_COUNT`).

**How the caller's IP is used.** The chat routes (`/v1/chat/completions`, `/api/chat`) capture the source IP into `client_meta["ip"]`; `/v1/embeddings` builds its `client_meta` without one. Its uses include:

1. **Identity anchor.** Rung 2 derives the thread identity from it, so under rung 2 clients with different source addresses get different ids.
2. **Fast Lane.** `TurbohaulManager.submit` passes it to `match_fastlane` at admission, where it can select a priority rule ([docs/FAST_LANE.md](docs/FAST_LANE.md)). It also keys the Fast Lane address census, staleness-grant ledger and per-client fairness-floor timer (`_floor_client_key`).
3. **Observability.** It appears in the per-request `R2B_REQ_IDENTITY` log line, the `/status` `request_identity` block, and the dashboard's residents-box identity strip.
4. **Never authentication.** Under the network-perimeter trust model (§8) the IP is a routing and observability hint only; no allowlist or authentication path reads it. `EventBus.REDACTED_KEYS` drops the top-level keys `ip`, `thread_id` and `client_meta` from every event published on the redacted `/ws/state` bus.

An **identity recompose** layer (`kv_classify.recompose_identity`) appends hashed suffixes to the derived id, `-s=<sha256(session_id)[:12]>` and `-r=<sha256(role)[:8]>`, each only when that field is present. With `TURBOHAUL_M2B_ACTIVE` on (default off; accepts `1`/`true`/`yes`/`on`) the recomposed id drives the slot identity. The suffix is append-only: a derived id is extended, never rewritten.

### 4.2 Role tags, classification, and the save_kv contract

> User-facing guide with examples, the per-role behavior table, and the no-tags fallback ladder: [docs/TAGS_AND_IDENTITY.md](docs/TAGS_AND_IDENTITY.md).

**The client_meta contract.** Identity enters through one shared parser, `_derive_client_meta_identity` (`chat_completion.py`), used at all three `client_meta` build sites (OpenAI non-stream, OpenAI stream, `/api/chat`). Dual-source fields read the **top-level payload first, falling back to the nested `payload["client_meta"]`**: `role`, `session_id`, `is_main`, `is_sub_agent`, `is_curator`, `is_compression`, `save_kv`. `turn0_meta` and `context_size` are top-level-or-derived. **None of these keys are in the knob-forwarding allowlists** (`_COMMON_FORWARDED_KNOBS`), so identity never reaches the llama-server payload.

**Role resolution.**

- `kv_classify._class_from_label` resolves labels by priority, `is_curator > is_compression > is_sub_agent > is_main`, then a literal `role` string that names a known class. A curator can carry both labels, so order matters.
- The bin-keying wrapper `_bin_role` (`manager.py`) returns `None` for unlabeled traffic, which keys by raw `thread_id`, and never infers main by default.
- **Bin-keying guarantee:** `main` is reachable only through an explicit `is_main` label. `_class_from_label` still resolves a literal `role: "main"`, but `_bin_role` post-filters it: a resolved class of `main` without a truthy `is_main` returns `None`. `_bin_identity` therefore never derives a session's `main` bin key for an unlabeled or bare-literal request.

**Bin identity.** `_bin_identity` (`manager.py`) converts `(thread_id, client_meta, admission chain)` into the KV bin key:

```
no session_id, or unlabeled role     → raw thread_id                    (per-thread bin)
sub-agent / curator, chain present   → sess:{session_id}:{role}:{fp8}   (fp8 = hash of the first two chain entries,
                                                                          so concurrent same-role siblings get distinct bins)
sub-agent / curator, no chain        → raw thread_id
any other role (main, compression)   → sess:{session_id}:{role}
```

The save meta stamps this string as the bin's owner and the restore owner-match compares it, so there is **one KV copy per (session, role)**. The manager treats `session_id` as opaque, so a sub-agent given its own id (for example `{parent_session_id}-sub-<nonce>`) gets its own bin.

**Per-role saving, the `save_kv` contract** (`_role_save_enabled`, `manager.py`). The sole source is the per-request boolean `client_meta["save_kv"]`; there are no environment variables. With the field absent, disposable roles (sub-agent, curator, compression) do not save, and main is never gated by it. Unlabeled traffic whose `thread_id` starts with `hermes-sub-`, `agent-ip-` or `auto-` is also treated as disposable and is not flushed at an unload seam (`_seam_flush_allowed`). Setting `save_kv: true` keeps a disposable role's KV.

**The identity proof line.** Each admitted request emits one structured log line, `R2B_REQ_IDENTITY {...}` (`TurbohaulManager._emit_request_identity`). It carries `ip`, `model_tag`, `session_id`, the `is_*` flags, `resolved_class`, `thread_id` and the Fast Lane label.

### 4.3 The resolve_kv decision chokepoint

> **Cache matching happens at two levels.** This turn-level manager gate decides *whether* a saved bin is offered to the engine. The engine's token-level `get_common_prefix` then decides *how much* is reused (§4.6). On the cold-restore path a bin that fails any rule below is skipped and the request prefills fresh; the gate never restores on doubt. Full treatment: [docs/KV_CACHE_MATCHING.md](docs/KV_CACHE_MATCHING.md).

The per-slot save (`_save_slot_kv_inner`) and each cold-restore candidate (`_restore_slot_kv_inner`) are decided by **one function**, `kv_policy.resolve_kv(action, identity, sizes)`. It returns an immutable **`KVDecision(do_it, action, reason, resolved_from)`**, so every decision carries its provenance and is logged.

**Save rules** (`_resolve_save`): refuse zero-token saves (degenerate) and empty-identity saves (cross-thread collision risk); otherwise save.

**Restore rules** (`_resolve_restore`) are evaluated per candidate bin, in order:

```
 candidate bin ─▶ owner differs from the incoming identity? ───────────▶ REJECT           (rule 1)
               ─▶ no data, no identity, no size or no chain? ──────────▶ skip             (rules 2-5)
               ─▶ saved chain longer than the incoming chain? ─────────▶ FRESH            (rule 6)
               ─▶ saved chain a valid prefix of the incoming chain? ───▶ RESTORE          (rule 7)
               ─▶ otherwise (divergence before the saved chain ends) ──▶ FRESH            (rule 8)
```

| # | Rule | `resolved_from` |
|---|---|---|
| 1 | saved owner ≠ incoming identity → **REJECT** | `restore-owner-mismatch` |
| 2 | `saved_tokens ≤ 0` → skip | `restore-no-data` |
| 3 | incoming `thread_id` empty → skip | `restore-no-identity` |
| 4 | incoming admission size never threaded → skip (fail safe, never restore blindly) | `restore-no-incoming-size` |
| 5 | empty incoming admission chain → skip | `restore-no-incoming-chain` |
| 6 | **physics belt:** saved chain longer than incoming → FRESH (the engine would CLEAR) | `restore-physics-belt-saved-longer` |
| 7 | saved chain is a valid **prefix** of the incoming chain → **RESTORE** | `restore-prefix-valid` |
| 8 | otherwise (divergence before the saved chain ends) → FRESH | `restore-diverged-fresh` |

Rule 1 applies only when both the saved owner and the incoming `thread_id` are non-empty. Rule 7 is the prefix gate: validity is decided by the **prefix-hash chain**, an order-sensitive rolling hash `H_i = SHA256(H_{i-1} ‖ role_i ‖ content_i)` over the rendered messages (`kv_policy._prefix_hash_chain`, the single implementation, with `\x00` delimiters). A turn's hash also covers `tool_calls`, `function_call` and `tool_call_id` when present, so requests that differ only in tool name, arguments or call correspondence never share a chain position. `compute_ctx_len` is the single source of truth for context size at admission and at save.

### 4.4 Bin naming & per-model segregation

There are **no per-model directories**. Both KV tiers are flat, and segregation is carried by the filename, minted by the single source `kv_policy.kv_save_fn`:

```
{model_tag}.p{port}.{thread_hash}.slot{sid}.bin      ← engine state
{model_tag}.p{port}.{thread_hash}.slot{sid}.json     ← meta sidecar (owner identity, sizes, hash_chain,
                                                        clean_prefix flag, engine fingerprint, model_tag)
<bin>.ckpt                                            ← engine checkpoint-ladder sidecar
```

`thread_hash` is the first 16 hex characters of the SHA-256 of the bin identity. Cross-model isolation is enforced at every lookup layer:

- The cold-restore listing (`_restore_slot_kv_inner`, `manager.py`) keeps only files that start with `{model_tag}.` and end with `.bin`.
- The warm scan cache and the clean-bin finder (`_find_clean_bin`) key on the exact tuple `(model_tag, thread_hash, port)`.
- Every save stamps `model_tag` into the meta, and tier crossings move files by their tag-prefixed basenames.
- `validate_tag` (`manifest.py`) restricts tags to `^[a-z0-9][a-z0-9._-]{0,63}$`, and both KV inner functions refuse tags containing `/`, `\` or `..`.

A second layer handles *same tag, different model*. Every save stamps a five-field **engine fingerprint** (`_engine_fingerprint`, `manager.py`):

- `gguf_sha256`: the manifest's precomputed blob digest, so the save path never hashes the model file.
- `engine_build_id`: the pinned llama-server binary digest, `None` in dev mode.
- `n_ctx`: the configured context size.
- `n_rs_seq`: the engine's recurrent-state rollback budget: the speculative draft length for speculator types in `SPEC_TYPES_NEEDING_RS_SEQ` (3 when the manifest omits `spec_draft_n_max`), else 0.
- `fp_gen`: a fingerprint-format generation marker.

The opt-in sweep `_purge_mismatched_bins` (`TURBOHAUL_FINGERPRINT_PURGE`, default off) deletes bins whose stamp cannot match the current engine. It checks each sidecar against its own model's current fingerprint, so one model's re-quantization never touches another model's bins. The fingerprint is **not** consulted by the restore decision; it is a file-hygiene sweep only.

**Operational naming rule:** do not create a model tag that equals another tag plus a `.`-suffix (for example `my-model` and `my-model.moe`). Dots are legal in tags, and the cold-path prefix filter would co-list such a pair.

### 4.5 KV save path: unload-seam writes, guards, tiers, GC

**When main's KV is written.** KV is saved right before a model is unloaded, not when a sub-agent enters the queue; the live cache stays in VRAM until then.

The per-turn probe `_probe_and_save_clean_kv` runs before decode. With `save_to_disk=False` it normally only stamps the VRAM anchor (`_kv_vram_anchor`), tracks the dirty tip (§4.7) and returns. The exceptions are the displacement-seam flush (writer 3) and the ownership-transfer persist (see Storage tiers). Every clean save also passes a **single-series gate**: the engine runs one slot and no other resident engine is ACTIVE, otherwise the save is skipped.

The writers are:

1. `_unload_teardown`, run by every resident eviction (make-room unload, idle expiry, dead-engine reap). For a single-slot resident whose engine is alive it calls `_flush_clean_kv_at_unload` and then `_save_slot_kv`, both before SIGTERM.
2. The exit path of `_drive_resident`: the same sequence when a resident driver ends, including at shutdown.
3. **The displacement seam**, a same-model, no-unload displacement. When a disposable visitor (sub-agent or curator) takes a live engine away from a main identity, `_displacement_save_allowed` permits one clean save of the outgoing main identity from inside the probe path. `is_compression` visitors are excluded: a compression turn rewrites main's context, making a pre-compression bin obsolete.

**Clean by construction.** With `kv.covered_scaffold_strip` (default on, Settings, General tab), the seam flush re-renders the transcript think-stripped (`_render_strip_prefill_probe`: `/apply-template`, strip the assistant `<think>` scaffold, `n_predict=0` prefill), so the saved bin byte-matches the client's next think-stripped resend. Save floor: `_MIN_CTX_LEN = 40000` characters (about 13k tokens); only substantial contexts are clean-saved.

**The single save chokepoint.** `_save_slot_kv_inner` is the one funnel for every anchor and clean save. Guards, in order:

```
 D1 dirty-tip ─▶ D2 disposable ─▶ clean-present / never-demote ─▶ resolve_kv("save")
   ─▶ T-GUARD ─▶ clean-stamp evidence gate ─▶ PAIR-GUARD ─▶ atomic finalize (bin, ckpt, then meta JSON)
```

- **D1 dirty-tip refusal**: port flagged dirty, save refused, previous bin kept (§4.7).
- **D2 disposable refusal**: `_seam_flush_allowed` rejects a disposable role whose per-role save is off (the default; a `client_meta` `save_kv` flag overrides it), or an unlabeled identity named `hermes-sub-*`, `agent-ip-*` or `auto-*`.
- **Clean-present skip / never-demote**: a plain save is skipped when a `clean_prefix` bin exists for the thread; a forced re-save that would lower `clean_prefix` from True to False is aborted.
- **`resolve_kv("save", …)`**: the policy decision (§4.3).
- **T-GUARD**: the expected size is calibrated from the thread's previous meta. A pre-flight skips the save when free space is below 1.05 times it; a post-save check rejects a bin under 0.5 times it, keeping the previous bin.
- **Clean-stamp evidence gate** (`TURBOHAUL_CLEAN_STAMP_EVIDENCE_GATE`, default on): a forced save is stamped `clean_prefix=True` only when the probe's think-free token count agrees with the slot's within 8 tokens and the chain fingerprint matches. No probe evidence declines the save; a mismatch writes the bin non-clean.
- **PAIR-GUARD** (when `TURBOHAUL_CKPT_SIDECAR` is on): the `.ckpt` sidecar must land with the bin, at least 10% of its size, and must not be missing when the previous pair had one.
- **Atomic finalize**: temp file then `os.replace` for bin and ckpt, then the meta JSON.

**Storage tiers**

| Tier | Path | Role |
|---|---|---|
| 0, VRAM | (engine memory) | native `get_common_prefix` reuse; back-to-back tool calls never touch disk |
| 1, system RAM | `/var/lib/turbohaul/kvcache` (`SLOT_SAVE_DIR`, a tmpfs) | live save and restore directory; zero SSD wear; emptied on every container restart |
| 2, SSD | `/var/lib/turbohaul/kvcache_persist` (`SLOT_PERSIST_DIR`) | receives bins on ownership transfer; hydrated back to RAM absent-only (`_hydrate_ram_from_persist_for_key`, "RAM always wins while present") |
| none | `/var/lib/turbohaul/kv_quarantine` | poisoned-bin evidence, moved and never deleted (§4.7) |

SSD persistence fires on **ownership transfer**: when a different identity takes the port, `_persist_clean_bin_to_ssd_by_hash` copies the outgoing identity's bins to the SSD tier in the background, before its anchor record is overwritten.

**GC and retention.** `_gc_kv_cache` runs on the background sweeper, throttled to 300 s. The tmpfs pass never deletes the pinned clean anchor or any bin owned by a live thread, then prunes by age (`kv.ram_cache_max_age_hours`, default 6), count (100 files) and a bytes ceiling (`kv.ram_cache_max_bytes`, default 20 GiB, also settable as `TURBOHAUL_KVCACHE_MAX_BYTES`), counting `.bin` and `.ckpt` as a pair. The SSD pass applies the same age cutoff plus `persist.max_bytes` (default 40 GiB), oldest first, skipping files younger than 60 s (copy in progress).

**Size limits.** There is no absolute maximum bin size; the guards self-calibrate. The GC settings are adjustable through `PUT /api/config` and re-read on the next pass; `kv.ram_cache_max_bytes <= 0` disables the byte ceiling only. Save-guard thresholds are code constants with no config field. The per-role `save_kv` toggle is per-request `client_meta`, not config (§4.2).

### 4.6 KV restore path: wave-return and warm reuse

**Cold restore** runs when an engine has just been (re)spawned and passed its health check: `_spawn_for_resident` calls `_restore_slot_kv` right after health, and the dirty-tip remediation (§4.7) calls it once inside the `save_to_disk=True` seam flush, which unload teardowns and the displacement seam both reach. Warm decode never calls it.

Per restore, `_restore_slot_kv_inner`:

1. **Hydrates** the key from the SSD tier when RAM holds no bin for it (once per RAM-empty gap, never a stale-marked bin), then lists the RAM tier for `{model_tag}.*.bin`.
2. **Filters candidates.** A candidate needs its meta sidecar, a `bin_bytes` stamp equal to the file size when the meta carries one (an absent stamp is not checked), and no `stale` mark (a labeled `is_compression` turn marks the session's main bin stale at admission).
3. **Decides.** Every candidate goes through `resolve_kv` (§4.3) and **only the single best bin is restored** (sort: `clean_prefix` first, then token count).
4. **Guards.** Two decline-only guards run before the POST, both default on: a size guard (`TURBOHAUL_COLD_RESTORE_SIZE_GUARD`) declines a clean bin more than 32 tokens larger than the incoming request renders to, and a tools-fingerprint guard (`TURBOHAUL_COLD_RESTORE_TOOLS_FINGERPRINT_GUARD`) declines a bin whose recorded tools fingerprint differs from the request's.
5. **Restores.** The engine call is `POST /slots/{id}?action=restore {"filename": …}`, status-checked; the engine's `get_common_prefix` on the next decode decides actual reuse. It returns the restored bin's saved token count for load verification (§4.8); a declined or failed restore returns nothing and leaves a fresh prefill.

**Outcomes.** Each decision is counted in `/status` under `kv_classifier` (counts by event type, `wave_return_restores`, the last decision):

- `wave-return-clean-restore` (POST succeeded on a think-free clean bin) and `wave-return-shadow-restore` feed `wave_return_restores`.
- `restore-no-bins`: first-seen identity or fresh sub-agent nonce. `restore-no-anchor-for-identity`: bins belong only to other identities. `restore-diverged-fresh`: the owned bin is not a valid prefix.
- `restore-guard-skip`, `cold-size-decline`, `cold-tools-fingerprint-decline`, `restore-post-failed`: a policy skip, either guard above, or an engine error.

**Warm path.** A follow-up served on a resident that already holds the thread (`_serve_on_resident`) decodes on live VRAM state: native prefix reuse, delta-only prefill, no dependence on saves.

### 4.7 Seam integrity and poisoned-bin defense

A disposable role's serve **extends the shared VRAM tip** past main's canonical chain. On hybrid-recurrent models that tail cannot be trimmed afterwards beyond a small bounded rollback (removing a partial sequence range aborts the engine, `common_context_seq_rm`), so a seam save of that state would persist an inconsistent bin that passes the prefix gate and aborts the engine at strict extension.

Defense in depth:

- **D1, dirty-tip tracker** (`_kv_dirty_tail`, per port). A disposable turn that passes the probe's early exits (single-series gate, context floor, messages present) SETs `{role, pid, thread, ts}`; the role comes from the **admission stamp** `slot.admission_role`. A main or unlabeled turn CLEARs it, because the engine reprocessed from the divergence point. The record carries the engine pid; a seam that finds another pid drops it as stale.
- **Seam remediation.** At a dirty seam the manager reads `/slots` (`_classify_dirty_tip_probe`). No populated slot: proceed without an erase. Exactly one: `POST /slots/{id}?action=erase` (a whole-sequence erase is legal on a recurrent context). A failed probe or erase, or more than one populated slot: save refused, previous good bin kept. After the erase (or the nothing-to-erase proceed) it first tries restoring an existing clean snapshot and verifying it (§4.8); otherwise the unchanged strip probe **re-prefills the full canonical render onto the empty slot**, so the bin equals the canonical chain by construction, at a one-time seam cost. An incomplete reprefill or save re-arms the dirty record.
- **D2, disposable-identity refusal** at the same chokepoint (§4.5).

- **3-strikes auto-quarantine.** Each successful restore records `_last_restored_bin[port]`. When `_idle_engine_liveness_sweep` finds an idle resident's engine dead, `_record_bin_death_strike` blames that bin, **one blame per attribution** (the record is popped after use). At 3 strikes `_quarantine_bin_triplet` moves its `.bin`, `.bin.ckpt` and `.json` out of both tiers into `/var/lib/turbohaul/kv_quarantine`, so the next load goes fresh instead of looping. Each strike also sets an `engine_stall` field in `/status`.

### 4.8 Load verification (LOAD_VERIFY)

`load_verify_log.py` is the **observability-only** proof layer; it never gates behavior.

- `verify_kv_restored(handle, slot_id, expected_tokens, *, actual_n_past=None, threshold=0.98, timeout=2.0)` compares restored depth with expected depth. The actual value is the engine's restore-response count when the caller holds one, otherwise `n_prompt_tokens` from `/slots`.
- `kv_restore_ok` is computed in one place (`_kv_verdict`): `True` or `False` only when `expected` is a positive real number and `actual` is a real number (`actual >= expected * 0.98`), otherwise `None` (not measurable). For `kv_restore` records, `log_load_verify` refuses a passing `final_status` (`ok` or `retried_ok`) next to anything but `True`, coercing it to `failed` (verdict `False`) or `unverified` (verdict `None`).
- `verify_model_resident` (pid alive, `/health` 200, `/slots` readable) and `verify_model_identity` are pure read helpers.
- One greppable `LOAD_VERIFY {json}` line per event, plus a 64-record ring; `/status` returns the newest 20 as `load_verify`, rendered per model by the dashboard resident card (`LoadVerifyWidget`).

Emission: `_spawn_for_resident` emits a `kv_restore` record with trigger `session-return` after each restore attempt, with the restored bin's saved token count as the expectation. A spawn that restores nothing reads `kv_restore_ok: null`, `final_status: unverified`. The dirty-tip remediation runs the same check without a record. `retry_count` is always 0; there is no retry loop.

## 5. Engine sidecar supervision

Each model runs in a `llama-server` child process (a sidecar) that the manager spawns, health-checks, tears down and reaps.

### 5.1 Spawn: three-layer flag pipeline and pinned-binary exec

A sidecar's argv passes three enforcement layers before exec:

```
 manifest ─▶ 1 manifest validation ─▶ 2 argv build ─────────▶ 3 spawn ───────────▶ llama-server
             closed flag allowlist     flags_to_argv           spawn_sidecar         own process group
             (manifest.py)             re-checks, --kebab      manager flags first   (start_new_session)
```

1. **Manifest validation** (`manifest.py`): `llama_server_flags` is a **closed allowlist**. Over fifty path-, URL-, credential- and RCE-class flags are denied (`DENIED_FLAGS`, including `slot_save_path`, `log_file`, `model`, `port`, `host`). A suffix-pattern guard (`^slot_save_`, `^api_key`, `^ssl_`, `.*_model$` and others) rejects future path-bearing names. Membership in `SAFE_LLAMA_FLAGS` is then required and values are validated; `parallel` > 1 requires `kv_unified: true`.
2. **argv build** (`flags_to_argv`): allowlist and denylist are re-checked; snake_case becomes `--kebab-case`; `True` emits a bare flag, `False` omits it; `flash_attn` is a tri-state special case.
3. **Spawn** (`spawn_sidecar`, `subprocess_mgr.py`): the manager injects its own flags first, `--port ... --host 127.0.0.1 -m <gguf> --slot-save-path /var/lib/turbohaul/kvcache --log-file <engine_log>`, then the manifest flags. Manifests cannot set those. Projector and standalone-draft models are named by blob sha; the manager resolves them in the blob store and appends `--mmproj` / `--spec-draft-model`. `start_new_session=True` (setsid) gives the child its own process group.

**TOCTOU-pinned binary exec.** When `runtime.llama_server_binary_sha256` is set, boot verification (`open_and_verify_binary`) hashes the binary through an open file descriptor and holds it open. Every spawn execs `/proc/self/fd/<fd>`, so a path swap after the hash cannot redirect the exec. An empty pin execs by path (development mode).

**`--parallel` pinning.** `spawn_sidecar` parses the argv it passed into `SidecarHandle.parallel`; the in-flight cap in `_serve_on_resident` reads that field, never a later manifest read.

**Stdio capture.** Child stdout and stderr append to `<engine_log>.stdio`, because GGML abort output bypasses `--log-file` and a regular file never back-pressures the child.

### 5.2 Health, teardown, VRAM verification

- **Health:** `wait_until_healthy` polls `/health` every 2 s up to `queue.loading_health_timeout_s` (code default 60 s, range 10 to 7200 s; see [docs/MODEL_LOAD_TIMEOUT.md](docs/MODEL_LOAD_TIMEOUT.md)). Each poll checks liveness, so a child that exited during load fails at once; one stall warning is logged at the midpoint. A 200 response whose JSON is not an object with a `status` field raises `SchemaMismatch`. An unhealthy sidecar is reaped; an out-of-memory load failure re-queues the request if the client is still connected and the model is not known to exceed capacity; otherwise it fails once.
- **Teardown:** each resident is torn down exactly once (`torn_down`), by its driver on exit or by the detached `_unload_teardown` for an eviction.

  1. KV flush and save before any signal, for a live engine with an idle thread id (best effort); a dead one skips it.
  2. Drained SIGTERM to the **process group** (`drained_sigterm`), escalating to SIGKILL, with an explicit `wait` so no zombie remains. The window is `drained_sigterm_window_cold_s` (default 5 s); `drained_sigterm_window_active_s` (default 15 s) exists, but every call site passes `is_active=False`, so the 5 s window governs. A still-booting sidecar is reaped by pid (`_reap_booting_pid`).
  3. Orphan passes (`_reap_cap2_orphans`, §5.3), protecting every resident's live, idle and booting pid.
  4. **VRAM-clear verify** (`verify_vram_cleared`) after an eviction. The victim's reserved VRAM stays credited as pending reclaim on its card until `nvidia-smi` shows at least a 90% drop (clamped to what the card held), polled after a 5 s settle floor for up to 30 s. Confirmation releases the credit and wakes waiters; a timeout releases only the measured drop. An unreadable `nvidia-smi` trusts the kill.

### 5.3 Singleton and orphan enforcement

`singleton.py` implements "only writer to the GPU" as follows.

- **Durable engine identity.** After each spawn the manager records the engine's `(pid, port, starttime)` in `state.sqlite` (`record_engine_identity`). The boot reaper kills only a live process matching a record; a recycled pid has a different starttime. The boot reaper reports, and does not reap, an unrecorded `llama-server` in the managed range, even when orphaned to init.
- **Boot orphan reaper** (`boot_orphan_reaper`): scans `/proc` for `llama-server` processes with `--port` in the managed range (`default_port_base`, 100 ports) and reaps those with a matching record, whether orphaned to init or a subreaper (tini, systemd) or still parented to a dying manager. `reap_orphan` signals the whole process group, SIGTERM then SIGKILL. It runs at boot and after each resident teardown.
- **Intra-lifetime orphan scan** (`intra_lifetime_orphan_scan`): SIGTERMs managed-range `llama-server` processes (matched by port alone) whose pid is not in the live-handle set, such as sidecars still parented to the running manager after a lost handle; the final safety net after a failed teardown.
- **Foreign GPU applications** are detected at boot; this is informational (logged, not refused).
- `acquire_state_lock` (an exclusive `fcntl.flock` on `state.sqlite`) is implemented and unit-tested but not called by the manager.

### 5.4 Vendored engine and build

The engine is [Tom's TurboQuant fork of llama.cpp](https://github.com/TheTom/llama-cpp-turboquant) (MIT), **vendored** at `engine/llama-cpp-turboquant/` as a history-free snapshot. **`engine.lock`** pins the engine revision, build number, branch and image tag, and notes that checkpoint-ladder cold restore is proven on this engine (`TURBOHAUL_CKPT_SIDECAR` requires it). It is not read at runtime.

- **`Dockerfile.engine-src`** is the production build. It compiles the vendored tree (`CUDA_ARCH`, default `89;120`), installs Python dependencies offline from `vendor/pywheels/` and copies the committed frontend build from `src/frontend/dist`: no PyPI, npm or external git. Build guards `scripts/check_engine_symbols.sh` (every undefined engine symbol resolves) and `scripts/check_engine_kv_variant.sh` (turbo-K auto-upgrade present) fail the build on a partial or wrong-variant engine.
- **`Dockerfile.cuda-multi`** compiles the same tree for CUDA architectures 75, 80, 86, 89, 90 and 120.
- **`Dockerfile`** is the slim manager-only image; `docker-compose.yml` mounts the binary at `/opt/turbohaul/bin/llama-server`.

**TurboQuant and speculative decoding.**

- **KV quantization.** `turbo2`, `turbo3` and `turbo4` are accepted for `cache_type_k` and `cache_type_v` (§9.1); `safety.py` scales them 0.125, 0.1875 and 0.25 of f16 and costs K and V independently (§6). With a GQA ratio of 6 or more and identical K and V turbo types, the engine upgrades K to `q8_0` unless `TURBO_AUTO_ASYMMETRIC=0`. See [docs/TURBOQUANT_FLAGS.md](docs/TURBOQUANT_FLAGS.md).
- **Speculative types.** `spec_type` accepts `draft-mtp`, `draft-dflash` and `draft-dspark`. The engine fingerprint records `n_rs_seq` as `spec_draft_n_max` (3 when omitted) for these types and 0 otherwise.
- **Standalone draft.** `draft-dflash` and `draft-dspark` load a separate drafter named by `spec_draft_gguf_blob_sha256`; for other types the hash is ignored and a downgrade is recorded. The `spec_draft_*` tuning knobs are allowlisted flags.
- **Downgrade, not abort.** When engine fit-params is on (default; manifest flag `fit`) and the draft context cannot be initialised for the target's architecture, the engine logs `SPEC_DOWNGRADED` (`draft_context_init_failed`) and serves without speculation. The manager exposes it as `spec_downgrade` records on `/status` and in the dashboard (§11). See [docs/SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md).

## 6. Admission control and safety gates

```
 request needs a new engine
   ─▶ reservation: cross-resident VRAM gate (_vram_admits_locked)     does not fit: stays queued, retried with a short back-off
   ─▶ before each cold spawn: seven safety gates (all_safety_gates)   refused: audit events and an error naming each failed gate
   ─▶ spawn (§5)
```

At reservation, under the registry lock, the cross-resident gate (`_vram_admits_locked`) budgets the new resident's footprint against live free VRAM per card, minus VRAM reserved by still-loading siblings (for the eviction decision only, plus VRAM pledged by evictions in flight). Co-residence requires every resident to be single-card (`split_mode: none`). A request that does not fit stays queued, retried with a short back-off, until the client disconnects.

Before each cold spawn (`queue.safety_enabled`, default on), `_run_spawn_safety_gate` calls `all_safety_gates` (`safety.py`). It runs **seven** gates without short-circuiting; the audit record and client error list only the failed ones:

| Gate | Passes when |
|---|---|
| `ram` | `MemAvailable` ≥ `safety_min_free_ram_mib` (default 1024 MiB) |
| `vram` | effective free VRAM ≥ the larger of `safety_min_free_vram_mib` (default 512 MiB) and the manifest's expected VRAM |
| `kv_cache_fit` | model body + KV + overhead (default 1024 MiB) + per-slot floor ≤ free VRAM |
| `moe_ram_fit` | host RAM covers CPU-offloaded experts plus KV when `no_kv_offload` is set; no-op otherwise |
| `cpu_util` | sampled host CPU busy ≤ `safety_max_cpu_busy_percent` (default 90) |
| `iowait` | sampled iowait ≤ `safety_max_iowait_percent` (default 30) |
| `tensor_split_devices` | `tensor_split` element count equals the visible device count |

- Free VRAM aggregates all cards for a layer-split model and reads the pinned card for `split_mode: none`.
- A gate without a probe passes (`passed-no-probe`), except `kv_cache_fit` with `parallel` > 1, which refuses.
- **`occupied_vram_mib`**, VRAM reserved by a same-card sibling still loading, is subtracted in both VRAM checks.
- **`mtp_draft_active`** adds an MTP draft-context KV term on the parsed-dimensions path below, when the GGUF reports `nextn_predict_layers` > 0: `nextn_predict_layers` × `n_head_kv` × (key + value length) × 2 × 1.05 × ctx. That cache is always f16 K/V, so its quant scale is 1.0.

**The KV estimator** (`estimate_kv_cache_mib`) scales the K and V halves by their own cache types (f16 = 1.0, q8_0 = 0.5, turbo2 = 0.125). Precedence:

1. A measured `kv_bytes_per_token` manifest override, used verbatim.
2. Parsed GGUF attention dimensions (`_gguf_meta.py`): attention layers only, plus a window-capped term for sliding-window layers. Used when the GGUF has sliding-window layers or the manifest sets a non-empty `arch`.
3. A file-size heuristic of about 9 KB per token per GiB of body at f16. `hybrid_kv_ratio` (§9.1, [docs/HYBRID_KV_RATIO.md](docs/HYBRID_KV_RATIO.md)) scales this path only. At the defaults (`arch: ""`, ratio 1.0, no override) a model without sliding-window layers takes this path.

With `no_kv_offload`, KV is costed against host RAM and a ctx-scaled scratch term replaces it in VRAM. When a spawn is refused only for VRAM that a live eviction is still releasing on the gated card, it waits up to `queue.spawn_reclaim_wait_max_s` (default 10 s, 0 disables), re-running the gates as releases complete; any other refusal is terminal at once. A terminal refusal writes audit events `safety_gate_refused` and `safety_gate_detail` and fails the request with an error naming each failed gate, unless another engine of the same model takes it. See [docs/SAFETY_GATE_VRAM_MATH.md](docs/SAFETY_GATE_VRAM_MATH.md). HTTP-level admission is bounded by the body-size middleware (§7) and acceptance-buffer cap (§3.1).

### 6.1 Fast Lane (priority admission)

Fast Lane is off by default (`fastlane.enabled`). It orders which waiting request is served next and, when the server is full, chooses which idle model gives way at a turn boundary. It never interrupts or cancels a generating request.

```
 admission  source address ─▶ match_fastlane ─▶ ordered rule table (at most ten rules), stamped once on slot.fastlane
 next       unlisted waiter older than max_normal_wait_s first (unless its client was served within that window);
            otherwise listed waiters sorted by (rule_index, rank, arrival)
 gives way  unlisted first ─▶ highest rule_index ─▶ worse tag rank ─▶ least recently active
```

**Matching.** `match_fastlane` tests a request's source address as the server sees it on the connection (`request.client.host`) against an ordered rule table built by `compile_fastlane`. A rule names its client by exactly one of `address` (a single host) or `container_name`. `FastLaneNameResolver` resolves names forward on a background timer into a cache that admission only reads. A name that fails to resolve yields an empty set: the rule matches nothing and matching continues. The table holds at most ten rules (`MAX_FASTLANE_RULES`).

**Tiers.** The first tier is the rule's list position (`rule_index`, lower is served first); unmatched clients rank below every listed one. The second is a tag rank of 1-5 inside one rule for the classes `main`, `curator`, `compression`, `sub_agent` and `unclassified`; an unset tag sorts last. Tag ranks are compared only between requests of the same rule.

**Stamping.** `TurbohaulManager.submit` stamps the match on `slot.fastlane` once at admission and never reassigns it; a loaded model's rank is resolved against the current rule table each time eviction considers it.

**Ordering.** `TurbohaulQueue._pick_fastlane_locked` runs on each `pop_next` while Fast Lane is enabled. An unlisted waiter older than `max_normal_wait_s` is served first, unless its client was served within that window. Otherwise listed waiters sort by `(rule_index, rank, arrival)`. Listed requests are exempt from the `cross_model_switches_per_min` swap budget. A same-thread follow-up (served by `pop_matched_thread`, not the picker) reuses the warm model unless a strictly better request of its own client is waiting.

**Eviction.** Candidates are `IDLE_EVICTABLE` residents with no running turn and no unexpired staleness grant (`_idle_unload_candidates`), plus a designated victim after its turn ends. `_unload_priority_key_for_meta` orders them: unlisted first, then the highest `rule_index`, then the worse tag rank, then least recently active. A listed claimant may unload the worst idle resident on any card only if it strictly outranks it. A resident in its grace window loses the rest of it (`_serve_on_resident`) when it is the designated victim, holds a strictly better request in its own inbox, or an unlisted request for another model has waited past `max_other_model_wait_s` (`queue` section, default 20); a Fast Lane request never ends grace that way. The victim is torn down after its turn completes, and `_unload_teardown` saves its KV cache first when it can.

**Surface.** The `fastlane` section of `GET` and `PUT /api/config` applies to the next request: `enabled`, `rules` (`address` or `container_name`, `label`, `tag_ranks`), `max_normal_wait_s` (default 3600), `cross_model_switches_per_min` (default 2) and `census_ttl_hours`. `GET /api/fastlane/census` lists observed addresses.

See [docs/FAST_LANE.md](docs/FAST_LANE.md) and [docs/design/FAST_LANE_ARCHITECTURE.md](docs/design/FAST_LANE_ARCHITECTURE.md).

## 7. API surface

`create_app` (`api/main.py`) fronts one `TurbohaulManager`. The lifespan boots the audit pool, runs boot reconcile, verifies the llama-server binary, optionally purges mismatched fingerprints, then starts the worker loop, sweeper and live monitor. The only middleware is `BodySizeLimitMiddleware`; there is no app-layer auth (§8). FastAPI's default `/openapi.json`, `/docs` and `/redoc` routes stay enabled.

Every route, with its method and notes (including the 501 stub, the routes that are not implemented and the limits), is listed in [docs/API_REFERENCE.md](docs/API_REFERENCE.md). This section covers how the API layer is built.

**Chat-completion compat layers** (both chat endpoints unless noted):

- **`response_format`**: `json_object` passes through. `json_schema` is bounded (64 KiB, depth 16, 64 properties), rejects `$ref` and requires `additionalProperties: false`; on non-streaming requests, thinking models get a one-retry `enable_thinking: false` fallback.
- **Knob forwarding**: the allowlist `_COMMON_FORWARDED_KNOBS` carries samplers and tools; identity keys are excluded.
- **Reasoning**: wrapped back as `<think>…</think>` on non-streaming responses; streams carry `reasoning_content` and `content` as separate deltas.
- **Tool-call recovery** (`api/tool_call_recovery.py`; non-streaming results only): when tools were advertised and the model emitted the call as text (`{"name":…,"arguments":…}` JSON, or `<tool_call>` XML after `</think>`), a brace-balancing parser recovers it into structured `tool_calls` against the advertised-name allowlist, strips the consumed span and sets `finish_reason`. A response with upstream-populated `tool_calls` is left unchanged. See [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).
- **`keep_alive`**: accepts Ollama forms (`0`, `-1`, `"5m"`), clamped to `KEEP_ALIVE_MAX_S` (1800).
- **Typed errors**: slot eviction before activation is 499; sidecar unavailable and capacity over-commit (`VramOverCommitError`) are 503 with `Retry-After`; sidecar timeout is 504; an upstream sidecar 4xx/5xx is 502; queue overflow or closure and load or safety-gate failure are 500.
- **Streaming** (`/v1/chat/completions` only; `/api/chat` never streams): raw SSE bytes pass through from the sidecar. Keep-alive comments go out during prefill every `HEARTBEAT_INTERVAL_S` (12 s); the wait for a ready slot is bounded by `SLOT_READY_TIMEOUT_S` (7200 s), except that a slot parked after an OOM requeue is exempt while parked. Each chunk also feeds the live-output tee, fail-open and after the client yield. A failure after the response starts arrives as an SSE error frame plus `data: [DONE]` under HTTP 200.

## 8. Security model & hardening

**Trust boundary: network perimeter.** Every endpoint runs without app-layer auth; the bind address is the boundary. `server.host` defaults to `127.0.0.1`, and the config validator rejects `0.0.0.0` from yaml; any other `server.host` is accepted as given. The `--allow-public-bind` CLI flag or `TURBOHAUL_ALLOW_PUBLIC_BIND=1` (either suffices; the env var seeds the flag's default) selects the public bind: `__main__.py` binds a dual-stack `::` socket (`IPV6_V6ONLY=0`) on `server.port`. A deployment crossing a perimeter should put a bearer-auth reverse proxy in front of every path.

```
 outside the perimeter ─▶ bearer-auth reverse proxy ─▶ bind address = THE BOUNDARY ─▶ manager
                          (when a deployment crosses     server.host, default 127.0.0.1    every endpoint, no app-layer auth
                           a perimeter)
```

Enforcement stack:

- **SSRF guard** (`ssrf_guard.py`, `api/pull.py`): `https` only; the hostname resolves to its first record, which is checked against deny-listed networks (RFC1918, loopback, link-local including the metadata range, CGNAT, `0.0.0.0/8`, multicast and reserved space; IPv6 loopback, unique-local, link-local, multicast, NAT64, IPv4-compatible, IPv4-mapped and documentation ranges). `_double_resolve_check` resolves twice and refuses on divergence (DNS-rebinding and round-robin defense). Redirects are re-validated per hop (at most 5), stripping `Authorization` on cross-host hops. The HuggingFace bearer token attaches only to hosts in `pull.hf_host_allowlist`.
- **Plugin egress guard** (work in progress feature; `plugin_invoke.py`): `resolve_endpoint` is the address-policy chokepoint shared by `invoke_plugin` and the exec WebSocket.
  - The destination is never caller-supplied: a manifest names a `resource_key`, which resolves against `plugins.registry`, a boot-only section (`BOOT_SECTIONS`) that `PUT /api/config` cannot write.
  - The IP policy is narrower than the pull guard's: a plugin endpoint is the operator's own internal container, so blocking private and CGNAT ranges would refuse every plugin.
  - Still blocked: `0.0.0.0/8`, multicast, reserved, the link-local metadata range and its IPv6 analogue, NAT64, IPv4-compatible `::/96` (which also covers `::1`), the IPv6 documentation range, and loopback in both address families, so a plugin entry cannot point at the manager's own API.
  - An IPv4-mapped IPv6 address is unwrapped and re-checked against the IPv4 policy. The check covers IP-literal hosts only.
  - Per-endpoint credentials are an `auth_token_file` path read fresh on every call, so the config never holds the token. The exec forwarder sends a bearer token from `EXEC_PLUGIN_AUTH_TOKEN` when set.
- **Import safety** (`api/import_.py`): absolute path, `DENIED_PATH_PREFIXES` (`/proc/`, `/sys/`, `/dev/`, `/etc/`, `/root/` and others), symlinks rejected plus `O_NOFOLLOW`, resolved path must stay under `import_allowed_root`, 4-byte `GGUF` magic check.
- **Blob integrity** (`blob_store.py`): streamed to `incoming/` under a per-stream byte ceiling, sha256 verified, renamed atomically into the content-addressed store, then `chmod 0o400`. `verify_blob_on_stage` re-hashes a stored blob, but no runtime path calls it; model load relies on the write-time hash and the read-only mode.
- **Manifest hardening**: tag regex validation; a closed spawn-flag allowlist with denied path and credential flags plus a suffix-pattern defense (§5.1); rejection of Jinja constructs (`{%`, `{{`) in the `chat_template` flag value; atomic ETag/`If-Match` writes.
- **Config write protection**: boot sections are rejected on PUT with 403, so no runtime change can swap the binary; the binary itself is verified against `runtime.llama_server_binary_sha256` (empty skips the check) and executed through a pinned file descriptor (§5.1).
- **Redaction: streams and reads**:
  - The event bus strips the top-level `REDACTED_KEYS` (`prompt`, `response`, `context`, `stderr`, `stdout`, `messages`, `thread_id`, `ip`, `client_meta`, `idle_client_meta`) from every event before `/ws/state` delivery, and the initial `connected` snapshot carries only an 8-character thread-id prefix.
  - `/v1/logging` payloads pass a recursive redaction against the same keys to depth 10 (`_REDACT_DEPTH_CAP`; deeper values are returned as-is).
  - The denylist is a tripwire; emitters keeping prompts, responses and identifiers out of payloads is the real protection.
  - Live generation text flows only on its dedicated SSE stream, and `GET /api/config` reduces paths to basenames.
- **Redaction: error bodies.** An HTTP error `detail` is an exposure surface (`yaml.YAMLError` carries parser position and file content, pydantic `ValidationError` echoes manifest field values, `OSError` carries an absolute path), so posture differs by audience:
  - **Discovery routes redact.** `/v1/models/{model}` answers a corrupt manifest with the fixed text `manifest for 'X' is present but unreadable` and logs the exception server-side. The handled 500 keeps a corrupt manifest from reading as a mistyped tag.
  - **Management routes do not.** `GET /api/manifests/{tag}` returns `str(e)` with a 400 for parse and validation failures: the caller receives the whole manifest on success, and an operator needs the parser's line number. `OSError` gets a fixed message and a server-side log.
  - **Listing routes survive one bad file.** `/v1/models`, `/api/tags` and `/api/plugins` skip an unreadable manifest and log its type name; `/api/manifests` returns a placeholder row.
  - [docs/PLUGIN_REFERENCE.md](docs/PLUGIN_REFERENCE.md) §5 tabulates the plugin routes' errors.
- **The plugin exec WebSocket is inside the trust boundary.** `WS /api/plugins/{model_tag}/exec` (§9.2, `api/exec_ws.py`; work in progress) forwards exec frames to the resolved plugin container. With no app-layer auth, anyone who can reach the bind address can open it. The same `resolve_endpoint` guard as `invoke` applies, but only the bind address bounds who can reach the route.
- **Request bounds**: `BodySizeLimitMiddleware` (`body_size_limit.py`) returns 413 before routing on `/v1/chat/completions` and `/api/chat`, and the embeddings route (`embeddings.py`) has the equivalent `_content_length_gate`; both read `http.max_body_bytes` (§9, default 64 MiB, `TURBOHAUL_MAX_BODY_BYTES`).
  - Both compare the declared `Content-Length`, so a chunked request without one is not size-checked.
  - `http` is a runtime-adjustable section, `max_body_bytes` has no upper bound and `PUT /api/config` carries no auth, so the cap bounds accidents, not an adversary inside the perimeter, who can raise it first. It is a resource guard, not a security control.

## 9. Configuration

`TurbohaulConfig` (pydantic, `extra='forbid'`, `config.py`) is loaded from a YAML file (`/etc/turbohaul/turbohaul.yaml` by default; `TURBOHAUL_CONFIG_PATH` or `--config` override it), overridden by a fixed 15-entry `TURBOHAUL_*` environment map (`_ENV_MAP`), then **split** into:

- a frozen **BootConfig** (`server`, `storage`, `runtime` paths, `ui`, `plugins`), which changes only on restart;
- a mutable **RuntimeConfig** (`queue`, `pull`, `persist`, `monitor`, `kv`, `fastlane`, `http`, `plugin_runtime`), which `PUT /api/config` updates at runtime.

`PUT /api/config`:

- rejects boot sections (HTTP 403) and unknown sections (HTTP 400);
- validates the merged result against `RuntimeConfig` and swaps the runtime object atomically;
- persists the touched fields to `runtime_config.yaml` beside the state database (merged at boot, runtime sections only).

Resident grace and idle timers (§3.6) take new `queue` values at that resident's next turn; a running timer keeps its window. The writable set (`RUNTIME_SECTIONS` in `api/config_put.py`) is derived from `RuntimeConfig`'s fields. `GET /api/config` reports every section, a per-field `_provenance` (the supplying layer: `default`, `yaml`, `env` or `persisted`, later winning) and a `_provenance_stamp` (when a field was last written).

**Environment map** (`_ENV_MAP`), with code defaults:

| Variable | Field | Code default |
|---|---|---|
| `TURBOHAUL_HOST` | `server.host` | `127.0.0.1` |
| `TURBOHAUL_PORT` | `server.port` | 11401 |
| `TURBOHAUL_MAX_PARALLEL` | `queue.max_parallel_sidecars` | 1 (range 1..32) |
| `TURBOHAUL_STAGING_DEPTH` | `queue.staging_queue_depth` | 100 |
| `TURBOHAUL_ACCEPT_MAX` | `queue.acceptance_buffer_max` | 10000 |
| `TURBOHAUL_GRACE_S` | `queue.grace_seconds` | 30 |
| `TURBOHAUL_IDLE_HOT_S` | `queue.idle_hot_load_seconds` | 600 |
| `TURBOHAUL_MAX_GRACE_EXT` | `queue.max_grace_extensions` | 5 |
| `TURBOHAUL_PREFIX_TOKEN_COUNT` | `queue.prefix_token_count` | 256 |
| `TURBOHAUL_GRACE_IP_SHARED_ADDRESSES` | `queue.grace_ip_match_shared_addresses` | empty |
| `TURBOHAUL_MAX_CONSECUTIVE_SAME_MODEL` | `queue.max_consecutive_same_model` | 3 |
| `TURBOHAUL_MAX_OTHER_MODEL_WAIT_S` | `queue.max_other_model_wait_s` | 20.0 |
| `TURBOHAUL_KVCACHE_PERSIST_MAX_BYTES` | `persist.max_bytes` (SSD KV tier) | 40 GiB |
| `TURBOHAUL_KVCACHE_MAX_BYTES` | `kv.ram_cache_max_bytes` (RAM KV tier) | 20 GiB; `<= 0` disables; read fresh each GC pass |
| `TURBOHAUL_MAX_BODY_BYTES` | `http.max_body_bytes` | 64 MiB |

Every `max_parallel_sidecars` value uses the resident dispatcher (§3.6); at 1 it spawns one sidecar at a time and queues the rest. Engine ports are allocated lowest-free from `runtime.default_port_base` (default 11500) over 100 ports.

**Feature flags read directly from the environment** (outside `_ENV_MAP`). These nine default off and turn on with `1`, `true`, `yes` or `on`:

- `TURBOHAUL_M2B_ACTIVE`: the role-and-session-keyed identity drives the slot identity (§4.1); off, it is only computed and logged.
- `TURBOHAUL_DURABLE_RING`: an in-memory index of the last 3 saved bins per (role, session); it indexes on-disk bins, restarts empty, and is not a storage tier.
- `TURBOHAUL_CKPT_SIDECAR`: finalizes the engine's checkpoint-ladder sidecar beside each saved bin; it mirrors the engine-side switch and needs the engine pinned in `engine.lock`.
- `TURBOHAUL_CURATOR_REUSE_MAIN`: a labelled curator turn restores the main bin and skips its own save.
- `TURBOHAUL_SHADOW_REPREFILL`, `TURBOHAUL_SHADOW_RESTORE_PREFER`, `TURBOHAUL_WARM_NATURAL_SKIP`, `TURBOHAUL_TOOLTAIL_RESTORE_SKIP`: alternative save and restore strategies, kept for A/B comparison and as containment switches.
- `TURBOHAUL_FINGERPRINT_PURGE`: sweeps bins whose engine fingerprint is stale (§4.4).

Five guards default on and turn off with `0`, `false`, `no` or `off`: `TURBOHAUL_COLD_RESTORE_SIZE_GUARD` (declines a cold restore of a bin larger than the request), `TURBOHAUL_CLEAN_STAMP_EVIDENCE_GATE` (declines a force-clean save lacking a recorded think-free render count), `TURBOHAUL_STALETAIL_RESTORE_SKIP` (skips a forced warm restore when resident state holds more turns than the request), `TURBOHAUL_WARM_FORCE_CLEAN_RESTORE` (clean-bin restore on a warm follow-up) and `TURBOHAUL_RAM_REANCHOR` (per-turn RAM save and restore re-anchor).

Per-request knobs ride the payload: `keep_alive`, `thread_id`, `response_format`, the forwarded sampler and tool set, and the `client_meta` identity contract (§4.2) including the per-role `save_kv` toggle. The KV contract is request-level, not environment-level: without `save_kv`, the main role saves and disposable roles do not (`_role_save_enabled`).

**`http.max_body_bytes`** (default 64 MiB, enough for base64-inline multimodal input, since base64 adds about 33%) is the one live field that bounds request bodies: the ASGI middleware and the embeddings route answer HTTP 413 when the declared `Content-Length` exceeds it (see Request bounds in §8). The uvicorn server in `__main__.py` sets no body-size option.

### 9.1 Per-model configuration (manifests)

Every model is configured by **one YAML manifest per tag**, `<model_tag>.yaml` under `storage.manifests_path` (`/var/lib/turbohaul/manifests` in `docker/turbohaul.default.yaml`). The manifest is what makes a stored model file runnable: it names the blob, sizes the model for admission (§6) and carries the engine flags.

| Group | Fields |
|---|---|
| Identity and sizing | `model_tag`, `display_name`, `description`, `prompt_template`, `gguf_blob_sha256`, `gguf_size_bytes`, `context_size`, `expected_vram_bytes` |
| Hybrid and KV sizing | `arch`, `hybrid_kv_ratio`, `kv_bytes_per_token` |
| Kind and visibility | `kind` (`model` or `plugin`), `hidden` |
| Companions and placement | `mmproj_blob_sha256`, `spec_draft_gguf_blob_sha256`, `auto_place` |
| Engine flags | `llama_server_flags`, a closed allowlist (`SAFE_LLAMA_FLAGS` in `manifest.py`, §5.1) |
| Lifecycle | `revision` (the ETag) |

What to know at this level:

- **Closed schemas.** `kind` selects `ModelManifest` or `PluginManifest`, both closed (`extra='forbid'`) on one base. A plugin manifest has no `llama_server_flags` or blob hash because those fields do not exist on it (§9.2).
- **Tag and blob.** `model_tag` matches `^[a-z0-9][a-z0-9._-]{0,63}$` (filename-safe by construction, §4.4). `gguf_blob_sha256` is required, 64 hex characters, and is the first field of the engine fingerprint (§4.4). Companion models are named by content hash, never by path or URL; the manager injects the argv itself (§5.1).
- **Sizing.** `expected_vram_bytes` (default 0) feeds the VRAM admission gate (§6); at 0 the gate uses the minimum-free-VRAM floor plus the context-derived KV-fit estimator. `arch`, `hybrid_kv_ratio` and `kv_bytes_per_token` steer the KV estimate for hybrid models (`_attn_kv_dims_for` in `manager.py`, §6).
- **Engine flags.** `cache_type_k` and `cache_type_v` accept the 12 names in `KV_CACHE_TYPES` (`safety.py`); this is where TurboQuant is dialed in per model (§5.4). Unset `ctx_checkpoints`, `cache_type_k` and `cache_type_v` take the `MANIFEST_FLAG_DEFAULTS` values (2, `turbo3`, `turbo3`), injected at load and never written back.
- **Visibility.** `hidden` (default false) omits a tag from discovery listings while it stays addressable by exact tag.
- **Lifecycle.** `revision` increments on every write and backs the ETag and `If-Match` contract (HTTP 412 on mismatch). Writes are atomic (temp file, `fsync`, rename). Edits arrive through the Models tab or `PUT /api/manifests/{tag}` and hot-reload on the next slot stage.

**To add and configure a model, on the backend and in the dashboard:** [docs/MODELS_AND_MANIFESTS.md](docs/MODELS_AND_MANIFESTS.md). Every field and flag: [docs/MODEL_CONFIG_REFERENCE.md](docs/MODEL_CONFIG_REFERENCE.md).

### 9.2 Resource plugins (Media Hook)

**Status: work in progress.** Present today:

- a `PluginManifest` variant (`manifest.py`);
- a boot-only endpoint registry (`config.PluginsConfig`);
- a runtime knob section (`config.PluginRuntimeConfig`);
- the routes `GET /api/plugins` and `POST /api/plugins/{model_tag}/invoke` (`api/plugins.py`);
- a WebSocket exec forwarder at `/api/plugins/{model_tag}/exec` (`api/exec_ws.py`);
- the invocation core and egress guard (`plugin_invoke.py`);
- a registry health probe (`plugin_health.py`);
- the Plugins tab.

Guides: [MEDIA_HOOK](docs/MEDIA_HOOK.md), [PLUGINS_SETUP](docs/PLUGINS_SETUP.md), [PLUGIN_REFERENCE](docs/PLUGIN_REFERENCE.md).

**Purpose.** An agent asks for a media capability (transcription, OCR, image or video generation) as it asks for a model. The manager resolves it to the operator-declared container serving it and returns the result inline, with no storage layer or path handoff.

```
 agent ─▶ POST /api/plugins/{model_tag}/invoke  {path, payload}
            │  model_tag ─▶ PluginManifest ─▶ resource_key ─▶ plugins.registry (boot-only)
            │  resolve_endpoint: the egress guard (§8)
            ▼
          one synchronous HTTP POST ─▶ the operator-declared plugin container ─▶ JSON body returned inline

 agent ─▶ WS /api/plugins/{model_tag}/exec ─▶ frames relayed to the container's /ws/exec
                                              (engine/ffshim is the other end, installed as ffmpeg and ffprobe)
```

**Direct call, not a queue lane.** A resource request does not traverse the queue (§3.1), Fast Lane ranking or the VRAM fit gate (§6). `invoke` resolves the tag to a `PluginManifest`, its `resource_key` against the boot registry, and issues one synchronous HTTP POST to that container, returning the JSON body inline. Declared but not enforced: `lane` (`cpu` or `gpu`) appears on the listing but selects no admission path, and `plugin_runtime.enabled` and `max_concurrent` are editable but never read. The only runtime knob the invoke path honors is `plugin_runtime.no_progress_timeout_s`.

**Addressing.** A plugin is addressed by `model_tag`, resolved server-side. The invoke body (`InvokeRequest`) is `{path, payload}`; the caller never names a host, port or URL. `plugin_invoke._validate_path` requires a leading `/`, no `..` and printable characters. The manifest's `provides_routes` is a discovery declaration, not an allowlist: the invoke route does not check `path` against it. The listing's `configured` flag means a registry entry exists for the `resource_key` and its address passes the egress policy. It never opens a socket; reachability is established by connecting at call time.

**No URLs in manifests.** Manifests are closed schemas, so a plugin manifest cannot carry an `endpoint_url`, `container_path` or `binary_path`. It names its container by an operator-declared `resource_key` into `PluginsConfig.registry`, the only place a host and port may live; this is the same by-reference pattern as `mmproj_blob_sha256` (§9.1). A `PluginEndpoint` holds a bare hostname or IP literal (URLs and shorthand spellings such as `127.1` are rejected), a port, a `health_path` (default `/health`) and an optional `auth_token_file`, a bearer-token file checked readable and non-empty at boot and read on each call. The registry is a boot section, so `PUT /api/config` cannot change it.

**Egress guard.** The manifest guards of §5.1 cover engine flags, so `resolve_endpoint` guards invoke and exec. An unknown `resource_key` fails as `unknown_resource`. An IP-literal host in a never-valid range (`0.0.0.0/8`, loopback, multicast, reserved, link-local, NAT64, IPv6 documentation; IPv4-mapped IPv6 is unwrapped and rechecked) fails as `blocked_target`. RFC 1918 and CGNAT ranges are allowed, since plugin containers live there. Hostnames are not resolved at this step. HTTP mapping: `unknown_resource` and `blocked_target` 400, `unreachable` and `plugin_error` 502, `no_progress` 504. Error details omit the registry host and port (server log only). See §8 for the security posture.

**Exec WebSocket and `ffshim`.** `WS /api/plugins/{model_tag}/exec` resolves the tag as `invoke` does, closes the upgrade before accepting on any resolution failure, then relays frames bidirectionally to the container's `/ws/exec`. It is a stateless forwarder with no session state or queue, so the admission notes above apply to it too. If the manifest declares `provides_executables`, a start frame naming another binary is refused at this hop; a lost target connection becomes an error frame with code 101. `engine/ffshim/` is the other end: one generic shim installed in the engine image as `ffmpeg` and `ffprobe` (`Dockerfile`, `Dockerfile.cuda-multi`). The image ships without ffmpeg; on an executable call the shim runs a real binary found on `PATH` if present, else forwards the call over this WebSocket.

**Health and timeouts.** `PluginHealthMonitor` probes each registry entry's `health_path` at boot and periodically (60 s, backing off to 600 s); results stay in-process and never feed `configured`. A resource job has no duration cap. `plugin_runtime.no_progress_timeout_s` (default 600 s, range 0 to 86400, runtime-adjustable) becomes the httpx read and write timeout, a per-operation inactivity limit; a connected job that produces nothing fails as `no_progress` (HTTP 504). The connect timeout is a fixed 10 s.

## 10. Observability

Six observe-only planes; none gates behavior.

```
 manager ─┬─▶ /status snapshot, 1 Hz ───────────────▶ the dashboard
          ├─▶ live monitor: metrics + text planes ──▶ /status generation, SSE live output
          ├─▶ event bus ────────────────────────────▶ /ws/state
          ├─▶ audit log (state.sqlite) ─────────────▶ GET /v1/logging
          ├─▶ flap telemetry (ring + rotating JSONL) ▶ GET /v1/telemetry/events, /v1/telemetry/status
          └─▶ spec-downgrade ring ──────────────────▶ /status, dashboard
```

**`/status`** (`status_snapshot()` in `manager.py`, await-free and lock-free) is the front end's 1 Hz source. It carries:

- `queue` depths and waiting requests;
- `active`, `loading`, `grace` and `idle_hot` blocks (`grace` is suppressed while a serve is active; `grace` and `idle_hot` resolve from the representative resident);
- `evictions`; `kv_classifier` counters; `request_identity` (the last record); `load_verify` (last 20 of a 64-record ring); `spec_downgrade` (below); `engine_stall`; `parallel_slots`;
- `generation` (live tok/s);
- `residents` (one entry per live model: phase, port, pid, inflight, idle countdown, generation);
- `vram` (per-GPU free MiB, null when the probe is down or the monitor is disabled);
- `persist_kvcache` (SSD tier usage);
- a `fastlane` badge (enabled flag, census counts, refusal and swap counts by reason).

**Live monitor** (`live_monitor.py`; `monitor.enabled` is the kill switch, `monitor.poll_interval_s` defaults to 1.0) has two planes keyed by one non-reversible 8-hex `generation_id = blake2b(pid:spawn_seq:thread-or-slot)`, so text and metrics cannot cross-attribute.

- The *metrics* plane is one `LiveResidentsSupervisor` task holding one `ResidentSlotsPoller` per live model. Each polls its engine's `/slots` as a pure observer: one await-free read of the identity scalars, then pid and `spawn_seq` re-validation after the await (guarding against fixed-port reuse). It derives tok/s (EWMA, alpha 0.4), prefill percent, a stall alarm (no decode or prompt progress for 10 s) and a prefill-stall alarm (60 s); a rising `n_prompt_tokens_processed` counts as activity, so a long warm prefill raises no false stall. The most recently active resident's generation is mirrored into `generation`.
- The *text* plane is `LiveOutputBuffer`: per-generation ring buffers (16 KB tail, LRU of 8) fed by the streaming tee and read only by the SSE endpoint `/ui/live/output/stream`, keeping token text off the event bus.

**Event bus and WebSocket** (`/ws/state`) carry state-level events only. `EventBus.publish_nowait` strips the `REDACTED_KEYS` before fan-out: `prompt`, `response`, `context`, `stderr`, `stdout`, `messages`, `thread_id`, `ip`, `client_meta`, `idle_client_meta`. Full subscriber queues drop events rather than block the worker.

**Audit log** (`state.py`, table `audit_events` in `state.sqlite`, WAL, autocommit) has one INSERT chokepoint, `record_audit_event`, behind a thread-local, sync-only connection pool (`audit_db_session`). The manager is the only emitter, with about 30 event types (submit, slot transitions, teardowns, safety-gate refusals, evictions, boot reconcile and others).

The read side is `GET /v1/logging`: keyset pagination (`since`, `next_since`), `slot_id` and `event_type` filters, `limit` (default 200, maximum 500), a page budget of 80,000 characters less envelope overhead, with an oversized-row escape hatch, and recursive redaction with the same `REDACTED_KEYS` as a tripwire. The load-bearing protection is emitter discipline: no prompts, responses or personal data in payloads.

**Flap telemetry** (`telemetry.py`) is an observe-only degradation pipeline: lifecycle hooks (request arrival, queue state, slot assign, prefill start, first token with TTFT, disconnects, completion, VRAM samples) append to a 10,000-entry in-memory ring and a rotating JSONL set (10 MiB per file, 5 files) under `DEFAULT_LOG_DIR` (`/var/lib/turbohaul/telemetry`), and every call site is fail-open. Read it through `GET /v1/telemetry/events` and `GET /v1/telemetry/status`; see [docs/telemetry_reader_guide.md](docs/telemetry_reader_guide.md).

**Spec-downgrade surfacing** (`spec_downgrade_log.py`). Speculative decoding exists for only some architectures; on the rest the engine downgrades cleanly, serving the target model without acceleration. This plane makes a silent downgrade visible. It scans both engine log sinks (`.log`, `.stdio`) for the tag `SPEC_DOWNGRADED`, emits one `SPEC_DOWNGRADE_RECORD {json}` line per event, and keeps a 64-record ring whose last 20 appear in `/status` under `spec_downgrade` and on the Dashboard. Two sources write the channel: the engine log, and the manager's own config check (a `spec_draft_gguf_blob_sha256` on a manifest whose `spec_type` takes no standalone draft); each record's `source` names which.

## 11. Front-end

A Vite + React 18 + TypeScript + Tailwind single-page app with five top-level tabs: **Dashboard**; **Queue** (Queue, Fast Lane); **Blob** (Blob, Models); **Plugins** (work in progress); **Settings** (General, Config, Schema, Logs). Legacy `/models`, `/config`, `/schema`, `/logs` paths redirect. Guide: [docs/frontend/README.md](docs/frontend/README.md).

```
┌────────────────┬───────────────────┬───────────────┬────────────────────┬──────────────────────────────────┐
│ Dashboard      │ Queue             │ Blob          │ Plugins            │ Settings                         │
│ resident cards │ Queue · Fast Lane │ Blob · Models │ (work in progress) │ General · Config · Schema · Logs │
└────────────────┴───────────────────┴───────────────┴────────────────────┴──────────────────────────────────┘
```

- **Build:** `npm run build` emits `src/frontend/dist`; `Dockerfile` builds it in a Node stage, `Dockerfile.engine-src` copies a pre-built copy.
- **Serving:** `create_app` (`api/main.py`) serves it single-port at `/ui` when `ui.enabled` is set and `ui.static_path` exists, with SPA fallback, a path-traversal guard, a Content-Security-Policy (`default-src 'self'`), `X-Frame-Options: DENY` and immutable caching for hashed asset types (`no-cache` for everything else).
- **Dependencies and rendering:** no CDN, chart library or code editor (sparklines are hand-rolled SVG); no `dangerouslySetInnerHTML`.

**Live state.** `useStatus` polls `/status` at 1 Hz; each `/ws/state` event triggers an immediate fetch. `useLiveStream` opens one SSE stream per resident; each card's output pane auto-follows only near the bottom.

**Resident card** (one per resident, by model tag):

```
┌─ resident card ─────────────────────────────────────────────────────────┐
│ badges     state · in-flight count · engine_op pill · one alarm badge   │
│ progress   prefill_pct during prefill, then decode progress             │
│ countdown  "grace" or "unload in"                                       │
│ identity   current and last request strips (ip, model, role, session)   │
│ checks     LOAD_VERIFY widget · tok/s and VRAM bars                     │
│ output     live generation text, one SSE stream per resident            │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Badges:** state (the backend `phase`, else `state`), in-flight count, the `engine_op` pill and one alarm badge. `residentAlarm` (`alarms.ts`) ranks STALLED, NO TELEMETRY, PREFILL HANG, BUSY. BUSY is suppressed during grace and escalates to NO TELEMETRY after 120 s (`BUSY_ESCALATE_S`); PREFILL HANG follows the backend's 60 s alarm (`PREFILL_STALL_AFTER_S`).
- **Progress:** during prefill, `prefill_pct`, computed in `live_monitor.py` as (restored-from-KV + processed tokens) over the prompt size recorded at admission; then decode progress against `max_tokens` when set.
- **Countdown:** one timer, "grace" (`phase`, `remaining_s` from `_resident_phase` in `manager.py`) or "unload in" (`idle_expires_in_s`).
- **Identity:** current- and last-request strips (ip, model, role, session); role priority curator, compression, sub-agent, main.
- **Verification:** the LOAD_VERIFY widget colours each model-load and KV-restore record, showing `n_past` actual/expected.
- **Placeholders:** tok/s shows "—" and VRAM bars a placeholder until data exists.

**Settings:**

- **General** sets the SSD KV cap `persist.max_bytes` in GiB via `PUT /api/config` (immediate, persisted in `runtime_config.yaml`) beside live usage and headroom.
- **Models** edits per-model `llama_server_flags` by category (including KV Cache: `cache_type_k/v`, checkpoint knobs) through ETag/`If-Match` manifest `PUT`s.
- **Config** edits the runtime sections; the boot sections (`server`, `storage`, `runtime`, `ui` and `plugins`) are read-only.
- **Logs** is a paginated audit feed with a redaction banner.
- **Plugins** (work in progress) lists plugin manifests with enable toggles.

## 12. Storage & persistence

All state lives under `/var/lib/turbohaul`:

```
/var/lib/turbohaul/
├── blobs/sha256/        # content-addressed GGUFs: incoming/*.tmp → <ab>/<hash> (0o400)
├── manifests/           # per-tag model YAML (ETag'd, atomic writes)
├── import-staging/      # sandboxed /api/import root
├── state.sqlite         # slots, audit_events, pull_history (WAL)
├── kvcache/             # KV RAM tier, expected tmpfs (§4.5)
├── kvcache_persist/     # KV SSD tier, written on ownership transfer (§4.5)
├── kv_quarantine/       # 3-strikes poisoned-bin evidence (§4.7)
├── engine_logs/         # per-spawn llama-server logs
└── telemetry/           # flap-telemetry rotating JSONL
```

**Boot-time mount assertion (fail loud, keep serving).** A missing mount fails silently: a RAM tier that is not tmpfs writes to the backing store, and a cold tier on tmpfs evaporates on restart. The manager asserts both, with opposite tests. A missing directory or unreadable mountinfo is reported first.

- `kv.kv_save_expected_fstype` (default `tmpfs`) requires `kvcache/` to be its own mount of that type; `null` skips only that assertion.
- `kv.kv_persist_forbidden_fstypes` (default `[tmpfs, ramfs]`) forbids volatile filesystems for `kvcache_persist/`, which may be a plain directory.
- Both are config fields, not env vars: a persisted runtime override outranks the shipped file and code default.
- A failure is logged as `KV_MOUNT_CHECK` and recorded in the `boot_reconcile` audit event; the manager keeps serving.

**Blobs.** A stream is written to `incoming/` under `pull.per_stream_max_bytes` (default 100 GiB), hashed while writing and fsynced, checked against the expected sha256 when supplied, atomically renamed, the directory fsynced, and made read-only (0o400); a failed write or hash mismatch unlinks its temp file. A manifest's `revision` is its ETag; a stale `If-Match` gets 412.

**KV persistence.** The SSD tier keeps what ownership transfer wrote; saving to it at unload (idle teardown, shutdown) is work in progress and currently not working. A container restart empties the RAM tier; cold restore copies the matching triplet back from `kvcache_persist/` per identity key (`_hydrate_ram_from_persist_for_key`), or loads fresh when none exists. Recovery: [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md).

## 13. Licensing & vendoring

- Manager: MIT (`LICENSE`); third-party terms in `THIRD_PARTY_LICENSES.md` and `THIRD_PARTY_NOTICES.md`.
- Backend: `llama-server`, a supervised child process built from **[Tom's TurboQuant fork of llama.cpp](https://github.com/TheTom/llama-cpp-turboquant) (MIT)**. Ollama compatibility is API shape only; no Ollama source is vendored.
- Vendored: the engine source (`engine/llama-cpp-turboquant/`, MIT) and Python wheels (`vendor/pywheels`, installed offline by `Dockerfile.engine-src`).

## Modules

`modules.toml` at the repository root registers the modules (`engine_launch`, `engine_budget`, `fastlane_client_names`) with folder, ownership, dependencies, public names and card path. Each module folder holds an `AGENTS.md` card (purpose, ownership, public interface, dependencies, invariants, test locations, gotchas) and a `README.md`.

`tools/sop_check.py` checks registered paths and sub-folders, card sections, card agreement with registry and code (`tools/doc_check.py`), import contracts (declared dependencies only; entry through `__init__.py`), line caps (300 soft, 500 hard), the `ARCHITECTURE.md` word budget, and that generated `MODULE_MAP.md` and `.sop/function_index.json` are current; `--write` regenerates them.
