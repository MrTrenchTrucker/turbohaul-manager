# Turbohaul-Manager — Architecture (As-Built)

**Version:** v0.6.0 (as-built).
**Scope:** this document describes the system **as it is built and deployed** — every mechanism below is grounded in code at the snapshot commit (receipts as `file:line`). Design history, phase plans, and superseded concepts live in §14 and in `docs/RETRO_*.md` + `CHANGELOG.md`, not here.
GitHub (`https://github.com/MrTrenchTrucker/turbohaul-manager`) is the public home.
**Quick start / per-client setup:** see [README.md](README.md) and [docs/AI_AGENT_SETUP.md](docs/AI_AGENT_SETUP.md) — not duplicated here.

---

## 1. What Turbohaul-Manager is

**Turbohaul-Manager is designed to serve an entire fleet of AI agents from one inference host — multiple agents, their sub-agents, and all of their requests at the same time.** Every incoming request lands in a two-tier queue (FIFO staging + a 10,000-deep acceptance buffer) with model-affinity scheduling, grace-window fast-pathing, and idle-hot model retention (§3), and the manager serves the load in whichever **residency mode** the hardware allows:

- **Single series** — one model resident, one context at a time. Requests serialize through the queue with warm fast-paths; the proven single-GPU configuration.
- **Series parallel** — one model resident, **multiple context windows served simultaneously** on the same engine: the sidecar's `parallel` slots let same-model requests fan out onto one running model and fully drain before any swap (§3.3, §5.1).
- **Double parallel** — **multiple models resident at once** (`max_parallel_sidecars`, up to 32; §3.6), each admitted under a VRAM budget with LRU eviction — and each resident can itself run series-parallel.

The mode is pure configuration, so one codebase **scales with the hardware and the user's needs**: a single 24 GB card time-slicing a whole agent fleet, up through multi-model hosts running several engines with parallel context windows each (see [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md)).

### 1.1 The three residency modes at a glance

**Single series** — one model, one context at a time; the queue serializes everything, warm fast-paths keep it quick:

```
 agents / clients                 TURBOHAUL MANAGER                            GPU
┌───────────┐
│ main agent│──┐    ┌────────────────────────────────┐    ┌───────────────────────────┐
└───────────┘  │    │  acceptance buffer (10,000)    │    │  llama-server :11500      │
┌───────────┐  ├──▶ │  ───────────────────────────   │    │  ┌─────────────────────┐  │
│ sub-agents│──┤    │  staging queue (FIFO, 100)     │──▶ │  │ slot 0 — ONE active │  │
└───────────┘  │    │   • model-affinity pop         │    │  │ context window      │  │
┌───────────┐  │    │   • grace-window fast-path     │    │  └─────────────────────┘  │
│ any app   │──┘    │   • idle-hot retention         │    │      Model A resident     │
└───────────┘       └────────────────────────────────┘    └───────────────────────────┘
                     one request decodes at a time; on model swap, the outgoing
                     session's KV is saved and restored when it returns (§4)
```

**Series parallel** — one model, **multiple context windows at the same time**: same-model requests fan out onto the one running engine (its `parallel` slots) and fully drain before any swap:

```
 agents / clients                 TURBOHAUL MANAGER                            GPU
┌───────────┐                                             ┌───────────────────────────┐
│ main agent│──┐    ┌────────────────────────────────┐    │  llama-server :11500      │
└───────────┘  │    │  queue (as above)              │    │  spawned --parallel N     │
┌───────────┐  ├──▶ │                                │──▶ │  ┌────────┐ ┌────────┐    │
│ sub-agent │──┤    │  same-model riders FAN OUT     │    │  │ slot 0 │ │ slot 1 │    │
└───────────┘  │    │  onto the ONE running engine   │    │  │ ctx A  │ │ ctx B  │    │
┌───────────┐  │    │  (up to N concurrent, fully    │    │  └────────┘ └────────┘    │
│ sub-agent │──┘    │  drained before any swap)      │    │  ┌────────┐               │
└───────────┘       └────────────────────────────────┘    │  │ slot 2 │   Model A     │
                                                          │  │ ctx C  │   resident    │
                                                          │  └────────┘               │
                                                          └───────────────────────────┘
```

**Double parallel** — **multiple models resident simultaneously** (`max_parallel_sidecars` ≥ 2), each admitted under a VRAM budget and each optionally series-parallel itself:

```
 agents / clients             TURBOHAUL MANAGER                             GPU(s)
┌───────────┐       ┌─────────────────────────────────┐    ┌───────────────────────────┐
│ agent 1   │──┐    │  queue + resident dispatcher    │──▶ │ llama-server :11500       │
└───────────┘  │    │  (max_parallel_sidecars ≤ 32)   │    │ Model A   [slot 0|slot 1] │
┌───────────┐  ├──▶ │                                 │    ├───────────────────────────┤
│ agent 2   │──┤    │   • VRAM-budget admission       │──▶ │ llama-server :11501       │
└───────────┘  │    │   • per-model Resident FSM      │    │ Model B   [slot 0]        │
┌───────────┐  │    │   • LRU idle eviction           │    ├───────────────────────────┤
│ advisors  │──┘    │   • per-resident grace/idle     │──▶ │ llama-server :11502       │
└───────────┘       └─────────────────────────────────┘    │ Model C   [slot 0|slot 1] │
                     each resident runs its own            └───────────────────────────┘
                     serve/grace/idle cycle (§3.6)
```

In every mode, the KV-cache tiers (§4.5) sit underneath: VRAM (native reuse) → system RAM (swap-seam saves) → SSD (post-unload persistence) — so scaling the residency mode never changes the KV contract.

Concretely, it is a **standalone HTTP inference manager** that:

- Presents an **Ollama-compatible + OpenAI-compatible API surface** (`/api/chat`, `/v1/chat/completions`, `/v1/embeddings`, tags/show, pulls/import) so Ollama- or OpenAI-aware clients can point at it transparently (full as-built route table in §7).
- Supervises **`llama-server` sidecars** built from **[Tom's TurboQuant fork of llama.cpp](https://github.com/TheTom/llama-cpp-turboquant)** (vendored in-repo at `engine/llama-cpp-turboquant/`, pinned by `engine.lock`) — one subprocess per resident model, spawned/health-checked/torn down by the manager.
- **Differentiator — KV-cache orchestration (§4):** the manager understands *who* each request is (main agent / sub-agent / curator / compression, per session) and *what that means for the precomputed KV cache*: the main agent's KV state is saved at model-swap seams to system RAM, persisted to disk after full unload, and restored on wave-return so follow-up turns re-use up to the full precomputed prefix instead of re-prefilling; disposable roles' KV is isolated per sub-agent and thrown away by contract — concurrent agents can no longer muddy each other's context.
- Ships **self-contained and offline-usable**: vendored engine source, vendored wheels, and a committed frontend; builds fully offline via `Dockerfile.engine-src` (README "Self-contained offline build").

Trust model in one line: single-tenant, network-perimeter security (bind address is the boundary; no app-layer auth in v1) — details in §8.

## 2. System map & repo layout

```
src/turbohaul/                  — the manager (Python, FastAPI + asyncio)
  manager.py                    — TurbohaulManager: queue worker, slot lifecycle, KV orchestration (9,537 lines)
  queue.py                      — TurbohaulQueue (two-tier FIFO + affinity), GraceTimer, IdleHotTimer
  slot.py                       — Slot dataclass, SlotState enum, thread-id prefix-hash derivation
  fsm.py                        — pure slot-state transition table + validators
  kv_policy.py                  — resolve_kv() decision chokepoint, prefix-hash chains, bin/meta filenames
  kv_classify.py                — role/label → class resolution, POLICIES registry, event taxonomy
  safety.py                     — pre-spawn admission gates (RAM/VRAM/load/iowait/KV-fit)
  singleton.py                  — state-lock flock, orphan reapers, foreign-GPU-app detection
  subprocess_mgr.py             — sidecar spawn/supervision, KV tier dir constants
  load_verify_log.py            — LOAD_VERIFY proof emitter + verify helpers + /status ring
  live_monitor.py               — live inference pollers (tok/s, prefill %, VRAM, residents)
  telemetry.py                  — flap-telemetry JSONL pipeline (rotating) + ring buffer
  config.py                     — TurbohaulConfig loader, env-override map, Boot/Runtime split (§9)
  __main__.py                   — uvicorn entry point (config path, public-bind opt-in, log filters)
  state.py                      — state.sqlite persistence + audit-event pool (NOT a state machine)
  manifest.py                   — per-model YAML manifests, tag validation, flag allowlist, ETag writes
  blob_store.py                 — content-addressed GGUF blob store (atomic staged writes, integrity GC)
  ssrf_guard.py                 — pull-URL validation (scheme/hosts/IP ranges/rebinding)
  api/                          — FastAPI routes: chat_completion, ollama, manifests, config_put, pull,
                                  import_, embeddings, logging, telemetry, live_stream, ws_state,
                                  tool_call_recovery, body_size_limit, main (app factory + lifespan)
src/frontend/                   — React + Vite + TypeScript FE (served from the same port, §11)
engine/llama-cpp-turboquant/    — vendored engine source snapshot @ 86771a58d (see engine.lock, §13)
docker/, Dockerfile*            — default config yaml + deployment images (slim / cuda / engine-src variants, §5.4;
                                  plus historical overlay Dockerfiles); persistence doctrine in §12
docs/                           — operator guides + retrospectives (linked throughout)
```

The manager is **one Python process** (uvicorn, port **11401**); each loaded model is a separate supervised `llama-server` process on ports from `default_port_base` (**11500+**). The FE is static-built and mounted by the same FastAPI app. All durable state lives under `/var/lib/turbohaul` (§12).

## 3. Request lifecycle & slot state machine

### 3.1 Admission

HTTP routes hand every completion request to the manager through two entry points: `submit_and_wait` (non-streaming; adds a completion-cache + single-flight layer) and `submit_for_streaming` (returns the slot immediately; the route later attaches to the sidecar stream) — both wrap **`submit()`** (`manager.py:1706`), the single admission chokepoint:

1. Refuses work after shutdown (`QueueClosed`).
2. Auto-derives a missing `thread_id` via the prefix-hash ladder (§4.1).
3. Creates the `Slot` (state `RECEIVED`) carrying `admission_ctx_len` and the admission prefix-hash chain used by the KV restore gate.
4. Stamps **`slot.admission_role = _bin_role(client_meta)`** before enqueue — the admission-time role record that survives the warm-inherit `client_meta` replacement (§4.7).
5. **Grace shortcut:** if the live grace window matches `(thread_id, model_tag)`, the slot enqueues at the **head** of the queue and the window re-arms (extension-capped); otherwise normal FIFO enqueue.
6. Persists the slot row (state.sqlite) and emits audit + telemetry events. (The per-request identity log line of §4.2 is emitted by the two HTTP-facing wrappers — `submit_and_wait` / `submit_for_streaming` — not by `submit()` itself.)

**Queue** (`queue.py:23` `TurbohaulQueue`): two deques under one lock — a **staging queue** (default cap 100) and an **acceptance buffer** (default cap 10,000). Overflow raises `QueueFull` (surfaces to the client as HTTP 500 via the routes' `RuntimeError` handler). Client disconnects are handled by **eviction**: a watcher sets the slot's `disconnect_event`; evicted slots are skipped at pop (bounded drain) and failed with a dedicated `SlotEvictedError` → HTTP 499.

**Model-affinity pop** (`queue.py:150` `pop_next`): when a warm model is resident, same-model staged work is preferred over the FIFO head — bounded by `max_other_model_wait_s` (starvation override, default 20 s) and `max_consecutive_same_model` (default 3; the count cap only forces the head when the head is same-model anyway, so a swap is never forced avoidably).

### 3.2 Slot state machine (as-built)

`SlotState` has 12 members; the legal transition table lives in `fsm.py:14` (`LEGAL_TRANSITIONS`) and every state change goes through validated `transition()`:

```
RECEIVED ─→ ACCEPT_BUFFER ─→ STAGED ─→ LOADING ─→ ACTIVE ─→ GRACE ─→ POPPED ─→ COLD
                │                │          │                  │        │
                └─→ STAGED       │          └─→ LOADING_FAIL   │        └─→ (idle-hot holder, manager-level)
                                 └─→ ACTIVE_MATCH (fast-track) │
                                                               └─→ (matched follow-up slot: STAGED → ACTIVE_MATCH → ACTIVE
                                                                    on the warm sidecar, while the anchor sits in GRACE)
```

Two table states are **legal but never entered by runtime code** (kept for FSM compatibility):

- **`GRACE_BUSY`** — exists in the enum, transition table, tests, and FE typings only. The behavior the original design called GRACE-BUSY is implemented as the **ACTIVE_MATCH promotion loop** instead (§3.4).
- **slot-level `IDLE_HOT`** — idle-hot retention is implemented as *manager-level holder scalars* (`_idle_handle`/`_idle_model_tag`/`_idle_expires_at`/`_idle_client_meta`), not a slot state; the slot's DB row ends as `grace-expired-held-idle`.

At `max_parallel_sidecars >= 2` a separate **Resident FSM** (5 states: `RESERVED_LOADING/ACTIVE/GRACE/IDLE_EVICTABLE/DEAD`) governs per-model residents (§3.6).

### 3.3 The worker loop

**`worker_loop`** (`manager.py:2252`) is the single dispatcher at the deployed cap of 1 (`max_parallel_sidecars=1`): *pop → spawn-or-inherit → serve → grace → idle*. Its idle tick doubles as a health reflex:

- **Proactive dead-idle sweep:** if the idle holder's engine process is dead, the death is attributed to the bin that port last restored (a "strike", §4.7), and the holder is torn down in seconds — a model that dies *between* requests is detected without any traffic.
- **Idle expiry:** an identity-guarded debounce tears down the holder when the idle window lapses.

**`_process_slot`** (`manager.py:3852`) drives one slot:

1. **Warm-inherit fast path:** if the idle holder matches the slot's model, is inside its window, and its process `is_alive()` (dead-handle recovery gate), the slot inherits the running sidecar — skipping spawn and health-wait entirely. **T3 identity-preservation rule** (`manager.py:3976-3998`): the holder's stashed `client_meta` is restored onto the inheriting slot **only** when the incoming request has no `session_id` or the same one; a *different* session keeps its own incoming `client_meta` — no cross-session identity clobber (this was the root fix for "main returns after a sub-agent wave and can't find its own bin").
2. **Cold/spawn path:** a different-model idle holder is first torn down (**this teardown is the model-swap seam where main's KV is saved — §4.5**); then the pre-spawn **safety gates** run (§6), the sidecar is spawned (§5), health is awaited, a **LOAD_VERIFY** record is emitted (`manager.py:4162-4188`, §4.8), and a best-effort KV restore is attempted (§4.6).
3. **Serve:** streaming requests hand the sidecar handle to the route and block until the stream closes (the slot stays `ACTIVE` for the whole stream — nothing else can promote onto the sidecar mid-stream); non-streaming requests complete inline and feed the completion cache. With `--parallel > 1` on the sidecar, same-model riders are fanned out and fully drained before any swap.

### 3.4 Grace window & ACTIVE_MATCH follow-ups

After a serve the slot enters **GRACE** (`grace_seconds`, deployed 60 s) owned by `(thread_id, model_tag)`. The grace loop atomically pops a matching queued follow-up and promotes it: `STAGED → ACTIVE_MATCH → ACTIVE` on the *same warm sidecar* — no reload, no re-prefill beyond the KV delta. Each promotion re-arms the window and increments the extension count up to `max_grace_extensions` (deployed 50). The anchor slot remains the grace driver; matched slots end (`POPPED`) after their serve.

### 3.5 Idle-hot retention & keep_alive

When grace expires, the **latest** request's `keep_alive` decides retention: `None` → `idle_hot_load_seconds` default (deployed 1800 s); negative ("pin") → capped at `KEEP_ALIVE_MAX_S = 1800` (never indefinite on a shared GPU); explicit values are honored up to the cap. Even at `keep_alive: 0`, if the queue head is the same model the sidecar is held for 30 s (`_SAME_MODEL_QUEUED_HOLD_S`) so queued same-model work warm-inherits instead of paying teardown + respawn. The idle window ends by **model swap** (different-model arrival), **expiry**, or the **dead-idle sweep** — all funnel into `_teardown_idle_holder` (`manager.py:4938`), whose live-holder sequence performs the KV unload-seam flush + SSD persist (§4.5) before the drained SIGTERM.

### 3.6 Multi-slot mode (cap ≥ 2)

`max_parallel_sidecars` (1..32; **deployed 1**) ≥ 2 replaces the worker body with **`_dispatch_loop`** (`manager.py:2482`): slots are routed to a live same-model **Resident** or a new resident is reserved under a VRAM budget (`_route_or_reserve` `manager.py:2633`, LRU idle eviction), each resident owned by a long-lived `_drive_resident` task (`manager.py:3141`) running its own serve/grace/idle cycle. The single-GPU deployment documented throughout this file runs the cap-1 path.

## 4. KV-cache orchestration (the differentiator)

The product's core promise: **one precomputed KV copy per (session, role), reused whenever physics allows.** The main agent's context is never silently recomputed: between tool calls it rides VRAM natively; across model swaps it is saved to system RAM at the unload seam; after a full unload it survives on SSD; on wave-return it is restored and only the new suffix is prefilled. Disposable roles (sub-agents, curator, compression) never pollute or evict it.

### 4.1 Identity derivation ladder (thread_id)

Every request resolves to a `thread_id` by a three-step ladder (`chat_completion.py:604-631`, mirrored on `/api/chat`):

1. **Explicit** `thread_id` in the payload (the reference agent integration sends this).
2. **IP + first-message fingerprint** for naive clients at single-residency: `agent-ip-<ip>-auto-<hash>` (bare `agent-ip-<ip>` when no usable first message exists) — distinguishes different callers behind the same deployment while keeping one caller's conversation together.
3. **Manager fallback** (`slot.py:139` `derive_thread_id_prefix_hash`): `auto- + sha256(model_tag ⊕ first-256-words)[:24]` — conversation *extensions* (same prefix, more tokens) map to the **same** thread_id, which is what makes grace matching fire for clients that send no identity at all. Width: `prefix_token_count` (default 256).

**How the caller's IP is used.** The manager captures the source IP of every request at the API route and carries it in the request's identity record — it is how the manager knows *which client sent what* when nothing else identifies the caller. Three roles: **(1) identity anchor** — for tag-less clients at single-residency the IP anchors the derived thread identity (rung 2 above), so two different machines never blur into one conversation while one machine's follow-ups stay together; **(2) observability** — the IP rides the per-request `R2B_REQ_IDENTITY` log line, the `/status` `request_identity` block, and the FE residents-box identity strip (`ip · model · role · session`), so an operator can see at a glance who is driving the GPU; **(3) explicitly not auth** — inside the network-perimeter trust model (§8) the IP is a routing/observability hint, never an authentication mechanism, and it is never broadcast on the redacted `/ws/state` event bus. User-facing treatment with request examples: [docs/TAGS_AND_IDENTITY.md](docs/TAGS_AND_IDENTITY.md).

An additional role+session **identity recompose** layer (append-only hashed suffix `-s=<sha12>-r=<sha8>`) is computed per request and, with `TURBOHAUL_M2B_ACTIVE` on (**enabled in the reference deployment**), drives the slot identity; a session-recovery rule re-attaches the idle resident's `session_id` to tool-call re-cues that arrive session-less after grace expiry.

### 4.2 Role tags, classification, and the save_kv contract

> User-facing guide with examples, the per-role behavior table, and the no-tags fallback ladder: [docs/TAGS_AND_IDENTITY.md](docs/TAGS_AND_IDENTITY.md).

**The client_meta contract.** Identity enters through one shared parser, `_derive_client_meta_identity` (`chat_completion.py:386-462`), used byte-identically at all 3 build sites (OpenAI non-stream / OpenAI stream / `/api/chat`). Dual-source fields read the **top-level payload first, falling back to the nested `payload["client_meta"]`**: `role`, `session_id`, `is_main`, `is_sub_agent`, `is_curator`, `is_compression`, `save_kv`; `turn0_meta` and `context_size` are top-level-or-derived (first-message metadata / `compute_ctx_len`), and the source `ip` is captured at the route. **None of these keys are in the knob-forwarding allowlists, so identity can never leak into the llama-server payload.**

**Role resolution.** `kv_classify._class_from_label` (`kv_classify.py:303`) resolves labels with a fixed priority ladder — `is_curator > is_compression > is_sub_agent > is_main`, then a literal `role` string — because harness labels are historically not mutually exclusive (a curator can carry both `is_sub_agent` and `is_curator`). The bin-keying wrapper `_bin_role` (`manager.py:126`) returns `None` for unlabeled traffic (which then keys by raw thread_id and can never collide with a session's main bin) and never *infers* main by default. One honest caveat: "main" is intended to be reachable only via an explicit `is_main` label, and `_bin_role`'s own literal-role branch does block a bare `role:"main"` — but the upstream `_class_from_label` fallback honors any literal role naming a registry class, *including* `"main"`, and it runs first; so a request carrying `client_meta={"role":"main"}` with no flags does key the main bin as-built. Trusted fleet clients always send the `is_*` flags, but the documented only-via-`is_main` guarantee is not mechanically enforced.

**Bin identity.** `_bin_identity` (`manager.py:146`) converts `(thread_id, client_meta)` into the KV bin key:

```
no session_id or unlabeled role      → raw thread_id                    (legacy per-thread bin)
sub-agent / curator (with chain)     → sess:{session_id}:{role}:{fp8}   (fp8 = first-two-chain-entry hash:
                                                                          concurrent same-role siblings get distinct bins)
main / compression                   → sess:{session_id}:{role}
```

This string is what the save meta stamps as owner and what the restore owner-match compares — **one KV copy per (session, role)** by construction.

**Per-role saving — the `save_kv` contract** (`_role_save_enabled`, `manager.py:422`): the *sole* source is the per-request setting delivered as `client_meta["save_kv"]` (bool) — **deliberately no environment variables**, so the toggle is runtime- and per-request-switchable. Absent field → disposable roles (sub-agent, curator, compression) do **not** save; **main always saves**. A big-VRAM deployment can flip a role's toggle client-side and that role's KV starts being kept — per request, at runtime, no container rebuild.

**The R2B identity proof line.** Exactly one greppable structured log line per admitted request — `R2B_REQ_IDENTITY {ip, model_tag, session_id, is_*, resolved_class, thread_id}` (`manager.py:1429` `_emit_request_identity`) — also surfaced at `/status` as `request_identity`. This is the first thing to grep when validating harness tagging end-to-end.

**Synthesized sub-agent sessions.** The agent harness plugin mints `{parent_session_id}-sub-<nonce>` session ids for spawned sub-agents (plugin-side; exactly-one-true `is_*` flags). The manager treats them as opaque: each nonce mints a distinct `sess:{parent}-sub-{nonce}:sub-agent:{fp8}` bin, classified fresh, never cross-restored; the `hermes-sub-*` thread-name prefix additionally serves as disposable-provenance proof for the unload seam when labels are missing (§4.7).

### 4.3 The resolve_kv decision chokepoint

> **Cache matching happens at two levels** — this turn-level manager gate decides *whether* a saved bin is offered to the engine; the engine's token-level `get_common_prefix` then decides *how much* is actually reused (§4.6). Matching decides all of the speed and none of the safety: every mismatch fails safe to a fresh prefill. Full treatment — what must match, what breaks it, client rules, diagnostics: [docs/KV_CACHE_MATCHING.md](docs/KV_CACHE_MATCHING.md).

Every save and every restore decision funnels through **one function**: `kv_policy.resolve_kv(action, identity, sizes)` (`kv_policy.py:102`). It returns an immutable **`KVDecision(do_it, action, reason, resolved_from)`** — every decision is loggable with provenance, and the manager logs each one.

**Save rules** (`_resolve_save`): refuse zero-token saves (degenerate) and empty-identity saves (cross-thread collision risk); otherwise save.

**Restore rules** (`_resolve_restore`) — evaluated per candidate bin, in order:

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

The prefix gate (rule 7, the WS2 classifier) replaced the older length-compaction heuristic: validity is decided by the **prefix-hash chain** — an order-sensitive rolling hash `H_i = SHA256(H_{i-1} ⊕ role_i ⊕ content_i)` over the rendered messages (`kv_policy.py` `_prefix_hash_chain`; canonical single implementation). `compute_ctx_len` is the single source of truth for context size, so admission-time and save-time numbers stay comparable. Worst case of every skip/reject path is one fresh prefill — the design never restores on doubt. Note: the chain hashes role+content only, so tool-call turns with null content are hash-invisible — the dormant tool-tail restore-skip guard (`TURBOHAUL_TOOLTAIL_RESTORE_SKIP`, §9; default OFF since the save-side byte-match fix made it unnecessary) sits in the restore paths for exactly that seam.

### 4.4 Bin naming & per-model segregation

There are **no per-model directories** — both KV tiers are flat, and segregation is carried by the filename, minted by the single source `kv_policy.kv_save_fn` (`kv_policy.py:258`):

```
{model_tag}.p{port}.{thread_hash}.slot{sid}.bin      ← engine state
{model_tag}.p{port}.{thread_hash}.slot{sid}.json     ← meta sidecar (owner identity, sizes, hash_chain,
                                                        clean_prefix flag, engine fingerprint, model_tag)
<bin>.ckpt                                            ← engine checkpoint-ladder sidecar
```

Cross-model isolation is enforced at every lookup layer: the cold-restore listing keeps only `fn.startswith(f"{model_tag}.")` for the resident's own tag (`_restore_slot_kv_inner`, `manager.py:7830`); the warm scan cache and finders key on the exact tuple `(model_tag, thread_hash, port)` (`_find_clean_bin`, `manager.py:6719`); every save stamps `model_tag` into the meta; tier crossings (SSD persist, hydration, quarantine) move files by their tag-prefixed basenames. `validate_tag` (`manifest.py`) guarantees tags are filename-safe, and both KV inner functions refuse path-metacharacter tags outright.

A second, deeper layer handles *same tag, different model*: every save stamps a 4-field **engine fingerprint** (`gguf_sha256`, `engine_build_id`, `n_ctx`, `n_rs_seq` — `_engine_fingerprint`, `manager.py:8861`); the opt-in sweep `_purge_mismatched_bins` (`TURBOHAUL_FINGERPRINT_PURGE`, default OFF) deletes bins whose stamp cannot belong to the current engine, grouped per tag so one model's re-quant never touches another's valid bins. The fingerprint is **not** consulted by the restore decision — it is a file-hygiene sweep only.

**Operational naming rule:** never create a model tag that equals another tag plus a `.`-suffix (e.g. `mymodel` and `mymodel.moe`) — dots are legal inside tags, and the cold-path prefix filter would co-list such a pair. No deployed manifest does this; keep it that way.

### 4.5 KV save path: unload-seam-only writes, guards, tiers, GC

**When main's KV is written — at the unload seam, never per-turn.** The design principle, recorded verbatim in the code: KV is saved right before the model is unloaded, *not* at the time a sub-agent enters the queue — the precomputed KV cache stays in VRAM until then. The per-turn probe `_probe_and_save_clean_kv` (`manager.py:5387`) still runs pre-decode on the hot paths, but with `save_to_disk=False` it only maintains bookkeeping (VRAM anchor stamp + dirty-tip tracking, §4.7) and returns. The **seam writers** (`save_to_disk=True`) are:

1. `_teardown_idle_holder` (`manager.py:4938`) — model swap / idle expiry / shutdown: flush → SSD persist → belt save, all sequenced **before** the SIGTERM.
2. `_teardown` (`manager.py:4832`) — non-idle teardowns, same sequence.
3. Resident eviction teardown (cap ≥ 2).

**Clean-by-construction:** with `TURBOHAUL_COVERED_SCAFFOLD_STRIP` (default ON), the seam flush re-renders the historical transcript think-stripped (engine `/apply-template` → strip assistant `<think>` scaffold → `n_predict=0` prefill) so the saved bin byte-matches the harness's future think-stripped resend. Save floor: `_MIN_CTX_LEN = 40000` chars (~13k tokens) — small contexts are cheaper to re-prefill than to manage.

**The single save chokepoint** — `_save_slot_kv_inner` (`manager.py:6314`) is the single funnel for every anchor/clean save: all seam and probe writers route through it. (The dormant shadow A/B mechanism has its own deliberately separate writer, `_save_shadow_slot_kv` `manager.py:6182`, which writes only distinct `.shadow` bins and must never touch the clean-anchor never-demote path.) In-order guards:

- **D1 dirty-tip refusal** — port flagged dirty → save REFUSED, previous bin kept (§4.7).
- **D2 disposable refusal** — `_seam_flush_allowed` says the identity is a disposable role (or unlabeled with `hermes-sub-*`/`agent-ip-*`/`auto-*` provenance) → REFUSED.
- **Clean-present skip / NEVER-DEMOTE** — a save never overwrites a longer clean bin and never demotes `clean_prefix` True→False.
- **resolve_kv("save", …)** policy decision (§4.3).
- **T-GUARD** — ENOSPC pre-flight (expected size self-calibrated from the thread's previous meta; skip if tmpfs headroom < 1.05×) and post-save truncation reject (< 0.5× expected → delete tmp, keep previous bin).
- **PAIR-GUARD** — the `.ckpt` sidecar must land with the bin (≥ 10% of bin size when the model produces ckpts).
- Atomic finalize: tmp → `os.replace` for bin, ckpt, then meta JSON with the full owner/chain/fingerprint stamp.

**Storage tiers:**

| Tier | Path | Role |
|---|---|---|
| 0 — VRAM | (engine memory) | native `get_common_prefix` reuse; back-to-back tool calls never touch disk at all |
| 1 — system RAM | `/var/lib/turbohaul/kvcache` (tmpfs mount) | the live save/restore directory; zero SSD wear |
| 2 — SSD | `/var/lib/turbohaul/kvcache_persist` | **one write per session** at controlled unload/ownership-transfer; hydrated back to RAM absent-only (`_hydrate_ram_from_persist`, `manager.py:5232` — "RAM always wins while present"); survives idle teardown and controlled swaps; a hard container restart deliberately loses the RAM tier and recomputes fresh |
| — | `/var/lib/turbohaul/kv_quarantine` | poisoned-bin evidence, moved (never deleted) by the 3-strikes automation (§4.7) |

RAM→SSD persistence also fires on **ownership transfer**: when a different identity takes the port, the outgoing identity's bins are background-persisted *before* its anchor record is overwritten — so a main agent's state survives a sub-agent taking the GPU even mid-burst.

**GC / retention** (`_gc_kv_cache`, throttled to 300 s, piggybacked on the background sweeper): tmpfs pass protects the per-(model, thread) pinned clean anchor + every live-thread bin, then prunes by age (6 h), count (100 files), and a bytes ceiling (`TURBOHAUL_KVCACHE_MAX_BYTES`, code default 20 GiB, deployed 24 GiB); SSD pass mirrors the age prune with its own FE-adjustable ceiling (`persist.max_bytes`, default 40 GiB) and a 60 s mid-copy safety window. `.bin`+`.ckpt` are pair-accounted on the tmpfs tier (an invisible ckpt once caused ENOSPC truncation). The tmpfs pass emits a structured `KVGC` line naming what pinned the floor; the SSD pass logs its own `persist KV GC: total_bytes=… ceiling=… deleted=…` summary (different grep token, no pin attribution).

**Size limits & FE adjustability:** there is no absolute max-bin-size — guards self-calibrate. Of the size knobs, `persist.max_bytes` is runtime-adjustable via `PUT /api/config` (the FE Config tab); the tmpfs ceiling and guard thresholds are env/code-level. The per-role `save_kv` toggle is per-request `client_meta`, not config (§4.2).

### 4.6 KV restore path: wave-return and warm reuse

**Cold restore fires exactly at the wave-return/swap-back moment:** once per (re)spawned sidecar, right after health passes — from the `_process_slot` cold-spawn path and from `_spawn_for_resident` (cap ≥ 2). The warm path never calls it.

Per restore: list the RAM tier for `{model_tag}.*.bin` → require the meta sidecar → check the `bin_bytes` consistency stamp → honor the compression **stale mark** (a labeled `is_compression` turn marks the session's main bin stale at admission, so the next main turn recomputes at the new compressed baseline instead of restoring the pre-compression copy) → run every candidate through `resolve_kv` (§4.3) → **restore only the single best bin** (sort: `clean_prefix` first, then token count — restoring multiple bins under one identity is how stale-CLEAR bugs happened). The engine call is `POST /slots/{r_sid}?action=restore {"filename": …}`, status-checked, with a deliberately honest log: the manager reports *"POSTed … engine determines actual n_past"* — the engine's `get_common_prefix` on the next decode is the final authority on reuse.

**Outcome classification** (per restore, surfaced on `/status` under `kv_classifier`): `wave-return-clean-restore` (2xx on the think-free anchor — the headline counter `wave_return_restores`), `wave-return-shadow-restore`, `restore-no-bins` (first-seen identity / fresh sub-agent nonce), `restore-no-anchor-for-identity` (bins exist but belong to other identities — own-anchor rule, no cross-restore), `restore-diverged-fresh`, `restore-post-failed`. Every branch also writes a `KV_RESTORE` diag line (`chosen=`, `common_prefix_turns`, KV-pressure fields).

**Warm path (no cold restore involved):** grace/ACTIVE_MATCH follow-ups decode on the engine's live VRAM state — native prefix reuse, delta-only prefill; this is why back-to-back tool calls are instant and never depend on saves. The one warm seam is `_maybe_force_clean_restore`: if the harness's next render is known to diverge from the engine's natural think-carrying state, the manager force-restores the think-free clean bin — but only when the clean chain is a valid prefix of the incoming, the warm state is known, and the warm state doesn't already cover it (`warm-native-reuse-longer` wins otherwise). Default ON (`TURBOHAUL_WARM_FORCE_CLEAN_RESTORE`), hard-coupled OFF if scaffold-strip is off; hot kill-switch files exist for ops.

**Measured validation results**: in repeated multi-agent validation runs, wave-returns on one session restored the same growing main bin — ~25k → ~47k tokens across consecutive waves (`wave-return-clean-restore, chose_clean` every time) with interleaved sub-agents on a second model correctly going fresh (`restore-no-anchor-for-identity`); under the full hardening validation load, a **105,446-token** main bin was seam-saved at the swap and wave-return-restored at **46/48 common-prefix turns (~96% reuse)**; and the proof document records a 154,647-token bin reused with a 29-token prefill — **629× less prefill work** than recomputing — after a full unload. The reasoning-model `<think>` case that makes this the hard problem, and the full benchmark, are covered in [docs/REASONING_KV_REUSE.md](docs/REASONING_KV_REUSE.md).

### 4.7 Seam integrity & poisoned-bin defense (dirty-tip + 3-strikes)

The failure class this system kills: a disposable role's serve **extends the shared VRAM tip** past main's canonical chain; on hybrid-recurrent models (MTP with `n_rs_seq=2`) that tail cannot be trimmed post-hoc — a seam save of that state would persist an internally inconsistent bin that passes the prefix gate and then **aborts the engine at strict extension** (observed live: `common.cpp:1498` × 3 → engine death loop → fresh slow-load).

Defense in depth:

- **D1 — dirty-tip tracker** (`_kv_dirty_tail`, per port): every disposable turn SETs `{role, pid, thread, ts}` (the role comes from the **admission stamp** `slot.admission_role`, which survives the warm-inherit client_meta replacement); every main/unlabeled turn CLEARs it (the engine reprocessed from divergence — tip canonical again). pid-stamped so a respawned engine never inherits a stale flag; dropped at engine teardown.
- **Seam remediation:** at a dirty seam the manager POSTs `/slots/{id}?action=erase` (whole-sequence erase is legal on recurrent ctx; only mid-sequence trims abort) and lets the unchanged strip-probe **re-prefill the full canonical render onto the empty slot** — bin == canonical chain by construction, one-time seam cost. Erase failure → save refused, previous good bin kept.
- **D2 — disposable-identity refusal** at the same chokepoint (§4.5) — belt to D1's suspenders.
- **3-strikes auto-quarantine**: each 2xx restore records `_last_restored_bin[port]`; the worker's proactive dead-idle sweep attributes an engine death while idle-hot to that bin (**one blame per attribution** — popped after use so unrelated deaths never strike an old restore), and at 3 strikes the bin's triplet is moved out of both tiers into quarantine — the next load goes fresh instead of looping. Combined with **ENGINE STALLED surfacing**: the death-strike path records an `engine_stall` status field (cleared on the next successful serve) so the FE can show the condition instead of a silent retry loop.

Validated end state: the exact curator + sub-agent-wave + heavy-context scenario that previously produced a 3-death engine storm now runs with **zero engine deaths, zero strikes, zero aborts** and 96% wave-return reuse (validation run, 2026-07-10).

### 4.8 Load verification (LOAD_VERIFY)

`load_verify_log.py` is the **observability-only** proof layer — it never gates behavior:

- `verify_model_resident(handle)` — pid-alive + `/health` 200 + `/slots` readable;
- `verify_kv_restored(handle, expected, actual)` — actual `n_past` (via `/slots` `n_prompt_tokens`) vs expected at a 0.98 threshold; never-raise, numeric-guarded;
- one greppable `LOAD_VERIFY {json}` line per event + a 64-record ring surfaced at `/status` (`load_verify`) and rendered by the FE LoadVerifyWidget (§11).

Emission as-built: the records emit on the **`_process_slot` cold-spawn path** (`manager.py:4162-4188`) — a `model_load` record after health, and a `kv_restore` record after the restore attempt. Honest caveat: the V1 wiring passes `expected_tokens=None` on this path, so `kv_restore_ok` degenerates to a "did `/slots` return a numeric `n_prompt_tokens`" check — records show `kv_actual_n_past: null / kv_restore_ok: false` whenever the just-restored slot doesn't yet report the count, and cannot verify restore *depth* against an expectation either way. The wave-return counters (§4.6) and the engine log are the truthful reuse signals today; threading real expected/actual values is the queued V2 follow-up. `_spawn_for_resident` (cap ≥ 2) performs the same spawn+restore sequence without LOAD_VERIFY emission. A bounded verify+**retry** loop that would consume these records is design intent recorded in code comments, not shipped behavior — `retry_count` is always 0 today.

## 5. Engine sidecar supervision

### 5.1 Spawn: three-layer flag pipeline + pinned-binary exec

A sidecar's argv passes three enforcement layers before exec:

1. **Manifest validation** (`manifest.py`): `llama_server_flags` is a **closed allowlist** — ~50 path/URL/credential/RCE-class flags are DENIED outright (including `slot_save_path`, `log_file`, `model`, `port`, `host`), a suffix-pattern guard (`^slot_save_`, `^api_key`, `^ssl_` …) forward-defends against future path-bearing flags, then membership in `SAFE_LLAMA_FLAGS` is required and values are validated (including a `parallel`-vs-ctx floor cross-check).
2. **argv build** (`flags_to_argv`): allowlist+denylist re-checked at build time; snake_case → `--kebab-case`; bool mapping; `flash_attn` tri-state special case.
3. **Spawn** (`spawn_sidecar`, `subprocess_mgr.py:83`): the manager **unconditionally injects its own flags first** — `--port … --host 127.0.0.1 -m <gguf> --slot-save-path /var/lib/turbohaul/kvcache --log-file <engine_log>` — and since those are denied to manifests, only the manager can ever set them. `start_new_session=True` (setsid) puts the child in its own process group for clean `killpg` teardown.

**TOCTOU-pinned binary exec:** when `runtime.llama_server_binary_sha256` is pinned, boot verification hashes the **open fd's inode** and spawn execs `/proc/self/fd/<fd>` — a post-hash path swap cannot redirect the exec. Empty pin = dev mode. **`--parallel` pinning:** the spawn parses the argv it actually passed and pins `SidecarHandle.parallel`; the in-flight admission cap reads the handle field, never a later manifest re-read (drift-proof across warm-inherit).

**Stdio "dying words" capture:** child stdout+stderr append to `<engine_log>.stdio` — GGML abort output bypasses `--log-file`, and a regular file never back-pressures like a pipe. This is what turned previously-silent engine deaths into same-day root causes.

### 5.2 Health, teardown, VRAM verify

- **Health:** `wait_until_healthy` polls `/health` every 2 s up to `loading_health_timeout_s` (default 600 s — a cold 21 GB load is legitimately slow), with a **fast-fail liveness check**: a child that exited during load returns False immediately instead of burning the timeout. A 200 with a drifted JSON shape raises `SchemaMismatch` loudly (fork-contract defense).
- **Teardown ladder** (`_teardown`, `manager.py:4832`): unload-seam KV flush + SSD persist (live engines only) → drained SIGTERM on the **process group** → SIGKILL escalation with explicit `waitpid` reaps (no zombie can absorb a reparented grandchild) → **VRAM-clear verify** (`nvidia-smi` polled until used drops ≥ 90% of the manifest's expected VRAM — derived dynamically, not hardcoded; 30 s timeout) → post-teardown orphan reaper pass → intra-lifetime orphan scan. Window note: config carries a 15 s active / 5 s cold SIGTERM window pair, but every as-built call site passes `is_active=False`, so the 5 s cold window governs all teardowns at this snapshot — the KV flush having already completed *before* the SIGTERM is what makes that safe.
- **Dead holders:** teardown captures aliveness at entry — a dead holder skips the KV flush and skips expecting a VRAM drop (it already freed VRAM at death).

### 5.3 Singleton & orphan enforcement

`singleton.py` implements "only writer to GPU 0" as-built:

- **Boot orphan reaper**: /proc walk for `llama-server` processes orphaned to init/subreaper (tini/systemd handled) with ports in the manager's range; SIGTERM→SIGKILL with process-starttime comparison to defeat PID reuse. Runs at boot, after every teardown, and — via the idle-holder teardown path — at shutdown when a holder is held (shutdown has no unconditional reaper pass of its own).
- **Intra-lifetime orphan scan**: catches sidecars still parented to the *running* manager whose PID left the live-handle set (lost-handle bugs, cancelled unwinds) — the final safety net after any failed teardown.
- **Foreign-GPU-app detection** at boot is informational (logged, not refused).
- As-built caveat: the exclusive `fcntl.flock` on state.sqlite (`acquire_state_lock`) exists and is tested but has **no production call site** at this snapshot — the singleton invariant is enforced operationally by the reapers + deployment (one container), not by the lock.

### 5.4 Vendored engine & build lineage

The inference engine is **[Tom's TurboQuant fork of llama.cpp](https://github.com/TheTom/llama-cpp-turboquant)** (MIT), **vendored in-repo** at `engine/llama-cpp-turboquant/` (git-archive snapshot of `86771a58d`, savestate branch; provenance in `engine/llama-cpp-turboquant/VENDORED.md`). **`engine.lock`** at the repo root pins SHA/version/branch and records that the checkpoint-ladder cold-restore was proven on this engine (`TURBOHAUL_CKPT_SIDECAR` requires it); it is pin metadata — no runtime code reads it. Build variants: **`Dockerfile.engine-src`** compiles the vendored tree fully offline (no PyPI/npm/external git; vendored wheels; committed FE dist); **`Dockerfile.cuda-multi`** compiles the vendored engine with broad GPU-architecture coverage (Turing through Blackwell); a slim `Dockerfile` ships manager-only (mount your own llama-server binary).

**TurboQuant + MTP together.** The fork's two headline capabilities compose, and the reference configuration runs both at once:

- **TurboQuant KV-cache quantization** — the `turbo2` / `turbo3` / `turbo4` cache types are accepted for `cache_type_k` and `cache_type_v` in per-model manifests (§9.1), shrinking KV memory to as little as ~1/8 of f16 (`safety.py` scales: turbo2 = 0.125, turbo3 = 0.1875, turbo4 = 0.25). The admission gate budgets the K and V halves independently (§6), so mixed configurations like K=f16 + V=turbo3 are costed correctly.
- **MTP (multi-token prediction) models** — MTP GGUFs run as ordinary manifests with the draft-MTP speculative config (`spec_type: draft-mtp`; `n_rs_seq` equals the configured `spec_draft_n_max` when draft-MTP is enabled, 0 otherwise). Their hybrid-recurrent contexts are what motivated the seam-integrity machinery in §4.7 (a recurrent tip can't be partially trimmed — hence whole-sequence erase + canonical re-prefill). The engine's checkpoint-ladder `.ckpt` sidecars carry a per-entry draft-MTP validator (recorded in `engine.lock`) so saved MTP state restores correctly across unloads.
- **Together:** an MTP model with TurboQuant-quantized KV gets both the speculative-decode throughput and the compressed cache — this is the reference deployment's standard configuration, and the full KV save/restore lifecycle (§4) is validated on exactly that combination. See [docs/TURBOQUANT_FLAGS.md](docs/TURBOQUANT_FLAGS.md) for the production flag doctrine.

**Weight quantization is auto-detected.** Nothing is added to the spawn-flag allowlist (§5.1) for a model's *weight* quant — the engine reads it straight from the GGUF header (`general.file_type`), so a model needs no extra `llama_server_flags` on account of how its weights are stored. This is distinct from the KV-cache quant types above, which quantize the *cache* and are selected explicitly per manifest.

**Hybrid (SSM + attention) models.** A `qwen35` hybrid interleaves state-space (SSM) layers with attention layers. Such a model serves in every residency mode (§1) — single series, series-parallel (`--parallel N`), and double-parallel — and reuses the existing KV-cache types (no new KV type is introduced). Its manifest sets the `arch`, `hybrid_kv_ratio` and `kv_bytes_per_token` fields (§9.1) so its SSM layers are costed correctly at admission (§6): SSM layers hold a fixed-size recurrent state rather than a cache that grows per token, so a hybrid's true KV footprint is markedly smaller than its file size implies. Left at the defaults (`arch: ""`, `hybrid_kv_ratio: 1.0`, no override), costing is byte-identical for every existing model.

## 6. Admission control & safety gates

Before any cold spawn (`queue.safety_enabled`, default ON), `all_safety_gates` (`safety.py:530`) checks free RAM (≥ 1024 MiB), free VRAM (≥ 512 MiB headroom), load-per-core (≤ 0.9), iowait (≤ 30%), and **KV-cache fit**. The KV estimator (`estimate_kv_cache_mib`, `safety.py:290`) is quant-aware **per K and V half**: the cache is ~half K, half V, so each half scales by its own quant type (`f16=1.0 … turbo2=0.125 …`) — a K=f16 + V=turbo3 manifest is no longer over-counted as full-f16 and wrongly refused. It is also **dimension-aware for hybrids**: for a `qwen35` model the manager parses the real GGUF attention dims (via the stdlib `_gguf_meta.py` KV-header reader) and sizes the KV cache from the actual attention layers. The estimate follows a strict precedence — a measured `kv_bytes_per_token` manifest override (used verbatim), then parsed GGUF dims, then the legacy file-size heuristic — and `hybrid_kv_ratio` (§9.1) scales the **file-size fallback path only** (the dims and override paths ignore it, since the attention-layer count already reflects the hybrid fraction). At the defaults (`arch: ""`, `hybrid_kv_ratio: 1.0`, no override) the estimate is unchanged for every existing model. Fit = model body + KV + overhead + per-parallel-slot floor ≤ free VRAM; `parallel>1` with unreadable `nvidia-smi` refuses a blind spawn. Any failed gate → `LOADING_FAIL` → audit `safety_gate_refused` with per-gate detail → the client gets a clean failure instead of an OOM'd GPU. HTTP-level admission is bounded by the body-size middleware (§7) and the acceptance-buffer cap (§3.1).

## 7. API surface (as-built route table)

The app (`create_app`, `api/main.py`) fronts one `TurbohaulManager`; lifespan boots the audit pool → boot reconcile → binary verify → optional fingerprint purge → worker loop + background sweeper + live monitor. The **only** middleware is the body-size limit — no app-layer auth (§8).

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compat, non-stream + SSE stream |
| POST | `/api/chat` | Ollama-compat chat (streaming+tools currently 400-deferred) |
| POST | `/v1/embeddings` | llama-server embeddings passthrough (own ~2 MB gate, batch ≤ 64, capability-checked) |
| GET | `/v1/logging` | paginated audit events, server-side redaction, ~80k-char token budget |
| GET | `/v1/telemetry/events`, `/v1/telemetry/status` | flap-telemetry ring/file reader (§10) |
| GET | `/api/tags`, `/api/show` | Ollama model listing / detail from manifests |
| POST | `/api/pull-url`, `/api/pull-hf` | SSRF-guarded downloads into the blob store (§8) |
| POST | `/api/pull` | **501 stub** — Ollama registry protocol not implemented in v1 |
| POST / DELETE | `/api/import`, `/api/delete` | sandboxed local import (GGUF magic, O_NOFOLLOW) / blob delete |
| GET/PUT/DELETE | `/api/manifests/{tag}` | ETag revision + `If-Match` (412 on mismatch), atomic writes |
| GET / PUT | `/api/config` | boot values read-only (paths redacted to basename); PUT limited to `queue`/`pull`/`persist` — boot sections → 403 |
| GET | `/status`, `/health`, `/api/version` | `/status` = full `status_snapshot()` (§10) |
| WS | `/ws/state` | redacted state event bus |
| GET | `/ui/live/output/stream` | SSE live generation text (kept OFF the WS bus so redaction holds) |
| GET | `/ui…` | SPA static serving with CSP + traversal guard |

Endpoints that do **not** exist as-built (despite older design docs): `/api/generate`, `/v1/completions`, `/api/ps`, `/api/create`, `/api/copy`, `/api/push`.

**Chat-completion compat layers** (both chat endpoints): `response_format` validation (`json_object` pass-through; `json_schema` DoS-bounded — 64 KiB / depth 16 / ≤ 64 properties / `$ref` rejected / `additionalProperties:false` required — with a one-retry `enable_thinking:false` fallback for thinking models); **knob-forwarding allowlist** (`_COMMON_FORWARDED_KNOBS` — samplers + tools; identity keys deliberately excluded); reasoning wrapped back as `<think>…</think>` via one formatter; **tool-call recovery** (`api/tool_call_recovery.py`): when tools were advertised and the model emitted the call as text JSON (`{"name":…,"arguments":…}` or `<tool_call>` XML after `</think>`), a brace-balancing parser recovers it into structured `tool_calls` against the advertised-name allowlist, strips the consumed span, and flips `finish_reason` — idempotent when upstream already populated the field (details: [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md)). `keep_alive` accepts Ollama forms (`0`, `-1`, `"5m"`), clamped to `KEEP_ALIVE_MAX_S=1800`. Typed error mapping: eviction → 499, sidecar unavailable → 503+Retry-After, timeout → 504, upstream → 502, queue overflow/closed → 500. Streaming: raw SSE byte pass-through from the sidecar with keep-alive comments during prefill (12 s cadence, slot-ready wait up to 7200 s), a per-chunk live-output tee, and mid-stream errors delivered as SSE error frames.

## 8. Security model & hardening

**Trust boundary (unchanged in spirit from the original design):** single-tenant, **network-perimeter trust**. Every endpoint runs without app-layer auth; the boundary is the bind address. `server.host` defaults to `127.0.0.1` and the config validator rejects `0.0.0.0` from yaml — public binding requires an explicit opt-in via the `--allow-public-bind` CLI flag or `TURBOHAUL_ALLOW_PUBLIC_BIND=1` (either alone suffices; the env var seeds the flag's default — the yaml `server.allow_public_bind` field exists but is not consulted at bind time). The reference deployment sets it and scopes exposure at the Docker/network layer. Adding auth on one surface without gating all of them would be worse than the uniform posture; a future cross-perimeter deployment should put a reverse proxy with bearer auth in front of *all* paths.

As-built enforcement stack:

- **SSRF guard** (`ssrf_guard.py` + `api/pull.py`): `https` only; hostname resolution pinned to the first record; deny-listed IPv4/IPv6 ranges (RFC1918, loopback, link-local, CGNAT, IMDS, NAT64 `64:ff9b::/96`, IPv4-compat/mapped v6, multicast/reserved); **double-resolve check** (validate+resolve twice, refuse on divergence — DNS-rebinding defense); redirects re-validated per hop (max 5) with `Authorization` stripped on cross-host hops; the HF bearer token attaches only to allowlisted HF hosts.
- **Import safety**: absolute path, prefix denylist (`/proc /sys /dev /etc /root …`), symlinks rejected + `O_NOFOLLOW`, realpath must stay under `import_allowed_root`, 4-byte `GGUF` magic check.
- **Blob integrity**: streamed to `incoming/` with per-stream byte ceiling → sha256 verified → atomic rename into the content-addressed store → `chmod 0o400`. `verify_blob_on_stage` (`blob_store.py:237`) implements a stage-time re-hash (TOCTOU-swap defense) but is currently **unwired** — no runtime caller at this snapshot (tests only); the model-load path trusts the write-time hash + read-only chmod. Wiring it into the spawn path is an open gap.
- **Manifest hardening**: tag regex validation; closed spawn-flag allowlist + denied path/credential flags + suffix-pattern forward defense (§5.1); jinja-injection guard on template-bearing fields; atomic ETag/If-Match writes.
- **Config write protection**: boot sections rejected on PUT (403) — no config-driven binary swap; the binary itself is sha256-pinned with fd-pinned exec (§5.1).
- **Redaction discipline**: `/ws/state` never carries prompts, responses, stderr, full thread ids, or IPs; `/v1/logging` payloads pass a recursive redaction against the event bus's redacted-key set; live generation text flows only on the dedicated SSE stream; `GET /api/config` redacts filesystem paths to basenames.
- **Request bounds**: ~2 MB body cap on the chat endpoints (ASGI middleware, pre-routing) and embeddings; JSON-schema caps; audit/telemetry pagination budgets.

## 9. Configuration

`TurbohaulConfig` (pydantic, `extra='forbid'`) is loaded from `/etc/turbohaul/turbohaul.yaml` (`TURBOHAUL_CONFIG_PATH`/`--config`), env-overridden via a 12-entry `TURBOHAUL_*` map, then **split** into a frozen **BootConfig** (`server`, `storage`, `runtime` paths, `ui` — restart to change) and a mutable **RuntimeConfig** (`queue`, `pull`, `persist`, `monitor`) that `PUT /api/config` can update at runtime (per-section re-validation, atomic swap, live grace/idle timer refresh). Known asymmetry: `monitor` is in RuntimeConfig but not PUT-able (and absent from GET) — changing it requires a restart.

### 9.1 Per-model configuration (manifests)

Every model is configured by **one YAML manifest per tag** (`/var/lib/turbohaul/manifests/<model_tag>.yaml`), the unit the scheduler, safety gates, and spawn pipeline all read:

- **Identity & sizing:** `model_tag` (validated `^[a-z0-9][a-z0-9._-]{0,63}$` — filename-safe by construction, §4.4), `display_name`, `description`, `gguf_blob_sha256` (required — must name a blob in the content-addressed store; the model's true identity and the first field of the engine fingerprint §4.4), `gguf_size_bytes`, `context_size`, and `expected_vram_bytes` (optional, default 0 — when set it feeds the VRAM admission gate §6; when omitted the gate falls back to the min-free-VRAM floor plus the independent ctx-derived KV-fit estimator). Three optional architecture fields describe hybrid models: **`arch`** (string, default `""`) names the model architecture — set `"qwen35"` for an SSM+attention hybrid, which makes the manager size KV from the model's real GGUF attention dims (§6); **`hybrid_kv_ratio`** (float 0.0–1.0, default `1.0`) is the fraction of layers that contribute a *growing* per-token KV cache and scales the **file-size fallback** estimate only (the dimension-aware path ignores it); and **`kv_bytes_per_token`** (float ≥ 1024.0, optional/unset) is an operator-**measured** effective KV cost in BYTES/token that overrides the estimate verbatim (highest precedence; a 1 KiB/token floor rejects a KiB-vs-bytes typo). SSM layers keep a fixed-size recurrent state instead of a growing cache, so a hybrid's per-token KV is smaller than a pure-attention model of the same size. The defaults (`arch: ""`, `hybrid_kv_ratio: 1.0`, `kv_bytes_per_token` unset) mean pure attention — byte-identical costing for every existing model.
- **`llama_server_flags`** — the closed-allowlist spawn flags (§5.1): performance/memory layout (`ctx_size`, `n_gpu_layers`, `flash_attn`, `threads`, `parallel`, `mlock`, `cache_reuse`…), the **KV-cache quant pair `cache_type_k` / `cache_type_v`** (f16/q8_0/q4/turbo2/turbo3/turbo4 — this is where TurboQuant is dialed in per model, §5.4), chat-template selection (`chat_template`, `jinja`, `reasoning_format` — jinja-injection-guarded), and MoE/MTP-specific knobs (e.g. `n_cpu_moe`). Path-bearing and credential flags are denied outright — only the manager sets those.
- **`prompt_template`** — default system prompt + stop tokens.
- **Lifecycle:** `revision` auto-increments on every write and powers the ETag/`If-Match` concurrency contract (412 on mismatch); writes are atomic (temp + rename + fsync); edits land via the FE **Models** tab or `PUT /api/manifests/{tag}` and **hot-reload on the next slot stage** — no restart. Different models on one host can therefore run completely different context sizes, KV quants, and template stacks, swapped in and out by the scheduler with per-model safety costing.

Key defaults (code) and the **reference deployment's** live values (captured 2026-07-10):

| Knob | Code default | Deployed | Notes |
|---|---|---|---|
| `queue.grace_seconds` | 30 | **60** | `TURBOHAUL_GRACE_S` |
| `queue.max_grace_extensions` | 5 | **50** | `TURBOHAUL_MAX_GRACE_EXT` |
| `queue.idle_hot_load_seconds` | 600 | **1800** | history 120→300→600; `TURBOHAUL_IDLE_HOT_S` |
| `queue.max_parallel_sidecars` | 1 (1..32) | **1** | ≥2 switches to the resident dispatcher (§3.6) |
| tmpfs KV ceiling | 20 GiB | **24 GiB** | `TURBOHAUL_KVCACHE_MAX_BYTES` (env-only) |
| `persist.max_bytes` (SSD KV) | 40 GiB | 40 GiB | FE-adjustable via PUT /api/config |
| `TURBOHAUL_M2B_ACTIVE` | off | **on** | role+session-keyed slot identity (§4.1) |
| `TURBOHAUL_DURABLE_RING` | off | **on** | in-memory last-3 recency index per (role, session) — an *index over* on-disk bins, rebuilt empty at restart; NOT a storage tier |
| `TURBOHAUL_CKPT_SIDECAR` | — | **on** | engine checkpoint-ladder sidecars (requires the pinned engine, see `engine.lock`) |
| `TURBOHAUL_COVERED_SCAFFOLD_STRIP` | on | **off** | deployed harness already sends think-stripped renders |
| `TURBOHAUL_CURATOR_REUSE_MAIN` | off | off | curator save_ok route (flag- and label-gated) |
| `TURBOHAUL_SHADOW_REPREFILL` / `TURBOHAUL_SHADOW_RESTORE_PREFER` / `TURBOHAUL_WARM_NATURAL_SKIP` / `TURBOHAUL_TOOLTAIL_RESTORE_SKIP` | off | off | dormant A/B and emergency-floor flags |
| `TURBOHAUL_FINGERPRINT_PURGE` | off | (unset) | opt-in stale-bin sweep (§4.4) |
| `server.port` / engine ports | 11401 / 11500+ | same | `default_port_base`, range +100 |

Per-request knobs ride the payload: `keep_alive`, `thread_id`, `response_format`, the forwarded sampler/tool set, and the `client_meta` identity contract (§4.2) including the per-role `save_kv` toggle — the KV contract is deliberately **request-level, not env-level**.

## 10. Observability

Five complementary planes, all observe-only (none gates behavior):

**`/status`** (`status_snapshot()`, lock-free by contract) — the FE's 1 Hz truth: `queue` depths, `active`/`loading`/`grace`/`idle_hot` blocks (grace is *suppressed while a serve is active* — display truth), `evictions`, `kv_classifier` counters (incl. `wave_return_restores`), `request_identity` (last R2B record), `load_verify` (last 20 records), `engine_stall`, `parallel_slots`, `generation` (live tok/s block), `residents` (cap ≥ 2), `vram`, `persist_kvcache` (SSD tier usage vs cap), background-sweeper and dormant-shadow diagnostics.

**Live monitor** (`live_monitor.py`, ~1 Hz, `monitor.enabled` kill-switch) — two planes keyed by one non-reversible 8-hex `generation_id = blake2b(pid:spawn_seq:slot_id)` so text and metrics can never cross-attribute: *metrics* — `LiveSlotsPoller` (cap ≤ 1; per-resident pollers at cap ≥ 2) polls the engine's `/slots` as a **pure observer** (await-free identity snapshot, post-await pid/spawn_seq re-validation against the fixed-port reuse race), derives tok/s (EWMA 0.4), prefill %, and stall (10 s) / prefill-stall (60 s) alarms — the freeze clock counts rising `n_prompt_tokens_processed` as activity so a warm follow-up's prefill never false-alarms STALLED; *text* — `LiveOutputBuffer` per-generation ring buffers (16 KB tail, LRU 8) fed by the streaming tee and read only by the SSE endpoint, keeping token text off the WS event bus entirely.

**Event bus / WS** — STATE-level events only; `EventBus.publish_nowait` hard-strips `{prompt, response, context, stderr, stdout, messages}` before fan-out, and full subscriber queues drop rather than block the worker.

**Audit log** (`state.py` → `state.sqlite` `audit_events`, WAL/autocommit) — one INSERT chokepoint (`record_audit_event`) behind a thread-local sync-only connection pool; the manager is the sole emitter (~25 event types: submit, transitions, teardowns, safety refusals, evictions, boot reconcile…). Read side `GET /v1/logging`: keyset pagination, filters, ~80k-char page budget with an oversized-row escape hatch, poison-row sentinels, recursive redaction as a tripwire (emitter discipline is the load-bearing protection).

**Flap telemetry** (`telemetry.py`) — an observe-only degradation event pipeline: lifecycle hooks (request arrival, queue state, slot assign, prefill start, first token w/ TTFT, keep-alives, disconnects, completion, VRAM samples) append to a 10k-entry ring **and** a rotating JSONL set (10 MiB × 5 files) under `…/telemetry/`; every call site is fail-open. Read via `/v1/telemetry/*`; reading guide: [docs/telemetry_reader_guide.md](docs/telemetry_reader_guide.md). (Two hooks — `on_generation_tick`, `on_slot_state_change` — are defined but unwired as-built.)

Plus the **greppable proof lines** scattered through this doc: `R2B_REQ_IDENTITY`, `LOAD_VERIFY`, `KV_RESTORE` diag, `KVGC`, dirty-tip SET/CLEAR/DROPPED, `save GATED/REFUSED/SKIPPED` — the system narrates every KV decision with provenance.

## 11. Front-end

Vite + React 18 + TypeScript + Tailwind SPA (react-router, 8 tabs: **Dashboard, Models, Queue, Blob, Config, Schema, Logs, Settings**), built to `src/frontend/dist` (committed, for the self-contained image) and served single-port by the backend at `/ui` with CSP, traversal guard, and immutable-asset caching. No external CDN, no chart libs (hand-rolled SVG sparklines), no Monaco. Live state: 1 Hz `/status` polling with the WS as a refetch nudge (debounced), plus the SSE live-output stream demuxed into one pane per loaded model with sticky-bottom scroll (auto-follow only near the bottom; "↓ latest" jump button).

Dashboard truth-features (each one earned by a live incident):

- **State pill precedence**: PREFILL (BE state only — % deliberately not shown where it would lie) → engine-op pill (kv_restore/kv_save/decode/stream/unload; never outranks alarms) → **BUSY** ("engine busy — telemetry paused", escalating to red **NO TELEMETRY** at 120 s) → STALLED → RECENT → state colors.
- **PrefillBar ↔ progress swap**: during prefill the bar shows `(restored-from-KV + newly-processed) / last-completed-turn total` (denominator learned in the always-mounted parent — the fix for the bar pegging at 99%), then hands off to token progress at first decode; a red **MID-PREFILL HANG** banner rides the BE's 60 s prefill-stall alarm.
- **Residents box**: per-slot state badge, `engine_op` pill, the **request-identity strip** (`ip · model · ROLE · session` from `/status.request_identity`, role priority curator > compression > sub-agent > main), the **LOAD_VERIFY widget** (green/yellow/red per model-load and KV-restore record, with `n_past actual/expected`, dead-pid and not-resident flags), and the amber **"UNLOAD IN Ns"** timer rendered *only* when the model is genuinely parked (GRACE/IDLE_HOT) — never during a serve.
- **Honesty rules**: tok/s renders "—" rather than a fake number; VRAM bars render only from real backend data (placeholder text under cap ≤ 1 until the BE populates it); sparkline/peak reset per generation so numbers never blend runs.
- **Settings**: the SSD KV cap (`persist.max_bytes`) is the first-class runtime knob (GiB input → `PUT /api/config`, applies immediately) with live usage/headroom/over-cap readout; **Models** edits per-model `llama_server_flags` (including the KV-cache category: `cache_type_k/v`, checkpoint knobs) via ETag'd manifest PUTs; **Config** exposes the raw runtime-section editor (boot sections shown read-only); **Logs** is the paginated audit feed with a permanent redaction banner.

Rendering safety carried from the original design: all manifest-sourced strings render as text (no `dangerouslySetInnerHTML`); CSP `default-src 'self'`; `X-Frame-Options: DENY`.

## 12. Storage & persistence

All state under `/var/lib/turbohaul` (in the reference deployment: a bind-mount to SSD storage, with **tmpfs mounted over `kvcache/`** — the RAM tier is realized by the deployment mount, size 40g in prod):

```
/var/lib/turbohaul/
├── blobs/sha256/                 # content-addressed GGUFs: incoming/*.tmp → atomic rename → <ab>/<hash> (0o400)
├── manifests/                    # per-tag model YAML (revision/ETag'd, atomic writes)
├── import-staging/               # sandboxed /api/import root (RO bind of the host model dir in prod)
├── state.sqlite                  # slots + audit_events + pull_history (WAL)
├── kvcache/                      # KV RAM tier (tmpfs) — engine --slot-save-path (§4.5)
├── kvcache_persist/              # KV SSD tier — one write per session at unload (§4.5)
├── kv_quarantine/                # 3-strikes poisoned-bin evidence (§4.7)
├── engine_logs/                  # per-spawn llama-server --log-file (+ .stdio dying-words capture)
└── telemetry/                    # flap-telemetry rotating JSONL
```

Blob lifecycle: streamed to `incoming/` with a per-stream byte ceiling → sha256 verified → atomic rename + dir fsync → read-only; stale incoming temps GC'd. (A stage-time re-verify helper exists but is unwired as-built — see §8.) Deployment/persistence doctrine (bind-mounts, image tarballs, auto-recovery wiring) lives in [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md) — the reference `docker-compose.yml` is illustrative (named volume, no tmpfs); production runs the documented bind+tmpfs layout. KV persistence philosophy: the SSD tier survives controlled swaps and idle teardowns; a hard container restart deliberately starts the RAM tier empty and recomputes fresh.

## 13. Licensing & vendoring

- Manager: see `LICENSE`; third-party terms collected in `THIRD_PARTY_LICENSES.md` + `THIRD_PARTY_NOTICES.md`; contributors in `CONTRIBUTORS.md`.
- Inference backend: `llama-server` built from **[Tom's TurboQuant fork of llama.cpp](https://github.com/TheTom/llama-cpp-turboquant) (MIT)**; upstream llama.cpp MIT; Ollama (MIT) — API *shape* compatibility only, no source vendored. "Ollama-compatible" used as nominative fair use.
- Vendored for self-containment (all permissive): the engine source snapshot (`engine/llama-cpp-turboquant/`, MIT) and the Python wheels under `vendor/pywheels` consumed by `Dockerfile.engine-src`.

## 14. Lineage & superseded designs

- **Origin:** replaced an older sidecar-manager (archived internally), whose in-process-only lock raced across processes (documented 240 s hang, 2026-05-16). Two still-binding decisions from that era: **supervised `llama-server` subprocesses, not libllama bindings** (crash containment, fork-flag access) and **BYOM blob storage only** (no registry catalog).
- **v0.2 design doc → this doc:** the 2026-05-16 design built the skeleton that still stands — queue/grace/idle-hot, manifest+flag hardening, SSRF guard, WS redaction, config split, storage layout. Its *phase plan, shadow-mode soak strategy, migration/cutover runbooks, and closed-risks bookkeeping* are historical: the migration executed 2026-05-19 ([docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md)); no soak/shadow-mode *code* was ever built (today's "shadow" flags are dormant KV A/B mechanisms, unrelated); `GRACE_BUSY` shipped as an FSM state but the behavior landed as ACTIVE_MATCH (§3.2).
- **The KV era (2026-06-28 → this snapshot):** KV save/restore across swaps → unified identity-keyed chokepoint → prefix-chain restore validity → single-copy-per-(session, role) redesign → engine vendoring → the KV hardening ladder (per-role save gates, T3 identity preservation, LOAD_VERIFY, dead-handle recovery, unload-seam-only saves, 3-strikes quarantine, seam-dirty canonical integrity). Detailed history: the `CHANGELOG.md` v0.6.0 entry (landing in this release) and `git log` on main.

---

*Grounding: every mechanism above was verified against the code in this release (static analysis over the tree + reads of `manager.py`). Deployed-value column in §9 captured live from a reference container.*
