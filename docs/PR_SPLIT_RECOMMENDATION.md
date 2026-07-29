# Turbohaul controller hardening — PR split

Three independent PRs. PR 1 (this branch: `fix/controller-lifecycle-and-lanes`)
is lifecycle-only and safe to merge independently; it unblocks PR 2's
assumption of a trusted slot/port inventory.

---

## PR 1 (this branch: `fix/controller-lifecycle-and-lanes`) — lifecycle only

**Scope: lifecycle safety — orphan/stale-engine reaping + stuck-slot recovery.**

Smallest upstream-ready fix for the P12 incident's lifecycle seam. No
architecture rewrite; no lane partitioning; no reasoning-display changes.

### Changes
1. **`reap_orphan` is now process-group safe.** `spawn_sidecar` uses
   `start_new_session=True` (setsid), so an orphaned `llama-server` may have
   grandchildren in its own process group. The pre-hardening `reap_orphan`
   killed only the leader PID via `os.kill(pid, SIGTERM|SIGKILL)`, leaking
   grandchildren that keep holding GPU/port. The fix mirrors the live
   `drained_sigterm` contract: `killpg(getpgid(pid))` on the whole group,
   escalating to `killpg(SIGKILL)`. Injectable `kill_fn`/`killpg_fn`/
   `getpgid_fn`/`starttime_fn` seams (defaulting to real `os` functions) make
   the process-group behavior assertable without real subprocesses.

2. **`boot_orphan_reaper` uses ownership-aware reconciliation — NOT broad
   port-listener killing.** The original blocked candidate reaped ANY process
   listening in the managed port range, which risked killing unrelated foreign
   processes (nginx, python http.server, dev tools). The corrected design has
   two REAPING passes, both ownership-aware — they ONLY reap processes PROVEN
   to be Turbohaul-owned (not merely `llama-server --port N` in the managed
   range, which cmdline alone does NOT prove):

   - **PPid-based orphan scan** (`find_orphan_llama_servers`): reaps
     llama-server orphans with PPid in {1, subreaper} on our port range — the
     reparented-to-init signal is itself an ownership proof (the Turbohaul
     manager died, orphaning its engine).
   - **Proven-ownership stale-engine scan** (`find_llama_servers_in_port_range`,
     new): reaps stale Turbohaul-owned engines whose `--port` is in the managed
     range REGARDLESS of PPid, BUT ONLY when ownership is proven by a **DURABLE
     ENGINE-IDENTITY record** — the candidate's live `(pid, port, starttime)`
     must match a triple persisted atomically in `state.sqlite` at spawn time
     (`record_engine_identity`). This replaces the earlier parent-chain
     heuristic (PPid in {1, subreaper} OR parent chain contains a Turbohaul
     manager marker), which was insufficient: an unrelated `llama-server
     --port N` that is itself orphaned (PPid=1) — e.g. a foreign llama-server
     whose own parent died — would have been reaped, because reparenting to
     init only proves *some* parent died, not that the parent was Turbohaul.
     The durable record survives a manager crash (it lives in `state.sqlite`,
     not RAM) and prevents PID reuse (`starttime` is unique per process
     instance per boot and never changes for the life of a process; a recycled
     pid has a different starttime, so it can never match a recorded identity).
     This is the seam that catches the P12 "orphaned llama-server ... without a
     listener" + "stale occupied slot" case: the manager crashed AFTER
     recording the engine identity, the engine has no listener socket and may
     be reparented to init, but the recorded `(pid, port, starttime)` +
     live `/proc` starttime match proves ownership — not the parent chain, not
     the listener. A foreign or independently managed `llama-server --port N`
     has no recorded identity and is NOT matched and NOT reaped — it is
     reported only via the diagnostics-only listener scan. **A candidate
     lacking a matching recorded identity is NEVER reaped (report-only).**

   **Schema migration (v1 → v2):** `state.sqlite` gains an `engine_starttime`
   INTEGER column on the `slots` table (idempotent `ALTER TABLE` guarded by
   `PRAGMA table_info`, so a pre-v2 DB is migrated on first open and a v2 DB
   is a no-op). `SCHEMA_VERSION` bumps 1 → 2. Pre-v2 rows get `engine_starttime
   = NULL`, which degrades the identity proof to report-only for those slots
   (a NULL starttime can never match a live one) — conservative, no stale
   pre-v2 engine is reaped until it is re-spawned under v2 and gets a record.
   The `known_active_pids` reconcile path is unchanged; the new
   `known_engine_identities` helper mirrors its state filter.

   **Cmdline alone does NOT prove ownership; neither does PPid=1.** The
   previous revision's parent-chain heuristic was insufficient (a foreign
   orphaned llama-server with PPid=1, or one launched by a process whose
   cmdline happens to carry a manager marker, would be reaped). The durable
   identity record is the ONLY reaping authority. Adversarial tests
   (`TestForeignLlamaServerOwnershipProof`, `TestDurableEngineIdentityProof`)
   demonstrate: a foreign `llama-server --port` in range with PPid=1 is NOT
   reaped (no recorded identity); a recorded Turbohaul-owned no-listener orphan
   IS reaped (identity matches); a PID-reused replacement (same pid, different
   starttime) is NOT reaped (starttime cross-check fails).

   A third DIAGNOSTICS-ONLY pass (`port_listeners_in_range`) populates the
   `stale_listeners` count so an operator can SEE a port in the managed range
   is still occupied (and by whom). It does NOT reap. This replaces the blocked
   candidate's unsafe broad listener-killing pass, which reaped any listener in
   the range and risked foreign processes. `boot_reconcile` surfaces
   `stale_listeners` in its summary + audit event.

3. **Tests proving the failure paths** (`tests/test_lifecycle_hardening.py`):
   - `reap_orphan` sends `killpg` (SIGTERM + SIGKILL escalation), not PID-only
     `os.kill` (4 cases: group kill, SIGKILL escalation, already-gone,
     sigterm-clean).
   - `boot_reconcile` reaps a NO-LISTENER orphan (cmdline-identified
     `llama-server --port N`, no PPid match, no listener socket) — the real P12
     seam, not a listener fixture. The detector is the cmdline-based
     `find_llama_servers_in_port_range`, not the socket scan.
   - `boot_reconcile` does NOT reap a FOREIGN process (nginx) listening in the
     managed range. The foreign listener appears in the diagnostics-only
     `stale_listeners` count but is never killed.
   - `boot_reconcile` does NOT reap a FOREIGN `llama-server --port N` in the
     managed range — including one that is ITSELF orphaned (PPid=1). PPid=1
     is NOT ownership proof; only a recorded `(pid, port, starttime)` identity
     in `state.sqlite` proves Turbohaul ownership (adversarial
     `TestDurableEngineIdentityProof`). A genuine Turbohaul-owned no-listener
     orphan whose recorded identity matches the live `/proc` starttime IS
     reaped, proving the P12 ownership route survives the durable-identity
     hardening. A PID-reused replacement (same pid, different starttime) is
     NOT reaped.
   - `boot_reconcile` reports `stale_listeners` count (diagnostics).
   - A health-load timeout fails the request in bounded time (injected
     `_wait_healthy` returns False instantly → completion_future failed with
     `loading-fail-health-timeout`, `_active_handle` cleared). Proves the 600s
     silent wait does NOT happen on a failed transition.

### Non-goals (explicitly deferred to PR 2 / PR 3)
- Lane partitioning (primary-vs-aux isolation) — see PR 2.
- Reasoning-display policy — see PR 3.
- Changing the `loading_health_timeout_s` default (600s) — the bounded failure
  is already correct; the default is an ops tuning knob, not a bug.

---

## PR 2 (separate, recommended) — primary-vs-aux lane isolation

**Why it's a separate PR:** Lane isolation is a scheduling/admission change
that touches the queue, the worker_loop admission gate, and the resident
registry. It cannot be made safely within the lifecycle PR without expanding
the blast radius and the test surface beyond "smallest upstream-ready".
Building it on top of the reconciled, orphan-free baseline from PR 1 is the
correct ordering: isolation assumes the controller can trust its slot/port
inventory, which PR 1 guarantees.

### Recommended approach (strict main reservation, not full lane split)
A full two-lane (separate supervised queues + separate sidecars) split is a
large architecture change. The **smallest safe** approach is a **strict main
reservation with queue partitioning**:

1. **Config surface.** Add to `QueueConfig`:
   - `main_lane_reserved: bool = True` — reserve one admission slot for main-lane
     requests when aux work is in-flight.
   - `main_lane_identity_keys: list[str] = ["is_main"]` — the `client_meta`
     keys that classify a request as main-lane (interactive GRM). The
     `client_meta` identity plumbing already exists (`is_main`, `is_sub_agent`,
     `is_curator`, `is_compression` are read in `_emit_request_identity` and
     `kv_classify`).
   - `aux_lane_max_inflight: int = 0` — 0 disables aux admission while a main
     request is queued or active (strict reservation); >0 allows bounded aux
     concurrency alongside main.

2. **Admission gate** (in `submit` / `pop_next`): when `main_lane_reserved` is
   True and a main-lane request is queued/active, aux-lane requests are held
   in the staging queue and not popped until the main request reaches ACTIVE
   (or completes). This uses the existing `max_consecutive_same_model` /
   `max_other_model_wait_s` fairness knobs as the backstop so an aux request is
   never starved forever, only deprioritized behind main.

3. **No second sidecar required.** At `max_parallel_sidecars=1` (current
   production), the reservation is a queue-level admission control, not a
   second engine. A main request evicts/defers aux via the existing
   grace→cold→swap path, not a parallel lane. This keeps the single-sidecar
   invariant intact.

4. **Tests:** main request admitted ahead of a long aux request; aux does not
   evict a queued main request; aux is admitted once main reaches ACTIVE;
   `main_lane_reserved=False` preserves current FIFO behavior (back-compat).

### Acceptance
- Submit a long aux request then an interactive GRM request: main is admitted
  without aux eviction/starvation (PR 1 acceptance test 3 from the packet).

---

## PR 3 (separate, recommended) — reasoning-display policy

**Why it's a separate PR:** The reasoning-output policy is a config-level /
downstream concern. The Hermes CLI `--reasoning auto` raw panel issue is in
the Hermes consumer, not Turbohaul's controller. Turbohaul already supports
reasoning via the chat-completion proxy and the
`_merge_reasoning_into_content` / think-strip paths. A route-configurable
reasoning visibility flag is a small, additive config change but it belongs
with the Hermes picker PR (per the packet's PR separation note), not the
Turbohaul lifecycle PR.

### Recommended approach
1. **Config surface.** Add to `QueueConfig` or a new `RoutePolicyConfig`:
   - `expose_reasoning: bool = False` (default off for interactive routes).
   - `expose_reasoning_route_allowlist: list[str] = []` — routes where reasoning
     IS exposed (e.g. a debugging route).
2. **Chat-completion proxy:** when `expose_reasoning` is False for the matched
   route, strip the `reasoning` field from the forwarded response (the
   `_merge_reasoning_into_content` path already does think-strip; this adds an
   explicit route-level gate). Internal reasoning support is NOT removed
   globally — the engine still reasons; the field is just not surfaced to the
   client.
3. **Tests:** a thinking-capable model on the Kensei route returns a final
   response with no raw Reasoning panel; a route in the allowlist still
   surfaces reasoning.

### Acceptance
- Request a thinking-capable model on the Kensei route: final response
  displays normally without a raw Reasoning panel (packet acceptance test 5).

---

## Ordering
PR 1 (lifecycle) → PR 2 (lane isolation, built on the reconciled baseline) →
PR 3 (reasoning policy, can run parallel to PR 2). PR 1 is safe to merge
independently and unblocks PR 2's assumption of a trusted slot/port inventory.
