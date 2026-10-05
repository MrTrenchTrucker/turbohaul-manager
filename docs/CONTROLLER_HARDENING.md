# Turbohaul controller hardening — PR split

This document is a plan that splits the controller-hardening work into
three separate PRs: lifecycle (reaping orphaned and stale engines, recovering stuck slots),
primary-vs-aux lane isolation, and reasoning-display policy.

**What shipped, in this tree:** PR 1's changes are present. PR 2's main-lane reservation is
present in a simpler form: a queued main-lane request is admitted ahead of queued aux work
(`main_lane_reserved`), and request keys mark a request as main-lane (`main_lane_identity_keys`).

**What did not, in this tree:** PR 3's route-level reasoning gate is not implemented. Nor is the
bounded aux concurrency alongside main that was sketched for PR 2 (`aux_lane_max_inflight`).

Three independent PRs. PR 1 is lifecycle-only and safe to merge independently;
it unblocks PR 2's assumption of a trusted slot/port inventory.

**Status in this tree:** PR 1's changes are present, and PR 2's main-lane
reservation is present in a simpler form (see PR 2). PR 3's route-level reasoning
gate is not implemented.

---

## PR 1 — lifecycle only

**Scope: lifecycle safety — orphan/stale-engine reaping + stuck-slot recovery.**

Smallest upstream-ready fix for the orphaned-engine / stale-slot lifecycle seam. No
architecture rewrite; no lane partitioning; no reasoning-display changes.

### Changes
1. **`reap_orphan` is now process-group safe.** `spawn_sidecar` uses
   `start_new_session=True` (setsid), so an orphaned `llama-server` may have
   grandchildren in its own process group. Killing only the leader PID
   via `os.kill(pid, SIGTERM|SIGKILL)` would leak grandchildren that keep holding
   GPU/port. The fix mirrors the live
   `drained_sigterm` contract: `killpg(getpgid(pid))` on the whole group,
   escalating to `killpg(SIGKILL)`. Injectable `kill_fn`/`killpg_fn`/
   `getpgid_fn`/`starttime_fn` seams (defaulting to the real `os` functions and the
   `/proc` starttime reader) make
   the process-group behavior assertable without real subprocesses.

2. **`boot_orphan_reaper` uses ownership-aware reconciliation — NOT broad
   port-listener killing.** Reaping ANY process listening in the managed port
   range would risk killing unrelated foreign processes (nginx, python
   http.server, dev tools), so that is not done. There are two REAPING passes,
   both ownership-aware — they ONLY reap processes PROVEN
   to be Turbohaul-owned (not merely `llama-server --port N` in the managed
   range, which cmdline alone does NOT prove):

   - **PPid-based orphan scan** (`find_orphan_llama_servers`): nominates
     llama-server processes with PPid in {1, subreaper} on our port range, and
     reaps only those whose live identity matches a recorded engine identity
     (see the next pass). PPid=1 alone is a nomination, not an ownership proof.
   - **Proven-ownership stale-engine scan** (`find_llama_servers_in_port_range`): reaps stale Turbohaul-owned engines whose `--port` is in the managed
     range REGARDLESS of PPid, BUT ONLY when ownership is proven by a **DURABLE
     ENGINE-IDENTITY record** — the candidate's live `(pid, port, starttime)`
     must match a triple persisted in `state.sqlite` right after spawn
     (`record_engine_identity`). A parent-chain heuristic (PPid in {1,
     subreaper} OR parent chain contains a Turbohaul manager marker) is not
     used, because it is insufficient: an unrelated `llama-server
     --port N` that is itself orphaned (PPid=1) — e.g. a foreign llama-server
     whose own parent died — would have been reaped, because reparenting to
     init only proves *some* parent died, not that the parent was Turbohaul.
     The durable record survives a manager crash (it lives in `state.sqlite`,
     not RAM) and prevents PID reuse (`starttime` is unique per process
     instance per boot and never changes for the life of a process; a recycled
     pid has a different starttime, so it can never match a recorded identity).
     This catches the "orphaned llama-server ... without a
     listener" + "stale occupied slot" case: the manager crashed AFTER
     recording the engine identity, the engine has no listener socket and may
     be reparented to init, but the recorded `(pid, port, starttime)` +
     live `/proc` starttime match proves ownership — not the parent chain, not
     the listener. A foreign or independently managed `llama-server --port N`
     has no recorded identity and is NOT matched and NOT reaped — if it is
     listening in the range it shows up in the diagnostics-only listener scan.
     If the manager crashes between spawning an engine and recording its
     identity, that engine is unrecorded and is likewise left alone. **A
     candidate lacking a matching recorded identity is NEVER reaped by the boot
     reaper (report-only).**

   **Schema migration (v1 → v2):** `state.sqlite` gains an `engine_starttime`
   INTEGER column on the `slots` table (idempotent `ALTER TABLE` guarded by
   `PRAGMA table_info`, so a pre-v2 DB is migrated on first open and a v2 DB
   is a no-op). `SCHEMA_VERSION` bumps 1 → 2. Pre-v2 rows get `engine_starttime
   = NULL`, which degrades the identity proof to report-only for those slots
   (a NULL starttime can never match a live one) — conservative, no stale
   pre-v2 engine is reaped until it is re-spawned under v2 and gets a record.
   The `known_active_pids` reconcile path is unchanged; the new
   `known_engine_identities` helper mirrors its state filter.

   **Cmdline alone does NOT prove ownership; neither does PPid=1.** A
   parent-chain heuristic would be insufficient (a foreign
   orphaned llama-server with PPid=1, or one launched by a process whose
   cmdline happens to carry a manager marker, would be reaped). The durable
   identity record is the ONLY reaping authority. The tests
   (`TestForeignLlamaServerOwnershipProof`, `TestDurableEngineIdentityProof`)
   demonstrate: a foreign `llama-server --port` in range with PPid=1 is NOT
   reaped (no recorded identity); a recorded Turbohaul-owned no-listener orphan
   IS reaped (identity matches); a PID-reused replacement (same pid, different
   starttime) is NOT reaped (starttime cross-check fails).

   A third DIAGNOSTICS-ONLY pass (`port_listeners_in_range`) populates the
   `stale_listeners` count so an operator can SEE a port in the managed range
   is still occupied (and by whom). It does NOT reap: reaping any listener in
   the range would risk foreign processes. `boot_reconcile` surfaces
   `stale_listeners` in its returned summary and in its audit event (under the
   key `stale_listeners_reaped`, which carries the same diagnostics count;
   nothing is reaped from this pass).

3. **Tests proving the failure paths** (`tests/test_lifecycle_hardening.py`):
   - `reap_orphan` sends `killpg` (SIGTERM + SIGKILL escalation), not PID-only
     `os.kill` (4 cases: group kill, SIGKILL escalation, already-gone,
     sigterm-clean).
   - `boot_reconcile` reaps a NO-LISTENER orphan (cmdline-identified
     `llama-server --port N` with a matching recorded identity, no PPid match,
     no listener socket) — the real seam, not a listener fixture. The detector is the cmdline-based
     `find_llama_servers_in_port_range`, not the socket scan.
   - `boot_reconcile` does NOT reap a FOREIGN process (nginx) listening in the
     managed range. The foreign listener appears in the diagnostics-only
     `stale_listeners` count but is never killed.
   - `boot_reconcile` does NOT reap a FOREIGN `llama-server --port N` in the
     managed range — including one that is ITSELF orphaned (PPid=1). PPid=1
     is NOT ownership proof; only a recorded `(pid, port, starttime)` identity
     in `state.sqlite` proves Turbohaul ownership
     (`TestDurableEngineIdentityProof`). A genuine Turbohaul-owned no-listener
     orphan whose recorded identity matches the live `/proc` starttime IS
     reaped, proving the ownership route works through the durable-identity
     check. A PID-reused replacement (same pid, different starttime) is
     NOT reaped.
   - `boot_reconcile` reports `stale_listeners` count (diagnostics).
   - A health-load timeout fails the request in bounded time (injected
     `_wait_healthy` returns False instantly → completion_future failed with
     `loading-fail-health-timeout`, `_active_handle` cleared). Proves the request
     does not wait silently for the full health-load timeout on a failed
     transition.

### Non-goals (explicitly deferred to PR 2 / PR 3)
- Lane partitioning (primary-vs-aux isolation) — see PR 2.
- Reasoning-display policy — see PR 3.
- Changing the `loading_health_timeout_s` default (60s) — the bounded failure
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

1. **Config surface.** Present in `QueueConfig`:
   - `main_lane_reserved: bool = True` — admit a queued main-lane request ahead
     of queued aux work.
   - `main_lane_identity_keys: list[str] = ["is_main"]` — the `client_meta`
     keys that classify a request as main-lane (interactive main agent). The
     `client_meta` identity plumbing already exists (`is_main`, `is_sub_agent`,
     `is_curator`, `is_compression` are read in `_emit_request_identity` and
     `kv_classify`).
   - An `aux_lane_max_inflight` knob (bounded aux concurrency alongside main) was
     sketched for this PR and is not implemented.

2. **Admission gate** (in `pop_next`): when `main_lane_reserved` is True and a
   main-lane request is queued, the oldest queued main-lane request is popped
   ahead of FIFO order and warm-model affinity (after any Fast Lane pick);
   aux-lane requests stay queued and are popped once no main-lane request is
   queued. This is admission-only: a running aux request completes normally.
   The `max_consecutive_same_model` / `max_other_model_wait_s` fairness knobs
   apply only on the path after this reservation, so they do not bound how long
   an aux request waits while main-lane traffic keeps arriving.

3. **No second sidecar required.** At `max_parallel_sidecars=1` (the
   default), the reservation is a queue-level admission control, not a
   second engine. A queued main request is admitted ahead of queued aux work;
   it does not interrupt a running request, and no parallel lane is created.
   This keeps the single-sidecar invariant intact.

4. **Tests** (`tests/test_queue.py`): a main request is popped ahead of a
   queued aux request and the aux request is popped next;
   `main_lane_reserved=False` preserves FIFO behavior (back-compat).

### Acceptance
- Queue an aux request and then an interactive main-agent request: the main
  request is admitted first, without interrupting a running aux request.

---

## PR 3 (separate, recommended) — reasoning-display policy

**Why it's a separate PR:** The reasoning-output policy is a config-level /
downstream concern. The raw reasoning-panel display issue is in
the downstream client, not Turbohaul's controller. Turbohaul already supports
reasoning via the chat-completion proxy: `_merge_reasoning_into_content` wraps
`reasoning_content` inline as `<think>...</think>` tags in `content`, and
separate helpers (`_strip_thinking_wrapper`, `_strip_thinking_all`) remove such
wrappers. A route-configurable
reasoning visibility flag is a small, additive config change but it belongs
with the downstream client change, not the Turbohaul lifecycle PR.

### Recommended approach
1. **Config surface.** Add to `QueueConfig` or a new `RoutePolicyConfig`:
   - `expose_reasoning: bool = False` (default off for interactive routes).
   - `expose_reasoning_route_allowlist: list[str] = []` — routes where reasoning
     IS exposed (e.g. a debugging route).
2. **Chat-completion proxy:** when `expose_reasoning` is False for the matched
   route, strip the `reasoning` field from the forwarded response (the
   existing `_merge_reasoning_into_content` path does not strip anything - it
   wraps reasoning inline as `<think>` tags and is skipped for `json_object` and
   `json_schema` requests - so this adds an explicit route-level gate). Internal reasoning support is NOT removed
   globally — the engine still reasons; the field is just not surfaced to the
   client.
3. **Tests:** a thinking-capable model on an interactive route returns a final
   response with no raw reasoning panel; a route in the allowlist still
   surfaces reasoning.

### Acceptance
- Request a thinking-capable model on an interactive route: the final response
  displays normally without a raw reasoning panel.

---

## Ordering
PR 1 (lifecycle) → PR 2 (lane isolation, built on the reconciled baseline) →
PR 3 (reasoning policy, can run parallel to PR 2). PR 1 is safe to merge
independently and unblocks PR 2's assumption of a trusted slot/port inventory.
