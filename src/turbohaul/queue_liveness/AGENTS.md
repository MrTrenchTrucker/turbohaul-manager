# Module: queue_liveness

## Purpose
Know whether the client of a waiting request is still there, and get a request whose client has gone out of the system.
A request can wait in the staging queue, in the acceptance buffer, in the inbox of an engine, or hold a Fast Lane claim.
When its client hangs up meanwhile, the request must not be served, must not make the box start an engine, and must not
keep a claim that blocks a live request. The module gives every place that can take a waiting request ONE shared answer
(`client_gone`), the one-shot failure of a gone request (`drop_if_gone`), the eviction of a waiting request at the
source (`evict_unstarted`) and the synchronous entry a route calls when it sees the hang-up (`notify_client_gone`).
The module is a small set of plain functions that read and write what the manager hands them.

## Owns
- The liveness predicate (`client_gone`): the slot is flagged evicted, or its disconnect event is set.
- The stamp (`stamp_if_disconnected`): raise the evicted flag when the event is set, never clear it.
- The one-shot fail (`drop_if_gone`): stamp, and fail a gone slot once through the host.
- The source-first eviction of one waiting slot (`evict_unstarted`): flags, claim release, removal from the queues and
  from a busy engine's inbox, failure of the completion future.
- The synchronous notify entry (`notify_client_gone`) and the lookup of the waiting slots that carry a given event.

## Does Not Own
- The queues, the inbox storage and the claims: the queue and the manager own them; the module removes a slot through the
  host and asks the host to release the claim (the host moves it to the next best waiter or drops it).
- The locks: the host owns the registry lock; the module takes the one the host hands it.
- How the manager fails a slot (its audit, its counters): the module calls the host.
- The route code that sees the hang-up and calls `notify_client_gone`.
- The Fast Lane rank logic and the engine reservation.
- The meaning of a slot's states beyond the three that mean "not started yet".

## Public Interface
- `QueueLivenessHost`: the host contract (see Depends On).
- `SlotLike`: the request attributes the module reads and writes: `slot_id`, `state`, `disconnect_event`, `is_evicted`,
  `completion_future`.
- `ResidentLike`: the engine attributes the module reads: `state`, `active_slot`, `inbox`.
- `client_gone(slot) -> bool`: True iff `slot.is_evicted` or the disconnect event is set. Pure read, await-free.
- `stamp_if_disconnected(slot) -> bool`: set `slot.is_evicted = True` when the disconnect event is set (nothing else
  changes); returns `client_gone(slot)`. Await-free.
- `drop_if_gone(host, slot) -> bool`: `stamp_if_disconnected`; when the client is gone, fail the slot once through
  `host._handle_unloaded_slot` unless its completion future is already done. Returns True iff the client is gone. Await-free,
  idempotent. Does not touch queues, inboxes or claims.
- `evict_unstarted(host, slot) -> bool` (async): evict one waiting slot at the source. Returns True iff this call evicted
  it; False for a slot that has begun, or was already failed. Idempotent.
- `notify_client_gone(host, event) -> None`: set the event and schedule, on the host, the eviction of every waiting slot
  that carries it. Synchronous, never awaits, safe in a cancelled task, idempotent.

## Depends On
- Nothing inside TurboHaul: no module file imports the project. The module is handed one `host` (the manager) and may
  touch only the members declared in `liveness_host.py`; the contract test records every member used on a recording host
  and fails on an undeclared one, and on a declared one that no scenario uses.
- Host attributes: `queue` and `_registry_lock` (an async context manager).
- Queue members: `waiting_snapshot()` (await-free list of the staging queue and the acceptance buffer, live slot objects) and
  async `remove(slot_id)` (takes the slot out of either buffer).
- Host methods: `_model_residents()`, `_release_fastlane_claim_locked(slot, reason)` (caller holds the registry lock),
  `_handle_unloaded_slot(slot)` and `_spawn_bg(coro)`.
- Resident attributes read: `state` (its `name`), `active_slot`, `inbox` (an `asyncio.Queue`; only `get_nowait` and
  `put_nowait` are called). Request attributes read: `slot_id`, `state` (its `name`), `disconnect_event`, `is_evicted`,
  `completion_future` (its `done()`).

## Invariants
- `client_gone` changes nothing. `stamp_if_disconnected` writes only `is_evicted = True`, and only when the event is set;
  it never clears the flag.
- `drop_if_gone` and `evict_unstarted` fail a slot at most once: a slot whose completion future is already done is not
  failed or audited again.
- `evict_unstarted` does nothing for a slot that has begun (a state other than RECEIVED, ACCEPT_BUFFER or STAGED, compared
  by name) or whose completion future is done, and returns False.
- The event and the evicted flag are set before the first await, so no take path can pick the slot while it is being removed.
- A slot leaves a resident's inbox only when that resident is mid-turn: its state is ACTIVE and its `active_slot` is set.
  For a parked driver, or an engine still loading, the slot stays in the inbox, flagged and failed, and the driver discards it.
- The claim is released in every case, under the registry lock, through the host. The module has no claim logic of its own.
- The queue removal runs outside the registry lock.
- `notify_client_gone` never awaits and never raises for a missing running loop or a failed schedule.
- One failing eviction does not stop the eviction of the other slots that carry the same event.
- Every file is under 300 lines.

## Test Locations
- Unit: tests/test_queue_liveness_unit.py
- Contract: tests/contract/test_queue_liveness_contract.py

## Known Gotchas
- Removing the last queued item from the inbox of a PARKED driver wedges that engine: a producer flips an idle engine to
  ACTIVE before it puts, and only the driver's own loop flips it back. That is why the inbox removal is conditional.
- A slot already handed to a parked getter (taken off the inbox, not yet begun) cannot be reached. It is served as before;
  this is a known residual.
- A hang-up seen before the manager attached the event to a slot finds no slot to evict. The event is set, and the take paths
  check it later.
- A request with no completion future is not protected by the "already failed" check; it is failed on every call.
- The inbox is read and edited by draining it with `get_nowait` and putting the items back in order, all without an await.
