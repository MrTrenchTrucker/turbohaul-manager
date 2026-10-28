# queue_liveness

Finds out whether the client of a waiting request is still there, and removes the requests whose client has gone.

## The problem it solves
A request can wait for a long time: in the staging queue, in the acceptance buffer, in the inbox of a busy engine, or
holding a Fast Lane claim. If its client hangs up meanwhile, nothing used to notice until the request was taken. It could
then still win an engine turn, make the box start a further engine, end a running engine's grace window, or keep a claim
that blocked a request whose client was still there. This module makes one rule for all of those places: a request whose
client has gone is not takeable, and it is removed at the source as soon as the hang-up is seen.

## How it is used
The manager hands itself to the functions as the `host`; the request queue and the claims stay where they are.
- `client_gone(slot)` is the shared answer: the slot is flagged evicted, or its disconnect event is set. Take paths ask it.
- `stamp_if_disconnected(slot)` raises the evicted flag when the event is set and returns the answer. It is exactly the
  two-line check the queue used to repeat inline.
- `drop_if_gone(host, slot)` stamps, and fails a gone slot once through the host. A second call, or a call that races an
  eviction, finds the completion future done and does nothing more.
- `evict_unstarted(host, slot)` is the source-first eviction of one waiting slot: it sets the event and the flag before any
  await, takes the slot out of a busy engine's inbox, releases its claim (the host moves the claim to the next waiter of the
  same client or drops it), removes the slot from the queue and the acceptance buffer outside the registry lock, and fails
  the completion future so the route answers "client closed request". A slot that has already begun, or was already failed,
  is left alone and the call returns False.
- `notify_client_gone(host, event)` is what a route calls when it sees the hang-up. It is synchronous, so a cancelled task
  can call it: it sets the event and schedules, on the host, the eviction of every waiting slot that carries that event.

## When a slot is taken out of an inbox
A slot is taken out of a resident's inbox only when that resident's driver is busy with a turn: its state is ACTIVE and
`active_slot` is set. `active_slot` is set when a turn begins and cleared in the same locked step in which the driver parks
the engine, and the engine has no separate grace state, so a set `active_slot` means the driver will come back to its own loop.

A parked driver is different. A producer flips an idle engine to ACTIVE before it puts a request into the inbox, and only
the driver's loop flips it back. If the last queued item were taken away from a parked driver, the engine would stay ACTIVE
with an empty inbox: it would never unload when idle, never be chosen to make room, and hold its memory. The same holds for
an engine still loading, whose first request sits in the inbox until the driver takes it. For these the slot stays in the
inbox, flagged evicted, its claim released and its future failed; the driver's own drain discards it.

## Idempotence
Failing a request twice would double the audit record and the counters. `drop_if_gone` and `evict_unstarted` therefore look
at the completion future first and do nothing when it is done. A request without a completion future has nothing to look at
and is failed on every call.

## The host
The module imports nothing from the rest of TurboHaul. It is handed the manager and may use only the members listed in
`liveness_host.py` (and in the card's Depends On). A test can therefore drive the whole module with a small fake host.

## Files
`liveness_host.py` the host contract; `liveness.py` the predicate, the stamp and the one-shot fail; `eviction.py` the
source-first eviction and the notify entry; `__init__.py` the public names.
