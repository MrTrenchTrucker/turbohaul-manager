# resident_identity

Remembers which request each running engine is serving, finds the engine that serves a given request, and says how much work an engine has pending.

## The problem it solves
The status display shows, for every running engine (a "resident"), the request it is serving. When two engines of one model
run side by side, the model name is no longer enough to tell them apart: each engine needs its own entry. This module keeps
that entry under the engine's own key, so the two never overwrite each other. It also answers "which engine serves this
request handle?" and "what is that engine's spawn counter?", which the live stream uses to name a generation.

When a new request has to be given to one of several engines of the same model, the manager prefers the one with the least
pending work. This module counts that work, so the rule lives in one place that a test can reach without the manager.

## How it is used
The manager owns a dict of last request identities and hands itself to the functions as the `host`.
- `stash_identity(host, r, slot, singleton_key)` is called when a request is bound to engine `r`. It builds the identity
  through the host and stores it under `r.resident_key`. An empty key and the legacy singleton key are skipped. A failure
  is logged at debug level and swallowed, because the entry is for display only and must never fail a request.
- `identity_for(host, r)` reads the entry for the status display.
- `forget_identity(host, r)` drops the entry when the engine is evicted. It removes only that engine's own key, so a sibling
  engine of the same model keeps its entry.
- `serving_resident(host, handle)` returns the first engine in the host's snapshot whose handle is the given one (compared
  by identity). A None handle, or a handle nobody owns, gives None.
- `spawn_seq_for_handle(host, handle, model_tag)` returns that engine's spawn counter, or the host's counter for the model
  when no engine owns the handle.
- `pending_load(r)` is the pending work of engine `r`: the requests riding along in flight, plus one when it is running a
  turn right now, plus the requests waiting in its inbox. An engine in its grace loop keeps the slot of the last request it
  served; while that slot is in its finished state the request is done and nothing is running, so the slot adds nothing. A
  running turn always counts, including a follow-up turn served inside the grace loop, and so do riders in flight and queued
  requests. The engine is only read, never changed.

## The host
The module imports nothing from the rest of TurboHaul. It is handed the manager and may use only the members listed in
`identity_host.py` (and in the card's Depends On). A test can therefore drive the whole module with a small fake host.

## Files
`identity_host.py` the host contract; `identity.py` store, read and forget the identity; `serving.py` the handle lookup and the spawn
counter; `engine_load.py` the load of one engine; `__init__.py` the public names.
