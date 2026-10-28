# Module: resident_identity

## Purpose
Remember which request each running engine (a resident) is serving, so the status display can show it, and find the
resident that serves a given request handle. When two engines of one model run side by side, each one shows its own last
request: the identity is kept under the resident's own key, never under the model tag alone. It also gives the load of one
resident, the number the manager uses to pick an engine for a new request. The module is a small set of plain functions
that read and write what the manager hands them.

## Owns
- The bookkeeping of the last request identity per resident: store it (`stash_identity`), read it (`identity_for`) and
  forget it (`forget_identity`), always keyed by the resident's own key.
- The lookup of the resident that serves a request handle (`serving_resident`) and the spawn counter of that resident
  (`spawn_seq_for_handle`), with the fall-back to the model's counter when no resident owns the handle.
- The load of one resident (`pending_load`): riders in flight, plus one for a running turn, plus the inbox depth, where the
  finished last-served slot of a resident in its grace loop (the slot is in its finished state) is not a running turn.

## Does Not Own
- The storage and lifetime of the identity dict: the manager creates it, may replace it, and decides when a resident is
  evicted and so when its entry is forgotten.
- Building the identity dict (`_identity_dict`): the host builds it; the module only passes it the fields of the request.
- The resident registry and `_model_residents`: the manager owns the registry and the snapshot, which leaves out the legacy
  singleton and dead residents.
- The generation blocks and their key (`live_monitor`), and the status snapshot shape.
- The legacy singleton key constant: the caller passes it in.
- Where the calls are made (the driver's bind step, the status snapshot, the eviction path, the stream tee, the engine pick).
- Choosing the engine: the pick key (readiness, then the load, then the oldest last activity) and the affinity record stay
  in the manager; the module only answers the load number.
- The resident's fields themselves (`inflight`, `active_slot`, `in_grace_loop`, `inbox`) and when they change.

## Public Interface
- `ResidentIdentityHost`: the host contract (see Depends On).
- `ResidentLike`: the resident attributes the module reads: `resident_key`, `model_tag`, `handle`, `spawn_seq`.
- `SlotLike`: the request attributes the identity is built from: `client_meta`, `model_tag`, `thread_id`, `fastlane`.
- `stash_identity(host, r, slot, singleton_key) -> None`: store the identity of the request just bound to resident `r`
  under `r.resident_key`. An empty key and `singleton_key` are skipped. Any failure is swallowed.
- `identity_for(host, r) -> dict | None`: the stored identity of `r`; a resident with no key reads its model tag's entry.
- `forget_identity(host, r) -> None`: drop the entry under `r.resident_key`, and only that one.
- `serving_resident(host, handle) -> ResidentLike | None`: the first resident of the host's snapshot with `r.handle is
  handle`; None for a None handle or no match.
- `spawn_seq_for_handle(host, handle, model_tag) -> int`: the serving resident's `spawn_seq`, else the host's counter for
  `model_tag`.
- `LoadLike`: the resident attributes the load reads: `inflight` and `active_slot`, and the optional `in_grace_loop` and
  `inbox`; the active slot is read for its optional `state`.
- `pending_load(r) -> int`: `len(r.inflight)`, plus 1 when `r.active_slot` is set unless `r` is in its grace loop and the
  slot's `state` is the finished state (the string "GRACE"), plus the depth of `r.inbox` (`qsize()`). Read-only; takes no
  host.

## Depends On
- Nothing inside TurboHaul: no module file imports the project. The module is handed one `host` (the manager) and may touch
  only the members declared in `identity_host.py`; the contract test records every member used on a recording host and fails on an
  undeclared one, and on a declared one that no scenario uses.
- Host attribute: `_last_request_identity_by_model` (a dict of identity dicts by resident key; read at every call, never
  cached, because the manager may replace it).
- Host methods: `_identity_dict(*, ip, model_tag, client_meta, thread_id, fastlane=None)`, `_model_residents()` and
  `_spawn_seq_for_model(model_tag)`.
- Resident attributes read: `resident_key`, `model_tag`, `handle`, `spawn_seq`. Request attributes read: `client_meta`,
  `model_tag`, `thread_id`, `fastlane`.
- `pending_load` takes no host. It reads four attributes of the resident it is handed: `inflight` and `active_slot` (always
  present), and `in_grace_loop` and `inbox` (optional: missing reads as False and as no inbox, so a test fake without them
  works). The inbox is only asked for its `qsize()`. From the active slot it reads one optional attribute, `state`, and
  compares it with the plain string "GRACE" (a string enum equals its value, so the module imports nothing); a slot with no
  `state` counts as running.

## Invariants
- The legacy singleton key and an empty key are never stored: `stash_identity` returns before it builds an identity or
  touches the dict.
- A failed stash never raises and leaves the dict unchanged. The key read is inside the guard too. It is logged at debug
  level on the module logger with the traceback. Only `Exception` is swallowed.
- Forgetting one resident never removes a same-model sibling's entry: `forget_identity` pops exactly the resident's own key
  and never looks at the other residents.
- The identity is stored and forgotten under the resident's key, never under the model tag. Only a resident with no key
  reads the model tag's entry.
- The handle lookup is identity based (an equal but different handle does not match), takes the first match of one pass over
  the snapshot, matches nothing for a None handle, and is read-only.
- When no resident owns the handle, `spawn_seq_for_handle` falls back to the host's counter for the model tag. When one
  does, it returns that resident's counter, even when it is 0, and does not ask the host for the model's.
- The functions read the dict from the host at call time: a dict swapped in after the host was built is the one used.
- The active slot counts 0 only when both hold: the resident is in its grace loop and the slot is in its finished state
  ("GRACE"), which means the request is already served and nothing is running. Every other active slot counts 1: a live turn
  outside the grace loop, a follow-up turn served inside the grace loop, the finished state outside the grace loop, and a
  slot with no `state`. Riders in flight and the inbox depth count in every case.
- `pending_load` is read-only, returns an `int`, and reads `in_grace_loop` and `inbox` with defaults (False and None) the same
  way the manager always has, so a resident built without them still has a load.
- Every file is under 300 lines.

## Test Locations
- Unit: tests/unit/resident_identity/ (shared fakes: tests/_resident_identity_fakes.py; `pending_load` is in
  test_engine_load.py and uses its own small fakes)
- Contract: tests/contract/test_resident_identity_contract.py

## Known Gotchas
- A resident that has no handle yet (idle) has `handle` None. A None handle must not match it, so the None check comes
  before the scan.
- `_model_residents()` is a snapshot that leaves out dead residents and the legacy singleton. A request on one of those has
  no owner, and `spawn_seq_for_handle` then answers with the model's counter.
- Two residents of one model may be asked for their identity with the same `model_tag`; only the key tells them apart. A
  resident with an empty key is the exception and reads the model tag's entry.
- The module does not remove entries for residents that vanish without a `forget_identity` call; the manager decides when
  to forget.
- A resident in its grace loop keeps its finished last-served slot as `active_slot` until the next request is bound. That
  slot is not work, so `pending_load` must not count it; counting it makes a waiting engine tie with one that is running a
  turn, and a new request can be sent to the busy engine.
- `in_grace_loop` is True for the whole grace window, which includes a follow-up turn served inside it, so the flag alone
  cannot tell a finished slot from a follow-up being served. The slot's state does: a finished slot is in the state "GRACE",
  a follow-up turn is in another state, so a resident serving a follow-up still counts 1 and an idle engine is picked
  before it.
