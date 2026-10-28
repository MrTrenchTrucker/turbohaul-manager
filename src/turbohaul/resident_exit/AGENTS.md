# Module: resident_exit

## Purpose
Decide whether a finishing same-model co-resident stays warm for a bounded window instead of being torn down at once, and
name every resident death with exactly one exit-class log line that carries the cause the caller gives.

## Owns
- The keep-warm decision for a resident whose request just ended on a box that allows more than one engine: its rules, the
  order they are checked in, and the reason each answer carries.
- The exit line: its single text format, its field order, and the cleaning and cut rule for each field.
- The rule for which deaths get that line: every cause except MAKE_ROOM.
- The `ExitReason` names, the text a log reader sees for each cause.

## Does Not Own
- Measuring the inputs: the idle window, the live sibling count, the designated victim, outranking claims, capacity demand,
  the engine budget. The manager measures them under its registry lock and passes plain values in.
- Evicting, deregistering or parking a resident (IDLE_EVICTABLE and its deadline): manager.py.
- The make-room eviction line (MAKE_ROOM_EVICTION) and every other log line: manager.py writes all logging.
- Whether a KV save is allowed for a disposable identity: manager.py.
- Engine placement, cards and the engine count: engine_budget and manager.py.

## Public Interface
- `ExitReason`: the causes a resident death can carry, as text. Each member and its meaning are in reasons.py.
- `needs_exit_line(reason) -> bool`: False only for MAKE_ROOM, whether passed as the enum member or as the plain string.
- `format_resident_exit(*, reason, resident_key, model_tag, thread, port, state_before, state_after) -> str`: the one exit
  line, with no trailing newline.
- `RetentionInputs`: the measured inputs for the keep-warm decision, keyword-only, every field required.
- `RetentionDecision`: the answer: whether to hold, the idle window to use, and the rule that decided it.
- `decide_retention(inputs) -> RetentionDecision`: the keep-warm decision.

## Depends On
- Nothing inside TurboHaul. Standard library only: plain values in, a decision or a line out.

## Invariants
- Pure: no I/O, no logging, no locks, no globals, no live state, no imports from turbohaul. The caller measures and logs.
- The decision order is fixed and the first match wins: cap_le_1, idle_window_positive, designated_victim,
  outranking_claim, capacity_demand_pending, no_active_sibling, no_hold_window, and only then the hold
  co_resident_active_sibling. Every answer that is not a hold returns the caller's own idle window unchanged.
- A cap below 2 never holds, because cap_le_1 is checked first: a cap of 1 gives the same answer as before this module.
- The identity class of the finished request is never an input. A disposable identity decides only whether its KV cache is
  saved, never whether the resident lives.
- The hold is bounded: it returns a finite window that the caller parks with a real deadline, so make-room can still evict
  the resident for a higher-ranked waiter.
- The exit line is one line. Its fields come in a fixed order, reason first, then resident_key, model_tag, thread, port,
  state_before and state_after. None, an empty value, or a value that is only whitespace is "-", whitespace runs become one "_", and the thread is cut at 60
  characters.
- One exit line per death, written by the caller at its DEAD transition. Never for MAKE_ROOM.

## Test Locations
- Unit: tests/unit/resident_exit/
- Contract: tests/contract/test_resident_exit_contract.py
- Manager level (the legacy wiring): tests/test_resident_exit_guards.py and
  tests/test_resident_exit_wiring.py

## Known Gotchas
- manager.py (legacy) keeps the registry lock, every live-state read, the parking, the make-room line and all logging.
  This module only decides and formats.
- `_begin_unload_locked` is not the only death site. The finally block of `_drive_resident` also marks a resident DEAD and
  removes it. Both must log, each only at its own DEAD transition, so one death never logs twice.
- `_drive_resident` is the live path at every legal cap. The `max_parallel_sidecars >= 2` check inside decide_retention is
  what keeps a cap of 1 unchanged; there is no separate code path for it.
- This module covers the keep_alive 0 immediate-evict path; other unload triggers are not decided here.
- "Card has room" is not an input. Holding keeps capacity the resident already has and cannot add any. The question is only
  whether someone is waiting for that capacity, which outranking_claim and capacity_demand_pending carry.
- The exit line cuts the thread at 60 characters, but IDLE_TEARDOWN keeps its thread_hash untruncated. For a thread id longer
  than 60 characters the two lines do not match on the thread.
