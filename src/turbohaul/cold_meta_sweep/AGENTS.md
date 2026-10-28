# Module: cold_meta_sweep

## Purpose
Reduce, in the background and in small steps, the stored metadata of old ended slots to their identity keys, so an old
database sheds the request bodies it still holds. The module is the manager-facing side of that pass: one bounded run over
the table, and the decision whether a run is due now, with its single log line. Everything it touches is handed to it.

## Owns
- One bounded run of the pass (`run_once`): picking up to a batch of ended slots older than a given age, in row order after
  a watermark, from cheap columns only; calling the strip for each; counting the outcomes; and returning the counts with the
  watermark for the next run.
- The decision whether a run is due (`maybe_sweep`): the setting is on, the idle time after a full pass has passed. It runs
  the work in a worker thread, updates the manager's timer and lifetime counters, and writes the one log line (warning if any
  row was invalid or failed, info otherwise, nothing if nothing happened).

## Does Not Own
- Which keys are kept and how a stored body is reduced to them: slot_meta_projection.
- The strip of one row and its outcome counts (`strip_slot_meta`): state.py, legacy.
- The setting `kv.cold_meta_purge_age_hours` and its validation: config.py.
- The sweeper tick that calls the pass, the batch size, the idle time, the watermark, the timer and the lifetime counters:
  manager.py, legacy. The manager keeps them and hands them in.
- The boot census and the slot-end strip.

## Public Interface
- `run_once(session, strip_one, db_path, watermark, age_hours, batch) -> (counts, new_watermark)`: `session` opens the state
  database as a context manager (the manager's `state_db_session`), `strip_one(conn, slot_id)` returns one of `stripped`,
  `unchanged`, `invalid`, `missing`, `error`. `counts` has those five keys plus `rows` and `pass_done`. The new watermark is
  0 when the run reached the end of the table, else the last row handled.
- `maybe_sweep(host) -> dict | None`: async. Returns the counts of the run it started, or None when the pass is off or not
  yet due. `host` is the manager.

## Depends On
- Nothing inside TurboHaul. Standard library only. Everything else arrives as an argument.

## Invariants
- The watermark is returned, never stored here. If the run fails part way, the caller's watermark is unchanged.
- A row is picked only if it is ended (`COLD`) and its `updated_at` is older than the cutoff; nothing else is touched.
- Bodies are read one at a time, by the strip, never as a list.
- The batch size, the idle time and the timer are read from the manager at call time, so a change applies on the next run.
- The database work is handed to `asyncio.to_thread` as the manager's own `_cold_meta_sweep_once`.
- The log line goes to the manager's logger (`turbohaul.manager`) with a fixed text; no line is written when nothing was
  stripped, invalid or failed.
- No state of its own: no module-level counters, no cache.

## Test Locations
- Unit: tests/unit/cold_meta_sweep/
- Contract: tests/contract/test_cold_meta_sweep_contract.py
- The manager's use of the module (legacy code): tests/test_cold_meta_retention_sweep.py

## Known Gotchas
- `run_once` is blocking. The manager wrapper `_cold_meta_sweep_once` is what runs in the worker thread.
- A run with the idle timer set is skipped silently; the timer is only set after a run that reached the end of the table.
- A row whose body is not a JSON object is counted as `invalid` on every full pass, and so logged as a warning each time.
