# cold_meta_sweep

Reduces the stored metadata of old ended slots to their identity keys, a few hundred rows at a time, in the background.

## What it does
Every request becomes a slot row, and older rows still hold the full request body, which is most of the database. New
slots keep only their identity keys. This module is the part of the manager that works through the old rows. It is off
until the setting `kv.cold_meta_purge_age_hours` is above zero.

## How it works
1. **Decide.** `maybe_sweep` runs on every tick of the manager's sweeper. It reads the setting fresh. If the setting is 0, or
   the quiet time after the last full pass has not passed, it returns nothing.
2. **Run.** Otherwise it hands the manager's `_cold_meta_sweep_once` to a worker thread, which calls `run_once`. `run_once`
   picks up to one batch of ended slots older than the age, in row order after where the last run stopped, and strips each one
   through the function it is given. Only the rows are picked from cheap columns; each body is read alone.
3. **Remember.** `run_once` returns the counts and the place to resume from. A run that reached the end returns 0, so the next
   pass starts over, and the manager then waits an hour before starting it.
4. **Report.** `maybe_sweep` adds to the manager's lifetime counters and writes one log line when anything was stripped, was
   invalid or failed.

## What it does not do
- It does not decide what is kept (slot_meta_projection) or how one row is rewritten (state.py).
- It does not own the setting, the batch size, the timer or the counters. The manager does, and passes them in.
- It never deletes a row, and it changes only the metadata column.

## Who calls it
manager.py: the two methods `_cold_meta_sweep_once` and `_maybe_sweep_cold_meta` are thin wrappers over it, and the periodic
sweeper awaits the second on every tick. The public names are listed in AGENTS.md.
