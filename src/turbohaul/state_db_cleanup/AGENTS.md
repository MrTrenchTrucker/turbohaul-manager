# Module: state_db_cleanup

## Purpose
Return the space that old request bodies take in `state.sqlite`, once, safely. Older versions stored the whole request
in `slots.client_meta_json`. This module reduces the stored value of every ended (`COLD`) slot to the identity keys, then
rebuilds the file so the freed pages go back to the filesystem. It is run by an operator with the manager stopped, through
`tools/state_db_cleanup.py`. The manager never imports it.

## Owns
- The gated sequence of a run (`run`): preflight, exclusive lock, measure, plan, free-space gate, backup, verification of
  the backup, strip, check of the strip, rebuild with `VACUUM INTO`, verification of the new file, swap. Each step needs
  the one before it to have passed.
- The dry run: the same measure, plan and space gate on a read-only connection, with nothing written.
- The measurements the verification compares (row counts, counts by state, a hash of every column except the stored
  value, the schema, the total length of the stored values, `PRAGMA integrity_check`).
- The batched, per-value reduction of `COLD` rows (`strip.py`) and the JSON report of what was done.
- The files a real run leaves behind: the backup (owner-only), the JSON report beside it on every path (done, stopped,
  failed, interrupted), and the unfinished-run marker beside the database.
- The command line (`main`, `parse_args`) and its exit codes: 0 done, 2 stopped by a gate, 1 failed or unexpected error.

## Does Not Own
- Which keys are persisted and how a stored body is reduced to them: slot_meta_projection (`IDENTITY_KEYS`, `strip_stored_meta`).
- Opening the live database for the manager and its schema: state.py, legacy.
- The reduction when a slot ends and the background retention sweep: state.py and manager.py, legacy.
- Stopping and starting the manager, and deciding when the real run happens: the operator.
- The runbook the operator follows: docs/STATE_DATABASE_MEMORY.md.

## Public Interface
- `run(cfg, env=None) -> dict`: the whole sequence; returns the report. A failed gate returns a report with status
  `stopped`; any other error returns a report with status `failed` and a `failure` entry (step, type, message). Only an
  interrupt (`KeyboardInterrupt`, `SystemExit`) is raised again, after the report file has been written.
- `main(argv=None, env=None) -> int`: parse the command line, run, print and write the report, return the exit status.
- `parse_args(argv) -> Namespace`: the command-line options.
- `Config`: the settings of one run (database, backup directory, work directory, execute, swap, batch, pauses, margin).
- `Env`: everything the run reads from the outside world (free space, device, clock, sleep, SQLite version, the backup and
  rebuild functions, the owner id), so a test replaces one piece at a time.
- `Stop(step, message)`: a gate failed; `step` names the gate.

## Depends On
- slot_meta_projection: `IDENTITY_KEYS` and `strip_stored_meta` only.
- Standard library otherwise.

## Invariants
- Nothing is written to the database before the backup has been verified against the database it was taken from (the
  write-ahead log is first emptied into the database, which changes no stored data). The check includes a hash of the
  stored values themselves, not only their total length.
- The backup is owner-only (mode 0600): it is created empty and private before SQLite writes into it, a copy with any
  group or other permission bit stops the run, and a backup directory that group or others can write to is refused.
- Every real run ends with a report file (owner-only, beside the backup) whether it completed, stopped, failed or was
  interrupted, and the report names the backup. If the file cannot be written, the report is on stdout and one line on
  stderr names the backup and the marker.
- A real run creates an unfinished-run marker (`<db>.cleanup-unfinished`, beside the database, owner-only) before the
  backup. It is removed when the run completes or stops before any write; it stays after a failure, an interrupt, a kill,
  or a stop after the strip began. While it exists, a real run is refused at the preflight and nothing is touched. There
  is no resume: the operator reads the report, keeps the backup, and deletes the marker by hand.
- The swap (the rename over the database) happens while the run still holds the exclusive lock; the lock is released
  after the swapped file has been checked.
- The dry run opens the database read-only and writes nothing to it.
- Only `client_meta_json` of `COLD` rows is ever written. No row is deleted and no schema changes. A value that is not a
  JSON object is left exactly as it is and counted.
- The rebuilt file is a new file (`VACUUM INTO`), never an in-place `VACUUM`, and replaces the database only after every
  check of it has passed, by an atomic rename.
- The tool never deletes or overwrites a backup.
- Free space is measured live per volume, and the backup, the new file and the margin are summed per volume.
- A second run after a completed run strips nothing.
- Rows are handled a batch at a time, one stored value in memory at a time; a batch is committed whole or not at all.

## Test Locations
- Unit: tests/unit/state_db_cleanup/
- Contract: tests/contract/test_state_db_cleanup_contract.py

## Known Gotchas
- The manager must be stopped for the real run. The tool takes exclusive access and stops if anything else has the file
  open, even an idle connection.
- The tool reads the whole database several times (measure, plan, backup and its check, strip, check of the new file), and
  a real run also reads every stored value once more to hash it for the backup check, so on a large database it takes a
  long maintenance window.
- Stored values that are not JSON stay full size and are not sized in the report.
- A dry run on a write-ahead-log database can leave an empty `-wal` and `-shm` file beside it. They are harmless.
- Rowids may change in the rebuilt file, so the comparisons are order-independent.
- `os.replace` onto a single-file bind mount can fail with EBUSY. The swap reports that as a stop and leaves the verified
  new file and the backup in place.
- SQLite 3.27 or newer is needed for `VACUUM INTO`.
