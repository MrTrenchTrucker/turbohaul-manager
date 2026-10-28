# state_db_cleanup

A one-time tool that returns the space old request bodies take in `state.sqlite`.

## What it does
Older versions of TurboHaul stored the whole request body in `slots.client_meta_json`, and on a busy deployment that one
column became most of the database. New requests store only the identity keys. The ended (`COLD`) rows written before
that change still hold full bodies, and deleting text from a SQLite file does not shrink it. This module reduces every
`COLD` row to the identity keys and then rebuilds the file so the freed space goes back to the filesystem.

## How it works
Run `python3 tools/state_db_cleanup.py --db PATH --backup-dir DIR`. Without `--execute` it is a dry run: it opens the
database read-only and reports what it would do and whether the free-space check would pass. With `--execute` and the
manager stopped, the steps run in this order, each one only if the one before it passed:
1. **preflight** checks the paths, the SQLite version and the owner, then takes exclusive access to the database.
2. **measure and plan** count the rows, hash every column except the stored value, and work out what the strip would do.
3. **space** checks, per volume and live, that free space covers the backup, the new file and a margin.
4. **backup** writes a consistent snapshot with the SQLite online backup API, and **verify** compares it with the
   database (schema, counts, hashes, `integrity_check`). Nothing is written to the database before this passes.
5. **strip** reduces `COLD` rows a batch at a time, with a short pause, one stored value in memory at a time.
6. **rebuild** writes a new compact file with `VACUUM INTO`, and **verify the new file** repeats the comparison and
   checks that every `COLD` value is identity-only.
7. **swap** replaces the database with the new file by an atomic rename.

The backup is private (mode 0600) and is checked by content, including a hash of the stored values. The swap happens
while the run still holds the exclusive lock.

A JSON report of the sizes, counts and checks is printed and written (mode 0600) next to the backup on every real-run
path: done, stopped, failed or interrupted. A real run also keeps an unfinished-run marker beside the database; it is
removed when the run completes or stops before writing anything, and it stays after a failure, so the next real run is
refused until the operator has read the report and deleted the marker. Exit status is 0 for done, 2 for a stop by a gate,
1 for a failure or unexpected error.

## What it does not do
- It does not decide which keys are identity keys or how a body is reduced; slot_meta_projection does.
- It does not stop or start the manager, and it never deletes a backup.
- It is not used by the running manager.

## Files
`config.py` settings and the stop signal; `world.py` the outside world; `connect.py` how the database is opened;
`measure.py` the measurements; `strip.py` choosing and reducing rows; `checks.py` the preflight and space gates and the
comparison of a copy; `backup.py`; `private.py` owner-only files; `marker.py` the unfinished-run marker;
`rebuild.py` checks on the new file; `swap.py`; `report.py` one recorded step and where the report file goes;
`sequence.py` the gated sequence; `cli.py` the command line.

## Who calls it
An operator, through `tools/state_db_cleanup.py`. The runbook is docs/STATE_DATABASE_MEMORY.md.
