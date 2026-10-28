# The state database and the manager's memory at boot

TurboHaul keeps its state in one SQLite file, `state.sqlite`. This page explains what the file holds about requests, what
the manager reads back from it when it starts, and why that read used to be able to fill the manager's memory. If the
manager's resident size is large right after a restart and before any traffic, start here.

## What the database holds about requests

Every request becomes a **slot**, and a slot is one row in the `slots` table. A row has plain columns (the slot id, the
thread id, the model tag, timestamps, the engine port and process id, the state) and one text column,
`client_meta_json`, that holds the request's client metadata as JSON. A second table, `audit_events`, holds one small row
per event and is not where the size comes from. The file lives on the `state` mount, not in RAM.

Before v0.8.1, `client_meta_json` held the **whole request body**: every message, the system prompt, the tool list and the
sampling options. Request bodies are large and every request adds one, so on a busy deployment that single column
became most of the database. The manager also rewrote it on every state change of the slot.

## What the manager reads at boot

When the manager starts, the Fast Lane **census** is rebuilt from the `slots` table, so the Discovered list on the Fast
Lane screen survives a restart (see [FAST_LANE.md](FAST_LANE.md)). The census needs very little from each row: the
client's address, the thread id, the model tag, the time, and the few labels that decide the request's class (`is_main`,
`is_sub_agent`, `is_curator`, `is_compression` and `role`).

The rebuild used to select `client_meta_json` for every row and parse it in the manager. That loaded the stored
conversation text of every request ever made into the manager's memory, once, at boot, to read an address out of each.
Python and the C allocator under it can keep freed memory around for reuse rather than returning it to the operating
system, so the process can stay large long after the rebuild has finished. Under a container memory limit, the cold part
of that memory can end up in swap.

The rebuild now asks SQLite for only the address and the five class keys of each row, as one small value per row. The
request bodies never reach the manager. The census it builds is the same as before.

## What is stored now

For new requests the manager stores only the identity of the request in `client_meta_json`:

| Key | Meaning |
|---|---|
| `ip` | The client address the request came from |
| `session_id` | The harness's session identifier, when it sent one |
| `role` | The agent role, when the harness sent one |
| `is_main`, `is_sub_agent`, `is_curator`, `is_compression` | The class labels |
| `save_kv` | The per-role KV-save toggle |

The request body is not stored. A running request still carries its full client metadata in memory, which is what the
forwarding path reads, so nothing about serving a request changes.

When a slot ends, the manager also reduces whatever its row holds to those same keys. A row written by an earlier
version therefore sheds its body the next time its slot ends. A stored value that is not a JSON object is left exactly as
it is; the manager counts it and logs it, and never raises, so ending a slot does not depend on the reduction.

## Older rows and boot time

Rows written by v0.8.0 and earlier still hold full bodies until they are reduced: when their slot ends, by the background
pass below, or by the one-time cleanup further down. Until then they keep working: the census reads the same keys from
them. Reading them is cheaper in memory but not free in time. SQLite still has to read each stored body to pull the keys
out of it, so the time the rebuild takes grows with the size of the database until old bodies are reduced or the address
becomes a column of its own. The address is not yet a column of its own.

To see how much of a database is request text, a read-only query is enough. It reads the whole table, so on a large
database it takes a while:

```
sqlite3 state.sqlite "SELECT count(*), sum(length(client_meta_json)) FROM slots"
```

Do not bulk-update or delete rows in a database a running manager is using.

### Reducing old ended rows in the background

`kv.cold_meta_purge_age_hours` turns on a background pass that reduces the stored metadata of old ended (`COLD`) slots to
the identity keys above. It is **off by default** (`0`). Rows written before the identity-only change keep their full
bodies until an operator sets a positive number of hours, so the first run over an old database happens when you choose
it, not at an upgrade. The value is read on every run, so a change applies without a restart. Negative values are
rejected, and there is no environment variable for it.

The pass runs from the manager's periodic sweeper, in a worker thread, and is built to stay out of the way:

- It touches `COLD` rows whose `updated_at` is older than the setting. Slots in any other state, and ended slots newer
  than the setting, are never touched.
- It handles a fixed number of rows per run, in `rowid` order, and remembers where it stopped. After it has reached the
  end of the table it waits an hour before starting over, so a settled database is not rescanned continuously.
- Rows are chosen from small columns only, and each stored body is read on its own, so the pass never holds many large
  bodies in memory.
- Only `client_meta_json` is written. No row is deleted, and `updated_at` and every other column keep their values, so
  nothing about how old a slot looks changes.
- A body that is not a JSON object is left alone, counted, and reported in the run's log line.

Reducing rows does **not** shrink the database file. SQLite keeps the freed pages for reuse, so the file only gets
smaller after a `VACUUM`, which needs the manager stopped and a backup first. The pass is what makes the bodies small
enough for the rebuild to read quickly; the file size is a separate step.

## Cleaning out the old rows once, and getting the space back

The background pass above reduces old rows but cannot shrink the file. For a database that already holds a great
deal of old request text, `tools/state_db_cleanup.py` does the whole job once: it reduces every `COLD` row to the
identity keys, then rebuilds the file so the freed space goes back to the filesystem. Nothing is deleted, no schema
changes, and a value that is not a JSON object is left exactly as it is and counted. Such a value stays full size, and
the report does not size it, so if the database holds many of them the file will shrink less than the report's figures
for the other values suggest.

`tools/state_db_cleanup.py` is a thin launcher for the `state_db_cleanup` module (`src/turbohaul/state_db_cleanup/`, see its
README). The manager does not import it.

The manager must be **stopped** for the real run. The tool takes exclusive access to the database file and refuses
to go on if anything else has it open.

**1. Dry run (the default; safe to repeat).** It opens the database read-only and reports what would happen: rows by
state, how many values would be reduced, how many left alone, how much text would go, the expected size of the new
file, and whether the free-space check would pass.

```
python3 tools/state_db_cleanup.py --db /path/to/state.sqlite --backup-dir /path/to/backups
```

The backup directory must exist, must not be the database's own directory, and must be on a volume with room. For a
real run it must also not be writable by group or others (for example `chmod 700`): the backup is a copy of the whole
database, and another user who can write in that directory could replace it. A real run refuses such a directory at the
preflight, before anything is written; a dry run does not check it. A dry run can leave an empty `state.sqlite-wal` and a `state.sqlite-shm` beside the database; they are harmless.

**2. The real run.** Add `--execute`. The steps run in this order, and each one needs the one before it to have passed:

| Step | What it does |
|---|---|
| preflight | checks the paths, the SQLite version and exclusive access, and measures free space on the database, backup and work volumes: each must hold its share plus a margin (5 GiB by default) |
| backup | writes a consistent snapshot of the database into the backup directory. The file is created empty with mode 0600 (owner only) before SQLite writes into it, and a backup with any group or other permission bit stops the run |
| verify the backup | compares size, schema, row counts, the total length of the stored values, a hash of every other column, a hash of the stored values themselves (read only if the total lengths match), and `PRAGMA integrity_check`. **No row is written to the database before this passes.** One thing runs first: a write-ahead-log checkpoint moves already-committed data from the log into the main file so the copy is complete. It changes no row and is harmless |
| strip | reduces `COLD` rows, a batch at a time with a short pause, one value in memory at a time; only `client_meta_json` is written |
| rebuild | writes a new, compact file with `VACUUM INTO`. An in-place `VACUUM` is not used: it needs about twice the database's size free |
| verify the new file | integrity check, schema, row counts, hashes of every other column, the stored values, and that every `COLD` value is identity-only |
| swap | replaces the database with the new file by an atomic rename, only if everything above passed. The tool still holds the exclusive lock at that moment and lets go of it only after the swapped file has been checked, so nothing can open the old file between the last check and the rename |

A JSON report (sizes, counts, the steps taken and the checks made) is printed and also written, with mode 0600, next to
the backup. It is written on every real-run path: done, stopped, failed or interrupted, and it names the backup. If the
report file cannot be written (disk full, a missing directory given with `--report`), the result is not lost: the report
is on stdout, and one line on stderr beginning `REPORT NOT WRITTEN` names the backup and the marker below.
The run reads the database several times (to measure it, to plan, to copy and check the backup, to reduce, and to
check the new file). A real run also reads every stored value one more time, to hash it for the backup check, so the
window that already held six reads of the database now holds a seventh read of its largest column. On a large database
allow for a long maintenance window; progress lines are printed as it goes.

**3. If it stops or fails.** A failed check ends the run with a message saying which one, and nothing after it is
attempted. Exit status is 2 for a stop, 1 for a failure (an error the tool did not expect, such as a full disk) or any
other error, 0 otherwise. A failure is not a traceback: the report has `status` set to `failed` and a `failure` entry
with the step, the error type and the message. Before the strip starts nothing in the database has changed. After it
starts, the verified backup is the way back to the original. **Keep that backup**: it is the only copy of the old
request text, and the tool never deletes a backup.

**The unfinished-run marker.** Just before the backup, a real run creates a small file beside the database,
`state.sqlite.cleanup-unfinished` (mode 0600), that names the report file and the planned backup. The tool removes it
when the run completes, or stops before it has written anything. It stays when a run fails, is interrupted, is killed,
or stops after the strip began. While it exists, a real run is refused at the preflight and touches nothing, whichever
backup directory you give; a dry run ignores it. There is no resume option. If a run fails: read the report, keep the
backup, and delete the marker only after you have looked at them. Then run again; a second run after a partial strip
finds less to reduce and is safe to repeat.

**Options.** `--min-age-hours N` limits the run to rows not updated for N hours (default: every `COLD` row).
`--no-swap` stops once the new file has verified and leaves the swap to you. `--work-dir` chooses where the new file is
written; it has to be on the database's filesystem so the swap is a rename. Keep the manager stopped until the tool
reports `completed`.

## When the rebuild cannot run

The census rebuild is best-effort. If it fails, the manager logs the error and starts with an empty census; the list fills
again as traffic arrives. The projection needs SQLite's JSON functions, which SQLite includes by default from version 3.38.

A row whose stored text is not valid JSON, is valid JSON but not an object, or contains `NaN` or `Infinity` is treated as
a row with no address and is skipped. Skipped rows are counted. They no longer stop the rest of the rebuild.
