# Engine Lifecycle Ownership

How Turbohaul-Manager decides that an inference engine process belongs to it, and
what it will and will not kill.

## The problem

Turbohaul spawns `llama-server` processes on a managed port range and has to clean
up the ones it leaves behind — after a crash, after a model swap, after a handle is
dropped without a clean shutdown. The question "is this process mine?" has to be
answered before anything is signalled.

Answering it from the process tree does not work. A spawned engine is detached into
its own process group, and its parent is init whenever the manager is itself PID 1
in its container or has exited, so a **healthy, serving engine can have init as its
parent** and look exactly like an orphan. Having init as the parent only proves that
the parent is init or that *some* parent exited, not that the parent was Turbohaul. A port scan is worse: an unrelated process listening
on a managed port is indistinguishable from an engine by port alone.

Both of those are heuristics, and neither survives the case that matters — a manager
that crashed, whose process identity is gone.

## The rule

**An engine is Turbohaul's only if Turbohaul wrote down that it spawned it.**

At spawn time the manager records a triple in its state database:

| field | meaning |
|---|---|
| `pid` | the engine process id |
| `port` | the port it was given |
| `engine_starttime` | the process start time, in clock ticks since boot |

Cleanup then works in two stages:

1. **Nomination.** A scan proposes candidates — processes whose command line contains
   `llama-server` and whose `--port` falls inside the managed range, whether
   reparented to init or not.
2. **Proof.** A candidate is signalled *only* if its **live** `(pid, port, starttime)`
   matches a recorded triple. Anything unrecorded is reported and left alone.

Nomination is a filter. The recorded identity is the only authority.

### Why the start time is in the triple

A process id is recyclable. Once a process exits, the kernel is free to hand its
number to something else, and on a busy machine that can happen within seconds. A
start time is unique to a process instance for the life of a boot and never changes
while the process lives, so `(pid, starttime)` identifies a specific process rather
than a number that used to mean something.

Without it, a stale record could match a stranger that happened to inherit the id.

## What it refuses to do

The contract is deliberately one-sided: **when ownership cannot be proven, nothing
is signalled.**

- An unrelated `llama-server` in the managed range has no record. Report-only.
- A foreign process that is itself orphaned looks like the strongest possible
  candidate under a parent-chain heuristic. It still has no record. Report-only.
- A process whose start time cannot be read yields no proof. Report-only.
- A row whose `engine_starttime` is `NULL` — every row predating this feature —
  cannot match a live process, so historical state can never justify a kill.

The cost of that choice is real and worth stating: if the manager dies in the window
between spawning an engine and persisting its identity, that engine is alive,
unrecorded, and will **not** be cleaned up automatically. It keeps running and is
surfaced in the diagnostics below for an operator to deal with. That is preferred to
the alternative failure, which is killing something that was never ours.

## Live engines are preserved explicitly

Ownership is not the same as staleness. A healthy engine serving traffic right now
is recorded exactly like the one being torn down, so the ownership proof cannot tell
them apart — by design, because it answers a different question.

Callers therefore pass the set of currently-live engine pids, which cleanup skips:

- **At boot**, that set is empty. A lock file guarantees only one manager runs, so if
  this one started, any other manager is gone and its engines are stale by definition.
- **Everywhere else** — after a teardown, after an idle engine is released — the set
  is the manager's live engines. Without it, a co-resident engine would satisfy the
  ownership gate and be signalled while it was still answering a request.

If you add a new cleanup call site, it needs the live set unless it is boot.

## Diagnostics

A separate scan reports which processes are listening on the managed range and what
they are. It **never** signals anything; a foreign listener shows up here precisely so
that an operator can see the port is occupied and by what. This is also where an
engine that was spawned but never recorded becomes visible.

## Schema

The identity column is added by an idempotent migration that takes the state database
from version 1 to version 2. It adds one nullable column and is safe to run
repeatedly; existing rows are untouched and migrate with a `NULL` start time.

The code addresses the `slots` table by explicit column names rather than
`SELECT *`, so the additional column is inert to a build that does the same if you
roll back.

## Main-lane admission

Related but separate: when several requests are queued, an interactive request can be
admitted ahead of queued background work as soon as the active engine frees.

| setting | default | effect |
|---|---|---|
| `main_lane_reserved` | `true` | interactive requests are admitted first |
| `main_lane_identity_keys` | `["is_main"]` | request metadata keys that mark a request interactive |

This is a **queue-level reservation only**. It removes a queued request from the
buffer and admits it ahead of its neighbours; it never interrupts a running engine
and never changes how many engines are resident. Whatever concurrency the
deployment is configured for is unaffected — this changes admission order, not
execution.
Setting `main_lane_reserved` to `false` removes the reservation; admission then follows
the rest of the queue ordering (compression-class requests still skip ahead of queued
sub-agent work, and model affinity and first-in-first-out order apply). When Fast Lane
is enabled, its priority pick runs before this reservation (see
[FAST_LANE.md](FAST_LANE.md)).

Whether this is desirable depends on your workload. If background traffic (parallel
sub-agent fan-out, summarisation passes) matters as much as interactive latency, the
reservation will make that work wait behind interactive turns.

## Verifying it on a running system

Cleanup can be exercised without signalling anything, by supplying a reap function
that only records what it would have done:

```python
from pathlib import Path
from turbohaul.state import open_state_db, known_engine_identities
from turbohaul.singleton import boot_orphan_reaper

conn = open_state_db(Path("<state-db-path>"))
try:
    identities = known_engine_identities(conn)
finally:
    conn.close()

would_signal = []
def dry_run(pid, *args, **kwargs):
    would_signal.append(pid)
    return False, "dry-run"

boot_orphan_reaper(
    port_base=<managed-port-base>,
    known_pids={<live engine pids>},   # empty only at boot
    reap_fn=dry_run,
    known_engine_identities=identities,
)
print(would_signal)          # expected: [] while engines are healthy
```

Use `open_state_db` rather than a bare connection — the identity lookup addresses rows
by column name and needs the row factory that helper installs.

Two things worth asserting on a live system:

- with the live engine set supplied, the result is empty while engines are healthy;
- a torn-down engine drops out of `known_engine_identities` once its slot is closed,
  which is what keeps a recycled process id from matching an old record.

## Credits

The identity model and the reap-only-what-you-recorded contract were contributed by
**Sahil-SS9**, along with the main-lane admission reservation and the accompanying
test suite. See [CONTRIBUTORS.md](../CONTRIBUTORS.md).
