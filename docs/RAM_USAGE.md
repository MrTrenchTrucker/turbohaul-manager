# How TurboHaul uses RAM

TurboHaul keeps several different things in system memory, and they grow for different reasons. This page lists
each one, says when it grows, and names the settings that bound it. If a host is running short of memory, start here.

| Where | What lives there | Bounded by |
|---|---|---|
| The manager process | Python objects, request state, the in-memory caches | Request volume; the allocator setting below |
| The RAM disk (tmpfs) | Saved KV-cache files, so a conversation can be restored without re-reading its whole prompt | `kv.ram_cache_max_bytes`, `kv.ram_cache_max_age_hours` |
| The engine processes | The model and the live KV cache | The model, its context size and `llama_server_flags` |
| The state database | SQLite's page cache for `state.sqlite` | SQLite defaults |
| Swap | Anything the kernel pushes out under pressure | The host, not TurboHaul |

## The manager process

The manager is a Python process. Its memory follows the work it is doing: queued and in-flight requests, the model
registry, telemetry and the caches it keeps to answer the dashboard quickly. It grows with traffic and settles when
traffic drops.

It can look bigger than the work it is doing, because of how the C allocator under Python behaves. On Linux, glibc
gives threads that allocate memory their own arenas, and an arena keeps freed blocks for reuse instead of always
returning them to the operating system. A process with many threads can therefore hold a lot of freed memory that is
neither in use nor returned.

`MALLOC_ARENA_MAX` is a glibc environment variable that caps the number of arenas. With a low cap (for example
`MALLOC_ARENA_MAX=2`), threads share arenas, so freed large blocks are less likely to stay pinned inside arenas that
nothing is reusing. It trades a little allocator contention for a smaller resident size. TurboHaul does not set it for
you: set it in the environment of the container or service that runs the manager, and restart.

## The RAM disk: saved KV caches

When a model is unloaded to make room for another, TurboHaul saves that conversation's KV cache so the conversation can
continue later without recomputing its prompt. The files go to a directory that is expected to be a tmpfs mount, which
means they live in system memory. See [KV_CACHE_RAM_LIFECYCLE.md](KV_CACHE_RAM_LIFECYCLE.md) for when saves happen, how
they are restored and what is reclaimed first.

A saved conversation is a **pair** of files: the cache itself (`.bin`) and, when the engine writes one, a checkpoint
sidecar (`.bin.ckpt`), plus a small metadata file. The sidecar is comparable in size to the cache, so both count.

### Settings

| Setting | Environment variable | Meaning |
|---|---|---|
| `kv.ram_cache_max_bytes` | `TURBOHAUL_KVCACHE_MAX_BYTES` | The size cap of the RAM disk in bytes. `0` means no ceiling. |
| `kv.ram_cache_max_age_hours` | | How long an idle conversation's pair may stay before the sweep reclaims it. Tunable at runtime. |

A value saved through the Settings page or the config API outranks the environment variable and takes effect without a
restart. The age and the byte ceiling are read fresh on every sweep.

Keep the cap at or below the size of the tmpfs mount. A cap above the mount size can never be reached, because the
mount fills first, and a full tmpfs is what makes a save come out truncated. Leave some room above the cap as well:
while a new pair is being written, the pair it replaces is still there.

### The size cap

The size cap is `kv.ram_cache_max_bytes`. Before TurboHaul asks the engine to write a pair, it measures the directory
(each cache counted together with its sidecar). If the directory has reached the cap, the sweep described below runs once
to make room, and the directory is measured again. If it is still at or over the cap, because the conversations holding
it are all live, the save is skipped: the conversation's cache stays in the engine and any earlier pair is kept. There is
no other size check on a save.

### Two edges of the size cap that are known and not yet fixed

Both behaviours below are known limitations of this release.

- **A directory exactly at the cap is declined, and the sweep frees nothing for size.** The save check counts a directory
  exactly at the cap as full, so it runs the sweep once and measures again. The sweep's byte ceiling acts only when the
  total is strictly over the cap. So the save stays declined until something else moves the bytes: age, file count, a
  conversation going idle, or a delete. This follows from the save check and the sweep as written.
- **The check does not add the pair about to be written.** It compares the current size to the cap. A directory one byte
  under the cap passes, and the save can end with the directory up to one whole pair over the cap. The next save then
  runs the sweep, which pays it back. For example, with a pair of about 4.5 GiB and a tmpfs only about 3 GiB
  larger than a 12 GiB cap, an over-cap save could run the tmpfs out of space and come out truncated.

A later release may add the incoming pair's size to the check and make the equal case consistent. Until
then, keep the tmpfs at least two largest pairs bigger than the cap, so there is room both for a save that lands up to
one pair over the cap and for the pair it replaces, which is still on disk while the new one is written.

### One pair per conversation

A conversation can move to a different engine port, for example after its model was unloaded and loaded again. Its pair
from the old port is of no further use, and used to stay on the RAM disk until a sweep reclaimed it. Now, once the new
pair has landed, TurboHaul deletes the same conversation's pairs on other ports (the cache, the sidecar and the metadata).

What this never touches:

- a pair on a port that a live turn is still serving for that conversation;
- pairs of other conversations, or of other models;
- other slots on the port the conversation moved to.

If TurboHaul cannot tell which ports a live turn serves, it deletes nothing.

### When the old pair is kept

The old pair is deleted only after the new one is in place, so a save that fails or is skipped leaves the conversation
with the pair it already had. A save that did complete is also checked:

- The engine reports how many bytes it wrote (`n_written`). TurboHaul compares that number with the size of the cache
  file now on disk. Only when both are the same whole number, and not zero, is the new pair taken as having landed whole
  and the old pair deleted.
- If the engine does not report the number, reports something that is not a number, or the two differ by even one byte
  (for example a write that was cut short), the old pair is kept. A different number logs one warning that names both
  sizes and says the old pair was kept. The new pair stays on disk either way.

Keeping the old pair costs RAM-disk space until the sweep reclaims it; deleting it on a doubtful landing could leave the
conversation with no usable pair at all, so the check errs towards keeping.

### The sweep

Independently of saves, a sweep reclaims entries by age, by file count and by byte ceiling, in that order, and never
takes an entry that belongs to a live conversation. Lowering `kv.ram_cache_max_age_hours` makes idle conversations leave
the RAM disk sooner, at the price of a full re-read of the prompt when one of them returns.

## The engine processes

Each model runs in its own engine process (`llama-server`). The model weights and the live KV cache sit mostly in the
GPU's memory. What an engine keeps in host memory is its own business: for example a model file mapped in from disk, or
layers a model's flags keep on the CPU. That memory belongs to the engine process, not to the manager, and it is
released when the engine exits.

## The state database

The manager keeps its state in a SQLite database (`state.sqlite`) in write-ahead-log mode. It sets no custom page-cache
size, so SQLite's defaults apply and the cache stays small. The database file lives on the `state` mount, not in RAM.

## Swap

TurboHaul has no swap setting. On a host with swap, the kernel may push idle pages out under pressure, and tmpfs pages
are no exception. A saved KV cache that has been swapped out is slower to read back than one in RAM. If restores slow
down on a loaded host, check swap activity before changing the cap.
