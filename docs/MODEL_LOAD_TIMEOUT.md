# Model Load Timeout

How long the manager waits for a newly-spawned engine process to report healthy
before it gives up on the load, and how to tune it.

## The setting

| | |
|---|---|
| Config key | `queue.loading_health_timeout_s` |
| Code default | `60` (seconds) - applies when nothing sets it |
| Shipped default | `120` (seconds) - what `docker/turbohaul.default.yaml` sets, and what any deployment using the shipped config actually gets |
| Range | `10` – `7200` |
| Applies to | every model load, including the load half of a model swap |

## What it does

When the manager needs a model that is not already resident, it spawns an engine
process and then polls that process's `/health` endpoint until it reports ready.
`loading_health_timeout_s` is the ceiling on that wait.

If the engine becomes healthy within the window, the load succeeds and the
request proceeds. If the window expires first, the load is treated as failed and
the caller gets an error rather than waiting indefinitely.

The wait also ends early — well before the timeout — if the spawned process
exits during loading. A crashed or mis-configured engine fails fast instead of
burning the full window.

## An out-of-memory load failure does not end the request

One exit-during-loading cause is treated differently: an engine that exited
because it could not allocate GPU memory for the model (a CUDA
out-of-memory failure, or the equivalent ggml allocator failure on a
different code path). This is judged from the engine's own log, the same
`E`-level scan the load-verify observability already runs, checked for the
allocator's own wording (`out of memory`, `failed to allocate`). If the
engine's log cannot be read at all — or was readable but empty — the cause
is **unknown**, and an unknown cause is treated exactly like any other
failure below, never guessed at as OOM.

- **A confirmed OOM failure, on a request the manager can still hear from
  (it has a live client-disconnect signal), goes back into the queue with
  the exact priority and Fast Lane standing it already had.** It is not an
  error the caller sees. It leaves the queue only two ways: the client
  disconnects, or it is served (a manager shutdown also ends it). It is retried only when something real
  changes — a resident unloading, being evicted, or its VRAM freeing — never
  on a bare timer, and at most once per such change (a burst of several
  frees in quick succession still produces exactly one retry, not one per
  free). This is the same "once queued, stays queued until the client
  leaves" rule the manager already applies to a model that simply doesn't
  have a free card *yet* (a "VRAM busy" defer) — an OOM failure just
  extends that rule to a load that started and then failed for the same
  underlying reason (no room), rather than never being attempted.
- **Every other load failure fails once, immediately**: a non-OOM error (a bad file, a missing weight blob, a crash with
  no allocator signature), an unreadable/empty engine log (cause unknown —
  an unprovable cause is never requeued), a request with no
  client-disconnect signal to wait on, and a model whose own declared
  need is more than the GPU capacity it could ever be placed on, even
  empty (one card for a single-card placement, all cards combined for a
  split placement; a "never fits" load, which would
  otherwise wait forever for a change that can never come — but see below:
  this only fails once when the box's total GPU capacity is *actually
  known*).
- **An unknown GPU capacity is never treated as never-fits.** If the
  manager has not yet read how much memory a card physically has — at
  boot, or on a probe failure — "never fits" cannot honestly be decided,
  and the request is requeued rather than failed. Ending a legitimate
  caller's request on a measurement the manager doesn't actually have would
  be worse than one extra retry once the capacity is known.
- **No new claim bookkeeping is involved.** A requeued request keeps
  whatever Fast Lane claim it already held; nothing new is registered, and
  nothing is released, until the client disconnects or the request is
  served — exactly the existing claim lifecycle, unchanged.
- In this one case a `queue.loading_health_timeout_s` failure does not end the
  request: the timeout above still ends the *load attempt* (the spawned process
  is torn down as for any failed load), it just does not end the *request* on
  its own. See
  [FAST_LANE.md](FAST_LANE.md) and
  [design/FAST_LANE_ARCHITECTURE.md §1.6](design/FAST_LANE_ARCHITECTURE.md)
  for how a requeued request's priority is re-evaluated fresh, the same way
  any other re-admission is: this is not a new placement rule, it reuses
  the one that already exists.

## Why the shipped value is 120, not 60

The code default and the shipped config deliberately differ, and the distinction
matters:

- **60s** is the code default -- the value a deployment gets when nothing
  configures it. It is a reasonable floor for a machine whose models load fast.
- **120s** is what `docker/turbohaul.default.yaml` sets, because 60 is tight
  for large models.

As an illustration, consider a healthy cold load of a large model that takes about
**38 seconds**.

The stall warning below fires at the **midpoint** of the timeout. At a 60s
timeout the midpoint is 30s, so a perfectly healthy 38s load would log a stall
warning *every single time* -- which is exactly the noise the midpoint test
exists to prevent. At 120s the midpoint is 60s, comfortably clear of such a 38s
load, while a genuine hang still surfaces in about two minutes rather than ten.

If you change this value, re-measure the slowest healthy load on the box and keep
the midpoint above it, or the warning stops carrying information.

## Why the default is small

The default assumes a normal model swap completes in well under a minute. The timeout exists to
bound the *failure* case, not the success case, so it should be only as long as
a healthy load plausibly needs.

A large default is deceptively expensive. It does not make slow loads faster; it
makes *stuck* loads slow to detect. With a ten-minute window a wedged engine
holds the caller for ten minutes and looks like poor performance rather than a
fault. At the shipped 120s the same failure surfaces in about two minutes and is
recognisable as a failure. The permitted range runs up to 7200s so that a
genuinely slow model can be accommodated by configuration alone, but that
headroom is for exceptional models, not a value to reach for by default.

## The stall warning

A load that has not become healthy by the **midpoint** of its timeout logs one
warning:

```
wait_until_healthy: model still not healthy after 31.2s for port 11500
(timeout=60.0s) - check the engine process and GPU memory for a stalled load
```

It fires at most once per wait, and only past the midpoint. That threshold is
deliberate: a warning that fired on every load — including healthy ones — would
appear in the log constantly and stop carrying information. Seeing this line
means the load is genuinely running long and there is still roughly half the
window left to investigate.

Common causes when it appears:

- the model is substantially larger than the timeout was sized for
- insufficient free GPU memory, causing the engine to thrash or stall
- slow storage on first read of a large model file
- an engine process that started but wedged before serving

## Tuning it

Raise the value if you routinely load models large enough that a healthy load
legitimately exceeds sixty seconds. The upper bound is deliberately generous so
that a genuinely slow model can be accommodated by configuration alone, with no
rebuild required.

Lower it if you want faster failure detection and know your models load quickly.
The floor is ten seconds.

### Where to change it

Edit the `queue` section of the runtime configuration file:

```yaml
queue:
  loading_health_timeout_s: 120
```

The value is also readable and writable through the configuration API, so it can
be changed from the settings interface without editing the file by hand:

```
GET  /api/config          # returns the current value under "queue"
PUT  /api/config          # accepts a "queue" section with a new value
```

Values outside the permitted range are rejected with a validation error rather
than silently clamped.

> **Note:** raising this value only widens the window before a load is declared
> failed. It does not speed up loading, and it does not affect an engine that
> has already exited — the wait still ends immediately in that case.

## Related

- `docs/MODEL_CONFIG_REFERENCE.md` — per-model configuration fields
- `docs/MULTI_GPU_PLACEMENT.md` — GPU memory and placement, a common cause of
  slow or stalled loads
