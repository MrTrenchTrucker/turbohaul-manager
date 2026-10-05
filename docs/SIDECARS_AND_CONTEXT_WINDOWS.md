# Sidecars vs. context windows

> **Read this if you have ever asked "how do I run two of the same model at once?"**
> The answer is two different settings, on two different levels, and they multiply.
> Getting this wrong is a common cause of "why is only one of my
> sub-agents running?".

Two settings decide how many conversations a model can serve at the same time. They
are not the same thing and they do not live in the same place. This document says what
each one does and how they compose.

---

## The one-sentence version

| Question | Setting | Level | Default |
|---|---|---|---|
| How many model processes may run on this box at once, across **all** models? | `queue.max_parallel_sidecars` | wrapper / manager | `1` |
| How many requests may one process answer **concurrently**? | `llama_server_flags.parallel` (manifest) | per manifest | not set (see below) |

There is no per-model limit on the number of processes. The number of engines the box
can hold for a model comes from the box-wide budget and from the cards (GPUs) the
model fits on right now. Whether a model can use more than one engine also depends on
its manifest's placement (one card per engine, see below), and on whether a copy fits
on a card at all.

**A "sidecar" (also called an engine) is one `llama-server` process. A "context window"
is one concurrent request slot inside that process.** `parallel` creates more slots
*inside* one process; the box-wide budget allows more *processes*. They are different
resources that happen to multiply:

```
conversations a model can serve = engines x windows per engine
```

---

## The two levels

### `queue.max_parallel_sidecars` - the box's budget of engines

Wrapper-level, in `turbohaul.yaml` under `queue:`, surfaced in the FE Settings tab and
editable there. Range 1-32, default 1.

It is the **total number of engines across every model on the machine**. The manager
treats the budget as full when the number of resident model engines is greater than or
equal to it. If the box is at 2 and two different models are resident, no third engine
starts, whatever any manifest asks for. If the box is at 2 and only one model is
resident, that model may run a second engine of its own (see the conditions below).

The value in force is set by three layers, lowest to highest: the `turbohaul.yaml` file,
then the `TURBOHAUL_MAX_PARALLEL` environment variable, then a value saved in the
Settings tab or with `PUT /api/config`. A saved value is kept in a separate runtime
override file that is loaded on top of the other two each time the manager starts, so it
survives a restart and beats the environment variable. Read the value actually in force
from `GET /api/config` (see the last section) rather than from the file or the
environment.

### The cards the model fits on

A model can only run as many engines as there are cards that can host one more copy
right now. The manager measures free VRAM per card and counts the cards where one more
copy fits (the model's own cards plus the other cards that currently have room). If no
card fits, the model still gets one engine and the VRAM check at spawn time decides
whether it loads.

### When a model runs a second engine

A second engine of the **same** model is started only when the manifest has **both**:

```yaml
auto_place: true                          # the placer picks the card
llama_server_flags:
  split_mode: none                         # explicit; an absent split_mode does not count
```

The reason is that only a one-card-per-engine placement can host a second copy. A model
that already spans cards (`layer`, `row` or `tensor` split) or is pinned to a card
(`auto_place: false`) keeps exactly one engine, placed as before. A model that already
has more than one engine stays on this path, so the decision never takes an engine away
from a model that has it.

With those two lines and room in the box budget, the manager prefers a card that does
not already host the model, keeps each session on the engine that holds its cached
context, and queues overflow instead of rejecting it.

If a second engine is refused by a host safety gate, fails to start or fails to load,
the request is served by the engine of that model that is already running instead of
failing (an out-of-memory load failure, with the client still connected, goes back in
the queue as before); the first engine of a model has nothing to fall back to and fails
as before, and so does a request whose only other engine cannot take it yet, for example
while it waits for VRAM to be freed before it starts.

### `llama_server_flags.parallel` - windows per sidecar

A spawn-argv flag, range 1-256. The context size comes from
`llama_server_flags.ctx_size`. When the manifest omits `parallel`, TurboHaul passes no
`--parallel` to the engine and its own accounting assumes one window per engine, while
the engine chooses its own slot count. Set `parallel` explicitly to pin the number of
windows.

This is the one that is genuinely about **concurrency inside one process**. One
cross-field rule binds it:

- **`parallel > 1` requires `kv_unified: true`.** A manifest that sets `parallel` above
  1 without a unified KV pool is rejected when it is saved. With a unified pool the
  limit for each window is the **full** `ctx_size`, and the windows share one pool of
  that many tokens: `ctx_size` is not divided between the windows, so it does not have
  to divide evenly by `parallel` and there is no per-window minimum. (The divisibility
  and per-window floor checks exist in the code but apply only to a non-unified pool,
  which the manifest validator does not let through.)

---

## How they compose

The number of engines the box can hold for a model comes from the box budget and the
cards; using more than one also needs a one-card-per-engine placement
(`auto_place: true` with `split_mode: none`). The windows come from `parallel`. The
model can then serve `engines x parallel` conversations.

In the table, "engines" is the most one model can run when no other model holds an
engine. "Cards that fit" is the number of cards that have room for a copy of the model.
The `parallel` values are set explicitly in the manifest.

| `max_parallel_sidecars` | `parallel` | cards that fit | placement | engines | windows | notes |
|---|---|---|---|---|---|---|
| 1 | 1 | 2 | any | 1 | 1 | one engine, one window: serial serving |
| 2 | 1 | 2 | `auto_place: true`, `split_mode: none` | 2 | 2 | two processes, one window each |
| 2 | 1 | 2 | `auto_place: false` (pinned) | 1 | 1 | a pinned model keeps one engine |
| 2 | 1 | 2 | `split_mode: layer` | 1 | 1 | a split model already spans the cards |
| 2 | 1 | 1 | `auto_place: true`, `split_mode: none` | 1 | 1 | only one card has room for a copy |
| 2 | 2 | 1 | any | 1 | 2 | one process, two windows: cheapest on VRAM |
| 2 | 2 | 2 | `auto_place: true`, `split_mode: none` | 2 | 4 | the most capable and the most VRAM-hungry |

Consequences worth internalising:

- **Raising `max_parallel_sidecars` is what allows more engines.** It does so only for
  a model that can use the room: `auto_place: true` with `split_mode: none`, and a
  second card that fits. A pinned or card-spanning model keeps one engine whatever the
  budget is.
- **The budget counts engines of all models.** Two models each holding an engine use
  the whole budget at 2. A model cannot take a slot another model holds; the manager
  evicts an idle resident to make room where it can, otherwise the request waits.
- **A second copy of a model competes for VRAM.** It carries its own copy of the
  weights and its own cache, so it takes memory the other models on that card could
  have used. Watch the free VRAM on every card.
- **Windows are not free.** Every window can use up to the full `ctx_size`, and the
  windows share one unified KV pool, so more windows mean more concurrent work on the
  one process: more compute, and more KV cache in use when several windows hold long
  conversations at once. Two engines x two windows is four concurrent contexts, with
  the weights loaded twice (once per engine).
- **Three layers set the budget, in this order of precedence:** the config file, then
  the `TURBOHAUL_MAX_PARALLEL` environment variable, then a value saved in the Settings
  tab (highest, and it survives a restart). With the environment variable set, editing
  the file does not change the value the manager starts with, but a value saved in the
  Settings tab still does.

### Which shape should you use?

Choose by what you are trying to protect:

- **Isolation** - a slow or long request must not block the others -> more engines
  (raise `max_parallel_sidecars` and use `auto_place: true` with `split_mode: none`).
  Costs weights in VRAM once more per engine.
- **Throughput on one copy of the weights** - contexts are mostly waiting on I/O or
  short generations -> more windows per engine (`parallel`). Cheaper, and the unified
  KV pool keeps the accounting flat.
- **Both** - a manager that runs a main turn plus several sub-agents - set both, and
  watch the VRAM headroom on every card.

Note that a model's *placement* interacts with all of this: a second engine is only
meaningful for a model that fits on one card (`split_mode: none` + `auto_place: true`).
A layer-split model already spans the cards and cannot be replicated per card.

---

## Retired setting

An older manifest field, `max_instances`, no longer exists. A saved manifest, a manifest
file or a `PUT` body that still contains it is accepted: the key is dropped and one
warning is logged, never an error. Saved manifest files are rewritten without the key
once, at the first start (the revision does not change). Nothing is carried over, so a
model that relied on the old value is governed by the budget and the cards as described
above.

---

## Verifying a change took effect

`parallel` is **spawn-argv**: patching a manifest does *not* affect a running
`llama-server` (`docs/MODEL_CONFIG_REFERENCE.md` section 2). A manifest `PUT` is not
enough - the process must be re-forked. Two different things can be read back:

- `GET /api/manifests/<model_tag>` returns the **stored** manifest: what the next
  engine will be started with, not what a running engine was started with.
- `GET /status` returns, under `residents`, one entry per running engine. Its
  `parallel` is the window count TurboHaul recorded when it admitted that engine: the
  manifest's `parallel` at that moment, or 1 when the manifest omits it. It is not read
  back from the engine, so it equals the engine's real slot count only when `parallel`
  is set explicitly. `main_gpu` and `split_mode` show where the engine was placed.

```bash
curl -s http://<host>:<port>/api/manifests/<model_tag> | \
  python3 -c 'import json,sys; m=json.load(sys.stdin); \
    print("parallel:", (m.get("llama_server_flags") or {}).get("parallel"), \
          "auto_place:", m.get("auto_place"), \
          "split_mode:", (m.get("llama_server_flags") or {}).get("split_mode"))'

curl -s http://<host>:<port>/status | \
  python3 -c 'import json,sys; [print(r["model_tag"], "parallel:", r["parallel"], "main_gpu:", r["main_gpu"], "split_mode:", r["split_mode"]) for r in json.load(sys.stdin)["residents"]]'

curl -s http://<host>:<port>/api/config | \
  python3 -c 'import json,sys; print("max_parallel_sidecars:", json.load(sys.stdin)["queue"]["max_parallel_sidecars"])'
```

The box budget is read at every routing decision, so a changed `max_parallel_sidecars`
is used by the next one. The engine-count decision never takes away an engine a model
already has; the budget only limits engines that would start afterwards.

A manifest `PUT` to an existing manifest requires `If-Match` with the current ETag (the
`ETag` header of the `GET`, quoted). The `GET` body can be sent back almost as it is:
`kind`, `hidden`, `revision` and `display_name` are accepted. The one exception is
`cache_reuse_inert_mmproj`, a derived marker the `GET` adds that is not a manifest
field: remove it before sending the body back, because it (like any unknown key) is
rejected as `extra_forbidden`. A YAML file on disk is not necessarily the file the server loaded;
the running server's own report is the truth.

---

## Related

- [docs/MULTI_GPU_PLACEMENT.md](MULTI_GPU_PLACEMENT.md) - card selection, co-residency, the VRAM gate
- [docs/MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md) - every manifest field and flag, with bounds
- [docs/design/FAST_LANE_ARCHITECTURE.md](design/FAST_LANE_ARCHITECTURE.md) - priority admission and turn-boundary unloads between clients
