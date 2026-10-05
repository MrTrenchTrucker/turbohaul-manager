# Placement and Co-residence

How Turbohaul decides **which GPU a model lands on**, and why two models can or cannot be resident at the
same time.

This document covers the **doctrine and the manifest data** — what an operator needs to know to configure a
model correctly today. For the VRAM arithmetic behind every refusal, see
[SAFETY_GATE_VRAM_MATH.md](SAFETY_GATE_VRAM_MATH.md); for the placement knobs themselves, see
[MULTI_GPU_PLACEMENT.md](MULTI_GPU_PLACEMENT.md).

---

## 1. The doctrine, in one table

| `split_mode` | what it means | can it co-reside? | can it be auto-placed? |
|---|---|---|---|
| `none` | tensor-isolated: the model's weights live **wholly on one card** | **yes** — with other `none` models, subject to per-card VRAM | yes, if `auto_place: true` (or while Fast Lane is enabled) |
| `layer` (also `row`, `tensor`) | layer-split: the model **spans every visible GPU** | **no** — never, with anything | no |

The rule that follows:

> **A model that fits whole on one card should be `split_mode: none`.**
> A model that genuinely does not fit one card must be `layer`, and accepts that it cannot co-reside.

### Why layer-split can never co-reside
`manager._vram_admits_locked` refuses co-residence unless the incoming model **and every existing sibling**
are `none`:

```py
new_split = (split_mode or "layer").lower()
if siblings:
    if new_split != "none": return False
    for r in siblings:
        if (r.split_mode or "layer").lower() != "none": return False
```

A layer-split sibling occupies every card, so per-card budgeting cannot prove the incoming model has room on
its target card. The gate refuses rather than guessing. That check runs **before** VRAM, `tensor_split` or
`main_gpu` are consulted — so no placement setting can rescue a `layer` model from it.

### Why `auto_place` matters
`manager._auto_pick_gpu` picks the **most-free card that fits**, which is what spreads models across cards
instead of piling them on `main_gpu: 0`. But it is gated:

```py
if (auto_place or self._fastlane_enabled()) and split_mode == "none":
```

`auto_place` defaults to **`False`** (`manifest.py`). A model that is `none` but leaves `auto_place` unset stays
**hard-pinned** to its manifest `main_gpu` while Fast Lane is off (with Fast Lane enabled the card pick also runs
for a `split_mode: none` model, whatever `auto_place` says), and the fit gate will defer the spawn (see §6) at that card
without ever considering the other one.

---

## 2. The common misconfiguration

The usual cause of a model that will not co-reside is not a manager defect — it is manifest data. A model that
comfortably fits one card but is declared `split_mode: layer` spans every visible GPU by definition, which
makes it ineligible for co-residence and for auto-placement both. Neither condition is visible from the
model's behaviour until something else tries to load beside it.

The fix is per-manifest: set `split_mode: none` and `auto_place: true` on any model whose measured footprint
fits a single card with slack, and leave `layer` only where it is genuinely required. Re-verify the fit
predicate against the edited file before loading it.

---

## 3. ⛔ `auto_place` is NOT safe for models near the card limit

Adding `auto_place: true` to models that are already `none` but hard-pinned can regress a pairing that works.
The reason is the useful part:

Auto-placement can move a near-limit model to the other card. Inspecting per-process GPU memory then shows a
single process holding memory on **both** cards: roughly 1.3 GB of CUDA context on the card it was moved off,
and its weights on the card it was moved to. The context it leaves behind can put the first card below
its partner's need, and the pairing that had been working stops fitting.

> **A `split_mode: none` model still consumes a per-card CUDA context on the card it is NOT placed on.**

`expected_vram_bytes` describes the footprint on the card the model is placed on. It does not describe the
~1.3 GB the process leaves behind on the other card. For models with headroom this is irrelevant. For
models within ~1.5 GB of the card limit, the manual `main_gpu` pins are **load-bearing**: they keep each
model's context and weights on the same card so that two models that each nearly fill a card still fit on two cards.

**Guidance:** set `auto_place: true` when the model has comfortable slack. Leave near-limit models pinned until
placement logic accounts for context overhead as well as weight footprint. (A deliberate `main_gpu` pin holds
only while Fast Lane is off; with it on, the card pick also runs for a `split_mode: none` model.)

---

## 4. Configuring a model — the short version

```yaml
expected_vram_bytes: <MEASURED footprint>   # a GATE PREDICATE, see §5
auto_place: true            # only if the model has comfortable card slack
llama_server_flags:
  split_mode: none          # if it fits one card
  main_gpu: 0               # a fallback once auto_place is on; load-bearing without it
```

- **Fits one card, lots of slack** → `none` + `auto_place: true`.
- **Fits one card, near the limit** → `none` + `auto_place: false` + a deliberate `main_gpu` pin.
- **Does not fit one card** → `layer`. It will run alone. That is correct, not a bug.

---

## 5. ⚠ `expected_vram_bytes` is a gate predicate, not documentation

Both `_vram_admits_locked` and `_auto_pick_gpu` **decide from a footprint derived from this field**:
`max(expected_vram_bytes, gguf_size_bytes + estimated KV cache)` plus a per-slot allowance when `parallel` is
above 1 (for an expert-offload model, `cpu_moe` / `n_cpu_moe` set, the measured value is used on its own). The
per-spawn free-VRAM check also requires at least `expected_vram_bytes` to be free, and never less than
`safety_min_free_vram_mib`. It is directional in both directions, and both directions can bite:

- **Too high** → the fit gate refuses a model that would actually have fit (a value padded "for
  documentation" refuses the very spawn it was written for).
- **Too low** → for an expert-offload model, or one with no `gguf_size_bytes`, the gate admits a model onto a
  card that cannot hold it. For other models the size-derived estimate takes over as a floor.

A manifest declaring `expected_vram_bytes: 0` with no `gguf_size_bytes` tells placement the model is free, so
the placement/fit gate admits it onto any card in any state; only the per-spawn free-VRAM check still applies, and it
refuses a card with less than the `safety_min_free_vram_mib` floor (default 512 MiB) free. A hidden or never-loaded model makes that harmless right up until the day it is loaded.
Treat this field as a load-bearing number and measure it.

---

## 6. Reading the deferral

A spawn that cannot land on its target card right now is **deferred, not refused**. The manager re-queues it
and keeps retrying with backoff for as long as the client stays connected — it does not give up on its own.

Log line: `cross-resident VRAM gate deferred (not refused) spawn for <tag> (need=<N>MiB parallel=<p> gpu=<g>
split=<mode>) -- requeued, awaiting VRAM-confirmed teardown wake`

- `split=layer` → the co-residence doctrine is what's blocking it; no placement setting will help while
  anything else is resident. It re-admits automatically once every other resident has vacated.
- `split=none` with a card that has room elsewhere → the model is hard-pinned (`auto_place` false and Fast Lane
  off) to a card that does not have room. It re-admits automatically once that specific card frees enough VRAM.
- **It is not an OOM.** Nothing failed to allocate; the manager just hasn't found room yet.
- **A single line in the log is the manager working, not the request dying.** There is no bounded retry
  budget and no client-facing 503 for this condition any more — a request that can never actually fit (the
  card it's pinned to is permanently occupied, or nothing will ever vacate) queues indefinitely rather than
  erroring out. If a request seems stuck, the fix is on the manifest side (`auto_place`, a different
  `main_gpu` pin, or an offload knob — `n_gpu_layers` / `cpu_moe` / `n_cpu_moe` — for a model too big for any
  single card), or the caller cancelling the request itself.

---

## 7. Checking a manifest

There is no bundled audit script. To classify a model by hand: read its `expected_vram_bytes`, compare it
against the free VRAM a single card can offer, and check `llama_server_flags.split_mode`. A model whose
footprint fits one card but is declared `layer` can neither co-reside nor be auto-placed — that is the
mispinned case §1 describes, and it is a manifest problem, not a manager one.
