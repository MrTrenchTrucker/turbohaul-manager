# KV-Cache Offloading

*Companion to [TurboQuant Flags](TURBOQUANT_FLAGS.md) · [Project README](../README.md)*

---

KV-cache offloading moves the **KV cache out of VRAM and into host system RAM** while keeping the **model weights on the GPU**. It trades a little per-token latency for a lot of VRAM headroom — enough to run a longer context, a larger model, or **multiple parallel slots** on a card that otherwise couldn't fit them. This page explains what it is, the `no_kv_offload` mechanism, the sizing math, parallel serving while offloaded, and when to use it.

## 1. What the KV cache is, and why offload it

During generation, a transformer caches the **keys and values** it computed for every token it has already seen, so each new token attends to the past without recomputing it. That **KV cache** is separate from the model weights and grows **linearly with context length** (× layers × attention dimension). At long contexts it can rival — or exceed — the size of the model weights themselves.

By default the KV cache lives in **VRAM**, right next to the weights. That's fastest, but it means VRAM has to hold **weights + KV + compute scratch** all at once. On a single consumer-class card, a large model at a long context can run out of room — not because the *weights* don't fit, but because the *KV cache* pushed past the VRAM ceiling.

**Offloading** relocates the KV cache to **host system RAM**, which is typically far more plentiful (and cheaper) than VRAM. The weights stay GPU-resident, so the heavy matrix math still runs on the GPU at full speed; only the cached keys/values live in RAM and are read back across the PCIe bus during attention.

You want this when you are **VRAM-bound, not RAM-bound** — e.g. a long-context workload, a model that *almost* fits, or when you want to serve **several requests in parallel** on one card (each parallel slot needs its own slice of KV).

And when a model is simply **too big for VRAM**, offloading the *KV cache* is the **smarter of the two ways to make room** — because it keeps **all the weights**, and therefore the heavy per-layer matmuls, **on the GPU**. The alternative (offloading *weights/layers* to CPU/RAM) is far slower. [§5](#5-when-to-use-it--and-the-perf-cost) explains why, and what the speed cost actually is.

## 2. The `no_kv_offload` mechanism

Turbohaul-Manager exposes offloading through a single manifest flag:

```yaml
llama_server_flags:
  no_kv_offload: true     # → llama-server --no-kv-offload : KV cache in host RAM, not VRAM
```

What happens at spawn:

- **Model weights stay in VRAM** — every matmul still runs on the GPU. Offloading does **not** move the weights; it moves only the KV cache.
- **The KV cache is allocated in host RAM.** During each decode step the attention layer reads the cached keys/values from RAM over PCIe, computes attention on-GPU, and writes the new token's KV back to RAM.
- The trade is **bandwidth/latency**: PCIe is much slower than on-card VRAM bandwidth, and the KV is touched on every decode step (see [§5](#5-when-to-use-it--and-the-perf-cost)).

> **Use `no_kv_offload: true` — not `kv_offload: false`.** The two are *not* equivalent. A boolean flag set to `false` is omitted from the spawn command line entirely (a no-op that leaves the default, KV-in-VRAM, in place). `no_kv_offload: true` is the canonical, affirmative way to request RAM-resident KV, and it is what the manager's VRAM-fit pre-check keys on.

Like all spawn-argv flags, this takes effect on a **cold-spawn** — a manifest change doesn't reconfigure a running `llama-server` (see [TurboQuant › Spawn vs request](TURBOQUANT_FLAGS.md)).

## 3. The VRAM-vs-RAM trade-off and the sizing math

Offloading changes **where each cost lands**, and the manager's fit pre-check models that explicitly. The KV-cache size estimate itself is unchanged from the [Architecture: VRAM/KV math](ARCHITECTURE.md):

```
gguf_mib          = gguf_size_bytes / (1024 * 1024)
kv_cache_mib      ≈ (9 KB/token per GiB of body, at f16) × quant_factor × ctx_size
```

### KV resident in VRAM (default)

Everything must fit one budget — VRAM:

```
needed_vram = body + kv_cache + overhead          # overhead floor ≈ 1024 MiB
refuse if needed_vram > free_vram
```

### KV offloaded to host RAM (`no_kv_offload: true`)

The KV term **drops out of the VRAM budget** — but a **context-linear VRAM scratch** term stays, because the on-GPU attention scratch still grows with context even when the cache lives in RAM. Then a **complementary host-RAM check** ensures the KV cache fits free system RAM:

```
vram_need = body + (overhead + ctx_size / 128)    # KV term removed; ctx-linear scratch stays in VRAM
refuse if vram_need > free_vram

# complementary host-RAM fit:
refuse if kv_cache > free_host_ram
```

This is **not a safety relaxation** — the VRAM requirement is *genuinely* lower (the KV really isn't in VRAM anymore), and the requirement that moved is *re-checked* against host RAM. Without this branch the pre-check would wrongly count the RAM-resident KV against VRAM and refuse high-context configurations that actually fit.

**Worked intuition** — a long-context config that's ~24 GiB "needed" with KV in VRAM might be only ~17 GiB of VRAM with the KV offloaded (body + a few GiB of ctx-scaled scratch), with the multi-GiB KV cache now sitting in host RAM instead. The card goes from "won't fit" to "fits with headroom," at the cost of PCIe reads.

## 4. Serving requests in parallel while offloaded

Offloading's biggest payoff is **concurrency on one card**. A single warm `llama-server` sidecar can serve **N requests at once** when configured with:

```yaml
llama_server_flags:
  parallel: 2            # N internal server slots → N concurrent requests on one sidecar
  kv_unified: true       # one unified KV-cache buffer shared across the slots
  no_kv_offload: true    # that N-slot KV buffer lives in host RAM
  cont_batching: true    # continuous batching across the slots (recommended with parallel)
```

The reason offloading and parallelism pair so well: **N parallel slots need roughly N× the KV cache.** In VRAM that multiplied cache is often what blows the budget. Move it to RAM and the GPU only has to hold the **weights + a modest ctx-scaled scratch** — which doesn't multiply with the slot count the way the KV does — so the parallel slots become affordable.

> **Where this sits in the architecture.** This is **intra-sidecar** parallelism: the manager still supervises **one** `llama-server` at a time (its single-slot-per-GPU invariant is unchanged), but *that* sidecar, configured with `parallel:N`, holds N concurrent generations internally. Offloading is what frees the VRAM to make the N-slot unified KV fit. See [Architecture › The single-slot serialization model](ARCHITECTURE.md).

### Measured example

A **35B-class sparse-MoE model** run at **`parallel: 2` with RAM-resident KV** on a **24 GB-class GPU**:

| Metric | Value |
|---|---|
| VRAM used (weights + 2-slot scratch) | **~21,903 MiB** |
| VRAM free | **~2,084 MiB** |
| KV cache location | host system RAM (both slots) |
| Concurrent slots served | **2 / 2 — both served, no OOM** |

The model body plus the two slots' on-GPU scratch fit inside ~21.9 GiB, leaving headroom on a 24 GB card, while the two slots' KV caches lived in host RAM. Without offloading, the two-slot KV would have had to share that same ~2 GiB of remaining VRAM — which it cannot — so the parallel configuration would have failed to allocate. Offloading is precisely what turned "can't fit two slots" into "two slots served cleanly."

Sparse-MoE models are an especially good fit here: only a fraction of their parameters are active per token, so the compute (and the per-step pressure of reading KV over PCIe) is lower relative to a dense model of the same total size.

## 5. When to use it — and the perf cost

### Use RAM-offloaded KV when…

- You are **VRAM-bound, RAM-rich** — the weights fit but KV (or N×KV for parallel slots) does not.
- You need a **long context** that a VRAM-resident KV can't hold on your card.
- You want **concurrent serving** (`parallel:N`) on a single card.
- Throughput/capacity matters more than minimum single-request latency.

### Keep KV in VRAM when…

- You are **latency-critical** and single-request (every millisecond of per-token decode counts).
- The KV cache **fits VRAM comfortably** at your context length with room to spare.
- You're running **one slot** with no concurrency pressure.

### KV-offload vs offloading the weights — pick the right thing to move

If a model is too big for VRAM you have two relocation choices, and they are **not** equal:

- **Offload the KV cache** (`no_kv_offload`) — **all the weights stay on the GPU**, so the heavy per-token, per-layer matmuls stay on-GPU at full speed. Only the KV cache is read across PCIe, and only for the attention step.
- **Offload weights/layers to CPU/RAM** — the offloaded layers' weights must be streamed back **every single token**, a huge sustained bandwidth cost, and that layer's compute leaves the GPU entirely.

The weights are read far more often, and in far greater volume, than any one token's KV slice. So **for a too-big-for-VRAM model, KV-offload is the better offload choice** — it keeps the compute where it's fast and pays PCIe bandwidth only for attention's KV reads, instead of for the whole weight matrix on every token.

### The perf implication

KV-offload is the *smart* offload — but it is **not free.** Offloading buys **capacity, not speed.** The KV cache is read on **every decode step**, so moving it to RAM puts PCIe bandwidth in the hot path:

> **Measured (a 35B-class sparse-MoE):** decode runs at roughly **~112 tok/s with the KV cache in VRAM** versus **~44 tok/s with the KV cache in host RAM** — about **2.5× slower**. That gap is the price of capacity: you trade decode throughput for the context/concurrency that wouldn't fit otherwise.

- **Decode** (token-by-token generation) is the most affected — it is KV-read-bound, so tokens/sec drops, and the gap widens at **longer contexts** (more KV to stream) and **higher parallelism** (more slots reading KV).
- **Prefill** (processing the prompt) is comparatively less sensitive.
- **Sparse-MoE models tolerate it better** than dense models of equal total size, because less compute per token leaves the PCIe reads less exposed.

The right mental model: offloading is a **VRAM-for-bandwidth trade**. Reach for it when the alternative is "won't run at all" or "can't serve in parallel" — and prefer **KV compression first** ([TurboQuant cache types](TURBOQUANT_FLAGS.md)) when you only need to shave the cache rather than relocate it. Compression and offloading also **compose**: a compressed KV that's also offloaded is both smaller and out of VRAM, which stretches context and slot count the furthest on a constrained card.

## Quick reference

| Goal | Flags |
|---|---|
| Move KV to host RAM | `no_kv_offload: true` |
| Serve N requests in parallel on one sidecar | `parallel: N` + `kv_unified: true` + `cont_batching: true` |
| Parallel **and** VRAM-constrained | the above **+** `no_kv_offload: true` |
| Shrink KV without leaving VRAM | `cache_type_k` / `cache_type_v` (see [TurboQuant cache types](TURBOQUANT_FLAGS.md)) |

---

*Companion to [TurboQuant Flags](TURBOQUANT_FLAGS.md) · [Project README](../README.md)*
