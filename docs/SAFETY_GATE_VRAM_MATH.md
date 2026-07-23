# How the Safety Gate Estimates VRAM Need

Before the manager launches an inference server process (a *sidecar*) for a model, it runs a battery of **pre-spawn safety gates**. Their job is to predict whether the machine can actually run the model, and to **refuse the spawn up front** rather than let the process start and then be killed by an out-of-memory (OOM) error, stall on a thrashing disk, or overcommit GPU memory.

The load-bearing part of this system is the VRAM math: a closed-form estimate of how much GPU memory a model will need — **model body + KV cache + overhead** — weighed against the **free VRAM actually reported by the hardware**, adjusted for how many GPUs are visible and how the model is configured to spread across them. If the predicted requirement does not fit, no server process is ever started; the caller's request fails with a human-readable explanation of every gate that objected.

The same VRAM math also backs **cross-resident reservation** — deciding whether a second model may be loaded alongside one that is already warm.

This document explains exactly how that estimate is computed and how the decision is made. Everything below is deterministic integer arithmetic, and every rounding step truncates **downward**, so the design deliberately errs toward *refusing* a risky spawn rather than admitting one that could OOM.

---

## 1. Inputs to the math

The estimate draws on three sources: the **per-model configuration** (a validated manifest), the model's **real attention geometry** parsed from the GGUF header, and the **live system state** (free VRAM, GPU count, free RAM).

### 1.1 Per-model configuration

The manifest carries top-level sizing fields plus an allowlisted dictionary of inference-server flags. The fields that steer the VRAM estimate are:

| Field | Source | Type / default | Role in the estimate |
|---|---|---|---|
| `gguf_size_bytes` | manifest | int, default 0 | On-disk model-body size. Used as the body VRAM reservation and as the base of the legacy KV heuristic. |
| `context_size` | manifest | int, default 2048 | Fallback context window if no server-flag `ctx_size` is set. |
| `expected_vram_bytes` | manifest | int, default 0 | Operator-declared total GPU footprint. Feeds the coarse VRAM-floor check and is the authoritative footprint for expert-offload configs. |
| `hybrid_kv_ratio` | manifest | float in [0.0, 1.0], default 1.0 | Fraction of layers that grow a per-token KV cache. Applied **only** on the legacy path (see §4). |
| `kv_bytes_per_token` | manifest | float or unset, floor ≥ 1024 | Operator-**measured** effective KV cost in bytes/token. Highest-precedence KV tier when present. |
| `arch` | manifest | str, default `""` | Architecture identifier. The hybrid discount activates only when this matches a designated hybrid-architecture identifier. |
| `ctx_size` | server flags | int [1, 2_000_000] | The actual context window handed to the server; preferred over `context_size`. |
| `cache_type_k` | server flags | enum, absent ⇒ `f16` | KV-cache quant type for the K half. |
| `cache_type_v` | server flags | enum, absent ⇒ falls back to `cache_type_k` | KV-cache quant type for the V half. |
| `parallel` | server flags | int [1, 256], default 1 | Concurrent server slots; each extra slot adds a flat compute floor. |
| `split_mode` | server flags | enum {none, layer, row, tensor}, default `layer` | GPU placement — single card vs. spanning all cards. |
| `main_gpu` | server flags | int [0, 16], default 0 | Card index used when `split_mode == none`. |
| `no_kv_offload` | server flags | bool, default false | Places the KV cache in host RAM instead of VRAM. |
| `cpu_moe` / `n_cpu_moe` | server flags | bool / int, default false / 0 | Offload expert weights to host RAM; switches the fit test to trust the declared footprint. |

### 1.2 GGUF-derived attention dimensions

A small, dependency-free GGUF header reader pulls the model's real attention geometry so the per-token KV cost can be computed from first principles instead of guessed from file size. It reads only the key/value **header** block — never tensor data or engine-custom quant type-ids — and it is best-effort: it **never raises**, returning nothing on any missing, malformed, oversized, or mismatched input, which cleanly drops the estimate back to the legacy path.

| GGUF header key | Meaning |
|---|---|
| `general.architecture` | Architecture string (`<arch>`). |
| `<arch>.block_count` | Total transformer layers. |
| `<arch>.full_attention_interval` | In a hybrid arch, every Nth layer is a full-attention layer. |
| `<arch>.attention.head_count_kv` | Number of **KV** heads (after grouped-/multi-query sharing). |
| `<arch>.attention.key_length` | Per-head K dimension. |
| `<arch>.attention.value_length` | Per-head V dimension. |
| `<arch>.embedding_length`, `<arch>.attention.head_count` | Fallback used to derive per-head dims when `key_length`/`value_length` are absent (`embedding_length // head_count`). |

The parsed dims are considered usable only when `block_count > 0`, `n_head_kv > 0`, `key_length > 0`, and `value_length > 0`; otherwise the reader yields nothing and the estimate falls back.

### 1.3 Live system state

- **Free VRAM per GPU** — `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`, one integer (MiB) per visible CUDA device, in index order. Total VRAM is probed at boot.
- **Free system RAM** — `MemAvailable` from `/proc/meminfo` (KiB).
- **CPU load / IO-wait** — `os.getloadavg`, `os.cpu_count`, and two samples of the `/proc/stat` `cpu` line for the sibling load and IO-wait gates.

---

## 2. Estimating the model body

The model body reservation is simply the on-disk GGUF size converted to MiB:

```
gguf_mib = gguf_size_bytes // (1024 * 1024)
```

This assumes every weight is GPU-resident. That assumption is deliberately broken in the **expert-offload** case (§5.4), where the body term is dropped because the experts live in host RAM.

---

## 3. Estimating the KV cache — overview

A KV cache grows one K entry and one V entry **per token, per attention layer**. Its byte size per element depends on the KV-cache quantization type. The estimator has **three tiers** with strict, short-circuiting precedence:

```
measured bytes/token   >   dimension-aware GGUF dims   >   legacy file-size heuristic
```

The higher a tier, the more accurate it is; each falls back to the next when its inputs are missing. All three scale **linearly** with context length.

A guard short-circuits the whole thing:

```
if ctx_size <= 0 or gguf_size_bytes <= 0:  KV estimate = 0 MiB   (gate passes "insufficient-input")
```

### 3.1 KV quant scale factors

Every tier that is not a raw measured value scales the cache by a per-type factor relative to f16 (where f16 = 1.0 = 2 bytes/element). The K half and V half are looked up **independently**; the V half falls back to the K type when unset. Any unrecognized label maps to **1.0 (treated as f16)** — this over-counts an unknown low-bit cache rather than under-counting it, which is the safe direction for a gate.

| Cache type | Scale | | Cache type | Scale |
|---|---|---|---|---|
| `f32` | 2.0 | | `q5_0` | 0.32 |
| `f16` | 1.0 | | `q5_1` | 0.32 |
| `bf16` | 1.0 | | `q4_0` | 0.25 |
| `q8_0` | 0.5 | | `q4_1` | 0.25 |
| `turbo4` | 0.25 | | `iq4_nl` | 0.25 |
| `turbo3` | 0.1875 | | `turbo2` | 0.125 |

```
scale_k = QUANT_SCALE[cache_type_k]
scale_v = QUANT_SCALE[cache_type_v or cache_type_k]
```

---

## 4. Estimating the KV cache — the three tiers

### Tier 1 — measured bytes/token (highest precedence)

Taken when `kv_bytes_per_token` is set and > 0. The operator-measured value already reflects real hardware, post-quant and post-hybrid, so it is used **verbatim** — no quant scaling, no hybrid multiply:

```
total_bytes = int(kv_bytes_per_token * ctx_size)
kv_mib      = total_bytes // (1024 * 1024)
```

`kv_bytes_per_token` is in **bytes** per token. A configuration floor of 1024 rejects a KiB-value mistakenly entered as a raw number, which would silently under-count.

### Tier 2 — dimension-aware (real GGUF geometry)

Taken when parsed attention dims are present and the derived attention-layer count is > 0. K and V are computed as two separate terms, each with its own per-head dimension and its own quant scale (the constant `2` is the byte size of an f16 element):

```
k_bytes_f16        = n_attn_layers * n_head_kv * key_length   * 2
v_bytes_f16        = n_attn_layers * n_head_kv * value_length * 2
eff_bytes_per_token = k_bytes_f16 * scale_k + v_bytes_f16 * scale_v
total_bytes        = int(eff_bytes_per_token * ctx_size)
kv_mib             = total_bytes // (1024 * 1024)
```

The attention-layer count is derived from the hybrid interval. When no interval is declared, **every** layer is counted as attention — an intentional over-estimate that never under-reserves:

```
if full_attention_interval > 0:
    n_attn_layers = max(1, block_count // full_attention_interval)
else:
    n_attn_layers = block_count
```

`n_head_kv` is the number of KV heads **after** grouped-/multi-query sharing, which is what actually determines cache size — using the larger query-head count would over-count a GQA/MQA model.

### Tier 3 — legacy file-size heuristic (fallback)

Used when neither a measured value nor usable dims are available. It sizes the cache off the model file, calibrated at roughly **9 KB/token per GiB of body at f16**, and averages the two quant factors (it has only a single lumped figure and cannot separate K from V):

```
gguf_mib             = gguf_size_bytes // (1024 * 1024)
bytes_per_token_kb   = (9 * gguf_mib) // 1024              # KB/token at f16
scale                = (scale_k + scale_v) / 2.0
bytes_per_token_kb   = int(bytes_per_token_kb * scale)
total_kib            = int(bytes_per_token_kb * ctx_size * hybrid_kv_ratio)
kv_mib               = total_kib // 1024
```

### 4.1 `hybrid_kv_ratio` and why the first two tiers ignore it

Some architectures interleave **attention** layers (whose KV grows with sequence length) with **SSM/recurrent** layers (which hold a fixed-size state that does **not** grow per token). `hybrid_kv_ratio` is the attention fraction of layers — a discount that stops the legacy path from sizing every layer as a growing cache. The default of `1.0` is a no-op multiply, keeping non-hybrid models byte-identical to the pre-hybrid behavior. It is honored only when `arch` matches the designated hybrid-architecture identifier; for any other arch it is forced to `1.0`.

Crucially, **Tier 1 and Tier 2 do not apply it**:

- Tier 2's `n_attn_layers` **already counts only the growing attention layers** (e.g. 16 of 64).
- Tier 1's measured value **already reflects real effective usage**.

Multiplying either by `hybrid_kv_ratio` (~0.25) a second time would **double-discount** the cache — the estimate would come out roughly 4× too small, and the gate would admit an over-commit that later OOMs. The ratio is therefore confined to the legacy path, which sizes off total file bytes and has no other way to know that most layers do not grow. Because the legacy heuristic sizes off file bytes and was calibrated on mid-bit weight quants, it badly under-counts an ultra-low-bit model (a small file whose true attention dims are large) — which is exactly why supplying real dims (Tier 2) or a measured value (Tier 1) is preferred for such models.

---

## 5. Fitting it into VRAM

### 5.1 Measuring free VRAM

Free VRAM is read per card from `nvidia-smi` and reduced to a single budget by placement mode:

```
if split_mode == "none":                 # single-card pin
    free_for_fit  = free[main_gpu]        # main_gpu clamped to 0 with a warning if out of range
else:                                     # layer / row / tensor / absent
    free_for_fit  = sum(free over ALL cards)
    min_per_card  = min(free over all cards)

if nvidia-smi is unreadable:  free_for_fit = None
```

With exactly one physical GPU, `sum([x]) == min([x]) == x`, so every branch collapses to that single card and behavior is identical to a naive GPU-0-only check. The multi-GPU logic is a strict, behavior-preserving superset.

### 5.2 The coarse floor check (`check_free_vram`)

A first, manifest-driven check compares free VRAM against the larger of a fixed floor and the model's declared footprint:

```
expected_mib = expected_vram_bytes // (1024 * 1024)
threshold    = max(min_free_vram_mib, expected_mib)     # min_free_vram_mib default 512
REFUSE if free_for_fit < threshold
```

`min_free_vram_mib` is a **floor on the required-free amount** — it is *not* subtracted from measured free VRAM. Even a tiny model must leave at least that floor free. If the probe is missing, this check passes.

### 5.3 The closed-form fit check (`check_kv_cache_fit`)

This is the load-bearing gate. Instead of trusting a hand-declared footprint, it re-derives the requirement from first principles. It first computes a flat per-extra-slot compute floor:

```
p             = max(1, parallel)
par_extra_mib = (p - 1) * 256          # PER_SLOT_COMPUTE_FLOOR_MIB = 256
overhead_mib  = 1024                    # activation/scratch floor (configurable)
```

The context-linear KV term is the **aggregate** window that the inference server splits across slots, so it is counted **once** and never multiplied by the slot count. Only the flat 256 MiB buffer is charged per *additional* slot.

**Default branch — KV resident in VRAM:**

```
total_mib = gguf_mib + kv_mib + overhead_mib + par_extra_mib
REFUSE if total_mib > free_for_fit
```

**No-probe handling:** if `ctx_size <= 0` or `gguf_size_bytes <= 0`, the gate passes ("insufficient-input"). If VRAM cannot be probed, a single-slot spawn (`p == 1`) passes ("no-probe"), but a `p > 1` spawn is **refused outright** — blind-spawning concurrent slots with no VRAM visibility is treated as a guaranteed OOM risk.

### 5.4 Special placement branches

**Host-RAM KV (`no_kv_offload = true`).** The KV cache lives in system RAM, so it is dropped from the VRAM total. A context-linear scratch term is still added (attention buffers grow with context even when the KV is offloaded), and the KV is separately checked against free host RAM:

```
vram_scratch_mib = overhead_mib + ctx_size // 128 + par_extra_mib
vram_need        = gguf_mib + vram_scratch_mib
REFUSE if vram_need > free_for_fit
# complementary host-RAM check:
ram_avail_mib    = MemAvailable_KiB // 1024
REFUSE if kv_mib > ram_avail_mib
```

Note this keys strictly on `no_kv_offload = true`; a `kv_offload = false` is **not** equivalent, because false booleans are dropped when flags are turned into CLI arguments.

**Expert offload (`cpu_moe` or `n_cpu_moe > 0`, with a declared footprint).** When expert weights are offloaded to host RAM, the `body = file-size` term over-counts by many GiB. The gate instead trusts the operator's measured footprint:

```
vram_need = expected_vram_mib + overhead_mib + par_extra_mib
REFUSE if vram_need > free_for_fit
```

This override applies only when `expected_vram_bytes > 0`; otherwise the gate falls back to the closed form (which over-counts for offload configs), so keeping that value accurate is the operator's responsibility for context-bump safety.

### 5.5 Cross-resident reservation

The same math backs co-residence of two warm models. This is intentionally narrow: a second model may load alongside existing ones **only** when the incoming model and every already-loaded sibling are all `split_mode == none` on **distinct** `main_gpu` cards. The reservation charges only siblings still in a loading state on the **same** card (an already-loaded sibling is already reflected in the live free-VRAM reading; charging it again would double-count):

```
reserve = sum(reserved_need_mib of still-loading siblings on the SAME main_gpu)
ADMIT if (free_for_fit - reserve) >= need
```

A note on the reservation footprint: pre-spawn budget accounting uses `max(declared_footprint, closed_form_body_plus_kv)` as a conservative floor, and the live gate then re-checks against real free VRAM at spawn time.

---

## 6. The full gate sequence

The fit checks above run as part of a five-gate battery, evaluated in a **fixed order** with **no short-circuiting** — every gate always runs, so the audit trail and the error returned to the caller carry the complete picture. **All gates must pass** for the spawn to proceed; if any gate reports failure, the manager marks the load failed, writes an audit record, moves the slot to a failed/popped state **without launching any server process**, and fails the caller's request with a message concatenating every failed gate's detail. The whole subsystem can be disabled with a single runtime flag, in which case the manager spawns unconditionally.

| # | Gate | Passes when | Missing-probe behavior |
|---|---|---|---|
| 1 | Free system RAM | `MemAvailable // 1024 ≥ min_free_ram_mib` | Pass |
| 2 | Free VRAM (floor vs. declared) | `free_for_fit ≥ max(min_free_vram_mib, expected_mib)` | Pass |
| 3 | Closed-form KV-cache fit | `predicted total ≤ free_for_fit` | Pass at `parallel = 1`; **refuse** at `parallel > 1` |
| 4 | CPU load per core | `loadavg[0] / cpu_count ≤ max_load_per_core` | Pass |
| 5 | Disk IO-wait % | sampled `%iowait ≤ max_iowait_percent` | Pass (also passes on zero delta) |

Every gate **degrades open** — a missing or unreadable probe returns a pass rather than blocking — with the single exception of the multi-slot no-probe case in Gate 3. Gates 2 and 3 are **complementary, not redundant**: Gate 2 uses the hand-declared footprint, while Gate 3 recomputes from context size + body + quant. Bumping the context window in the manifest is caught by Gate 3 even if the declared footprint was never re-tuned to match.

```
gates  = [free_ram, free_vram, kv_cache_fit, cpu_load, iowait]
failed = [g for g in gates if not g.ok]
spawn proceeds only if failed == []
# on refusal:
detail = "; ".join(f"{g.name}: {g.detail}" for g in failed)
fail caller with: f"safety gates refused spawn: {detail}"
```

---

## 7. Worked examples

Both examples use **generic** model sizes. Free VRAM figures are the values `nvidia-smi` would report on the given card.

### 7.1 Default-path model (no measured override)

A 27B-class dense model with a **17 GiB** body, an **f16** KV cache, a **64K** context (`ctx_size = 65536`), single slot, default 1024 MiB overhead. No parsed dims and no measured value are available, so the **legacy** KV tier is used.

```
gguf_mib           = 17 GiB               = 17408 MiB
bytes_per_token_kb = (9 * 17408) // 1024  = 153 KB/token   (f16)
scale              = (1.0 + 1.0) / 2       = 1.0
kv_mib             = (153 * 65536) // 1024 = 9792 MiB       (~9.56 GiB)
par_extra_mib      = (1 - 1) * 256         = 0
total_mib          = 17408 + 9792 + 1024 + 0 = 28224 MiB    (~27.6 GiB)
```

- On **one 24 GB card** (~23,000 MiB free): `28224 > 23000` → **REFUSE**.
  Detail: `kv_cache_fit: need ~28224 MiB (body=17408 + KV@ctx65536=9792 [f16] + overhead=1024); only 23000 MiB free`.
- On **one 32 GB card** (~31,000 MiB free): `28224 ≤ 31000` → **PASS**.
- On **two 24 GB cards, layer split** (budget `≈ 46000` MiB): `28224 ≤ 46000` → **PASS** — the exact model that failed on one card fits once the budget aggregates both.

Two knobs lower the requirement and can flip a refusal into a fit on the 24 GB card. Quantizing the KV to `q8_0` (`scale = 0.5`): `kb_per_token = 76`, `kv_mib = (76 * 65536) // 1024 = 4864 MiB`, `total = 23296 MiB`. Or dropping the context to 8K with `q8_0`: `kv_mib = (76 * 8192) // 1024 = 608 MiB`, `total = 19040 MiB ≤ 23000` → **PASS**.

### 7.2 Large-context model that fits only via a measured override

A 27B-class **hybrid** model with a **17 GiB** body, run at a **128K** context (`ctx_size = 131072`), f16 KV, on **one 24 GB card** (~23,000 MiB free).

**Without an override (dimension-aware tier).** With parsed dims of `n_attn_layers = 16`, `n_head_kv = 16`, `key_length = value_length = 128`:

```
k_bytes_f16 = 16 * 16 * 128 * 2 = 65536 bytes/token
v_bytes_f16 = 65536 bytes/token
eff/token   = 65536 * 1.0 + 65536 * 1.0 = 131072 bytes/token
kv_mib      = (131072 * 131072) // (1024*1024) = 16384 MiB   (16 GiB)
total_mib   = 17408 + 16384 + 1024 = 34816 MiB               (~34 GiB)
```

`34816 > 23000` → **REFUSE** (and it would still be refused on a 32 GB card).

**With a measured override.** The operator measures the model's real effective KV cost on hardware — the hybrid recurrent layers hold fixed state, so the true per-token cost is far below the geometry-only estimate — and records `kv_bytes_per_token = 32768` (32 KiB/token). Tier 1 uses it verbatim:

```
kv_mib    = (32768 * 131072) // (1024*1024) = 4096 MiB       (4 GiB)
total_mib = 17408 + 4096 + 1024 = 22528 MiB                  (~22 GiB)
```

`22528 ≤ 23000` → **PASS**. The same 128K spawn that the closed-form geometry refuses now fits, purely because the authoritative measured value replaces the conservative geometry estimate. This is the intended escape hatch: when the closed-form tiers over-count a model whose real behavior is known, a measured bytes/token value lets it load safely — while still refusing anything that genuinely will not fit.

---

## See also

- [MODEL_CONFIG_REFERENCE.md](./MODEL_CONFIG_REFERENCE.md) — the full per-model manifest and server-flag reference (every field summarized in §1.1).
- [MULTI_GPU_PLACEMENT.md](./MULTI_GPU_PLACEMENT.md) — how `split_mode`, `main_gpu`, and GPU count shape the VRAM budget (§5.1) and cross-resident placement (§5.5).
- [ARCHITECTURE.md](../ARCHITECTURE.md) — where the safety gate sits in the spawn/admission pipeline.
