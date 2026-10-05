# TurboQuant KV Quantization — what `turbo2` / `turbo3` / `turbo4` actually are

This document covers the **KV cache quantization types** themselves: what the algorithm is, what
the `cache_type_k` / `cache_type_v` values mean, and — most importantly — **the one case where the
engine will silently give you a different K type than the one you asked for.**

For the *operational serving flags* (`flash_attn`, `no_context_shift`, `cache_reuse`,
`slot_prompt_similarity`, `no_perf`), see [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md). This document is
about the KV data format; that one is about how the server is launched.

---

## What TurboQuant is

TurboQuant is a data-oblivious vector quantization algorithm from Google Research, published at
ICLR 2026. It compresses the attention key/value cache to roughly **3 bits per coordinate**.

The published method works in two stages:

1. **PolarQuant** — a random rotation is applied to each vector, which induces a Beta distribution on
   the coordinates. Because the resulting distribution is known and fixed, an optimal scalar quantizer
   can be applied per coordinate without needing to look at the data first.
2. **A 1-bit QJL residual pass** — a Quantized Johnson–Lindenstrauss transform is applied to what the
   first stage got wrong, reducing the residual to a single sign bit per dimension.

The turbo types in this repository's engine (`ggml/src/ggml-turbo-quant.c`) implement the first stage:
a Walsh–Hadamard rotation followed by PolarQuant quantization. `turbo2` and `turbo3` carry no QJL
residual, and `turbo4` is 4-bit PolarQuant with no QJL in the default build (`TURBO4_USE_4BIT`).

The published results report roughly **6x memory reduction** on the KV cache and up to **8x faster
attention** on H100-class hardware, with accuracy matched against 16-bit baselines on long-context
benchmarks. The algorithm is training-free and requires no calibration data, which is what makes it
usable as a drop-in cache format rather than a per-model tuning exercise.

**Primary sources:**

- Google Research announcement — <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>
- Paper: *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*, arXiv:2504.19874 — <https://arxiv.org/abs/2504.19874>
- ICLR 2026 camera-ready — <https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f.pdf>
- Component papers: QJL, arXiv:2406.03482 · PolarQuant (AISTATS 2026), arXiv:2502.02617

## The practical consequence

A turbo type is **much smaller than the integer quant it replaces**. That is the main reason to
prefer it: `q8_0` spends 8 bits per value on a format that was never designed for attention caches,
while a turbo type spends ~3, and the published results report accuracy matched against 16-bit
baselines at that size. Accuracy still depends on the model — see the two overrides and the
per-model verification below. When choosing a cache type, the turbo variants should be the default
and an integer quant should be the exception that carries a reason.

| value | approx. bits/coord | notes |
|---|---|---|
| `turbo2` | ~2 | most aggressive; largest quality risk |
| `turbo3` | ~3 | the balanced default for production serving |
| `turbo4` | ~4 | most conservative turbo variant |
| `q8_0` | 8 | integer fallback; larger than a turbo type |
| `f16` | 16 | unquantized reference |

---

## ⚠ The engine can override your declared K type — and it does so silently

This is the part that is easy to miss, because a manifest can declare `cache_type_k: turbo3` and the
running engine can still be using `q8_0` for K.

**The rule, implemented in `llama-kv-cache.cpp`:** when a *turbo* K type is requested, the engine
computes the model's GQA ratio (`n_head / n_head_kv`, read from the first layer). If **all** of the following hold, it replaces
the requested K type with `q8_0`:

- the requested K type is `turbo2`, `turbo3` or `turbo4`, **and**
- the GQA ratio is **>= 6**, **and**
- **`cache_type_k` and `cache_type_v` are the same value**, **and**
- the environment variable `TURBO_AUTO_ASYMMETRIC` is not set to `0`

### Why the guard exists

Key quantization error is amplified by the GQA broadcast factor: a small number of KV heads is shared
across many query heads, so error in K is multiplied. The engine's own reference figures record a model
at a **7:1** ratio degrading catastrophically under turbo K (perplexity in the thousands against a
single-digit baseline), while a **4:1** model works acceptably (about +4.4% perplexity). The threshold is set at 6:1.

**A model at 8:1 is on the worse side of the ratio that was measured as catastrophic.** Treat any
decision to bypass the guard on such a model as requiring its own quality measurement.

### How to tell what you actually got

The engine emits a warning at load time when it overrides. Its absence is as meaningful as its presence:

```
llama_kv_cache: auto-asymmetric: GQA ratio 8:1 (n_head=16, n_head_kv=2) —
  upgrading K from turbo3 to q8_0 to prevent quality degradation.
  Disable with TURBO_AUTO_ASYMMETRIC=0
```

Grep the engine log for `auto-asymmetric` after a **cold spawn**. If the line is present, your declared
K type was discarded. Checking the manifest is not sufficient — the manifest records the request, not
the outcome.

### The shipped default is itself the trigger — and the sizing gate does not know

Two things compound here. First, a model that sets neither cache type inherits `turbo3` for **both** K
and V, which is symmetric turbo — precisely the condition the guard fires on. So on any model at a
GQA ratio of 6 or more, the out-of-the-box configuration is the one that gets its K silently upgraded.

Second, the manager's pre-spawn KV-fit gate sizes the K half from the type the manifest *declares*. It
has no view of the engine's GQA decision, which happens later and inside the engine process — the gate
never computes a GQA ratio at all. When the override fires, the gate has budgeted K at 0.1875x f16
while the engine allocates it at 0.50x: an under-count of roughly **2.7x on the K half**, or roughly
**1.8x on the cache as a whole** where the K and V head dimensions are equal. At long context that is
real headroom the gate believes it has and does not.

This is about the *estimate*, not about which type you end up serving. To keep the gate's number
truthful on a high-GQA model, either declare K and V as different types — not overridden by the GQA guard, so the
declared K type and the allocated K type agree (a `turbo2` V is still subject to the layer-adaptive
rewrite described below) — or set a measured `kv_bytes_per_token` on the manifest,
which is used as-is with no quant scaling applied and is therefore the only input that reflects what
the engine actually allocated.

### The two ways to actually get a turbo K on a high-GQA model

**1. Declare K and V as different types.** The `cache_type_k == cache_type_v` condition is the guard's
opt-out: it only overrides when *symmetric* turbo was requested. Setting them to different turbo variants is
not touched by this guard (the layer-adaptive mechanism below can still rewrite a `turbo2` V).
This is **per-model, needs no environment change, and affects nothing else** — prefer it.

**2. Set `TURBO_AUTO_ASYMMETRIC=0`.** This is the only route to a *literally symmetric* turbo K and V.
Note the scope: the engine reads it from its own environment, which it inherits from the manager
process, so **it applies to every model that process spawns** — it is not a per-model setting. Changing
it requires recreating the manager container, and disables the quality guard for every model.

## ⚠ A SECOND override, on V: layer-adaptive (`TURBO_LAYER_ADAPTIVE`)

The GQA guard above is not the only place the engine replaces a declared cache type. The same file
carries a second, independent mechanism that rewrites types **per layer** — and one of its modes
**turns itself on with no configuration at all**.

**The auto-enable case, which is the one that will surprise you:** if `cache_type_v` is `turbo2` and
the model has **8 or more layers**, the engine enables *Boundary V* (mode 7) by itself. V then becomes
`q8_0` on the first two and last two layers, and stays `turbo2` on the rest. No environment variable is
involved, and nothing in the manifest says so.

```
llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)
llama_kv_cache: Boundary V mode 7: first2+last2 V=q8_0, rest V=turbo2
```

The explicit modes are selected by setting `TURBO_LAYER_ADAPTIVE=<n>`; `0` disables the mechanism
entirely, including the auto-enable above.

| Mode | Requires | Effect |
|---|---|---|
| `1` | turbo **K**, >= 8 layers | first 4 and last 4 layers: **K and V** -> `q8_0` |
| `2` | turbo **K**, >= 8 layers | last 8 layers: **K and V** -> `q8_0` |
| `5` | turbo **V**, >= 8 layers | first 2 + last 2 layers V -> `turbo4`, all other layers V -> `turbo2` |
| `6` | turbo **V**, >= 8 layers | last 8 layers V -> `turbo4`, all other layers V -> `turbo2` |
| `7` | turbo **V**, >= 8 layers | first 2 + last 2 layers V -> `q8_0`, all other layers V -> `turbo2` — **auto-enabled for turbo2 V** |

Two properties worth knowing before you rely on this:

- **Modes 1 and 2 rewrite K as well as V**, and they key off whether **K** is a turbo type. They are not
  V-only tuning knobs despite sitting alongside modes 5-7.
- The mode is resolved **once per engine process** and reused for every cache built in it. Turbohaul
  spawns one engine process per model, so in Turbohaul that is equivalent to per-model — but it
  would not be if an engine were ever made to host more than one model.

## ⇒ Checking one log line is not enough

Because there are two independent override mechanisms, grepping for `auto-asymmetric` alone answers
only half the question. Its absence tells you K was not upgraded by the GQA guard; it tells you nothing
about V.

---

## Verification recipe

```bash
# 1. What the manifest REQUESTS
grep -E 'cache_type_[kv]' <manifest>.yaml

# 2. What the engine actually DID — cold spawn required; check the log it wrote
grep -iE 'auto-asymmetric|layer-adaptive|Boundary V' <engine-log>
#    'auto-asymmetric ... upgrading K'  -> K was overridden to q8_0, regardless of the manifest
#    'Boundary V' / 'layer-adaptive'    -> V (and under modes 1-2, K) was rewritten PER LAYER
#    neither line present               -> the declared K and V types are in effect
#
#    NOTE: older engine builds emitted an 'auto-asymmetric risk: ... K configured as ...'
#    advisory that did NOT override anything. If you are reading archived logs, match on
#    'upgrading K from' rather than on 'auto-asymmetric' alone, or you will read a warning
#    that K was KEPT as evidence that it was replaced.

# 3. Whether the guard is disabled process-wide
#    (inspect the manager container's environment for TURBO_AUTO_ASYMMETRIC)
```

## ⚠ A turbo type is not guaranteed to work on every model — and a bad pairing is SILENT

The two overrides above are cases where the engine *changes* your K or V type and still serves correct
output. This section is the opposite and more dangerous case: **you get exactly the type you asked for,
and the model produces incoherent text.**

There is no compatibility check. Turbohaul will pair any model with any `cache_type_k` / `cache_type_v`
value the manifest validator accepts. Whether a given model family actually survives a given turbo type
is an empirical property of that model, and it is **the operator's responsibility to verify**. New model
architectures appear far too often for this repository to carry a maintained compatibility matrix, and a
stale matrix would be worse than none.

### What a bad pairing looks like

Nothing in the serving path fails. A bad pairing can look like this:

- the engine starts normally and logs no error or warning
- `/health` returns 200 and the model reports as resident
- the request returns HTTP 200 with a normal-looking response envelope
- tokens-per-second is in the normal range

Only the **text** is wrong — typically degenerate repetition, stray or duplicated control tokens, or
fluent-looking output that never answers the prompt. Example of what such output can look like, from a
model paired with a turbo K/V type it had not been validated against:

```
</think>



</think>



Okay

Okay

Okay
```

Nothing in the serving path distinguishes this from a healthy response. **A liveness or smoke check that
asserts "the server answered" will pass.** Asserting on response *shape* is not enough here; the check
has to look at the content.

### Verify a new model before you rely on it

Do this once, per model, when you first introduce it or change its cache type:

1. Serve the model with the cache types you intend to use.
2. Send a prompt whose correct answer you can recognise at a glance — a greeting, or
   `"Count from one to twenty in words."` Deterministic settings (`temperature: 0`) make the result
   reproducible.
3. **Read the generated text.** Do not score the run on HTTP status, latency, or token count.
4. If it is incoherent, re-run the identical prompt with `cache_type_k: f16` and `cache_type_v: f16`.
   - coherent on `f16`, incoherent on the turbo type ⇒ **that pairing is unusable**; keep the model on a
     type it works with. This is a property of the model, not a bug in the manifest.
   - incoherent on both ⇒ the problem is not the cache type. Look at the model file, the chat template,
     or the sampling parameters.

Step 4 is the load-bearing one: without the `f16` arm you cannot tell a cache-type incompatibility from
a broken model, and the two have completely different remedies.

### Practical guidance

- Treat the turbo types as **validated per model, not globally**. A type that is correct for one model
  family carries no guarantee for another.
- When importing an unfamiliar model, start on `f16`, confirm the output is sane, then move to a turbo
  type and confirm again. The second confirmation is the one people skip.
- Keep the prompt you used. Re-run it after any engine upgrade that touches quantization.

## See also

- [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md) — the serving flag doctrine (launch flags, not cache format).
- [KV_CACHE_MATCHING.md](./KV_CACHE_MATCHING.md) — how a saved cache is matched to an incoming request.
- [HYBRID_KV_RATIO.md](./HYBRID_KV_RATIO.md) — KV sizing on hybrid architectures where only some blocks carry KV.
- [MODEL_CONFIG_REFERENCE.md](./MODEL_CONFIG_REFERENCE.md) — the full per-model configuration surface.
