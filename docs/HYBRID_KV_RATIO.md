# Hybrid KV Ratio

How to tell the VRAM fit gates that only part of a model's layers grow their KV
cache with context length.

## The setting

| | |
|---|---|
| Manifest key | `hybrid_kv_ratio` |
| Default | `1.0` (no discount — every layer counted) |
| Range | `0.0` – `1.0` |
| Scope | per model, set in that model's manifest |

## Background

For a conventional transformer, KV cache grows linearly with context across
**every** layer. The fit gates size a load by multiplying per-token KV cost by
the context length, and that estimate is accurate.

Hybrid architectures break that assumption. In a model that mixes attention
layers with state-space (SSM) layers, the SSM layers carry a **constant-size**
recurrent state — it does not grow at all as the context gets longer.

Sliding-window attention layers are **not** the same case and must not be
counted here. A sliding-window layer's KV is bounded by its window rather than
being free: it grows with context up to the window size and then stops. The
estimator models that separately when it works from the attention dimensions parsed from the model
file (path 2 under *What it affects*), capping those layers at the window instead of
zeroing them. Folding sliding-window layers into `hybrid_kv_ratio` treats them
as costing nothing and under-estimates the model.

Counting every layer as if it were full attention therefore over-estimates the
cache requirement, sometimes by a large margin. The practical result is a model
that fits comfortably in the available memory being **refused** by the fit gate,
or being given a far larger reservation than it needs.

`hybrid_kv_ratio` is the correction: the fraction of layers that actually
contribute to per-token KV growth.

## Choosing a value

The ratio is *layers whose KV grows with context* divided by *total layers*.
Count only genuinely non-growing layers — SSM/recurrent layers — in the
discounted remainder; sliding-window attention layers are handled by the
estimator's own window cap (on its dimension-based path) and are not discounted here.

For a model with 48 layers of which 12 are attention and 36 are SSM:

```
12 / 48 = 0.25
```

Set it in the model's manifest:

```yaml
hybrid_kv_ratio: 0.25
```

Leave it unset for any conventional model. The default of `1.0` means "count
every layer", which is the correct and safe behaviour for standard attention.

### Verifying it

When a model carries a ratio below `1.0`, the manager logs it once per model
the first time that model is sized:

```
hybrid KV discount active (footprint): model=<tag> hybrid_kv_ratio=0.25
(SSM layers excluded from the per-token KV estimate)
```

The parenthesised word names which sizing point emitted it — `footprint`,
`spawn gate`, or `spawn safety gate`.

**Read this line as "a ratio is set", not as "a discount was applied."** It is
emitted from the manifest value before the estimator picks its path, so it also
appears on a model whose measured or dimension-derived estimate will ignore the
ratio entirely (see *What it affects* below). Absence of the line does mean no
discount is coming from this field: either the model sits at the default `1.0`,
or it has already been announced once in this process.

## What it affects

The ratio reaches the estimate through one function, which all three
memory-sizing points call:

- the **reservation footprint** — how much memory is set aside for the model
- the **per-spawn gate** — whether a spawn is permitted
- the **spawn safety gate** — the cross-resident over-commit check

Passing it at all three is deliberate. If the reservation used a discount the
admission gate did not, the two would disagree and the over-commit check would
be computed against a figure that no longer matched the reservation.

**But the ratio only takes effect on one of three estimation paths.** The
estimator resolves per-token KV cost in strict precedence order:

1. `kv_bytes_per_token` — an operator-measured figure. Used verbatim.
2. Attention dimensions parsed from the model file — computed from first
   principles.
3. The legacy file-size heuristic.

`hybrid_kv_ratio` applies to **path 3 only**. Paths 1 and 2 ignore it by
design: each already reflects only the layers that grow, so applying the ratio
again would discount the same layers twice. Dimension parsing is attempted for
every model with a resolvable model file, but the parsed dimensions are only
used when the file shows at least one derivable sliding-window layer or the
manifest sets a non-empty `arch`. Any other model (an SSM hybrid with no
sliding-window layers, for example) stays on path 3, where the ratio applies;
on a model that does use its dimensions the ratio is inert.

## Cautions

> **A ratio below 1.0 lowers the estimated memory requirement.** Setting it on a
> model that is *not* hybrid tells the fit gates the model needs less memory
> than it does, which can allow an over-commit and an out-of-memory failure
> under long contexts. Only set it for models whose architecture genuinely has
> non-growing layers, and derive it from the layer counts rather than by
> guessing.

If you have a **measured** per-token KV figure for a model, prefer
`kv_bytes_per_token` — a direct measurement is always better than a derived
ratio, and it takes precedence in the estimate. `hybrid_kv_ratio` is the option
when no measurement is available.

A value outside `[0.0, 1.0]` is **rejected** when the manifest is validated —
the manifest fails to load rather than being silently corrected to the nearest
legal value. There is no clamping: a typo here is a load error, not a quiet
change in behaviour.

## Related

- `docs/MODEL_CONFIG_REFERENCE.md` — the full manifest field reference
- `docs/MULTI_GPU_PLACEMENT.md` — how the sized footprint drives placement
- `docs/MODEL_LOAD_TIMEOUT.md` — bounding the wait when a load runs long
