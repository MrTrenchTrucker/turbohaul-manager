# Speculative decoding — MTP, D-Flash and D-Spark

Speculative decoding runs a small, cheap **drafter** ahead of the real model. The drafter proposes
several tokens; the target model verifies them in a single pass and keeps the ones it agrees with.
Accepted tokens are free. **Rejected tokens are not** — you paid to draft them and threw them away.

That trade is the whole subject of this document. Whether it wins is an empirical question about
*your* model, *your* hardware and *your* workload, and the answer is not stable across any of them.

---

## The three methods

| method | drafter lives | needs a second file | use when |
|---|---|---|---|
| `draft-mtp` | **inside the target GGUF** (a bundled multi-token-prediction head) | no | the model ships an MTP head |
| `draft-dflash` | a **separate** GGUF, block-diffusion | yes | the model has no MTP head |
| `draft-dspark` | a separate GGUF: D-Flash **+ a Markov head and a confidence head** | yes | as D-Flash, and a D-Spark checkpoint exists for your target |

`draft-dspark` is the more recent design. Its confidence head scores its own guesses and, when `spec_draft_p_min` is above 0,
truncates a block early at the first position below that threshold, which can raise acceptance per drafted
token. That does **not** automatically make it faster than D-Flash — acceptance per drafted token and raw
throughput are different measures, so measure both.

A drafter is trained **for a specific target**. It is not a general-purpose accelerator you can point
at any model. A mismatched pair either refuses to load or produces poor drafts.

---

## Choosing between them

**If the model ships an MTP head, start with MTP and make the alternative prove itself.**
The bundled head was trained with the model, shares its weights, and costs no extra VRAM for a second
model. In one measurement on a 27B, MTP accepted **66%** of drafted tokens where a separate
D-Flash drafter accepted **13%** on the same prompt.

**D-Flash and D-Spark exist for models with no MTP head.** There the comparison is not
"drafter vs better drafter", it is "drafter vs no acceleration at all", which is a much easier bar.

**Neither is a default.** Speculation is off unless a manifest opts in, and that is deliberate.

---

## Enabling it

Per-model, in the manifest. Nothing is global.

### MTP — bundled head, no second file

```yaml
llama_server_flags:
  spec_type: draft-mtp
  spec_draft_n_max: 3        # tokens drafted per step
```

### D-Flash / D-Spark — separate drafter

First import the drafter GGUF so it lands in the content-addressed blob store:

```bash
curl -X POST http://<manager>/api/import \
  -H 'Content-Type: application/json' \
  -d '{"path":"/var/lib/turbohaul/import-staging/<subdir>/<drafter>.gguf"}'
# -> {"sha256":"<DRAFTER_BLOB>", ...}
```

Then reference it from the target's manifest:

```yaml
spec_draft_gguf_blob_sha256: "<DRAFTER_BLOB>"
llama_server_flags:
  spec_type: draft-dflash      # or draft-dspark
  spec_draft_n_max: 15         # see "block size" below
```

Optional, and rarely needed — the drafter has its own placement and cache settings, independent of the
target's: `spec_draft_ngl`, `spec_draft_type_k`, `spec_draft_type_v`, `spec_draft_n_min`,
`spec_draft_p_min`.

### `spec_draft_n_max` and block size

A block-diffusion drafter is trained for a fixed **block size**, readable from the drafter GGUF as
`dflash.block_size`. `spec_draft_n_max` is clamped to the largest draft a block can yield — the block size for
D-Spark, one less for D-Flash — so setting it higher does nothing and setting it much lower wastes the
drafter's capacity. Match it to the file.

⚠ The default (3) comes from MTP. It is **wrong** for a D-Flash drafter trained at block size 16 —
set it explicitly.

### Confirming it actually engaged

Do not assume the flag took effect. The engine says so at load:

```
common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'
  - n_max=15, n_min=0, p_min=0.00
  - block_size=16, mask_token_id=..., n_extract=5
srv load_model: speculative decoding context initialized
```

If speculation could not be set up, the engine **downgrades rather than failing** — it serves the model
normally, without acceleration. When the draft context cannot be constructed it logs `SPEC_DOWNGRADED` with an
architecture, a component and a reason; other setup failures (for example a drafter GGUF with no
`dflash.block_size`) are logged as `failed to initialize speculative decoding context`, and the model likewise
serves without a drafter. A downgraded model looks completely healthy from the outside. Check the log, or the response
timings: `draft_n` and `draft_n_accepted` are present **only** when a drafter really ran.

---

## Results vary wildly. Measure, do not assume.

This is the part that matters most, and it is not a disclaimer — it is the main finding.

**Across models.** A drafter's acceptance depends on how predictable its specific target is. In one
comparison of two pairings, acceptance on the same prompt ranged from **13% to 66%**. Nothing about the
configuration explained it; the models did.

**Across hardware.** Speculation trades extra compute for fewer sequential steps. That trade improves
as your card has spare compute relative to memory bandwidth, and worsens when the GPU is already
saturated or when a second model competes for VRAM. A pairing that wins on one card can lose on
another with no software change at all.

**Across per-model configuration.** Context size, KV cache type, batch and parallel settings all move
the result, because they change what the extra draft work costs. Re-measure after changing any of them.

**Across workloads — and this one changes the sign, not just the size.** A drafter predicts repetitive
or formulaic text easily and novel reasoning poorly. In one example, measuring the same D-Spark pairing on a short
formulaic prompt and on a long open-ended reasoning task, acceptance fell from **51% to 27%**, and the
verdict flipped from *faster than no drafter* to **slower than no drafter**.

> **A short synthetic prompt does not merely give a noisy number. It can tell you a feature helps when
> on real work it hurts.** Benchmark with a task that resembles your actual traffic, and with a token
> budget large enough to cover reasoning plus answer.

### How to decide, concretely

1. Serve the model **without** a drafter. Long realistic prompt, generous `max_tokens`, note `tok/s`.
2. Serve it **with** the drafter, same prompt, same budget, same machine, nothing else running.
3. Compare `tok/s`, and read `draft_n` / `draft_n_accepted` to see *why*.
4. Keep the drafter only if it is meaningfully faster. A low acceptance rate is not a tuning problem to
   grind at — it usually means that drafter does not suit that target.

Run the arms **sequentially on an otherwise idle GPU**. A leftover process holding VRAM will change the
result, and it will look like a property of the feature rather than of your machine.

---

## See also

- [MODEL_CONFIG_REFERENCE.md](./MODEL_CONFIG_REFERENCE.md) — the full per-model configuration surface.
- [TURBOQUANT_KV_QUANTIZATION.md](./TURBOQUANT_KV_QUANTIZATION.md) — KV cache types, and the fact that a
  model paired with an unvalidated cache type fails silently.
- [TURBOQUANT_FLAGS.md](./TURBOQUANT_FLAGS.md) — the serving flags.
- [MTP_VIDEO_TRADEOFF.md](./MTP_VIDEO_TRADEOFF.md) — MTP's draft context and long video both draw from
  the same context window; measured numbers for the trade-off if the model also serves video.
