# Multi-GPU Placement & Model Co-Residency

Turbohaul-Manager runs on multi-GPU hosts and decides *which card each model lands on*. This guide covers three related capabilities:

1. Running several models **co-resident on one GPU**.
2. **Automatically spreading** models across all GPUs.
3. The **safety headroom** that keeps every card from over-committing.

For the per-model manifest fields see [MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md); for how the manager estimates a model's VRAM footprint see [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## Concepts

- **Sidecar / resident / engine** - a running model process. `max_parallel_sidecars` (wrapper config) caps how many can be resident at once, counted across all models.
- **`split_mode`** (per-model manifest flag under `llama_server_flags`):
  - `none` — the model loads **whole** onto a single card. Several `none` models can co-reside on one card, or be spread across cards.
  - `layer` — the model's layers are **split across all visible GPUs**, for a model too large to fit whole on one card. (`row` / `tensor` are also accepted for other split strategies.)
- **`main_gpu`** — for `split_mode: none`, which card the model pins to (default `0`).

---

## 1. Multiple models per GPU (co-residence)

More than one `split_mode: none` model can share a single GPU. When admitting a model, the manager does **per-card VRAM accounting**: it looks at the target card's free VRAM (accounting for models already resident there and any loads in flight) and only admits the new model if the card has room for its weights **and** KV cache. If the card would over-commit, the manager does not force the load: it will evict an idle resident to make room and re-queue the request, which keeps retrying on a backoff for as long as the client stays connected instead of failing it. Nothing is spawned onto a card that the free-VRAM check says cannot hold it (a lone spawn passes that check when the free-VRAM reading is unavailable).

Raise `max_parallel_sidecars` (wrapper config, default `1`) to allow more co-resident engines - for example on a two-card host serving many small models at once. A model with `auto_place: true` and an explicit `split_mode: none` may also use that room for a second engine of its own on another card; see [SIDECARS_AND_CONTEXT_WINDOWS.md](SIDECARS_AND_CONTEXT_WINDOWS.md). The number of concurrent conversations inside one engine is a separate setting, `llama_server_flags.parallel`.

---

## 2. Automatic placement (`auto_place`)

By default, each `split_mode: none` model pins to its manifest's `main_gpu` (default `0`). Without any intervention that means every model piles onto card 0. Set **`auto_place: true`** (a top-level manifest field) and the manager chooses the card for you:

- It reads the **free VRAM on every visible card**.
- It subtracts the per-card safety margin (`safety_min_free_vram_mib`) and any in-flight loads.
- It picks the **least-loaded card that still fits** the model (weights + KV) — so models spread out: as one card fills, the next model lands on the emptier card.
- If **no single card** fits the model right now, it next looks for one card that would hold the model whole once the idle models on that card are counted as free; if it finds one, it unloads those idle models and places the model there (see [Unloading idle models to keep a model on one card](FAST_LANE.md#unloading-idle-models-to-keep-a-model-on-one-card)). Only if no single card fits even so, but the cards *together* have room, does it fall back to **`split_mode: layer`** (splitting that one model across cards) — typically only the largest model in a mixed set. The unloading step does not act while a model that is split across cards is loaded.
- If nothing fits right now, the request is queued rather than failed — the manager frees an idle resident where it can and retries on a backoff for as long as the client stays connected.

`auto_place` is **opt-in and back-compatible**: the default (`false`) preserves the manifest's explicit `main_gpu` pin (except while Fast Lane is enabled, which also runs the card pick for a `split_mode: none` model), so existing manifests behave exactly as before. `auto_place` takes effect only with `split_mode: none` (an absent `split_mode` counts as `layer`); it does not depend on the value of `max_parallel_sidecars`.

Manifest example:

```yaml
model_tag: my-small-model
gguf_blob_sha256: <64-hex>
gguf_size_bytes: 1400000000   # REQUIRED: the closed-form KV-fit gate
                              # no-ops when this is 0
context_size: 8192
expected_vram_bytes: 1600000000
auto_place: true
llama_server_flags:
  ctx_size: 8192
  split_mode: none
  cache_type_k: turbo3
  cache_type_v: turbo3
  flash_attn: true
```

### Balancing by free VRAM

Auto-placement balances **free VRAM**, not model count. On a two-GPU host where several small models are loaded with `auto_place: true`, the manager spreads them so both cards keep roughly equal free space. If one card starts with more already in use, the manager places proportionally more on the emptier card until the free space evens out — so the resulting per-card *count* may differ while the *free VRAM* converges. Each card keeps its safety margin clear, and the models serve inference **concurrently across both GPUs**.

---

## 3. Safety headroom (`safety_min_free_vram_mib`)

`safety_min_free_vram_mib` (wrapper config, under `queue`, default `512`) is the amount of VRAM the manager keeps free on **each** card when it is *choosing* a card. Automatic placement subtracts it per card, so a model is only auto-placed where the card still has at least this much free after loading. The admission gates use the same number differently: it is a **floor on the required-free amount** — even a tiny model must find at least this much free VRAM — and is never subtracted from the measured reading. The cross-resident co-residence gate does not apply it at all. Raise it (for example to `4096` for 4 GB) to leave headroom for KV-cache growth and to stay clear of driver-level OOM. It is set in the `queue` section of the runtime configuration file, and is also readable and writable through the configuration API (`GET /api/config` / `PUT /api/config`) without editing the file by hand.

---

## Knobs at a glance

| Setting | Scope | Meaning |
|---|---|---|
| `queue.max_parallel_sidecars` | wrapper | max engines on the box, across all models (1-32, default 1); also the ceiling on how many copies of one model can run |
| `queue.safety_min_free_vram_mib` | wrapper | VRAM kept free on each card when auto-placing (default 512) |
| `auto_place` | per-model | opt into automatic card selection (default false) |
| `auto_place` with `split_mode: none` | per-model | together these allow a second engine of the same model, on a different card, while the box budget has room; a pinned (`auto_place: false`) or card-spanning model keeps one engine |
| `split_mode` | per-model | `none` (whole, one card) / `layer` (split across cards) |
| `main_gpu` | per-model | card pin when `auto_place` is off |
| `tensor_split` | per-model | per-GPU split ratio for a card-spanning model; its element count must equal the number of visible devices or the spawn is refused |

**Rule of thumb:** on a multi-GPU host, mark small `split_mode: none` models `auto_place: true`, set `max_parallel_sidecars` to the number you want co-resident, and set `safety_min_free_vram_mib` to the headroom you want on each card. Leave the single largest model to fall back to `layer` if it can't fit whole on one card even after idle models are unloaded.
