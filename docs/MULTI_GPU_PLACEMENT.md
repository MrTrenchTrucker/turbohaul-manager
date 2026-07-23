# Multi-GPU Placement & Model Co-Residency

Turbohaul-Manager runs on multi-GPU hosts and decides *which card each model lands on*. This guide covers three related capabilities:

1. Running several models **co-resident on one GPU**.
2. **Automatically spreading** models across all GPUs.
3. The **safety headroom** that keeps every card from over-committing.

For the per-model manifest fields see [MODEL_CONFIG_REFERENCE.md](MODEL_CONFIG_REFERENCE.md); for how the manager estimates a model's VRAM footprint see [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## Concepts

- **Sidecar / resident** — a running model process. `max_parallel_sidecars` (wrapper config) caps how many can be resident at once.
- **`split_mode`** (per-model manifest flag under `llama_server_flags`):
  - `none` — the model loads **whole** onto a single card. Several `none` models can co-reside on one card, or be spread across cards.
  - `layer` — the model's layers are **split across all visible GPUs**, for a model too large to fit whole on one card. (`row` / `tensor` are also accepted for other split strategies.)
- **`main_gpu`** — for `split_mode: none`, which card the model pins to (default `0`).

---

## 1. Multiple models per GPU (co-residence)

More than one `split_mode: none` model can share a single GPU. When admitting a model, the manager does **per-card VRAM accounting**: it looks at the target card's free VRAM (accounting for models already resident there and any loads in flight) and only admits the new model if the card has room for its weights **and** KV cache. If the card would over-commit, the load is refused rather than risking a driver-level out-of-memory.

Raise `max_parallel_sidecars` (wrapper config, default `1`) to allow more co-resident models — for example on a two-card host serving many small models at once.

---

## 2. Automatic placement (`auto_place`)

By default, each `split_mode: none` model pins to its manifest's `main_gpu` (default `0`). Without any intervention that means every model piles onto card 0. Set **`auto_place: true`** (a top-level manifest field) and the manager chooses the card for you:

- It reads the **free VRAM on every visible card**.
- It subtracts the per-card safety margin (`safety_min_free_vram_mib`) and any in-flight loads.
- It picks the **least-loaded card that still fits** the model (weights + KV) — so models spread out: as one card fills, the next model lands on the emptier card.
- If **no single card** can fit the model whole but the cards *together* have room, it falls back to **`split_mode: layer`** (splitting that one model across cards) — typically only the largest model in a mixed set.
- If nothing fits, the load is refused (rather than over-committing a card).

`auto_place` is **opt-in and back-compatible**: the default (`false`) preserves the manifest's explicit `main_gpu` pin, so existing manifests behave exactly as before. `auto_place` takes effect only with `split_mode: none` and when `max_parallel_sidecars >= 2` (there is nothing to spread across if only one model may be resident).

Manifest example:

```yaml
model_tag: my-small-model
gguf_blob_sha256: <64-hex>
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

Auto-placement balances **free VRAM**, not model count. On a two-GPU host where several small models are loaded with `auto_place: true`, the manager spreads them so both cards keep roughly equal free space. If one card starts with more already in use, the manager places proportionally more on the emptier card until the free space evens out — so the resulting per-card *count* may differ while the *free VRAM* converges. In practice a set of small models across two cards settles within a few hundred MiB of free VRAM of each other, each card keeping its safety margin clear, and the models serve inference **concurrently across both GPUs**.

---

## 3. Safety headroom (`safety_min_free_vram_mib`)

`safety_min_free_vram_mib` (wrapper config, under `queue`, default `512`) is the amount of VRAM the manager keeps free on **each** card. Both placement and admission respect it: a model is only placed on a card if, after loading, that card still has at least this much free. Raise it (for example to `4096` for 4 GB) to leave headroom for KV-cache growth and to stay clear of driver-level OOM. It can be set from the wrapper config or via the `TURBOHAUL_SAFETY_MIN_FREE_VRAM_MIB` environment variable.

---

## Knobs at a glance

| Setting | Scope | Meaning |
|---|---|---|
| `queue.max_parallel_sidecars` | wrapper | max co-resident models (1–32, default 1) |
| `queue.safety_min_free_vram_mib` | wrapper | VRAM kept free on each card (default 512) |
| `auto_place` | per-model | opt into automatic card selection (default false) |
| `split_mode` | per-model | `none` (whole, one card) / `layer` (split across cards) |
| `main_gpu` | per-model | card pin when `auto_place` is off |

**Rule of thumb:** on a multi-GPU host, mark small `split_mode: none` models `auto_place: true`, set `max_parallel_sidecars` to the number you want co-resident, and set `safety_min_free_vram_mib` to the headroom you want on each card. Leave the single largest model to fall back to `layer` if it can't fit whole on one card.
