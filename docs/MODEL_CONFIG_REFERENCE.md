# Turbohaul-Manager — Per-Model Configuration Reference

Every setting you can put in a per-model manifest: what it does, when to use it, why, and the exact type/bound/enum the manager enforces. Companion docs: [ARCHITECTURE.md](../ARCHITECTURE.md) (how the system works), [DEPLOYMENT_PATTERNS.md](DEPLOYMENT_PATTERNS.md) (which model for which role), [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) (wiring an agent to Turbohaul), and [TURBOQUANT_FLAGS.md](TURBOQUANT_FLAGS.md) (the recommended flag set in brief).

Grounded in the manager's own allowlist at commit time; the code (`src/turbohaul/manifest.py`) is the final authority — see the Preface.

**Settings docs:**

- [General and Config](frontend/07-settings-config.md) — every setting the server is running with, split into what you can change now and what needs a restart.
- [Schema](frontend/08-settings-schema.md) — a builder for the response_format envelope, so you can constrain what a model returns.
- [Logs](frontend/09-settings-logs.md) — the audit trail: what the server did, in order, with filters.
- Model configuration reference (this page) — every setting you can put in a per-model manifest.

---

## Preface — how to read this reference

**`src/turbohaul/manifest.py` is ground truth.** Every accepted flag, its type, its numeric bound, and its enum value set live in that one file — specifically in `SAFE_LLAMA_FLAGS` (the allowlist), `SAFE_LLAMA_FLAG_BOUNDS` (numeric min/max), `SAFE_LLAMA_FLAG_STRING_ENUMS` (string enums), `SAFE_CHAT_TEMPLATE_NAMES`, and `DENIED_FLAGS`. If a flag, bound, or enum value is not in that file, Turbohaul does not accept it — full stop. This document mirrors that file; when the two ever disagree, the file wins and this doc is the bug.

**Adding a flag is a code change, not a YAML edit.** The allowlist is *closed*: an unknown key in a manifest's `llama_server_flags` is rejected at load with `not in the closed allowlist`, never silently forwarded to `llama-server`. Introducing a new flag means editing `SAFE_LLAMA_FLAGS` (plus any bound/enum entry) and shipping the code. This is deliberate — it is the security boundary against flag-injection RCE. You cannot smuggle a capability in through a manifest.

**The 2-minute mental model.** A manifest is one YAML file per model. It has a handful of top-level identity/sizing fields (Section 1) and one `llama_server_flags` map (Section 2). The map is validated flag-by-flag through a fixed four-stage gauntlet: deny → suffix-guard → allowlist → value-check. Values that survive get encoded into `llama-server`'s argv at spawn time. The single most important operational fact: **most flags only take effect on a cold model spawn.** Editing a manifest for a model that is currently loaded does nothing until that process is torn down and re-forked. Section 2's spawn-argv-vs-request-body table tells you which is which, and how to force the reload.

---

## Section 1 — Manifest anatomy

A manifest is one YAML file per model at `<manifests_path>/<model_tag>.yaml` (the shipped default, in `docker/turbohaul.default.yaml`, is `/var/lib/turbohaul/manifests/`). It is edited via `PUT /api/manifests/{tag}` (with an `If-Match` ETag) or on disk, and is parsed into the `ModelManifest` pydantic model (the `kind: model` variant of the `Manifest` union), which is `extra="forbid"` — **any unknown top-level key is rejected**, not ignored.

### Field table

| Field | Type | Default | Why it exists | Gotcha |
|---|---|---|---|---|
| `model_tag` | `str` | **required** | Primary key + filename stem. Must match `TAG_RE` = `^[a-z0-9][a-z0-9._-]{0,63}$`. | Lowercase ASCII only, 1–64 chars, must start with `[a-z0-9]`, no `/`, no leading dot/dash (so no `..` path component). Re-validated at *every* path resolution (read/write/delete), not just on create, as an anti-traversal guard. An uppercase letter or a slash → `ManifestValidationError`. |
| `display_name` | `str` | `""` | Free-text human label shown in the UI. | Cosmetic; no validation beyond being a string. Example: `'My Model 27B parallel:2'`. |
| `description` | `str` | `""` | Free-text notes (provenance, quant, links). | Cosmetic only. |
| `kind` | `str` literal | `"model"` | Discriminates the manifest variant: `model` (this table) or `plugin` (a resource-plugin manifest, a completely different field set; resource plugins are a work in progress). | Absent in an on-disk YAML is fine for a model manifest — the loader supplies `"model"` before the union is resolved. A plugin manifest must declare `kind: plugin` explicitly. |
| `hidden` | `bool` | `false` | Listings-only visibility. `true` omits the manifest from the discovery endpoints (`/v1/models` and `/api/tags`) while leaving it fully loadable and servable by its exact tag. | Nothing in the serve or scheduler path reads it — a hidden model still serves normally. Lets a registry carry fixtures and variants without every consumer inheriting them. Absent/`false` = visible, so no existing manifest is affected. |
| `gguf_blob_sha256` | `str` | **required** | Content-address of the model blob in the store; ties the manifest to an exact GGUF. | Must `fullmatch` `[0-9a-f]{64}` — **lowercase hex, exactly 64 chars**. A pasted uppercase digest is rejected; lowercase it first. Wrong length or any non-hex char → `ManifestValidationError`. |
| `mmproj_blob_sha256` | `str` | `""` | Content-address of an optional vision projector blob, same store as `gguf_blob_sha256`. `""` = text-only. | Same validator shape as `gguf_blob_sha256` but optional: empty string passes; a non-empty value must still `fullmatch` `[0-9a-f]{64}`. The manager resolves it and injects `--mmproj <resolved path>` at spawn — the raw path-bearing `mmproj` CLI flag is in `DENIED_FLAGS` (Section 11.1), so this hash form is the *only* way to set a projector. Not currently exposed by the structured FE editor (raw-JSON manifest mode only). |
| `spec_draft_gguf_blob_sha256` | `str` | `""` | Content-address of an optional **standalone draft model** blob (D-Flash / D-Spark), same store and same pattern as `mmproj_blob_sha256` above. `""` = no standalone draft model. | Same validator: empty passes, non-empty must `fullmatch` `[0-9a-f]{64}`. **Both directions of the pairing with `spec_type` are enforced at spawn time, not at manifest-validation time**: the manager injects `--spec-draft-model <resolved path>` **only when** `llama_server_flags.spec_type` is a member of `manifest.SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL` (`draft-dflash`/`draft-dspark`) — a hash set with `spec_type` omitted, or set to `draft-mtp` (bundled head, never a standalone drafter), is **silently ignored but loudly logged** (`log.warning`, names both the hash and the spec_type), never injected, never a manifest-validation error. The other direction — `spec_type: draft-dflash`/`draft-dspark` with the hash omitted — still fails at the **engine**, not here (no `--spec-draft-model` to give it). See Section 7.1 and Section 11.4. Also raw-JSON-only, matching `mmproj_blob_sha256`. |
| `gguf_size_bytes` | `int ≥ 0` | `0` | Declared blob size; the KV-fit estimate uses it as the model body size. | Pydantic `ge=0`; a negative value is a validation error. It feeds the KV-fit pre-check (Section 4): with `0` that gate passes as "insufficient input" instead of estimating. |
| `context_size` | `int ≥ 1` | `2048` | Manifest-level declared model context length. | **DISTINCT from the `ctx_size` *flag*.** This top-level field is model metadata (`ge=1`, default 2048); `llama_server_flags.ctx_size` is what actually gets passed to `llama-server` as `--ctx-size`. They are usually set to the *same* number (e.g. both `250000`), but nothing in the schema forces them to agree — the field and the flag are validated independently. Do not assume setting one sets the other. |
| `expected_vram_bytes` | `int ≥ 0` | `0` | Footprint used by the VRAM-fit pre-check gate before a model is allowed to spawn. | `ge=0`. **A default of `0` removes the manifest-specific term from the VRAM gate** (the gate then reduces to the box-wide minimum-free-VRAM floor, `queue.safety_min_free_vram_mib`, default 512 MiB; the KV-fit estimate still runs from `ctx_size` and `gguf_size_bytes`); set a real value for the gate to protect you. Use the true device budget, e.g. `22500000000`, `24000000000`. |
| `auto_place` | `bool` | `false` | Opt-in smart GPU placer. `false` honours `llama_server_flags.main_gpu` verbatim (except while Fast Lane is enabled, which also runs the card pick for a `split_mode: none` model). `true` **with `split_mode: none`** lets the manager pick the least-loaded card at admit time. | A plain top-level field, **not** a `llama_server_flag`. Together with an explicit `split_mode: none` it is what allows a model to run a second engine (see [Retired settings](#retired-settings)). |
| `arch` | `str` | `""` | Declares the model's architecture family so the manager can size a non-standard KV shape correctly. Set `arch: "qwen35"` for a qwen35 hybrid (state-space/SSM + attention); leave it `""` for an ordinary pure-attention model. | The default `""` is byte-identical to prior behaviour (pure attention). Setting `arch` to any non-empty value (e.g. `qwen35`) is **load-bearing on its own**: it opts the model into parsing its real GGUF attention dims and sizing the KV cache from them (the dimension-aware path in Section 4), independent of `hybrid_kv_ratio`. The same path is also used, without `arch`, when the GGUF declares derivable sliding-window layers. |
| `hybrid_kv_ratio` | `float` `0.0`–`1.0` | `1.0` | The fraction of layers that contribute a **growing** per-token KV cache. In a qwen35 hybrid the SSM layers keep a fixed-size recurrent state (not a growing cache), so the model's per-token KV is smaller than a pure-attention model of the same size. **This ratio scales the file-size *fallback* estimate path only** (Section 4). | Bounded `0.0..1.0`. **Default `1.0` = pure attention = byte-identical sizing for every existing model.** For a model that takes the dimension-aware path (Section 4: `arch` set and the GGUF dims parse), that path takes over and **ignores `hybrid_kv_ratio`** — it then only matters as the fallback when dims can't be parsed and no `kv_bytes_per_token` override is set. |
| `kv_bytes_per_token` | `float ≥ 1024.0` or unset | *(unset)* | An operator-**measured** effective KV cache cost, in **BYTES per token**, used verbatim by the KV-fit estimate (Section 4) as the highest-precedence override. Derive it from a real measurement, e.g. `measured_marginal_MiB * 1048576 / ctx_size`. | **Units are BYTES/token, NOT KiB** — a measured 13.5 KiB/token is `13824.0`. A **1 KiB/token floor** (`Field ge=1024.0`) rejects a KiB-vs-bytes typo at load. Applied **verbatim**: no quant-scale, no `hybrid_kv_ratio` multiply. Leave **unset** for every existing model (the default) — byte-identical. |
| `revision` | `int ≥ 1` | `1` | The ETag value for optimistic-concurrency writes; server-incremented on every atomic update. | `ge=1`. You do not hand-manage this — `GET` returns it as `ETag: "<revision>"`, you echo it back in `If-Match` on `PUT`, and the server bumps it. A stale `If-Match` → HTTP 412. |
| `llama_server_flags` | `map` | `{}` (empty) | The closed allowlist of `llama-server` spawn flags — see Section 2. | Every key/value is gauntlet-validated. Unknown key → reject. This is where all the real tuning lives. |
| `prompt_template` | `object` | empty `PromptTemplate` | Server-side prompt scaffolding: `{system_default: str, stop_tokens: [str]}`. | Its own `extra="forbid"` model — unknown sub-keys are rejected. `system_default` is a plain system-prompt string (default `""`); `stop_tokens` is a list of stop strings (default `[]`, e.g. `["<|im_end|>", "<|endoftext|>"]`). |

### Retired settings

- **`max_instances`** was removed. A manifest file, a stored manifest or a `PUT` body that still contains it is accepted: the key is dropped and one warning is logged, never an error. Stored manifest files are rewritten without the key once, at the first start (the revision does not change). Nothing is carried over.
- The number of engines (`llama-server` processes) the box can hold for a model comes from the box-wide `queue.max_parallel_sidecars` budget and the cards the model fits on; see [SIDECARS_AND_CONTEXT_WINDOWS.md](SIDECARS_AND_CONTEXT_WINDOWS.md). Using more than one engine of the same model also needs a one-card-per-engine placement: the manifest must have `auto_place: true` and an explicit `llama_server_flags.split_mode: none`. A model that is pinned (`auto_place: false`) or split across cards keeps one engine.
- The windows (concurrent conversations) inside each engine are `llama_server_flags.parallel`, unchanged; when it is omitted no `--parallel` is passed to the engine and TurboHaul counts one window per engine, so set it explicitly to pin the count; see Section 2.

### The `context_size` vs `ctx_size` distinction (the #1 trap)

These are two different things that happen to be spelled almost the same:

- **`context_size`** (top-level Manifest field, default `2048`) — declared model context metadata.
- **`ctx_size`** (a key *inside* `llama_server_flags`, bound `1..2_000_000`) — the value forwarded to `llama-server` as `--ctx-size`, i.e. the KV window the process actually allocates.

Because they are validated by separate code paths, you can set one and forget the other. By convention they are set to the same number (both `16384`, both `250000`, both `500000`, etc.), but that is discipline, not enforcement. When you tune context length, change **both** — and if the model runs `parallel > 1`, remember that the cross-field rule in Section 2 (its `ctx_size`-based checks do not run while a unified pool is mandatory) keys off the **flag** `ctx_size`, not the top-level field.

### `expected_vram_bytes` and its gate role

`expected_vram_bytes` is the footprint the VRAM-fit pre-check consults before allowing a spawn. Its `ge=0` default of `0` means an un-set value declares a zero-byte footprint, which adds nothing to the gate's threshold (`max(box-wide minimum free VRAM, expected_vram_bytes)`) — so this manifest-specific check is only as good as the number you put here. Set it to the real device-budget target (for example `21500000000`–`29000000000` for 24 GB-class cards); leaving it `0` drops that model's manifest-specific VRAM check (the box-wide floor and the KV-fit estimate still apply).

### `revision` / ETag mechanics

`revision` *is* the ETag. The write path (`write_manifest_atomic`) enforces optimistic concurrency:

- **Create** (file does not exist): no `If-Match` is needed and a supplied one is not checked; the manifest is written as-is.
- **Update** (file exists): `If-Match` is **required**. A *missing* `If-Match` on an existing file raises `ConcurrencyError` (→ HTTP 412) — it does **not** silently overwrite. A *mismatched* `If-Match` also raises `ConcurrencyError`. On success the server sets `revision = existing.revision + 1`.

So the loop is: `GET` → read `ETag: "<n>"` → edit → `PUT` with `If-Match: "<n>"` → server writes and returns `<n+1>`. Out-of-band disk edits are caught by this same compare-on-write — there is no inotify watcher; concurrency is optimistic, not event-driven.

### `prompt_template` — `{system_default, stop_tokens}`

`prompt_template` is a nested `extra="forbid"` object with exactly two fields:

- `system_default: str` (default `""`) — a default system prompt injected when the request supplies none.
- `stop_tokens: list[str]` (default `[]`) — stop strings, e.g. `["<|im_end|>", "<|endoftext|>"]`.

It is usually left empty (`system_default: ''`, `stop_tokens: []`), letting the model's chat template handle stops. Any key other than these two inside `prompt_template` is a validation error.

### `arch` and `hybrid_kv_ratio` — hybrid (SSM + attention) model sizing

Two fields describe models whose KV footprint is **not** a plain function of size: hybrids that interleave state-space (SSM) layers with attention layers, such as a `qwen35` hybrid.

- **`arch`** (`str`, default `""`) — the architecture family. Empty for an ordinary pure-attention model (the default, unchanged). Set `arch: "qwen35"` for a qwen35 hybrid.
- **`hybrid_kv_ratio`** (`float`, default `1.0`, bounded `0.0..1.0`) — the fraction of layers that grow a per-token KV cache. In a qwen35 hybrid the SSM layers hold a **fixed-size recurrent state** rather than a cache that grows with context, so only the attention layers contribute the linear-in-tokens KV term. Setting `hybrid_kv_ratio` to that attention-layer fraction lets the KV-fit estimator scale the per-token KV down, so the VRAM/RAM gate doesn't over-estimate a hybrid's cache (see Section 4 for the exact math).

**The default `1.0` is a no-op.** A ratio of `1.0` means "every layer grows KV" = pure attention = the exact sizing every existing manifest already gets. Lower it only for a real hybrid. The two fields are used as a pair: set `arch` to the hybrid family **and** `hybrid_kv_ratio` to the growing-KV fraction together.

**Nothing new in `llama_server_flags`.** A hybrid model is content-addressed by `gguf_blob_sha256` like any other model. There is no hybrid-specific spawn flag to set and nothing added to the allowlist for it. A hybrid can use the existing TurboQuant KV-cache types (`turbo2`/`turbo3`/`turbo4` in `cache_type_k`/`cache_type_v`); there is no separate KV type. So the manifest side of configuring one is just: point `gguf_blob_sha256` at the blob, set `arch` + `hybrid_kv_ratio`, and carry the usual doctrine flags.

```yaml
# hybrid-model top-level fields (illustrative ratio):
arch: "qwen35"          # SSM + attention hybrid
hybrid_kv_ratio: 0.5    # only ~this fraction of layers grow a per-token KV cache;
                        # SSM layers hold a fixed recurrent state. 1.0 = pure attention (no change).
```

---

## Section 2 — How `llama_server_flags` works

`llama_server_flags` is a **closed allowlist** of 108 `llama-server` flags. Its security model is a *deny-by-default* gauntlet: a flag must survive four checks, in order, or the whole manifest is rejected. Adding a genuinely new flag is a code change (edit `SAFE_LLAMA_FLAGS`), never a YAML edit.

### The validation order (deny → suffix-guard → allowlist → value)

Per flag, `ModelManifest._flags_allowlist` runs these in exactly this sequence (order matters — the earliest failure wins):

1. **Deny check.** If the key is in `DENIED_FLAGS` (51 path/URL/credential/RCE flags — `model`, `host`, `port`, `lora*`, `model_url`, `hf_repo*`, `path`, `media_path`, `api_key*`, `ssl_*_file`, `chat_template_file`, `override_kv`, `tools`, `samplers`, `grammar`, `fit_target`, …) → reject with "explicitly denied". This is the first line and beats everything.
2. **Suffix forward-defense (`_suffix_guard_check`).** If the key *name* matches a dangerous pattern — regex `.*_file$` / `.*_path$` / `.*_dir$` / `.*_url$` / `.*_repo$` / `.*_key$` / `.*_model$`, or prefix `^hf_` / `^lora` / `^control_vector` / `^lookup_cache_` / `^ssl_` / `^api_key` / `^slot_save_` / `^webui_` / `^docker_` — it is rejected **even if a future edit mistakenly adds it to the allowlist**. This catches new upstream (Tom's Fork) flags that ship a path/credential before anyone denylists them. (`_SUFFIX_GUARD_EXCEPTIONS` is currently empty.)
3. **Allowlist membership.** If the key is not in `SAFE_LLAMA_FLAGS` → reject with "not in the closed allowlist". Unknown = rejected.
4. **Value validation (`_validate_flag_value`).** Type, then enum (if applicable), then numeric bounds. Details below.

After all per-flag checks, three **cross-field / model-level** passes run in order, and any of them can reject the manifest:

1. **`parallel` guard** (`_validate_parallel_ctx`) — the `parallel > 1` rules below (it runs at the end of the flags validator itself).
2. **Flag defaults** — `ctx_checkpoints`, `cache_type_k`, `cache_type_v` are filled in for any manifest that did not set them (`setdefault`, never overwrite).
3. **Reasoning-budget guard** — rejects `reasoning_budget >= n_predict` whenever `n_predict > 0` (Section 6.4).

A fourth pass is advisory only: an inert `cache_reuse` on a multimodal model is logged at INFO, never rejected.

### How values are typed, bounded, and enum-checked

`_validate_flag_value` applies, per flag:

- **Type spec** from `SAFE_LLAMA_FLAGS` (e.g. `int`, `bool`, `float`, `str`, or a tuple like `(int, str)` for `n_gpu_layers`).
- **String enums** from `SAFE_LLAMA_FLAG_STRING_ENUMS` — the value must be exactly one of a fixed set (e.g. `numa ∈ {none, distribute, isolate, numactl}`; `split_mode ∈ {none, layer, row, tensor}`; `cache_type_k/v ∈ {f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo2, turbo3, turbo4}`).
- **Numeric bounds** from `SAFE_LLAMA_FLAG_BOUNDS` — inclusive `(min, max)`, DoS-prevention (e.g. `ctx_size 1..2_000_000`, `parallel 1..256`, `top_p 0.0..1.0`, `temp 0.0..10.0`). A value outside range → reject.
- **`int → float` promotion is allowed** (an integer where a float is expected is fine).

**Bool-coercion rejection (load-bearing).** Python treats `bool` as a subclass of `int`, which would let `true`/`false` sneak into an int field (and vice-versa). Turbohaul explicitly blocks both directions: a `bool`-typed flag rejects a non-bool; an `int`-typed flag rejects a `bool` (the error message says Python's bool-is-int coercion is explicitly rejected). So `ctx_size: true` and `flash_attn: 1` both fail. Two flags are special-cased because they legitimately accept multiple types:

- **`flash_attn`** — accepts a `bool` *or* a string in `{on, off, auto, enabled, disabled}`. Anything else rejects.
- **`n_gpu_layers`** — accepts an `int` (bounded `-1..999`) *or* the strings `all` / `auto`. A `bool` is explicitly rejected.
- **`chat_template`** — must be a `str`, must **not** contain `{%` or `{{` (SSTI guard), and must be either a built-in name in `SAFE_CHAT_TEMPLATE_NAMES` or a short `^[A-Za-z0-9_.\-]+$` token ≤ 256 chars. Custom Jinja *bodies* are forced down the denied `chat_template_file` path — they cannot be inlined.

### The cross-field `parallel` guard

When `llama_server_flags.parallel > 1`, `_validate_parallel_ctx` enforces rule 1 in the same validation pass or the manifest is rejected (`parallel ≤ 1` is a no-op, back-compat). Rules 2 and 3 apply only without a unified KV pool, which rule 1 rules out, so they never fire today:

1. **`kv_unified: true` is required** — a unified KV pool keeps cache accounting exact and flat across concurrent slots.
2. **`ctx_size` (the flag) divisible by `parallel`** — would matter only without a unified pool (where the engine divides `ctx_size` between slots and a non-divisible pair silently truncates the per-slot window). With the mandatory unified pool each slot's limit is the full `ctx_size` (the slots share one pool of that size), so a non-divisible pair is accepted.
3. **The per-slot window `ctx_size // parallel` ≥ `PER_SLOT_CTX_FLOOR` (8192)** — likewise only without a unified pool; with one, there is no per-slot division and no minimum is checked.

Example: a model served at `ctx_size: 250000` with `parallel: 2` and `kv_unified: true` -> rule 1 is satisfied; each slot can use up to the full 250000 tokens, and the two slots share one pool of that size (rules 2 and 3 do not apply).

### The argv encoding

`flags_to_argv` re-checks the allowlist + denylist a second time (defense-in-depth) and then encodes each surviving flag into `llama-server`'s command line. Key mapping: snake_case → `--kebab-case` (`no_context_shift` → `--no-context-shift`). Value encoding:

| Value form | Argv result |
|---|---|
| `bool True` | `--flag` (bare, no value) |
| `bool False` | **omitted entirely** (not `--flag false`) |
| any other type | `--flag <value>` |
| **`flash_attn` (special)** | normalized to an explicit `on`/`off`: bool `True` → `--flash-attn on`, bool `False` → `--flash-attn off`, string → `--flash-attn <value>` |

So `flash_attn` always lands on the command line with an explicit value — useful when auditing the child process, because you never have to infer its state from a bare flag's presence.

### THE spawn-argv vs request-body distinction (the operational crux)

This is the single most consequential thing to internalize. Flags split into two layers by *when* they can change:

| Layer | Examples | Reload behavior |
|---|---|---|
| **Spawn argv** (process-fork) | `flash_attn`, `no_context_shift`, `cache_reuse`, `slot_prompt_similarity`, `no_perf`, `ctx_size`, `cache_type_k`, `cache_type_v`, `n_gpu_layers`, `parallel`, `kv_unified`, `no_kv_offload`, `cont_batching`, `spec_type`, `jinja`, and effectively **all** of `llama_server_flags` | **COLD-SPAWN ONLY.** A manifest `PUT` does *not* affect a running `llama-server`. The old cmdline persists until the process exits and is re-forked. |
| **Request body** (per-call) | `temperature`, `top_p`, `top_k`, `reasoning_budget`, `max_tokens`, `keep_alive` (request fields) | **Hot.** The values sent in the request body are applied per request through the forwarder and take effect on the next request immediately. A manifest has no such request-field keys: its `llama_server_flags` entries `temp`, `top_p`, `top_k` and `reasoning_budget` are allowlisted flags written into the spawn argv, so a manifest `PUT` of them is cold-spawn-only like every other flag. |

The five doctrine flags (`flash_attn`, `no_context_shift`, `cache_reuse`, `slot_prompt_similarity`, `no_perf`) and all the structural flags (`ctx_size`, `cache_type_k/v`, `parallel`, `n_gpu_layers`, `jinja`, `spec_*`) are **spawn argv** — patching them on a running model is inert until reload.

**Forcing the cold-spawn — three options** (also described in `docs/TURBOQUANT_FLAGS.md`):

- **Option A — `keep_alive: 0`.** Send any request to the same manifest tag with body `"keep_alive": 0` (Ollama-style, parsed by `parse_keep_alive` in `src/turbohaul/api/chat_completion.py`). This sets the slot's `IDLE_HOT` window to 0 → the running `llama-server` is torn down at the end of that request; the *next* request cold-spawns with the new flags. (If the next queued request is for the same model, the engine is instead held warm briefly for it.)
- **Option B — wait for natural `IDLE_HOT` teardown** (`idle_hot.remaining_s → 0`), then the next request triggers a cold-spawn.
- **Option C — `docker restart turbohaul-manager`** (nuclear; recovers cleanly but interrupts in-flight requests).

**`/proc/cmdline` verification recipe** — confirm the running process actually carries the flag (not just the manifest):

```bash
# 1. Manifest has the flag (post-PUT):
curl -s http://localhost:11401/api/manifests/my-model-27b | jq '.llama_server_flags'

# 2. The running child process reflects it (post-cold-spawn):
docker exec turbohaul-manager bash -c \
  'pgrep -af llama-server | head -1 | awk "{print \$1}" | xargs -I{} cat /proc/{}/cmdline | tr "\0" " "'
# Expect e.g.: --flash-attn on ... --no-context-shift ... --cache-reuse 256 ... --slot-prompt-similarity 0.5 ... --no-perf
```

If `/api/manifests` shows the flag but `/proc/<pid>/cmdline` does not, the running slot is **stale** (the manifest was patched after spawn) — trigger Option A/B/C and re-verify.

### The five doctrine flags — the tradeoffs

New manifests should ship these unless a model-specific reason says otherwise (`docs/TURBOQUANT_FLAGS.md`):

- **`flash_attn: true`** — the `turbo*` KV cache types need flash attention (the engine switches it on by itself if it is off, and a quantized V cache type refuses to start without it), so state it explicitly. Type: `bool | str-enum`. Argv: normalized to `--flash-attn on`. Spawn-argv.
- **`no_context_shift: true`** — turns off context shifting (the engine sliding its window forward by discarding old tokens once the context is full). Type `bool`. Spawn-argv. *Deviate* only for a model known to shift its context correctly; the recommended setting stays `true`.
- **`cache_reuse: 256`** — enables prefix-cache reuse across requests in the same warm slot; cuts long-tail prefill on follow-ups. Type `int`, bound `0..65536`. Spawn-argv. *Deviate* (omit) only for one-shot batch models with no follow-up traffic, where it gains nothing.
- **`slot_prompt_similarity: 0.5`** — lets a slot reuse its prefix cache even when the new prompt is not byte-identical (50% similarity threshold); improves `ACTIVE_MATCH` hit rate. Type `float`, bound `0.0..1.0`. Spawn-argv.
- **`no_perf: true`** — suppresses per-request perf logging (less log noise + minor CPU). Type `bool`. Spawn-argv. *Deviate* (`false`) when actively perf-debugging a model to surface per-request timings — flip back after.

These compose with the TurboQuant KV flags `cache_type_k` / `cache_type_v` (`turbo3` is the default).

---

## Section 3 — Identity & Sizing Flags

These flags set the shape of the process: how much context it holds, where the weights live, how many concurrent slots it serves, and how it threads work. Almost all of them are **spawn-argv** — they are baked into the `llama-server` command line at fork time, so a manifest `PUT` does **not** change a running model. You must trigger a cold-spawn (see the doctrine section for Options A/B/C) and confirm with `/proc/<pid>/cmdline`. Every value/bound/type below is from `manifest.py` (`SAFE_LLAMA_FLAGS`, `SAFE_LLAMA_FLAG_BOUNDS`); the example values are typical settings, not defaults (unless the row says so).

### The core sizing set

| Flag | Type | Bound (min, max) | Example value(s) | Apply | Purpose |
|---|---|---|---|---|---|
| `ctx_size` | int | `(1, 2_000_000)` | `8192`, `12288`, `16384`, `32768`, `250000`, `500000` | spawn-argv | The KV context size `llama-server` allocates. Drives the KV-cache VRAM/RAM estimate directly (see Section 4). With `parallel > 1` (which requires `kv_unified: true`) it is the size of one pool shared by all slots, so the slots together hold at most `ctx_size` tokens; no slot's limit is smaller than `ctx_size`. |
| `n_gpu_layers` | int **or** `"all"`/`"auto"` | int bounded `(-1, 999)`; bool rejected | `999` | spawn-argv | How many model layers offload to GPU. `999` = "all layers on GPU" (the ceiling; llama.cpp clamps to the real layer count). `-1` also means all in llama.cpp; strings `all`/`auto` are accepted, but `999` is the usual choice. |
| `parallel` | int | `(1, 256)` | `1`, `2` | spawn-argv | Number of concurrent same-model slots served by one sidecar. `>1` requires `kv_unified: true` (the slots then share one pool of `ctx_size` tokens) and pulls in the cross-field rule below. When omitted, no `--parallel` is passed and TurboHaul counts one window per engine (the engine picks its own slot count); set it explicitly to pin the number of windows. |
| `batch_size` | int | `(1, 65536)` | unset (engine default) | spawn-argv | Logical prompt batch (`--batch-size` / `-b`): tokens the server groups per prefill decode call. Larger = faster prefill, more transient compute VRAM. Usually left unset (engine default). |
| `ubatch_size` | int | `(1, 65536)` | unset | spawn-argv | Physical micro-batch (`--ubatch-size` / `-ub`): the sub-batch actually submitted to the GPU per kernel launch. Normally kept ≤ `batch_size`; the manager does not cross-check the two. Tunes prefill throughput vs. per-step memory; usually left unset. |
| `keep` | int | `(-1, 65536)` | unset | spawn-argv | `--keep`: number of prompt tokens (from the front) to retain when the context is full and old tokens are discarded. `-1` keeps the whole initial prompt. Only relevant when context-shift/truncation is active — with the recommended `no_context_shift: true` it is unused. |
| `n_predict` | int | `(-1, 1_000_000)` | `-1` | spawn-argv | `--n-predict` / `-n`: default max tokens to generate when a request doesn't specify. `-1` = unlimited (let the client's `max_tokens`/stop tokens govern). Typically `-1` on reasoning models so the request body controls length. |

**`ctx_size` vs `context_size` — do not confuse them.** `ctx_size` is the **llama_server_flag** that becomes `--ctx-size` on the child process, and it is also what the KV-fit gate sizes against. `context_size` is the **top-level manifest field** (`int ≥ 1`, default `2048`): it is declared model metadata, and the gate uses it only as a *fallback* when the `ctx_size` flag is absent (`ctx = ctx_size_flag or context_size`). Set both to the same number by convention; nothing in the schema forces them to agree, and the flag is the one that wins wherever both are present.

### Threading

| Flag | Type | Bound | Example value | Apply | Purpose |
|---|---|---|---|---|---|
| `threads` | int | `(-1, 256)` | `16` (e.g. on `cpu_moe` manifests) | spawn-argv | `--threads` / `-t`: CPU threads for generation/eval. `-1` = auto (all cores). Worth setting explicitly on `cpu_moe` manifests, where MoE experts run on CPU and thread count materially affects throughput. |
| `threads_batch` | int | `(-1, 256)` | unset | spawn-argv | `--threads-batch` / `-tb`: CPU threads for prompt prefill/batch processing (can differ from generation threads). `-1` = follow `threads`. Usually unset. |
| `threads_http` | int | `(-1, 256)` | unset | spawn-argv | `--threads-http`: HTTP server worker threads for the llama-server API layer. `-1` = auto. Usually unset; the fronting forwarder handles concurrency. |
| `cont_batching` | bool | — | `true` (with `parallel: 2`) | spawn-argv | `--cont-batching`: continuous batching so multiple in-flight slots interleave decode steps instead of running strictly serially. Normally set together with `parallel > 1`; the manager does not require it (the validator only requires `kv_unified`). |

### THE cross-field parallel rule (load-bearing)

When `parallel > 1`, `_validate_parallel_ctx` in `manifest.py` enforces one condition in the validation pass, or the manifest is **rejected** at load (it never reaches spawn). `parallel ≤ 1` is a no-op (back-compat). The rules (2 and 3 apply only without a unified pool, so they never fire today):

1. **`kv_unified: true` is REQUIRED.** Without a unified KV pool, `--parallel N`'s per-slot KV accounting diverges from the single count the VRAM gate assumes (that count is only accidentally correct because `--parallel` divides `ctx`). The unified pool keeps the cache exact and flat across concurrent slots. Omitting it raises `ManifestValidationError`.
2. **`ctx_size % parallel == 0`** — matters only without a unified KV pool, where the engine divides `ctx_size` across the N slots and a non-divisible pair silently truncates the per-slot window. Rule 1 makes the unified pool mandatory, so this check never runs: with a unified pool each slot's limit is the full `ctx_size` (one shared pool) and a non-divisible pair is accepted.
3. **`ctx_size // parallel ≥ PER_SLOT_CTX_FLOOR` (8192).** Also applies only without a unified pool (there each per-slot window would need at least 8192 tokens). It never runs today: with a unified pool there is no per-slot division and no minimum is checked.

**Examples of the rule satisfied:** a model served at `parallel: 2`, `ctx_size: 500000`, `kv_unified: true`, `cont_batching: true` -> each slot can use up to the full `500000` (the two slots share one pool of that size). Another model uses `parallel: 2` with `ctx_size: 250000`: the same, with a 250000-token pool. Note that a manifest may carry `parallel: 1` **explicitly alongside** `kv_unified: true` + `cont_batching: true` — that is legal because the guard is a no-op at `parallel: 1`; the extra flags are inert but harmless.

---

## Section 4 — KV-Cache & TurboQuant

The KV cache (per-token attention state) grows linearly with `ctx_size` and at long contexts can rival the model weights in size. These flags decide the KV **precision** (TurboQuant compressed types), **where it lives** (VRAM vs. host RAM), and the **checkpoint/reuse** machinery. All are **spawn-argv** except where noted. Enum sets and bounds are from `manifest.py` (`SAFE_LLAMA_FLAG_STRING_ENUMS`, `SAFE_LLAMA_FLAG_BOUNDS`); the KV size factors are `_KV_QUANT_SCALE` in `src/turbohaul/safety.py`, which also holds the sizing math.

### `cache_type_k` / `cache_type_v` — the TurboQuant knob

Both are `str`, and both accept the **same closed enum** (`SAFE_LLAMA_FLAG_STRING_ENUMS`):

```
f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo2, turbo3, turbo4
```

The KV-cache size factor (multiplier the manager's KV estimator applies relative to the f16 baseline):

| type | class | KV size factor |
|---|---|---|
| `f32` | uncompressed | 2.00 |
| `f16` / `bf16` | half precision (baseline) | 1.00 |
| `q8_0` | 8-bit | 0.50 |
| `q5_0` / `q5_1` | 5-bit | 0.32 |
| `q4_0` / `q4_1` / `iq4_nl` | 4-bit | 0.25 |
| **`turbo2`** | TurboQuant | **0.125** |
| **`turbo3`** | TurboQuant | **0.1875** |
| **`turbo4`** | TurboQuant | **0.25** |

**Quality/memory tradeoff — when to pick which:**
- **`turbo3`** — the balanced default. Good compression (0.1875×) at good quality.
- **`turbo2`** — the most aggressive TurboQuant (0.125×). Trades the most precision for the smallest cache; reach for it when VRAM is critically tight (for example to fit a very long context across cards, or to leave headroom for a co-resident model).
- **`turbo4`** — trades less precision for a **larger** cache (0.25×, same footprint as 4-bit `q4_0`). Highest-quality of the turbo family; use when you have VRAM to spare and want maximum KV fidelity.
- **`f16`** — baseline, no compression. A **mixed** pair is also valid, e.g. `cache_type_k: f16` + `cache_type_v: turbo2` (see the next note).

> **The estimator reads BOTH `cache_type_k` and `cache_type_v`.** The KV cache is treated as roughly half K and half V, and each half is scaled by its own cache type. So an asymmetric pair is sized correctly: `cache_type_k: f16` with `cache_type_v: turbo2` costs (1.00 + 0.125) / 2 = 0.5625× the f16 baseline, not the full 1.00 the K type alone would imply. When `cache_type_v` is unset it falls back to the K type (legacy single-quant behaviour). A *misspelled* type (e.g. `turbo33`) is rejected at manifest validation by the `cache_type_k`/`cache_type_v` string-enum check — it never reaches the estimator. The gate's refusal/accept message labels a mixed pair explicitly as `K=<type>/V=<type>`, so you can read the pair straight out of the log.

> **Neither type can be unset in practice.** Both `cache_type_k` and `cache_type_v` carry a default (see [New defaults](#new-defaults-ctx_checkpoints-and-kv-cache-type)), so both are always populated by the time the estimator reads them.

The `turbo*` KV types need flash attention: the engine switches it on by itself (with a warning) if it is off, and any other quantized V cache type refuses to start without it (see Section 5).

### KV placement — VRAM vs. host RAM

| Flag | Type | Bound | Example value | Apply | Purpose |
|---|---|---|---|---|---|
| `no_kv_offload` | bool | — | `true` | spawn-argv | `--no-kv-offload`: pushes the **entire KV cache into host RAM** while weights stay on the GPU. Removes the KV term from the VRAM equation (it's checked against host RAM instead). This is the recipe for fitting a very long context on a smaller card. The VRAM gate keys on `no_kv_offload: true` **specifically** — a `false` value is omitted from argv and has no effect. |
| `kv_offload` | bool | — | unset | spawn-argv | The complementary/inverse toggle (`--kv-offload`, default behavior = KV in VRAM). Redundant with the default; the manager's VRAM gate keys on `no_kv_offload` only. |
| `kv_unified` | bool | — | `true` (required when `parallel > 1`) | spawn-argv | `--kv-unified`: one shared KV pool across all slots instead of one-per-slot. **Required** whenever `parallel > 1` (enforced by the cross-field rule, Section 3). Keeps RAM-resident KV a single shared pool, not duplicated per slot. |
| `cache_ram` | int (MiB) | `(0, 262144)` | `32768` | spawn-argv | `--cache-ram`: the engine's prompt-cache size limit, in **MiB** (engine default 8192; the engine treats `-1` as no limit and `0` as prompt cache disabled, but the manifest bound only admits `0`..`262144`, so `-1` cannot be set from a manifest). It is not the KV-offload budget: `no_kv_offload` decides where the KV cache lives, and the manager does not read `cache_ram` anywhere else. |

**The RAM-KV recipe:** to serve a very long context that won't fit KV in VRAM —
```yaml
no_kv_offload: true
cache_ram: 32768        # 32 GiB engine prompt-cache limit (optional; not what moves the KV to RAM)
ctx_size: 250000
# for concurrency, add:
parallel: 2
kv_unified: true
cont_batching: true
```

### Checkpoint / reuse machinery

| Flag | Type | Bound | Example value | Apply | Purpose |
|---|---|---|---|---|---|
| `cache_prompt` | bool | — | unset | spawn-argv | Cache the processed prompt so an identical prefix isn't re-prefilled. Usually left unset; the `cache_reuse` + `slot_prompt_similarity` pair (Section 5) is the recommended mechanism. |
| `cache_idle_slots` | bool | — | unset | spawn-argv | Save idle slots to the engine's prompt cache when a new task arrives, and clear them when a unified KV pool is used (engine default: enabled; needs a non-zero `cache_ram`), so a returning request can warm-reuse them. Usually unset. |
| `ctx_checkpoints` | int | `(0, 1024)` | **defaults to 2** when not set explicitly | spawn-argv | Number of context checkpoints (`--ctx-checkpoints`) the server keeps for fast restore/rollback of KV state. See [New defaults](#new-defaults-ctx_checkpoints-and-kv-cache-type) below — a value set explicitly in a model's own configuration always wins. |
| `checkpoint_every_n_tokens` | int | `(1, 1_000_000)` | unset | spawn-argv | Cadence (in tokens) at which a context checkpoint is taken. Companion to `ctx_checkpoints`. The manager's allowlist accepts it, but the vendored engine defines no `--checkpoint-every-n-tokens` option (its spacing control is `--checkpoint-min-step`), so leave it unset. |

### New defaults: `ctx_checkpoints` and KV cache type

As of this release, any model whose own configuration doesn't set these explicitly gets:

- `ctx_checkpoints` defaults to **2**. Previously, an unset value inherited the inference engine's
  own default of 32.
- The KV cache type (`cache_type_k` / `cache_type_v`, see above) defaults to **`turbo3`**.

**These are defaults, not overrides — a value set explicitly in a model's own configuration always
wins.** Any model already tuned by hand is completely unaffected by this change.

**A default is never written into the file.** The values are injected when a manifest is loaded, so a
`GET` shows them — but on write they are stripped back out again if they still equal the default.
That keeps the distinction between "this model follows the default" and "an operator chose this
value": if the default is ever changed, every manifest that never set the flag picks up the new value,
instead of splitting the models between those that follow the default and those that only look like
they do. Set a flag explicitly and it is persisted verbatim, as always.

The 2-checkpoint ladder was chosen for warm restore. Cold restore at this setting has not been
characterised.

### VRAM/RAM sizing math

The manager estimates fit **without loading the model** — the `check_kv_cache_fit` gate (3rd of the 7 pre-spawn safety gates: free-RAM, free-VRAM, KV-cache-fit, MoE-RAM-fit, CPU busy %, IO-wait, tensor-split device count). KV size:

**KV-fit precedence.** The KV estimate is whichever of these applies first:
1. an explicit **measured `kv_bytes_per_token`** override — used verbatim (no quant-scale, no hybrid multiply);
2. **parsed GGUF attention dims** — used when the manifest sets `arch` (any non-empty value, e.g. `qwen35`) or the GGUF declares derivable sliding-window layers; sized from the real attention layers (sliding-window layers are capped at the window), and **ignores `hybrid_kv_ratio`** (the layer count already reflects the hybrid fraction);
3. the **legacy file-size heuristic** below — the *only* path that applies `hybrid_kv_ratio`.

The formula below is tier 3, the file-size fallback. Every model that neither sets `arch` nor has derivable sliding-window layers, and has no override, uses it:

```
gguf_mib          = gguf_size_bytes / (1024*1024)
bytes_per_tok_f16 = (9 * gguf_mib) / 1024            # ≈ 9 KB/token per GiB of model body, at f16
quant_factor      = (factor(cache_type_k) + factor(cache_type_v)) / 2   # KV is ~half K, ~half V — each half scaled by its OWN type
bytes_per_tok     = bytes_per_tok_f16 * quant_factor
kv_cache_mib      = (bytes_per_tok * ctx_size) / 1024 * hybrid_kv_ratio   # hybrid_kv_ratio applies to THIS fallback path only; 1.0 (default) = no change
```

**Hybrid models (`arch: "qwen35"`).** For a model with `arch` set the manager parses the real GGUF attention dims and sizes `kv_cache_mib` from the actual attention layers (tier 2 above) — this is the accurate path and it does **not** apply `hybrid_kv_ratio`. `hybrid_kv_ratio` matters only on the **file-size fallback** (tier 3): when the GGUF dims can't be parsed and no `kv_bytes_per_token` override is set, the estimator multiplies the per-token KV term by `hybrid_kv_ratio` so a hybrid's fallback `kv_cache_mib` is proportionally smaller than a pure-attention model of the same body size. The default `hybrid_kv_ratio: 1.0` leaves that factor at 1 — every existing pure-attention model is sized exactly as before.

**Expert-offload models (`cpu_moe: true` or `n_cpu_moe > 0`) take a different branch.** Because `cpu_moe`/`n_cpu_moe` move expert weights to host RAM, the closed form's `gguf_mib` body term — which assumes every weight is GPU-resident — grossly over-reserves and would wrongly refuse a co-resident that actually fits. For these configs the gate trusts the operator's measured number instead:
```
vram_need = expected_vram_mib + overhead_mib + par_extra
refuse if vram_need > free_vram_mib
```
This branch takes precedence over both branches below, and it only engages when `expected_vram_bytes` is non-zero. The trade is explicit: on an expert-offload model, context-bump safety is the operator's to maintain by keeping `expected_vram_bytes` accurate, because the closed form is no longer watching.

**KV resident in VRAM (default):**
```
needed_vram = gguf_mib + kv_cache_mib + overhead_mib + par_extra   # overhead floor = 1024 MiB
refuse if needed_vram > free_vram_mib
```

**KV offloaded to RAM (`no_kv_offload: true`):** the KV term drops out of VRAM, replaced by a context-linear scratch term, and KV is checked against host RAM:
```
vram_need = gguf_mib + (overhead_mib + ctx_size/128 + par_extra)
refuse if vram_need > free_vram_mib
refuse if kv_cache_mib > free_host_ram_mib          # complementary host-RAM check
```

where `par_extra = (parallel − 1) × 256 MiB` (the per-slot compute floor). The KV term is **not** multiplied by `parallel` — `ctx_size` is the size of one pool shared by the slots, not a window split between them, and a unified pool shares the RAM-KV rather than duplicating it.

**Worked example — a 20 GiB model body (`gguf_mib` = 20480) @ 128K ctx (`ctx_size: 131072`) on a 24 GB card, file-size fallback formula:** f16 KV = 23,040 MiB → needed ≈ 44,544 MiB (~43.5 GiB) ❌; `turbo3` KV = 4,224 MiB → needed ≈ 25,728 MiB (~25.1 GiB), still more than a 24 GB card holds, so it fits only with the KV offloaded to host RAM (`no_kv_offload`: needed ≈ 22,528 MiB of VRAM) or a lower `ctx_size`.

---

## Section 5 — Performance & GPU Doctrine

This section covers the five standing doctrine flags (recommended on for new manifests), the MoE / multi-GPU placement flags, and the performance long-tail. Doctrine and MoE flags are **spawn-argv**. Bounds/enums from `manifest.py`; doctrine from `docs/TURBOQUANT_FLAGS.md`.

### The five doctrine flags

These are the recommended settings for new manifests unless a model-specific reason requires deviation (`docs/TURBOQUANT_FLAGS.md`); the manager does not inject them (only `ctx_checkpoints`, `cache_type_k` and `cache_type_v` have injected defaults). All five are spawn-argv.

**`flash_attn`** (`bool | str`; str enum `{on, off, auto, enabled, disabled}`; default doctrine `true`). Enables the fused-attention path that the compressed KV cache types need. On argv-build a bool is normalized to an explicit `--flash-attn on` / `--flash-attn off` (not a bare flag), so the child process always carries a value — useful when auditing `/proc/<pid>/cmdline`. **When to deviate:** a model that does not use the compressed KV types can run `flash_attn: false` with `f16` KV. Otherwise leave it `true`; the TurboQuant `turbo*` types depend on it.

**`no_context_shift`** (`bool`; default doctrine `true`). Turns off context shifting (the engine sliding its window forward by discarding old tokens once the context is full). **When to deviate:** a model known to shift its context correctly *could* run `false`; the recommended setting is `true`.

**`cache_reuse`** (`int`; bound `(0, 65536)`; default doctrine `256`). Enables prefix-cache reuse across requests in the same warm slot — cuts long-tail prefill on follow-ups; the value is the minimum chunk size, in tokens, the engine tries to reuse via KV shifting. **When to deviate:** a pure one-shot batch workload with no follow-up traffic gains nothing from it. Doctrine value is `256`. **Note:** on some models this flag is accepted and then ignored — see below.

> **When `cache_reuse` is accepted but does nothing**
>
> Setting this flag does not guarantee it is active. llama.cpp turns it off at
> model-load time for two separate reasons:
>
> 1. **An mmproj is loaded** (the multimodal projector file). llama.cpp logs
>    `cache_reuse is not supported by multimodal`.
> 2. **The context cannot be shifted.** llama.cpp logs
>    `cache_reuse is not supported by this context`. Shifting is llama.cpp's
>    technique for sliding a conversation window forward without re-processing
>    the prompt from scratch. It is unavailable on the STEP35 model
>    architecture, and on models using M-ROPE or I-M-ROPE position encodings
>    — rotary encodings that reserve dimensions for image and video
>    position.
>
> On a typical multimodal model both reasons apply and both lines appear in a
> single load.
>
> **The KV cache type is not a cause.** `cache_type_k` / `cache_type_v`
> (`turbo2`, `turbo3`, `turbo4`) play no part in llama.cpp's shiftability check,
> and a text-only model using turbo KV reuses its cache normally. Because
> `turbo3` is the default cache type, it is usually present when the flag turns
> out to be inert, which can make it look responsible when it is not.
>
> The `cache_reuse_inert_mmproj` marker on the manifest-read API is a proxy for
> *inertness*, keyed on an mmproj being set. It cannot see one case: a model
> that uses M-ROPE but ships no mmproj. `cache_reuse` really is inert on that
> model, but there is no mmproj for the marker to key on, so **the marker will
> report it as fine**. For that case, llama.cpp's load-time log line is the
> only reliable signal.


**`slot_prompt_similarity`** (`float`; bound `(0.0, 1.0)`; default doctrine `0.5`). Lets a slot reuse the prefix cache even when the prompt isn't byte-identical — a 50% similarity threshold — improving the ACTIVE_MATCH / warm-reuse hit rate. **When to deviate:** raise toward 1.0 to require near-identical prompts (fewer false reuses), lower for looser matching.

**`no_perf`** (`bool`; doctrine `true`). Suppresses per-request perf logging — a small CPU + log-noise win. **When to deviate:** a model under active perf debugging or benchmarking wants `no_perf: false` to surface per-request timings (flip back after).

### MoE / multi-GPU placement

| Flag | Type | Bound / enum | Example value(s) | Apply | Purpose |
|---|---|---|---|---|---|
| `cpu_moe` | bool | — | `true` | spawn-argv | `-cmoe`: place **all** MoE expert layers on CPU (weights on host, attention on GPU). Frees GPU VRAM at the cost of CPU-bound expert compute. Typical pairing: `cpu_moe: true` + `no_mmap: true` + `mlock: true` + an explicit `threads`. |
| `n_cpu_moe` | int | `(0, 256)` | `10` | spawn-argv | `-ncmoe N`: place the **first N** MoE layers on CPU (a partial-overflow version of `cpu_moe`). Use to shave exactly enough VRAM to fit rather than dumping all experts to CPU. |
| `split_mode` | str | enum `{none, layer, row, tensor}` | `layer`, `none` | spawn-argv | `--split-mode`: how to spread the model across multiple GPUs. `layer` = split by layer across the cards. `none` = single-card, no split (pair it with `main_gpu` to pin a card). `row`/`tensor` are allowed. |
| `main_gpu` | int | `(0, 16)` | `0`, `1` | spawn-argv | `--main-gpu`: the primary GPU index (holds the non-split tensors / is the pin target when `split_mode: none`). e.g. `1` to pin a model to the second card so it can co-reside with a model on card 0. |
| `fit` | str | enum `{on, off}` | unset | spawn-argv | Tom's-Fork auto-memory-fit toggle (`--fit`). Unset does not mean off: the engine's own default is `on` (it adjusts unset parameters to fit device memory) and the manager injects no `--fit` flag, so set `fit: off` to disable the auto-fit. Sizing is normally explicit via the VRAM gate. |
| `fit_ctx` | int | `(1, 2_000_000)` | unset | spawn-argv | Target context for the `fit` auto-sizer. Unset; sizing is explicit via `ctx_size` + the fit gate. |

> **`tensor_split` is allowlisted; `fit_target` is not.** `tensor_split` takes a CSV per-GPU ratio (`"0.55,0.45"`) and is validated at manifest load by `parse_strict_csv_numeric`: 2–16 numeric elements, no negatives, not all-zero, 64 characters maximum. Because the visible device count is a runtime property a static validator cannot see, a second gate runs at spawn — `safety.check_tensor_split_devices` refuses the spawn when the element count does not match the number of visible devices, rather than risking a wrong placement and an OOM. `fit_target` (CSV `MiB,MiB`) remains in `DENIED_FLAGS` pending its own validator, so auto-fit targets still cannot be pinned from a manifest.

**The cpu-moe overflow recipe:**
```yaml
# Full offload of all experts to CPU:
cpu_moe: true
no_mmap: true      # force-load weights into RAM rather than mmap (see long-tail)
mlock: true        # lock them resident, no swap
threads: 16        # CPU expert compute needs the thread count set

# Partial overflow — shave just enough:
n_cpu_moe: 10      # first 10 MoE layers to CPU
split_mode: none   # single-card pin
main_gpu: 1
```

### The performance long-tail

Each of these is allowlisted; most are usually left unset (engine defaults). Value/bound + one-line purpose:

| Flag | Type | Bound / enum | Example value | Apply | Purpose |
|---|---|---|---|---|---|
| `mlock` | bool | — | `true` | spawn-argv | `--mlock`: lock model pages in RAM so the OS never swaps them out — steadier latency, needs the RAM headroom. Typically paired with `cpu_moe`. |
| `no_mmap` | bool | — | `true` | spawn-argv | `--no-mmap`: load weights fully into memory instead of memory-mapping the GGUF. Faster steady-state, higher load-time RAM. |
| `numa` | str | enum `{none, distribute, isolate, numactl}` | unset | spawn-argv | `--numa`: NUMA placement policy for multi-socket hosts. `distribute` spreads threads across nodes; `isolate` pins to one; `numactl` defers to an external `numactl` wrapper. Leave unset on a single-socket host. |
| `swa_full` | bool | — | unset | spawn-argv | `--swa-full`: use the full (non-windowed) sliding-window-attention KV cache for SWA models — more VRAM, avoids the windowed-attention approximation. Unset. |
| `warmup` | bool | — | unset | spawn-argv | `--warmup` / `--no-warmup`: run a dummy decode at spawn to warm kernels/allocations before serving. Unset (engine default). |
| `check_tensors` | bool | — | unset | spawn-argv | `--check-tensors`: validate tensor data on load (catches corrupt GGUF) at a load-time cost. Unset — blobs are already SHA256-gated by the manifest. |
| `repack` | bool | — | unset | spawn-argv | Repack weights into a CPU-optimized layout at load (throughput win for CPU-heavy paths). Unset. |
| `op_offload` | bool | — | unset | spawn-argv | Offload individual ops to the GPU where beneficial (fine-grained op placement). Unset (engine default). |
| `no_host` | bool | — | unset | spawn-argv | `--no-host`: bypass the host buffer so extra buffers can be used. Unset. |
| `direct_io` | bool | — | unset | spawn-argv | `--direct-io`: use direct/unbuffered I/O for GGUF reads (bypass page cache) — helps on fast NVMe, hurts on re-reads. Unset. |
| `sleep_idle_seconds` | int | `(-1, 86400)` | `-1` | spawn-argv | Idle timeout before the sidecar sleeps/tears down. `-1` = never sleep (pin the process resident). Example: `-1` on co-resident models that should stay hot. |

> **Boolean encoding (all bool flags above):** `True` → bare `--flag`; `False` → **omitted entirely** (not `--flag false`). So a `false` boolean has no effect on the spawned process — it's as if the flag were absent. The one exception is `flash_attn`, which always emits an explicit `on`/`off` value.

---

## Section 6 — Chat Template & Reasoning

These flags control how `llama-server` turns the OpenAI/Ollama request into the model's on-the-wire prompt, and how a thinking model's reasoning is parsed and budgeted. Every flag here is **spawn-argv** (it becomes part of the `llama-server` command line), including `reasoning_budget`, whose manifest value is the engine's `--reasoning-budget` launch flag. The per-call controls in this section are the request-body field `thinking_budget_tokens`, which needs no cold-spawn, and `reasoning_effort`, which is resolved once when the engine starts (both in 6.4). Changing a manifest value requires a cold-spawn (Option A `keep_alive: 0` / Option B natural idle teardown / Option C container restart) to take effect on a running slot.

### 6.1 `jinja` — load-bearing, keep it on

```yaml
jinja: true
```

`jinja` is a `bool` (allowlist type `bool`; argv `True` → bare `--jinja`, `False` → omitted). Every example manifest in Section 12 sets it to `true`.

`--jinja` makes `llama-server` use the model's own bundled Jinja chat template instead of its legacy hard-coded prompt formatter. In the vendored engine Jinja is already on by default for `llama-server` (`use_jinja = true` in `common/common.h`; the engine accepts `--jinja` and `--no-jinja`). Turbohaul omits the flag when `jinja` is `false` and does not emit `--no-jinja`, so the manifest key makes the setting explicit rather than being the only switch that turns it on. Two capabilities depend on Jinja being on:

1. **Tool calls.** Structured `tools` / `tool_choice` only work when the server renders the model's Jinja template, because that template is what emits the tool-call grammar the parser keys off. If Jinja were off in the engine (`--no-jinja`, which Turbohaul does not emit), `llama-server` would reject a request that carries `tools` (or a `tool_choice` other than `auto`) with the error "tools param requires --jinja flag"; text-JSON tool-call recovery also requires the request to have advertised `tools`.
2. **Thinking mode.** `llama-server` enables thinking only when Jinja is on (the engine default), the model's chat template supports `enable_thinking`, and `--reasoning` is not `off`.

**Doctrine:** if you copy a manifest, keep `jinja: true`. It is treated as load-bearing throughout these docs (for tool calls, see [TOOL_CALL_HANDLING.md](./TOOL_CALL_HANDLING.md)). There is no reason to set it to `false` for a chat model.

### 6.2 `chat_template` — built-in name or bounded token, never a Jinja body

`chat_template` is a `str`, but it is the most heavily gated flag in the allowlist because an inlined Jinja template body is a server-side-template-injection (SSTI) vector. Normally `jinja: true` selects the model's own bundled template, so you rarely set `chat_template` at all. Set it only to *override* the auto-selected template with a specific built-in.

Validation (from `manifest.py` `_validate_flag_value` / `_check_jinja_injection`) runs three gates in order:

1. **SSTI guard (hard reject).** If the value contains `{%` or `{{` it is rejected outright — those are Jinja constructs that a non-sandboxed Jinja env could exploit to read the filesystem. A custom Jinja template cannot be supplied from a manifest at all: `chat_template_file` is in `DENIED_FLAGS`.
2. **Built-in enum match.** If the value is one of the `SAFE_CHAT_TEMPLATE_NAMES` it is accepted immediately. The set is the subset of llama.cpp's bundled templates the allowlist recognizes:

   `chatml`, `llama2`, `llama3`, `llama3.1`, `llama3.2`, `llama3.3`, `gemma`, `gemma2`, `gemma3`, `gemma4`, `mistral`, `mistral-v1`, `mistral-v3`, `mistral-v3-tekken`, `mistral-v7`, `phi3`, `phi4`, `deepseek`, `deepseek2`, `deepseek-r1`, `qwen`, `qwen2`, `qwen2.5`, `qwen3`, `qwen3.5`, `qwen3.6`, `command-r`, `command-r-plus`, `vicuna`, `alpaca`, `zephyr`, `chatglm3`, `chatglm4`, `openchat`, `orion`, `yi`, `monarch`, `smollm`, `minicpm`, `exaone3`, `rwkv-world`, `granite`, `qwen3-thinking`, `qwq`, `default` (45 names).
3. **Bounded plain-token fallback.** If the value is not a known built-in name, it is accepted only if it is a plain identifier of **≤ 256 chars** matching `^[A-Za-z0-9_.\-]+$` (alphanumeric plus `. _ -`). Anything longer, or containing any other character, is rejected. This lets a newly-shipped built-in name that isn't yet in the enum through, while structurally forbidding a template body (which would need spaces, braces, or `>256` chars).

| Property | Value |
|---|---|
| Type | `str` (special-cased) |
| Accepted | built-in enum name, OR `≤256`-char `^[A-Za-z0-9_.\-]+$` token |
| Hard-rejected | any value containing `{%` or `{{` (SSTI guard) |
| Layer | spawn-argv (cold-spawn to apply) |
| Typical use | none — `jinja: true` + the model's bundled template |

### 6.3 `skip_chat_parsing`, `special`, `spm_infill` — parsing toggles

All three are `bool` (argv `True` → bare `--flag`, `False` → omitted). They are not part of the standard doctrine and are available for niche cases. Their semantics map directly to the corresponding llama.cpp / `llama-server` options:

| Flag | Type | What it maps to in llama.cpp | When you'd use it |
|---|---|---|---|
| `skip_chat_parsing` | `bool` | Forces a pure content parser (`--skip-chat-parsing`) even if a Jinja template is specified: the model's output, including any reasoning and/or tool calls, comes back in the content section without the tool-call / reasoning extraction pass. | Debugging what the model literally emitted, or a client that wants to do its own parsing. Leave off in normal use — it defeats tool-call recovery and reasoning-block extraction. |
| `special` | `bool` | Renders/allows **special tokens** to be emitted in output rather than being suppressed (`--special`). | Rare — inspecting a model's control tokens, or a downstream consumer that needs the raw special tokens in the stream. |
| `spm_infill` | `bool` | Uses the **Suffix/Prefix/Middle** pattern for infill (`--spm-infill`) instead of Prefix/Suffix/Middle, as some fill-in-the-middle (FIM) models prefer. Only meaningful for code/infill models. | A FIM-capable model that prefers Suffix/Prefix/Middle ordering. No effect on ordinary chat models. |

Because none of these is set by default, treat them as "known-to-llama.cpp, off by default" — set one only if you have a specific model behavior you are matching, and cold-spawn to apply.

### 6.4 Reasoning — `reasoning`, `reasoning_format`, `reasoning_budget`

This trio governs thinking models. A reasoning-capable manifest typically sets `reasoning: auto`, with `reasoning_format` and `reasoning_budget` alongside it.

**`reasoning`** — `str` enum, allowed values `{on, off, auto}` (from `SAFE_LLAMA_FLAG_STRING_ENUMS`). Spawn-argv.

- `on` — enable the reasoning path (the engine sets the chat template's `enable_thinking` to true).
- `off` — disable it (`enable_thinking` false).
- `auto` — let `llama-server` decide from the model's template/metadata. **This is the engine's default** (`--reasoning` defaults to `auto`).

**`reasoning_format`** — `str` enum, allowed values `{none, deepseek, deepseek-legacy, auto}`. Spawn-argv. Selects how the thinking segment is delimited/extracted so it can be surfaced as structured `reasoning_content` (and, on the non-stream path, wrapped as `<think>…</think>` in `message.content`).

| Value | Meaning | Engine behavior |
|---|---|---|
| `deepseek-legacy` | The older DeepSeek `<think>` delimiter convention. | Keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`. |
| `auto` | Let the server infer the format from the model. | The engine's default; the format is inferred. |
| `deepseek` | Current DeepSeek reasoning delimiter convention. | Puts thoughts in `message.reasoning_content`. |
| `none` | No reasoning-format extraction. | Leaves thoughts unparsed in `message.content`. |

A manifest may set `reasoning: auto` and omit `reasoning_format` entirely — the server then infers formatting itself.

**`reasoning_budget`** — `int`, bounds `(-1, 1_000_000)`. **Spawn-argv:** the manifest value becomes the engine's `--reasoning-budget` launch flag and stays fixed for that engine process's life, so a change needs a cold-spawn. The launch value is not always the raw manifest value: the manager resolves the first request that starts the engine, if it carries `reasoning_effort`, against the manifest `reasoning_budget` and uses the result as the launch flag (see [REASONING_EFFORT_LADDER.md](./REASONING_EFFORT_LADDER.md)). A client can also override the budget per request with the request-body field `thinking_budget_tokens`, which the forwarder passes through and the engine reads (when the field is absent, the launch value applies). It caps how many tokens the model may spend inside its thinking block before it must produce the answer.

Semantics:

- `-1` — unbounded thinking (no cap).
- `0` — no thinking budget (effectively suppresses the reasoning span).
- `N` (positive) — allow up to N reasoning tokens.

**Recommended convention:** set `reasoning_budget` to **about half of `max_tokens`**. This leaves the other half of the token budget for the actual answer, so a thinking model doesn't exhaust its window inside `<think>`. For example, `reasoning_budget: 8192` for a 16K `max_tokens`.

Related caution (see [AI_AGENT_SETUP.md](./AI_AGENT_SETUP.md)): keep the request's `max_tokens` **≥ 2000** for thinking-model agent loops, or the model can exhaust a small budget entirely inside its thinking block and never emit an answer.

**Cross-field rule — `reasoning_budget` must stay below a bounded `n_predict`.** If `n_predict` is a positive number and `reasoning_budget >= n_predict`, the manifest is **rejected at load**: the model could burn its entire allowance inside the thinking block and return no answer. Lower `reasoning_budget` below `n_predict`, or set `n_predict: -1` (unbounded), which is exempt from the check and is what most of the example manifests in Section 12 use.

---

## Section 7 — MoE + MTP / Speculative Decoding

Speculative decoding runs a cheap *draft* that proposes several tokens per step, which the main model then verifies in a single forward pass. When the draft is accurate this raises tokens/sec with **no quality loss** — the main model still makes every final decision. Turbohaul's allowlist exposes three speculative families: **draft-MTP** (multi-token prediction, bundled head), **draft-D-Flash**, and **draft-D-Spark** (D-Spark = D-Flash + a Markov head, a superset, not an alternative). Every flag in this section is **spawn-argv** (it becomes part of the `llama-server` command line), so applying or changing any of them requires a cold-spawn. They compose with the TurboQuant KV path (`cache_type_k`/`cache_type_v` = `turbo2/3/4`); keep `flash_attn` on — the engine requires flash attention whenever the V cache is quantized.

### 7.1 `spec_type` — the family selector

```yaml
spec_type: draft-mtp        # or draft-dflash, or draft-dspark
```

`spec_type` is a `str` enum; the allowlist accepts `draft-mtp`, `draft-dflash`, and `draft-dspark` (`SAFE_LLAMA_FLAG_STRING_ENUMS["spec_type"]`). The engine's real CLI name-map (`common/speculative.cpp`) also registers `draft-simple` (classic external draft model), `draft-eagle3` and several `ngram-*` types — Turbohaul deliberately stays narrower than the engine's own enum: `draft-simple` is unimplemented here, and `draft-eagle3` is out of scope. Any other value is rejected at manifest validation. This is a **single-select** field by construction — exactly one value, never a list — so no two speculative families can ever be configured for the same model at once.

Two structurally different drafter shapes hide behind that one field:

- **`draft-mtp`** uses the model's own bundled **nextn / MTP head** as the drafter — the draft weights live *inside* the target model's own GGUF (Qwen3.5/3.6-class models built with the nextn head). No second model file is involved. This is why there is no `model_draft` flag here — that flag is in `DENIED_FLAGS` (Section 11.1). On a model without a bundled nextn head, `spec_type: draft-mtp` has nothing to draft from.
- **`draft-dflash`** and **`draft-dspark`** are **standalone draft models** — the draft weights live in their *own*, separate GGUF file (the vendored engine's converter requires `--target-model-dir` to pull target metadata into a standalone D-Flash/D-Spark conversion). Selecting either requires **also** setting `spec_draft_gguf_blob_sha256` (Section 1) to the content hash of that draft model's own blob — the manager resolves it and injects `--spec-draft-model <path>` at spawn, the same content-addressed pattern as `mmproj_blob_sha256`. There is deliberately no path/URL/repo form of this (Section 11.4) — omitting the hash doesn't fail manifest validation, and no `--spec-draft-model` is injected, so the engine has no draft model to load. The pairing is enforced the OTHER way too, at the manager: `spec_draft_gguf_blob_sha256` set while `spec_type` is `draft-mtp` or omitted entirely does **not** inject `--spec-draft-model` (the engine's own `has_dft()` keys purely on the draft path (or repo) being non-empty and ignores `spec_type` entirely, so without this manager-side gate a hash-only manifest would load a full second, completely unbudgeted model at the target's full context size — an availability hazard, not a confidentiality/integrity one, but exactly the "bad config can be authored" shape the single-select design exists to prevent). The manager logs a warning rather than silently dropping the hash in that case.

A `-nomtp` manifest next to an MTP one — the same model with `spec_type` and its sub-knobs **omitted** — is a convenient way to A/B MTP on vs off. The `draft-dflash`/`draft-dspark` config surface is wired end-to-end (this reference, the FE dropdown, the manifest schema, the manager's dispatch sites); using it needs a real D-Flash/D-Spark draft GGUF for your target.

### 7.2 `spec_draft_*` sub-knobs

All draft-tuning knobs take effect for any of the three `spec_type` families (`draft-mtp`, `draft-dflash`, `draft-dspark`). Types and bounds are exact from `SAFE_LLAMA_FLAGS` + `SAFE_LLAMA_FLAG_BOUNDS`:

| Flag | Type | Bound | What it maps to in llama.cpp | Default |
|---|---|---|---|---|
| `spec_draft_n_max` | `int` | `(0, 64)` | Max draft tokens proposed per step (`--spec-draft-n-max`). MTP commonly ~2–3. Higher = more speculation per step (bigger win when the draft is accepted, bigger wasted-compute cost when rejected). | engine default `3` |
| `spec_draft_n_min` | `int` | `(0, 64)` | Min draft tokens per step (`--spec-draft-n-min`). Floors how many tokens the drafter must propose before verification. | engine default `0` |
| `spec_draft_p_min` | `float` | `(0.0, 1.0)` | Minimum probability to keep drafting (`--spec-draft-p-min`). The drafter stops proposing once its confidence falls below this — a higher value drafts fewer, higher-confidence tokens. | engine default `0.0` |
| `spec_draft_p_split` | `float` | `(0.0, 1.0)` | Draft-tree split-probability threshold (`--spec-draft-p-split`) — governs when the speculative tree branches. | engine default `0.1` |
| `spec_draft_ngl` | `int` | `(-1, 999)` | Draft-model GPU layers (`--spec-draft-ngl`, alias `-ngld` / `--gpu-layers-draft`). For bundled MTP the head normally lives on the same device as the main model, so this is rarely needed. | engine default: auto (`-1`) |
| `spec_draft_backend_sampling` | `bool` | — | Use backend-side sampling for the draft path (argv `True` → bare flag, `False` → omitted). | engine default is on; `false` only omits the flag (no `--no-spec-draft-backend-sampling` is emitted), so it cannot turn it off |

**In practice**, a typical MTP manifest sets only two of these: `spec_type: draft-mtp` and `spec_draft_n_max` (2 or 3). The remaining sub-knobs fall through to the server defaults.

### 7.3 `n_rs_seq` relationship

When `spec_type` is `draft-mtp`, `draft-dflash`, or `draft-dspark`, the engine's restored-state sequence count `n_rs_seq` equals `spec_draft_n_max`. In other words, `spec_draft_n_max: 2` implies `n_rs_seq = 2` and `spec_draft_n_max: 3` implies `n_rs_seq = 3`. This matters for KV-state accounting: a speculative-enabled slot tracks `n_max` speculative sequences, so the restored/checkpointed KV state carries that many parallel sequence lanes. Keep this in mind when reasoning about KV-reuse and restore behavior on a speculative slot — the sequence bookkeeping is driven by `spec_draft_n_max`, not by `parallel`.

This mirrors the vendored engine's `need_n_rs_seq()` (`common/common.h`), which returns nonzero iff the configured type is MTP, D-Flash, or D-Spark — `draft-eagle3` and `draft-simple` do **not** set `n_rs_seq`; neither is in our allowlist. `manager.py`'s `_engine_fingerprint`, `_read_model_footprint`, and its spawn-gate counterpart all key off one shared constant, `manifest.SPEC_TYPES_NEEDING_RS_SEQ = {"draft-mtp", "draft-dflash", "draft-dspark"}`, so this predicate cannot drift between the three call sites. Note: for `draft-dflash`/`draft-dspark` specifically, the KV-cache **byte-size** term (Section 4's dimension-aware path) stays inert today regardless of this widening — it's gated on the target GGUF's `nextn_predict_layers` field, which is specific to MTP's bundled-head shape and is not populated by a standalone draft model's own (separate) GGUF. Only `n_rs_seq` changes behavior for D-Flash/D-Spark today; the KV-byte accounting for a standalone draft model's own footprint is not yet wired (see Section 11.4).

### 7.4 When MTP helps vs hurts, and how it composes

MTP is a **workload-dependent** win, not a free one:

- **Helps most** on predictable, low-entropy continuations: structured output, code, JSON, repetitive formatting. There the draft's proposed tokens are frequently accepted, so several tokens land per verification pass and tokens/sec climbs.
- **Helps least (can even hurt)** on high-entropy creative text: draft acceptance is low, so most speculative tokens are rejected and the extra draft compute is wasted. This is the tradeoff `spec_draft_n_max` tunes — a higher `n_max` amplifies both the upside (when accepted) and the wasted work (when rejected). Measure it per model (for example with an A/B pair of manifests, one with and one without `spec_type`) rather than assume it.

**Composition with the rest of the stack:**

- **TurboQuant KV** (`cache_type_k`/`cache_type_v` = `turbo2/3/4`): MTP composes cleanly with compressed KV — the fragment below runs `spec_type: draft-mtp` alongside `cache_type_k: turbo3` / `cache_type_v: turbo3`.
- **`flash_attn`**: keep it on — the engine requires flash attention whenever the V cache is quantized, and MTP manifests are normally run with `flash_attn: true`.
- **Turbo KV + `flash_attn` + MTP together** is a common high-throughput recipe (turbo3 KV + `flash_attn: true` + `spec_type: draft-mtp`).

A minimal MTP manifest fragment:

```yaml
llama_server_flags:
  flash_attn: true
  cache_type_k: turbo3
  cache_type_v: turbo3
  spec_type: draft-mtp
  spec_draft_n_max: 3      # 2-3 is typical for MTP
```

---

## Section 8 — Context extension: RoPE / YaRN

These nine flags change how the model's **positional encoding** is scaled so `llama-server` can run a context window *longer than the length the model was trained on*. They are all **spawn-argv** — baked into the `llama-server` command line at process fork, so a change to any of them requires a **cold-spawn** to take effect (the running slot keeps the old cmdline; see the doctrine note at the end of this section). None of them are per-request knobs.

### When (and why) you'd touch these at all

A GGUF is trained at some native context length (say 32K). If you set `ctx_size` beyond that native length, the RoPE frequencies the model learned no longer line up with the token positions it now sees, and quality degrades — the model gets "lost" past its trained horizon. RoPE scaling and YaRN are the two techniques that **rescale the rotary frequencies** so the model degrades gracefully instead of falling off a cliff.

**The honest default: leave all nine unset.** Modern GGUFs bake their correct RoPE parameters into the file's metadata, and `llama-server` reads them automatically. You only reach for these flags when you are deliberately pushing a model past its trained window and the metadata alone isn't giving you what you want. The long-context examples in Section 12 (e.g. 12.2, at `ctx_size: 250000`) set no RoPE/YaRN flag — they rely on the model's baked metadata plus KV-in-RAM (`no_kv_offload`), not manual RoPE overrides. Treat this whole section as an advanced escape hatch.

**The tradeoff:** extending context is never free. Beyond the RoPE rescale, a longer window costs KV cache (linear in tokens) and, even with correct scaling, effective recall over the *extended* region is weaker than over the *native* region. YaRN is the better-quality extension method (it scales low- and high-frequency RoPE dimensions differently instead of uniformly), but it still can't manufacture attention the base model never learned. Extend only as far as you actually need.

### The knobs

`rope_scaling` selects the *method*; the rest are its parameters. Linear scaling and YaRN are mutually-relevant families — set `rope_scaling` first, then only the parameters that method reads.

| Flag | Type | Bound (from manifest.py) | What it does |
|---|---|---|---|
| `rope_scaling` | str enum | `none` / `linear` / `yarn` | Picks the extension method. `none` = use the model's native RoPE unchanged. `linear` = uniform position-interpolation (simple, lower quality at large factors). `yarn` = NTK-by-parts scaling (higher quality; the modern choice for large extensions). |
| `rope_scale` | float | `0.0`–`1000.0` | Linear-scaling factor as `1/scale` on positions. Maps to llama.cpp `--rope-scale`; a value of `N` means "stretch the trained window by ~N×". Read primarily under `linear`. |
| `rope_freq_base` | float | `0.0`–`10_000_000.0` | The RoPE theta base (`--rope-freq-base`). Overrides the model's base frequency directly; raising it is the "NTK-aware" way to extend without retraining. The wide upper bound accommodates models trained with very large theta (e.g. 1e6+). |
| `rope_freq_scale` | float | `0.0`–`100.0` | The inverse form of `rope_scale` (`--rope-freq-scale`); a frequency multiplier rather than a position divisor. Set one of `rope_scale` / `rope_freq_scale`, not both. |
| `yarn_orig_ctx` | int | `0`–`2_000_000` | Tells YaRN the model's **original trained context length** so it knows the ratio it's extending from. `0` = let llama.cpp infer it from metadata. Load-bearing for YaRN math — set it to the model's true native length when you override. |
| `yarn_ext_factor` | float | `-1.0`–`100.0` | YaRN extrapolation mix factor (`--yarn-ext-factor`). `-1.0` = use llama.cpp's default; `0.0` = pure interpolation; higher = more extrapolation. Controls how aggressively YaRN reaches beyond the trained window. |
| `yarn_attn_factor` | float | `-1.0`–`100.0` | YaRN attention-scaling factor (`--yarn-attn-factor`); scales attention magnitude to compensate for the temperature shift extension introduces. `-1.0` = default. |
| `yarn_beta_slow` | float | `-1.0`–`100.0` | YaRN "slow" boundary (`--yarn-beta-slow`) — the low-frequency ramp cutoff in the NTK-by-parts blend. `-1.0` = default. |
| `yarn_beta_fast` | float | `-1.0`–`100.0` | YaRN "fast" boundary (`--yarn-beta-fast`) — the high-frequency ramp cutoff. Together `beta_slow`/`beta_fast` define which RoPE dimensions get interpolated vs. extrapolated. `-1.0` = default. |

**Practical guidance.** If you must extend: set `rope_scaling: yarn`, set `yarn_orig_ctx` to the model's real native length, set `ctx_size` to your target, and leave the four `yarn_*` tuning factors at their `-1.0` defaults unless a specific model card tells you otherwise. Reserve `linear` + `rope_scale` for the rare case a model card explicitly specifies linear interpolation. The `rope_freq_base` override is the lowest-level lever — use it only when you know the exact theta you want. In all cases, **a cold-spawn is required** for the change to bind (patch the manifest, then trigger teardown per the TurboQuant doctrine's Option A/B/C, then verify via `/proc/<pid>/cmdline`).

---

## Section 9 — Sampling reference

This is the token-selection surface: temperature, truncation, repetition control, and the alternative sampler families (Mirostat, XTC, DRY, dynamic-temp, adaptive). **Read this box first:**

> **The common sampling flags in this section are REQUEST-BODY-HOT — do not bake them into a manifest.** llama.cpp accepts these as per-request parameters on the completions/chat API. The Turbohaul forwarder passes a fixed allow-list of request fields straight through from each client call (`_COMMON_FORWARDED_KNOBS` in `api/chat_completion.py`): `temperature` (manifest flag `temp`), `top_p`, `top_k`, `min_p`, `max_tokens`/`n_predict`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `repeat_last_n`, `typical_p`, `seed`, `mirostat`, `mirostat_lr` and `mirostat_ent`. The other samplers here (`top_n_sigma`, `xtc_*`, `dynatemp_*`, `dry_*`, `adaptive_*`, `ignore_eos`) are **not** in that allow-list, so for a client of Turbohaul the manifest is the only place to set them. If you put a sampling flag in `llama_server_flags`, all you are doing is **changing the server-side default** for that sampler — and because `llama_server_flags` is spawn-argv, that default only changes on a **cold-spawn**, and a client request can still override the forwarded ones per call. So the normal path is: set sampling per-request from the client; touch the manifest **only** when you genuinely want to move the *default* (and accept a cold-spawn to do it). This is why sampling is normally left to the caller. The bound column below is still enforced if you *do* set one in a manifest.

**Column key:** the rows for the forwarded fields listed above are **request-hot** (overridable per call); the others are manifest-only. The manifest bound applies only when you choose to set it as a spawn-time default. Bounds are from `SAFE_LLAMA_FLAG_BOUNDS`.

### Core truncation + temperature

| Flag | Bound | Purpose (one line) |
|---|---|---|
| `temp` | `0.0`–`10.0` | Softmax temperature. `0` = greedy/deterministic; `~0.7` typical chat; higher = more random. The primary creativity dial. |
| `top_k` | `0`–`10000` | Keep only the K highest-probability tokens before sampling. `0` = disabled (no top-k cut). |
| `top_p` | `0.0`–`1.0` | Nucleus sampling: keep the smallest set whose cumulative probability ≥ p. `1.0` = disabled. |
| `min_p` | `0.0`–`1.0` | Keep tokens whose probability ≥ `min_p × (top token's probability)`. A relative floor; robust alternative to `top_p`. `0` = disabled. |
| `typical_p` | `0.0`–`1.0` | Locally-typical sampling — keep tokens near the distribution's entropy rather than its peak. `1.0` = disabled. (Ollama-parity flag.) |
| `top_n_sigma` | `-1.0`–`100.0` | Sigma-based truncation: keep tokens within N standard deviations of the top logit. `-1.0` = disabled (llama.cpp default sentinel). |

### Repetition control

| Flag | Bound | Purpose (one line) |
|---|---|---|
| `repeat_penalty` | `0.0`–`10.0` | Divides logits of recently-seen tokens. `1.0` = no penalty; `~1.1` mild. Above ~1.3 tends to damage coherence. |
| `repeat_last_n` | `-1`–`65536` | How many trailing tokens `repeat_penalty` looks back over. `-1` = whole context; `0` = disabled. |
| `presence_penalty` | `-10.0`–`10.0` | OpenAI-style flat penalty for any token that has already appeared (encourages new topics). `0` = off. (Ollama parity.) |
| `frequency_penalty` | `-10.0`–`10.0` | OpenAI-style penalty scaled by how *often* a token appeared (suppresses over-use). `0` = off. (Ollama parity.) |

### Determinism + control

| Flag | Bound | Purpose (one line) |
|---|---|---|
| `seed` | `-1`–`2⁶³−1` | RNG seed for reproducible sampling. `-1` = random each run. Set a fixed value only when you need bit-reproducible output. |
| `ignore_eos` | bool | Suppress the end-of-sequence token so generation won't stop on EOS. **Spawn-safe but dangerous** — combined with an unbounded `n_predict` it runs to the context limit. Use only for benchmarking/forced-length tests. |

### Mirostat (adaptive perplexity control — an *alternative* to top-k/top-p)

Mirostat replaces the truncation samplers with a feedback loop that targets a constant output perplexity. When enabled, prefer it *instead of* `top_k`/`top_p`, not alongside.

| Flag | Bound | Purpose (one line) |
|---|---|---|
| `mirostat` | `0`–`2` | Selects the algorithm: `0` = off, `1` = Mirostat v1, `2` = Mirostat v2 (the common choice). |
| `mirostat_lr` | `0.0`–`1.0` | Learning rate (`eta`) of the feedback loop; how fast it corrects toward the target. Default ~`0.1`. |
| `mirostat_ent` | `0.0`–`100.0` | Target entropy (`tau`); the perplexity setpoint the loop holds. Default ~`5.0`. |

### Newer / experimental sampler families

All off by default; none is in the forwarder's allow-list (see the box at the top), so they are set in the manifest. Use one family deliberately — stacking many at once interacts unpredictably.

| Flag | Bound | Purpose (one line) |
|---|---|---|
| `xtc_probability` | `0.0`–`1.0` | XTC ("Exclude Top Choices") — probability of applying XTC on a given step. `0` = disabled. XTC drops high-probability tokens to boost diversity/creativity. |
| `xtc_threshold` | `0.0`–`1.0` | XTC threshold: tokens above this probability become eligible for exclusion. Pairs with `xtc_probability`. |
| `dynatemp_range` | `0.0`–`10.0` | Dynamic-temperature range: temperature varies within `temp ± range` based on per-step entropy. `0` = static temperature. |
| `dynatemp_exp` | `0.0`–`10.0` | Dynamic-temperature exponent shaping how entropy maps to the temperature within `dynatemp_range`. |
| `dry_multiplier` | `0.0`–`10.0` | DRY ("Don't Repeat Yourself") penalty strength — penalizes repeating multi-token sequences (n-grams), not just single tokens. `0` = DRY off. |
| `dry_base` | `1.0`–`10.0` | DRY penalty growth base; the exponential base applied as a repeated sequence gets longer. Note the **min is 1.0** (a base < 1 would be degenerate). |
| `dry_allowed_length` | `0`–`65536` | Max sequence length DRY tolerates before it starts penalizing. Short = aggressive anti-repetition. |
| `dry_penalty_last_n` | `-1`–`65536` | Lookback window DRY scans for repeated sequences. `-1` = whole context; `0` = disabled. |
| `adaptive_target` | `-1.0`–`100.0` | Target for llama.cpp's adaptive sampler (a self-tuning sampler); `-1.0` = default/off sentinel (any negative value disables it; llama.cpp's valid range is `0.0`–`1.0`). Semantics track llama.cpp's `--adaptive-*` implementation — set only if you're deliberately using that sampler. |
| `adaptive_decay` | `0.0`–`1.0` | Decay rate (0–1 factor; llama.cpp's valid range is `0.0`–`0.99`, default `0.90`) of the adaptive sampler's internal state. Pairs with `adaptive_target`. |

**Practical guidance.** For everyday use, set nothing here at the manifest level and let clients send `temperature` + one truncation sampler (`top_p` **or** `min_p`) plus a mild `repeat_penalty` per request. Reach for Mirostat, XTC, DRY, dynamic-temp, or the adaptive sampler only for a specific behavior problem (Mirostat for stable perplexity, DRY for stubborn loop-repetition, XTC/dynatemp for creative diversity) — and pick **one** family at a time. If you truly want a model to *default* to a non-standard sampler for every caller, that's the one legitimate reason to write a sampling flag into `llama_server_flags` (and then cold-spawn).

---

## Section 10 — Server toggles & debug

Two groups of process-level switches: **server toggles** that turn HTTP endpoints and modes on/off, and **debug/logging** switches that control `llama-server`'s console output. **All of these are spawn-argv** — they're part of the `llama-server` command line, so a change requires a **cold-spawn** to take effect. None are per-request. The manager sets none of these by default (llama.cpp's defaults apply) — they're here for when you need to expose an endpoint or debug a spawn.

### Server toggles

| Flag | Type / enum (manifest.py) | What it exposes / does | When to use |
|---|---|---|---|
| `metrics` | bool | Enables the `/metrics` Prometheus endpoint on `llama-server` (per-slot token throughput, timings, etc.). | Turn on when you want to scrape the sidecar directly for observability. Off = one less exposed endpoint. |
| `slots` | bool | Server-side toggle for the `/slots` endpoint that reports per-slot state (which is what the manager's `LiveSlotsPoller` reads). The engine serves it by default; because `false` is omitted from argv rather than sent as `--no-slots`, this flag cannot turn it off. | Normally unnecessary: the endpoint is on by default and the manager's monitor already polls `/slots`. |
| `props` | bool | Enables changing global properties via `POST /props`; `GET /props` (model properties + the loaded chat template) is served regardless. | `GET /props` is handy for confirming which template/props the server actually loaded; set `props` only if you need `POST /props`. |
| `embeddings` | bool | Restricts the server to the **embeddings** use case so `/embedding` (and `/v1/embeddings`) return vectors. | Turn on only for an embedding model. On a chat model it changes pooling behavior and is usually wrong — leave off for generative models. |
| `reranking` | bool | Enables **rerank** mode (the `/rerank` scoring endpoint). | Only for a reranker/cross-encoder model. |
| `pooling` | str enum: `none` / `mean` / `cls` / `last` / `rank` | Sets the embedding **pooling strategy** — how per-token hidden states collapse into one vector. `mean` = average, `cls` = first/CLS token, `last` = last token, `rank` = rerank pooling, `none` = no pooling. | Set to match what the embedding/rerank model expects (model card tells you). Irrelevant for pure generation. |
| `offline` | bool | Forces **offline** mode — no network fetches (no reaching out to Hugging Face, etc.) during load. | Belt-and-suspenders for an air-gapped/locked-down spawn. (Note: Turbohaul already denies all path/URL/hf fetch flags, so the server has no network-fetch flags to begin with — this is defense-in-depth.) |

### Debug / logging

| Flag | Type / enum (manifest.py) | What it does |
|---|---|---|
| `verbose` | bool | Turns on verbose `llama-server` logging (much more detail per request/load). Noisy — use only while diagnosing. |
| `log_disable` | bool | Disables `llama-server` logging entirely (silences the console). |
| `log_colors` | str enum: `on` / `off` / `auto` | ANSI color in log output. `auto` = color only on a TTY. Set `off` when logs go to a file/collector so escape codes don't pollute them. |
| `log_prefix` | bool | Adds a prefix (log-level / source tag) to each log line. Useful for parsing/grepping logs. |
| `log_timestamps` | bool | Prepends a timestamp to each log line. Turn on when correlating server logs with request timelines. |
| `log_verbosity` | int, bound `0`–`5` | Numeric log-level threshold; messages with a higher verbosity are ignored. `0` = generic output, `1` = error, `2` = warning, `3` = info, `4` = trace, `5` = debug (engine `LOG_LEVEL_DEBUG`). Finer-grained than the boolean `verbose`. |

**Relationship to the doctrine `no_perf` flag.** Note that per-request performance timings are governed by the separate `no_perf` doctrine flag (Section 5, "The five doctrine flags"), not by these debug toggles — the doctrine baseline is `no_perf: true`, which suppresses that specific per-request perf spam while leaving normal logging alone. If you're perf-debugging a model, flip `no_perf: false` (and cold-spawn) rather than cranking `verbose`/`log_verbosity`, which mostly add load/spawn noise, not per-request timings.

**Practical guidance.** Leave the whole section unset for normal generative serving. Enable a server toggle only to expose the specific endpoint you need (`embeddings`/`reranking`/`pooling` for the right model class; `metrics`/`slots`/`props` for observability). Reach for the debug switches only during a spawn investigation, and remember every one of them needs a cold-spawn to bind — patch the manifest, trigger teardown (doctrine Option A `keep_alive:0` / Option B natural idle-hot teardown / Option C `docker restart`), then confirm via `/proc/<pid>/cmdline`.

---

## Section 11 — Denied Flags & Why (Security Appendix)

`llama_server_flags` is a **closed allowlist**, but the allowlist is only half the defense. Before a key is even checked for membership, it runs a **deny gauntlet** in `manifest.py` (`ModelManifest._flags_allowlist` → per-key order): (1) explicit `DENIED_FLAGS` set → reject; (2) suffix-pattern forward-defense (`_suffix_guard_check`) → reject; (3) allowlist membership → reject if absent; (4) value/type/enum/bounds. A flag that clears (1)+(2) but is not in the allowlist still dies at (3). This section explains **why** whole classes of `llama-server` flags are forbidden, and gives the do-not-try list so you don't waste a `PUT` cycle discovering a `400` validation error the hard way.

The governing principle: **the manifest describes a model, not the machine.** Anything that would let a YAML file name a filesystem path, a credential, a network endpoint, or a code-execution primitive is the *manager's* concern (boot config, host paths, env-held tokens) — never a per-model manifest. A manifest is attacker-reachable via `PUT /api/manifests/{tag}`; the boot config is not. So the trust boundary is drawn exactly at "can this value point at something outside the model."

### 11.1 The explicit `DENIED_FLAGS` set (51 keys) — by class

Every one of these is hard-rejected with a `ManifestValidationError` (`llama_server_flags.<key> is explicitly denied ...`). Grouped by *why*:

| Class | Flags | Why forbidden |
|---|---|---|
| **Direct RCE** | `tools` | `tools` enables llama-server's server-side tool interface (`exec_shell_command` / `write_file` / `edit_file`). A manifest that could set it is a remote-code-execution primitive. Never negotiable. |
| **Arbitrary file READ (path-bearing)** | `path`, `media_path`, `models_dir`, `models_preset`, `model`, `alias`, `model_draft`, `model_vocoder`, `webui_config_file`, `grammar_file`, `json_schema_file`, `chat_template_file`, `in_prefix_file`, `in_suffix_file`, `cache_prompt_file`, `slot_save_path`, `log_file`, `api_key_file`, `ssl_key_file`, `ssl_cert_file`, `lookup_cache_static`, `lookup_cache_dynamic`, `control_vector`, `control_vector_scaled`, `binary_override` | Each takes a filesystem path. `path`/`media_path` are CRITICAL — they set llama-server's static-file serving root, so an attacker could point it at `/etc` and exfiltrate host secrets over HTTP. `ssl_*_file` leaks PEM private keys. `model`/`models_dir`/`binary_override` let a manifest load an arbitrary GGUF or swap the binary. The manager owns model paths (blob store + manifest resolution); a manifest may only *content-address* a blob via `gguf_blob_sha256`, never name a path. |
| **SSRF + remote fetch** | `model_url`, `model_url_draft`, `hf_repo`, `hf_repo_draft`, `hf_file`, `hf_repo_v`, `hf_file_v`, `docker_repo`, `webui_mcp_proxy` | These make llama-server fetch over the network at spawn — attacker-controlled URL = SSRF (probe internal services) plus arbitrary-download RCE (pull a poisoned GGUF). `webui_mcp_proxy` enables an experimental MCP CORS proxy (see the engine's `tools/server/README.md`) — a CORS-bypass / SSRF pivot. Model provenance is the manager's job (the `pull` subsystem: `POST /api/pull-hf` is restricted to a HuggingFace host allowlist, and `POST /api/pull-url` accepts only `https://` URLs and blocks private and denied addresses), not a per-model flag. |
| **Credential injection** | `api_key`, `hf_token` | A manifest that sets an API key or HF token injects/rotates a credential the operator didn't authorize (and would log it in plaintext YAML). Tokens live in env vars named by the boot config (`pull.hf_api_key_env`), never inline. |
| **Network bind / topology** | `host`, `port`, `rpc` | These control *where the process listens* and RPC backend wiring — the manager assigns ports (`default_port_base`) and binds; a manifest that could rebind is a takeover/pivot primitive. |
| **KV / weight override** | `override_kv` | Rewrites GGUF metadata at load (rope config, arch fields, etc.) — a smuggle-path to change model behavior or trigger loader bugs. |
| **LoRA / adapter injection** | `lora`, `lora_base`, `lora_scaled`, `mmproj` | Load arbitrary adapter/projector weights from a path — same arbitrary-file-load class as `model`, plus behavior-modification. |
| **Deferred — parser needed, not a hard "never"** | `grammar` (inline BNF), `fit_target` (CSV `MiB,MiB`), `samplers` (semicolon list), `dry_sequence_breaker` (str list), `chat_template_kwargs` (JSON string), `reasoning_budget_message` (mid-stream str) | These are *value-only* and not inherently path/RCE-bearing, but each carries a string with shell-meta / injection / recursive-scalar risk that the current validator can't safely parse. Denied until a dedicated pre-validator lands. If you need one, it's a code change + review — do not try to slip it into a manifest. `tensor_split` was in this class and has since graduated: it now has a strict CSV validator plus a spawn-time device-count gate, and is allowlisted. |

### 11.2 The suffix forward-defense — why it exists

Even if a *new* dangerous flag ships in a future engine update and nobody remembers to add it to `DENIED_FLAGS`, `_suffix_guard_check` rejects it by **name shape** before it can ever be enumerated. Any key matching one of these patterns is rejected (unless whitelisted in `_SUFFIX_GUARD_EXCEPTIONS`, which is currently empty):

| Pattern | Catches | Rationale |
|---|---|---|
| `.*_file$` | `*_file` | any path-read flag |
| `.*_path$` | `*_path` | any path flag |
| `.*_dir$` | `*_dir` | any directory flag |
| `.*_url$` | `*_url` | any SSRF/remote-fetch flag |
| `.*_repo$` | `*_repo` | HF/docker download flag |
| `.*_key$` | `*_key` | credential / PEM flag |
| `^hf_` | `hf_*` | HuggingFace fetch family |
| `^lora` | `lora*` | adapter-load family |
| `^control_vector` | `control_vector*` | steering-vector path family |
| `^lookup_cache_` | `lookup_cache_*` | arbitrary read/write cache files |
| `^ssl_` | `ssl_*` | TLS material |
| `^api_key` | `api_key*` | credentials |
| `^slot_save_` | `slot_save_*` | KV-dump-to-path family |
| `^webui_` | `webui_*` | web-UI config / proxy family |
| `^docker_` | `docker_*` | container-fetch family |
| `.*_model$` | `*_model` | arbitrary-GGUF-load flag (closes a real gap: `spec_draft_model` matched **none** of the other 15 patterns and is not in `DENIED_FLAGS` by exact name either, despite being the same arbitrary-local-path-read class as the already-denied `model`/`model_draft`. No key in the current 108-entry `SAFE_LLAMA_FLAGS` ends in `_model`, so this pattern rejects nothing that works today.) |

This is a **belt-and-suspenders** measure: the deny list is the known-bad list; the suffix guard is the "shape of bad" list. A flag has to survive **both** to reach the allowlist check. This is why you cannot introduce, say, a hypothetical `preset_dir` or `steering_url` even though neither is spelled out in `DENIED_FLAGS` — the suffix guard eats it.

### 11.3 The do-not-try list (fast reference)

If you find yourself wanting any of the below in a manifest, stop — it will be rejected, and the capability lives elsewhere:

- **A file path of any kind** (model, LoRA, grammar, chat template, log, SSL, KV save, static dir) → **denied.** Model blobs are content-addressed by `gguf_blob_sha256`; custom Jinja templates cannot be set from a manifest (`chat_template_file` is denied); logs/KV-dumps are the manager's.
- **A URL or `hf_*` / `docker_*` fetch** → **denied (SSRF/RCE).** Use the manager's `pull` subsystem (`POST /api/pull-hf` for HuggingFace hosts on an allowlist, `POST /api/pull-url` for HTTPS-only URLs behind an SSRF guard).
- **An API key, HF token, or SSL cert/key** → **denied.** Credentials are env-var-named in boot config, never inline.
- **`host` / `port` / `rpc`** → **denied.** The manager assigns and binds; see boot config `server.*` + `default_port_base`.
- **`tools`** → **denied (direct RCE).** No exception.
- **`override_kv`** → **denied.** GGUF metadata is fixed at build time.
- **`fit_target` / `samplers` / `grammar` (inline) / `dry_sequence_breaker` / `chat_template_kwargs`** → **denied *for now*** (unsafe string parse). These are on the roadmap behind dedicated validators; don't hand-edit them in. (`tensor_split` is the one that has since landed — it now has a CSV validator plus a spawn-time device-count gate, and is allowlisted; see Appendix A.4.)
- **`chat_template` containing `{%` or `{{`** → **rejected** by `_check_jinja_injection` (SSTI guard). Use a built-in template *name* from `SAFE_CHAT_TEMPLATE_NAMES` (45 options) or a short `^[A-Za-z0-9_.\-]+$` token ≤256 chars. Inline Jinja bodies are rejected by design.
- **Any brand-new flag whose name ends in `_file/_path/_dir/_url/_repo/_key/_model` or starts with `hf_/lora/ssl_/api_key/control_vector/lookup_cache_/slot_save_/webui_/docker_`** → **rejected by the suffix guard** before allowlist lookup, even if it looks harmless.
- **A standalone draft-model reference for `spec_type: draft-dflash`/`draft-dspark`** (`spec_draft_model`, `spec_draft_hf_repo`, or any raw path/repo form) → **denied.** Use `spec_draft_gguf_blob_sha256` (Section 1, Section 11.4) — the same content-address pattern as `mmproj_blob_sha256`, never a path or URL the manifest names directly.

Adding *any* flag to the allowlist is a **code change** that includes a security assessment of the new flag, never a YAML edit — that's the whole point of the closed allowlist.

### 11.4 Standalone draft models (D-Flash / D-Spark) and the allowlist

The `spec_type` allowlist (Section 7.1) has three values, and the manifest has one top-level field for a standalone draft model, `spec_draft_gguf_blob_sha256` (Section 1). Both are allowlist/schema surfaces, and both are safe for the reasons below:

**What the enum widening alone can and cannot do.** `SAFE_LLAMA_FLAG_STRING_ENUMS["spec_type"]` being `{"draft-mtp", "draft-dflash", "draft-dspark"}` rather than `{"draft-mtp"}` alone adds **zero new capability by itself** — it only changes which of the engine's `--spec-type` names pass a `value in {closed set}` check before being forwarded verbatim as `--spec-type <value>`. The value itself is not path-bearing, not a credential, not a network endpoint — it selects a compile-time-fixed dispatch branch inside `llama-server`, the same shape as `rope_scaling: yarn` or `pooling: mean` elsewhere in this allowlist. The governing principle from the top of this section ("does this value point at something outside the model") is satisfied trivially: an enum string cannot point anywhere.

**What `spec_draft_gguf_blob_sha256` can express: exactly one thing.** A 64-character lowercase-hex string (or empty), validated by `fullmatch(r"[0-9a-f]{64}", v)` — the identical validator shape as `gguf_blob_sha256`/`mmproj_blob_sha256`. It cannot be a path (no `/`, no `.`, no path separators pass the hex-only regex), cannot be a URL (no scheme, no host, no query string survives hex-only), cannot be a credential (fixed-length hex carries no key material), and cannot reach outside the blob store (the manager, not the manifest, decides the resolution: `blob_store_path / "sha256" / hash[:2] / hash` — a hard-coded path template the manifest never influences). This is the same "manifest content-addresses, manager resolves" split the doc's governing principle already states for `gguf_blob_sha256`.

**What happens on an unknown or malformed hash.** Malformed (wrong length, non-hex char, mixed case) → rejected at `Manifest(**data)` construction time, before the manifest is ever written — `ManifestValidationError`, same as every other sha256 field here. A *well-formed* hash with no matching blob in the store → not a manifest-validation failure; the manager resolves the path unconditionally (it does not check blob existence at manifest-write time, matching `gguf_blob_sha256`'s and `mmproj_blob_sha256`'s existing behavior) and the resulting `--spec-draft-model <path-to-nothing>` fails at the engine's own load of the draft model, at spawn time ("failed to load draft model") — not a manager crash, not a silently-wrong config, not a security boundary crossed. This is the same class of failure as pointing `gguf_blob_sha256` at a hash nothing ever uploaded.

**Why `spec_draft_model`/`spec_draft_hf_repo` are not allowlisted.** They are what the vendored engine's own CLI/env surface calls this parameter, and both are excluded on the same principle already governing `model_draft`/`hf_repo_draft` (Section 11.1): a manifest may content-address a blob, never name a path or a fetch target. `spec_draft_hf_repo` is caught by the `.*_repo$` suffix pattern (SSRF class), and `spec_draft_model` by the `.*_model$` entry in `_SUFFIX_GUARD_PATTERNS` (11.2); no key in the full 108-entry `SAFE_LLAMA_FLAGS` ends in `_model`, so that pattern rejects nothing that works today.

**Remaining limitation:** a standalone draft model's own VRAM/KV footprint is not yet accounted for by the spawn-time sizing gate (Section 7.3) — that's a correctness/capacity-planning gap, not a security one (an under-sized reservation can OOM the process; it cannot read a file or reach the network it wasn't already permitted to).

### 11.5 Gating draft-model injection on spec_type

The content-hash design in 11.4 holds: it cannot name a path, a URL, or a credential, and cannot escape the blob store. One **availability** hazard is handled separately, as described here:

**The rule.** Both `--spec-draft-model` injection sites in `manager.py` add the argument only when `spec_type` is one that loads a standalone draft model, not merely when `spec_draft_gguf_blob_sha256` is set. The vendored engine's own `has_dft()` (`common.h`) keys purely on the draft path (or repo) being non-empty and never consults `types` either, so without that condition a manifest that set `spec_draft_gguf_blob_sha256` and **omitted `spec_type` entirely** would validate, spawn, and load a full second model into VRAM — drafting nothing, budgeted at **zero** by the spawn-time sizing gate (`safety.py` has no `spec_draft`-keyed admission term at all), at the target's full context size. Availability only — no confidentiality or integrity angle — but an authorable bad config, exactly the shape the single-select design exists to prevent, one layer over from where the enum closes it.

**The condition.** It is a separate, independently-named constant, `manifest.SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL = frozenset({"draft-dflash", "draft-dspark"})` — deliberately **not** the same set as `SPEC_TYPES_NEEDING_RS_SEQ` (Section 7.3), even though the two overlap on two of three members. They answer different questions: `SPEC_TYPES_NEEDING_RS_SEQ` is "does this type need reserved recurrent-state sequences" (true for `draft-mtp` too); `SPEC_TYPES_WITH_STANDALONE_DRAFT_MODEL` is "does this type load a second, separate GGUF" (false for `draft-mtp`, whose head is bundled in the target's own GGUF). Gating the injection on the 3-member set instead would leave a narrower hole open: `spec_type: draft-mtp` plus a hash set anyway (nonsensical, but not rejected by any validator) would still trigger the same unbudgeted-second-model hazard.

**Not a validation-time reject.** Deliberately consistent with the precedent already set above for the *other* direction (spec_type set, hash omitted, fails at the engine, not at manifest validation): a hash set with a non-qualifying `spec_type` is **not** a `ManifestValidationError` either. It's ignored at spawn-argv construction, and **logged** (`log.warning`, names both the hash and the spec_type) rather than silently dropped — an authored-but-inert field should be loud, not a mystery to whoever set it.

---

## Section 12 — Worked Per-Model Configs (Annotated Example Manifests)

These are annotated example manifests; the model tags are illustrative. Each is a complete serving config, and the annotations explain *why* each setting is what it is. Every top-level field and flag is validated by `manifest.py` before it can be written. Note the common doctrine baseline shared by nearly all of them (the five TurboQuant flags + `jinja: true` + `reasoning_budget`) — the per-config *deltas* are what carry the engineering intent.

### 12.1 Dense 27B, KV-in-VRAM, single card — `my-model-27b-ivram`

The simplest realistic serving shape: a dense ~27B model with a small context and compressed KV entirely in VRAM. This is the "fast VRAM variant" — faster because there's no host-RAM KV round-trip.

```yaml
model_tag: my-model-27b-ivram
gguf_blob_sha256: <sha256 of your GGUF blob>
gguf_size_bytes: 16810713312
context_size: 16384              # manifest-level model context (distinct from ctx_size flag)
expected_vram_bytes: 21500000000 # ~21.5 GB — the VRAM-fit gate reads this
revision: 1
llama_server_flags:
  ctx_size: 16384                # 16K window. int, bound 1..2_000_000. spawn-argv.
  n_gpu_layers: 999              # "all layers on GPU" (999 ≥ any real layer count). bound -1..999.
  cache_type_k: turbo3           # TurboQuant K-cache quant, ~0.1875× f16 KV size. enum. spawn-argv.
  cache_type_v: turbo3           # matching V-cache; K and V independently quantized.
  n_predict: -1                  # no server-side gen cap; client sets max_tokens. bound -1..1_000_000.
  reasoning: auto                # thinking-block handling auto-detected. enum on/off/auto.
  flash_attn: true               # FP4 MMQ on Blackwell; argv → "--flash-attn on". doctrine flag.
  no_context_shift: true         # avoid shift_context loop bug. doctrine flag.
  cache_reuse: 256               # prefix-cache reuse across warm requests. doctrine flag.
  slot_prompt_similarity: 0.5    # 50% prefix-similarity threshold for cache hit. float 0..1. doctrine.
  no_perf: false                 # per-request perf timings LEFT ON here (this is an eval variant).
  jinja: true                    # LOAD-BEARING: tool_calls + <think> preservation need Jinja branch.
  reasoning_budget: 8192         # thinking-token cap, ~half of a 16K max_tokens. int, spawn-argv.
```

Why these choices: `turbo3`/`turbo3` KV at 16K fits comfortably in a 24 GB card with weights, so no `no_kv_offload` is needed — everything stays in VRAM for speed. `no_perf: false` is the tell that this is a benchmark/eval variant (serving variants typically set it `true` to cut log noise). No `spec_type` → MTP speculative decode is off in this "no-MTP" sibling.

### 12.2 Dense 27B, MoE-less, RAM-KV for 250K context, GPU-pinned, layer-split — `my-model-27b`

This is the flagship long-context dense config: **250K context** achieved by pushing the KV cache to host RAM (`no_kv_offload`), weights staying on GPU, plus MTP speculative decode and dual-card layer split.

```yaml
model_tag: my-model-27b
context_size: 250000
expected_vram_bytes: 24000000000    # full 24 GB budget claimed
revision: 25                         # manifest revision (also the ETag value)
llama_server_flags:
  parallel: 1                        # single slot — no concurrency here. bound 1..256.
  ctx_size: 250000                   # 250K window; K and V are quantised to turbo3 (below) to keep the KV cache small.
  n_gpu_layers: 999                  # all weights on GPU.
  split_mode: layer                  # split model layers across BOTH cards. enum none/layer/row/tensor.
  main_gpu: 0                        # card 0 is primary for the split. int, bound 0..16.
  cache_type_k: turbo3
  cache_type_v: turbo3               # turbo3 = ~0.1875x the size of an f16 KV cache.
  n_predict: -1
  reasoning: auto
  reasoning_budget: 9192             # roughly half of a ~18K max_tokens budget (half-of-max_tokens convention).
  jinja: true
  flash_attn: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true                      # perf logging OFF.
  spec_type: draft-mtp               # MTP speculative decode. enum draft-mtp/draft-dflash/draft-dspark. spawn-argv.
  spec_draft_n_max: 2                # draft up to 2 tokens/step. bound 0..64.
  reasoning_format: auto             # thinking-block parse format auto. enum none/deepseek/deepseek-legacy/auto.
```

Why: this config sets a 250K context window with the weights fully on GPU (`n_gpu_layers: 999`) and a compressed KV cache (`turbo3`, ~0.1875× the size of f16). `split_mode: layer` + `main_gpu: 0` spreads the model layers across every visible GPU so the model itself fits with headroom. To move the whole KV cache to host RAM instead, add `no_kv_offload: true` and `cache_ram` (see A.2); this manifest does not set them. `spec_type: draft-mtp` with `spec_draft_n_max: 2` uses the GGUF's bundled next-token-prediction head to draft up to 2 tokens per step (this needs a GGUF that ships an MTP head).

> Note: this manifest does not set `no_kv_offload` or `cache_ram`; it relies on the layer split across the cards rather than on host-RAM KV. For the flags that put the KV cache in host RAM, see A.2 (`no_kv_offload`, `cache_ram`).

### 12.3 35B MoE, `parallel: 2` concurrent slots, unified KV — `my-moe-35b`

The concurrency example: **one** 35B MoE resident, serving **two** same-model requests at once, with a **500K** context pool shared by the two slots (with `kv_unified: true`, each slot can use up to the full 500K).

```yaml
model_tag: my-moe-35b
context_size: 500000
expected_vram_bytes: 24000000000
revision: 19
llama_server_flags:
  parallel: 2                    # 2 concurrent slots for THIS one model. Triggers the cross-field guard.
  cont_batching: true            # continuous batching so the 2 slots interleave decode steps (the engine default is already on; set explicitly here).
  kv_unified: true               # REQUIRED when parallel>1 — one unified KV pool (guard rejects without it).
  ctx_size: 500000               # one pool shared by both slots; kv_unified: true below satisfies the guard.
  n_gpu_layers: 999
  split_mode: layer              # layers across both cards.
  main_gpu: 0
  cache_type_k: turbo3
  cache_type_v: turbo3
  n_predict: -1
  reasoning: auto
  reasoning_format: deepseek-legacy
  flash_attn: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true
  jinja: true
  reasoning_budget: 8192
  spec_type: draft-mtp
  spec_draft_n_max: 3            # up to 3 draft tokens/step (the dense example in 12.2 uses 2).
```

Why: this config exercises the **`_validate_parallel_ctx` cross-field rule**. `parallel: 2` mandates `kv_unified: true` (a unified pool keeps KV accounting exact and flat across slots), and it is set. The divisibility and per-slot-floor checks do not run for a unified pool, so nothing else applies. `cont_batching: true` makes the two slots' decode steps interleave (it is also the engine default). `spec_draft_n_max: 3` is simply a deeper draft than the dense example. `reasoning_format: deepseek-legacy` keeps the `<think>` tags in `message.content` while also populating `message.reasoning_content`.

### 12.4 35B MoE, CPU-MoE overflow — `my-moe-35b-cpumoe`

When even RAM-KV isn't enough headroom, the MoE *expert layers themselves* are pushed to CPU. This is the "cpu-moe overflow" shape: keep attention + shared layers on GPU, run the sparse expert FFNs on CPU.

```yaml
model_tag: my-moe-35b-cpumoe
context_size: 32768
expected_vram_bytes: 11000000000   # only ~11 GB VRAM — the point of cpu-moe
revision: 1
llama_server_flags:
  ctx_size: 32768
  n_gpu_layers: 999
  cache_type_k: turbo3
  cache_type_v: turbo3
  n_predict: -1
  reasoning: auto
  reasoning_format: deepseek-legacy
  flash_attn: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: false
  jinja: true
  reasoning_budget: 8192
  cpu_moe: true                    # ALL MoE expert layers on CPU (-cmoe). Drops VRAM hard.
  no_mmap: true                    # force full weight load into RAM (no lazy mmap paging).
  mlock: true                      # pin those pages in RAM so the OS can't swap them out.
  threads: 16                      # CPU threads for the now-CPU-bound expert compute. bound -1..256.
  spec_type: draft-mtp
  spec_draft_n_max: 3
```

Why: `cpu_moe: true` (emitted as `--cpu-moe`, short form `-cmoe`) moves the entire MoE expert stack off the GPU, which is why `expected_vram_bytes` here is only ~11 GB. Once experts run on CPU, three companion flags matter: `no_mmap: true` loads all weights eagerly instead of memory-mapping them (slower load; may reduce page-outs), `mlock: true` asks the OS to keep them in RAM rather than swap them, and `threads: 16` gives the CPU expert compute more parallelism. The `-ncmoe N` variant (`n_cpu_moe`, used in the GPU1 example in 12.5 below) is the *partial* form — offload only N expert layers to CPU instead of all — used when you want to trade a little VRAM for speed rather than dump the whole expert stack.

### 12.5 GPU-pinned co-residence pair — `my-model-27b-gpu0` + `my-moe-35b-gpu1`

The co-residence shape runs **two different models simultaneously**, each pinned to its own physical card (`split_mode: none` + a fixed `main_gpu`), so neither model's layers cross the PCIe bus. This is a deliberate departure from the single-model-resident norm.

**GPU0-pinned dense 27B:**
```yaml
model_tag: my-model-27b-gpu0
context_size: 250000
expected_vram_bytes: 22500000000   # ~22.5 GB claimed on GPU0
revision: 3
llama_server_flags:
  ctx_size: 250000
  n_gpu_layers: 999
  split_mode: none                 # DO NOT split — keep this model entirely on one card.
  main_gpu: 0                      # pin to card 0.
  sleep_idle_seconds: -1           # never idle-sleep — hold residency (co-residence needs it pinned). bound -1..86400.
  no_mmap: true
  cache_type_k: turbo2             # turbo2 = ~0.125× f16 KV, the smallest — squeezes 250K KV into VRAM.
  cache_type_v: turbo2
  flash_attn: true
  jinja: true
```

**GPU1-pinned 35B MoE (with partial CPU-MoE + parallel:2):**
```yaml
model_tag: my-moe-35b-gpu1
context_size: 250000                # fallback only; llama_server_flags.ctx_size (500000 below) takes precedence
expected_vram_bytes: 20500000000   # ~20.5 GB claimed on GPU1
revision: 3
llama_server_flags:
  parallel: 2
  cont_batching: true
  kv_unified: true                 # parallel:2 → unified KV required (guard).
  ctx_size: 500000                 # one pool shared by both slots.
  n_gpu_layers: 999
  split_mode: none                 # pinned to one card, NOT split.
  main_gpu: 1                      # card 1.
  sleep_idle_seconds: -1
  n_cpu_moe: 10                    # offload 10 expert layers to CPU to make room on GPU1. bound 0..256.
  no_mmap: true
  cache_type_k: turbo2
  cache_type_v: turbo2
  flash_attn: true
  jinja: true
```

Why the pair works: `split_mode: none` + distinct `main_gpu` (0 vs 1) pins each model to one card, so they don't contend for the same VRAM or cross PCIe (unless the placement logic overrides `main_gpu`/`split_mode`, e.g. with `auto_place`; see `docs/PLACEMENT_AND_CORESIDENCE.md`). `sleep_idle_seconds: -1` disables the idle-teardown that would otherwise free the slot — for a co-residence pair you *want* both pinned and hot. Both use `turbo2` KV (the smallest cache type, ~0.125× f16) to keep the KV cache within the per-card budget (a 250K window on GPU0, a 500K shared pool on GPU1). The 35B on GPU1 additionally uses `n_cpu_moe: 10` (partial CPU offload of 10 expert layers) to free VRAM on its card, and runs `parallel: 2` for its own two-slot concurrency. Note these two are minimal flag sets (no MTP, no `no_context_shift`/`cache_reuse` doctrine tail).

### 12.6 Thinking model with explicit reasoning budget — `my-moe-35b-reasoning` (turbo2 headroom variant)

The reasoning-heavy variant shows the interplay of `reasoning`, `reasoning_format`, `reasoning_budget`, and mixed K/V cache quants for VRAM headroom.

```yaml
model_tag: my-moe-35b-reasoning
context_size: 250000
expected_vram_bytes: 29000000000   # heavy — 250K + 35B MoE
revision: 11
llama_server_flags:
  parallel: 1
  cont_batching: true
  kv_unified: true
  ctx_size: 250000
  n_gpu_layers: 999
  split_mode: layer
  main_gpu: 0
  cache_type_k: f16                # K left at full f16 precision...
  cache_type_v: turbo2             # ...but V compressed to turbo2. Asymmetric K/V to trade quality vs VRAM.
  n_predict: -1
  reasoning: auto
  reasoning_format: deepseek-legacy
  flash_attn: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true
  jinja: true
  reasoning_budget: 10192          # large budget — roughly half of a ~20K max_tokens.
  spec_type: draft-mtp
  spec_draft_n_max: 3
```

Why: this is a **VRAM-headroom** tuning. The standout is the **asymmetric KV quant**: `cache_type_k: f16` keeps the key cache at full precision while `cache_type_v: turbo2` aggressively compresses the value cache — a quality/VRAM trade you can make per-side because K and V are independent flags. `reasoning_budget: 10192` is a large preserved-thinking budget, following the half-of-`max_tokens` convention for a ~20K generation cap. `reasoning: auto` lets the engine detect thinking blocks; `reasoning_format: deepseek-legacy` keeps the `<think>` tags in `message.content` while also populating `message.reasoning_content`. Unlike the `my-moe-35b` concurrency config, this one is `parallel: 1` — a single deep-reasoning slot rather than two shallow ones.

### 12.7 What to copy from these

- **Carry `jinja: true` for tool-call work** — `llama-server` only honours the chat-template `tools` placeholder when started with `--jinja` (see `docs/TOOL_CALL_HANDLING.md`).
- **The five doctrine flags** (`flash_attn`, `no_context_shift`, `cache_reuse: 256`, `slot_prompt_similarity: 0.5`, `no_perf`) are the shared baseline — flip `no_perf: false` only for eval/debug variants.
- **KV placement is the big lever:** KV-in-VRAM (`turbo2`/`turbo3`, no offload) = fast, small context; `no_kv_offload: true` + `cache_ram: N` = long context bounded by host RAM instead of VRAM.
- **`parallel > 1` is a package:** it needs `kv_unified: true` (a manifest without it is rejected at load); the divisibility and per-slot-floor rules do not apply to a unified pool. `cont_batching: true` is the usual companion.
- **MoE VRAM ladder:** all-GPU → `n_cpu_moe: N` (partial) → `cpu_moe: true` (all experts on CPU, + `no_mmap`+`mlock`+`threads`).

---

## Appendix A — Full flag index

Every key in `SAFE_LLAMA_FLAGS` (108 entries), with category, accepted type, numeric bound or string enum from `manifest.py`, whether it lands as **spawn-argv** (any value set in `llama_server_flags` is baked into the `llama-server` command line by `flags_to_argv`, so it needs a **cold-spawn** to take effect) or is also usable **request-body-hot** (sampling/reasoning knobs the forwarder can apply per-call), the code default where one exists (`MANIFEST_FLAG_DEFAULTS` in `manifest.py`) or otherwise a representative example value, and a one-line purpose.

**Spawn-argv vs request-body:** *Everything in a manifest is spawn-argv* — a `PUT` to `llama_server_flags` only affects the **next cold-spawn** (see the doctrine doc's Option A/B/C to force one). The "request-body-hot" tag marks flags that *also* correspond to per-request body fields (`temperature`, `top_p`, `top_k`, `reasoning_budget`, `n_predict`/`max_tokens`, `seed`, penalties, etc.) which the forwarder applies live without a respawn. Booleans encode as `--flag` when `true`, **omitted** when `false` (except `flash_attn`: a bool emits `--flash-attn on|off`, and a string value is passed through lower-cased).

**Encoding notes:** `flash_attn` bool→`--flash-attn on/off`; all other bools true→`--flag`, false→omitted; ints/floats/strings→`--flag <value>`. `bool↔int` coercion is rejected; `int→float` promotion allowed.

### A.1 Performance + memory layout

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `ctx_size` | int | 1..2,000,000 | spawn-argv | 250000 / 16384 | Context window (KV size driver). |
| `n_gpu_layers` | int \| str | -1..999, or `all`/`auto` | spawn-argv | 999 | Layers on GPU (999 = all). |
| `threads` | int | -1..256 | spawn-argv | 16 (cpu-moe) | CPU threads for generation. |
| `threads_batch` | int | -1..256 | spawn-argv | — | CPU threads for prompt batch. |
| `threads_http` | int | -1..256 | spawn-argv | — | HTTP server worker threads. |
| `parallel` | int | 1..256 | spawn-argv | 1 / 2 (no default) | Concurrent slots for one model (>1 triggers cross-field guard). |
| `batch_size` | int | 1..65536 | spawn-argv | — | Logical prompt batch (`-b`). |
| `ubatch_size` | int | 1..65536 | spawn-argv | — | Physical micro-batch (`-ub`). |
| `n_predict` | int | -1..1,000,000 | argv + body | -1 | Server-side gen cap (-1 = client-driven). |
| `keep` | int | -1..65536 | spawn-argv | — | Tokens kept when context truncates. |
| `flash_attn` | bool \| str | `on/off/auto/enabled/disabled` | spawn-argv | true | FlashAttention (FP4 MMQ on Blackwell). A bool always emits explicit `on`/`off`; a string is passed through lower-cased. |
| `mlock` | bool | — | spawn-argv | true (cpu-moe) | Pin weights in RAM (no swap). |
| `no_mmap` | bool | — | spawn-argv | true (cpu-moe/pinned) | Disable mmap; eager full load. |
| `numa` | str | `none/distribute/isolate/numactl` | spawn-argv | — | NUMA memory policy. |
| `swa_full` | bool | — | spawn-argv | — | Force full-size sliding-window-attention KV (no SWA shrink). |
| `no_perf` | bool | — | spawn-argv | true / false (eval example) | Suppress per-request perf logging. Doctrine flag. |
| `sleep_idle_seconds` | int | -1..86400 | spawn-argv | -1 (co-resident) | Idle-teardown timer; -1 = never sleep. |
| `cache_reuse` | int | 0..65536 | spawn-argv | 256 | Prefix-cache reuse across warm requests. Doctrine flag. |
| `no_context_shift` | bool | — | spawn-argv | true | Disable the context-shift loop. Doctrine flag. |
| `slot_prompt_similarity` | float | 0.0..1.0 | spawn-argv | 0.5 | Prefix-similarity threshold for slot cache reuse. Doctrine flag. |
| `warmup` | bool | — | spawn-argv | — | Run a warmup pass at load. |
| `check_tensors` | bool | — | spawn-argv | — | Validate tensor data on load. |
| `repack` | bool | — | spawn-argv | — | Repack quantized weights for the target CPU/GPU layout. |
| `op_offload` | bool | — | spawn-argv | — | Offload eligible ops to GPU (llama.cpp `--op-offload`). |
| `no_host` | bool | — | spawn-argv | — | Bypass the host buffer so extra buffers can be used (llama.cpp `--no-host`). |
| `direct_io` | bool | — | spawn-argv | — | Use direct I/O for model file reads. |
| `cont_batching` | bool | — | spawn-argv | true (parallel>1) | Continuous batching so `parallel>1` slots interleave decode steps; the engine default is already on (a `false` value is omitted from argv). |

### A.2 KV-cache

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `cache_type_k` | str | `f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1/turbo2/turbo3/turbo4` | spawn-argv | turbo3 (code default) / turbo2 / f16 | K-cache quant type. turbo2≈0.125×, turbo3≈0.1875×, turbo4≈0.25× f16 size. |
| `cache_type_v` | str | (same enum as K) | spawn-argv | turbo3 (code default) / turbo2 | V-cache quant type. Set independently from K (asymmetric OK). |
| `kv_offload` | bool | — | spawn-argv | — | Explicitly enable KV offload to GPU (llama.cpp default-on). |
| `no_kv_offload` | bool | — | spawn-argv | true (RAM-KV) | Put KV cache in host RAM, not VRAM — enables huge contexts bounded by RAM. |
| `kv_unified` | bool | — | spawn-argv | true (parallel>1) | Single unified KV pool across slots. REQUIRED when `parallel>1`. |
| `cache_idle_slots` | bool | — | spawn-argv | — | Save idle slots to the prompt cache on a new task and clear them when using unified KV (llama.cpp: default enabled; requires `cache_ram`). |
| `cache_prompt` | bool | — | spawn-argv | — | Cache the prompt tokens for reuse (llama.cpp `--cache-prompt`). |
| `cache_ram` | int | 0..262144 (MiB) | spawn-argv | 32768 | Host-RAM KV cache budget in MiB (pairs with `no_kv_offload`). |
| `ctx_checkpoints` | int | 0..1024 | spawn-argv | 2 (code default) | Number of context checkpoints retained for fast restore. |
| `checkpoint_every_n_tokens` | int | 1..1,000,000 | spawn-argv | — | Checkpoint cadence in tokens. The vendored engine defines no `--checkpoint-every-n-tokens` option (its nearest option is `--checkpoint-min-step`), so confirm your engine build accepts this flag before setting it. |

### A.3 Context / RoPE / YaRN

None of the example manifests above set these (models use their native context); they exist for context-extension tuning. All spawn-argv.

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `rope_scaling` | str | `none/linear/yarn` | spawn-argv | — | RoPE scaling method for context extension. |
| `rope_scale` | float | 0.0..1000.0 | spawn-argv | — | Linear RoPE context-scale factor. |
| `rope_freq_base` | float | 0.0..10,000,000.0 | spawn-argv | — | RoPE base frequency (θ) override. |
| `rope_freq_scale` | float | 0.0..100.0 | spawn-argv | — | RoPE frequency scale (inverse of `rope_scale`). |
| `yarn_orig_ctx` | int | 0..2,000,000 | spawn-argv | — | Original training context for YaRN. |
| `yarn_ext_factor` | float | -1.0..100.0 | spawn-argv | — | YaRN extrapolation mix factor (-1 = auto). |
| `yarn_attn_factor` | float | -1.0..100.0 | spawn-argv | — | YaRN attention-magnitude scaling. |
| `yarn_beta_slow` | float | -1.0..100.0 | spawn-argv | — | YaRN low-frequency ramp boundary. |
| `yarn_beta_fast` | float | -1.0..100.0 | spawn-argv | — | YaRN high-frequency ramp boundary. |

### A.4 MoE / multi-GPU

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `cpu_moe` | bool | — | spawn-argv | true (cpu-moe example) | Put ALL MoE expert layers on CPU (`-cmoe`). Big VRAM drop. |
| `n_cpu_moe` | int | 0..256 | spawn-argv | 10 (GPU1 example) | Offload N expert layers to CPU (`-ncmoe`); partial form of `cpu_moe`. |
| `split_mode` | str | `none/layer/row/tensor` | spawn-argv | layer / none | Multi-GPU split strategy. `none` = pin to one card; `layer` = split layers across cards. |
| `main_gpu` | int | 0..16 | spawn-argv | 0 / 1 | Primary GPU for the split / pin target. |
| `tensor_split` | str | CSV `N,N[,…]`, 2–16 elements, ≤64 chars | spawn-argv | — | Per-GPU split ratio (`--tensor-split`). Validated as strict numeric CSV at load; a second gate at spawn refuses when the element count ≠ the visible device count. |
| `fit` | str | `on/off` | spawn-argv | — | Tom's Fork auto-memory-fit toggle. |
| `fit_ctx` | int | 1..2,000,000 | spawn-argv | — | Minimum context size the engine's `--fit` option may reduce `ctx_size` to (llama.cpp `--fit-ctx`; engine default 4096). |

### A.5 Sampling

All are spawn-argv. Those marked `argv + body` are also forwarded per request by the chat-completion forwarder (the request field for `temp` is `temperature`); the rest (`top_n_sigma`, `xtc_*`, `dynatemp_*`, `dry_*`, `adaptive_*`, `ignore_eos`) are not forwarded and can only be set through the manifest. A manifest value is the default the cold-spawn bakes in.

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `temp` | float | 0.0..10.0 | argv + body | — | Sampling temperature. |
| `top_k` | int | 0..10000 | argv + body | — | Top-K sampling cutoff. |
| `top_p` | float | 0.0..1.0 | argv + body | — | Nucleus (top-P) sampling. |
| `min_p` | float | 0.0..1.0 | argv + body | — | Min-P sampling floor. |
| `typical_p` | float | 0.0..1.0 | argv + body | — | Locally-typical sampling (Ollama parity). |
| `top_n_sigma` | float | -1.0..100.0 | spawn-argv | — | Top-nσ logit-std sampling cutoff (-1 = off). |
| `repeat_penalty` | float | 0.0..10.0 | argv + body | — | Repetition penalty. |
| `repeat_last_n` | int | -1..65536 | argv + body | — | Window for repeat penalty (-1 = ctx). |
| `presence_penalty` | float | -10.0..10.0 | argv + body | — | Presence penalty (Ollama parity). |
| `frequency_penalty` | float | -10.0..10.0 | argv + body | — | Frequency penalty (Ollama parity). |
| `seed` | int | -1..2⁶³-1 | argv + body | — | RNG seed (-1 = random). |
| `mirostat` | int | 0..2 | argv + body | — | Mirostat mode (0 off, 1 v1, 2 v2). |
| `mirostat_lr` | float | 0.0..1.0 | argv + body | — | Mirostat learning rate (η). |
| `mirostat_ent` | float | 0.0..100.0 | argv + body | — | Mirostat target entropy (τ). |
| `xtc_probability` | float | 0.0..1.0 | spawn-argv | — | XTC (exclude-top-choices) trigger probability. |
| `xtc_threshold` | float | 0.0..1.0 | spawn-argv | — | XTC logit threshold. |
| `dynatemp_range` | float | 0.0..10.0 | spawn-argv | — | Dynamic-temperature range. |
| `dynatemp_exp` | float | 0.0..10.0 | spawn-argv | — | Dynamic-temperature exponent. |
| `dry_multiplier` | float | 0.0..10.0 | spawn-argv | — | DRY repetition-penalty multiplier. |
| `dry_base` | float | 1.0..10.0 | spawn-argv | — | DRY penalty base. |
| `dry_allowed_length` | int | 0..65536 | spawn-argv | — | DRY max allowed repeat length before penalizing. |
| `dry_penalty_last_n` | int | -1..65536 | spawn-argv | — | DRY lookback window (-1 = ctx). |
| `adaptive_target` | float | -1.0..100.0 | spawn-argv | — | Adaptive-sampling target (maps to Tom's Fork adaptive knob; -1 = off). |
| `adaptive_decay` | float | 0.0..1.0 | spawn-argv | — | Adaptive-sampling decay rate. |
| `ignore_eos` | bool | — | spawn-argv | — | Ignore EOS token (keep generating). |

### A.6 Chat / template

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `chat_template` | str | built-in name (45-set) OR `^[A-Za-z0-9_.\-]+$` ≤256 chars; no `{%`/`{{` | spawn-argv | — | Chat template by name. Jinja bodies are rejected (SSTI guard); `chat_template_file` is itself in `DENIED_FLAGS`, so a manifest cannot supply a custom template body. |
| `jinja` | bool | — | spawn-argv | true | Use Jinja chat-template branch. Needed for tool_calls (`llama-server` honours the template `tools` placeholder only with `--jinja`). |
| `skip_chat_parsing` | bool | — | spawn-argv | — | Bypass server-side chat parsing (raw prompt). |
| `special` | bool | — | spawn-argv | — | Render special/control tokens in output. |
| `spm_infill` | bool | — | spawn-argv | — | SentencePiece infill token ordering (FIM). |

### A.7 Reasoning

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `reasoning_format` | str | `none/deepseek/deepseek-legacy/auto` | spawn-argv | auto / deepseek-legacy | Thinking-block parse format. |
| `reasoning` | str | `on/off/auto` | argv + body | auto | Enable/detect reasoning mode. |
| `reasoning_budget` | int | -1..1,000,000 | argv + body | 8192 / 9192 / 10192 | Preserved-thinking depth (-1 unlimited, 0 off). Common convention ≈ half of max_tokens. |

### A.8 Speculative decoding / MTP

All spawn-argv. `draft-mtp` requires a target GGUF with a bundled next-token-prediction head; `draft-dflash` and `draft-dspark` instead load a standalone draft GGUF referenced by the top-level `spec_draft_gguf_blob_sha256` field.

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `spec_type` | str | `draft-mtp` / `draft-dflash` / `draft-dspark` | spawn-argv | draft-mtp | Speculative-decode family (single-select). |
| `spec_draft_n_max` | int | 0..64 | spawn-argv | 2 (dense) / 3 (MoE) | Max draft tokens per step. |
| `spec_draft_n_min` | int | 0..64 | spawn-argv | — | Min draft tokens per step. |
| `spec_draft_p_min` | float | 0.0..1.0 | spawn-argv | — | Min probability to continue drafting. |
| `spec_draft_p_split` | float | 0.0..1.0 | spawn-argv | — | Draft split-probability threshold. |
| `spec_draft_ngl` | int | -1..999 | spawn-argv | — | Draft-head GPU layers (usually same device). |
| `spec_draft_backend_sampling` | bool | — | spawn-argv | — | Backend-side sampling for the draft path. |
| `spec_draft_type_k` | str | same enum as `cache_type_k` | spawn-argv | — | K-cache quant for the *draft* context, set independently of the target's. |
| `spec_draft_type_v` | str | same enum as `cache_type_v` | spawn-argv | — | V-cache quant for the *draft* context, set independently of the target's. |

### A.9 Server toggles

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `metrics` | bool | — | spawn-argv | — | Enable Prometheus `/metrics` endpoint. |
| `slots` | bool | — | spawn-argv | — | Enable `/slots` introspection endpoint. |
| `props` | bool | — | spawn-argv | — | Enable changing global properties via `POST /props` (llama.cpp `--props`). |
| `embeddings` | bool | — | spawn-argv | — | Enable embeddings endpoint (embedding mode). |
| `reranking` | bool | — | spawn-argv | — | Enable rerank endpoint. |
| `pooling` | str | `none/mean/cls/last/rank` | spawn-argv | — | Embedding pooling strategy. |
| `offline` | bool | — | spawn-argv | — | Offline mode (no network fetches). |

### A.10 Debug

| Flag | Type | Bound / enum | Argv/body | Code default / example | Purpose |
|---|---|---|---|---|---|
| `verbose` | bool | — | spawn-argv | — | Verbose server logging. |
| `log_disable` | bool | — | spawn-argv | — | Disable logging entirely. |
| `log_colors` | str | `on/off/auto` | spawn-argv | — | ANSI color in logs. |
| `log_prefix` | bool | — | spawn-argv | — | Prefix log lines with level/source. |
| `log_timestamps` | bool | — | spawn-argv | — | Timestamp log lines. |
| `log_verbosity` | int | 0..5 | spawn-argv | — | Log verbosity level. |

> **Coverage note:** this table is the complete `SAFE_LLAMA_FLAGS` allowlist (108 keys) as of `src/turbohaul/manifest.py` in this release. 58 keys carry numeric `SAFE_LLAMA_FLAG_BOUNDS`; 13 carry `SAFE_LLAMA_FLAG_STRING_ENUMS`; `flash_attn`, `n_gpu_layers`, `chat_template` and `tensor_split` are special-cased. Any key **not** in this table is rejected at manifest load. For the forbidden classes and the suffix forward-defense, see §11.
