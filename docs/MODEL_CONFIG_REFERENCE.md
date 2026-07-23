# Turbohaul-Manager — Per-Model Configuration Reference

Every setting you can put in a per-model manifest: what it does, when to use it, why, and the exact type/bound/enum the manager enforces. Companion docs: [ARCHITECTURE.md](../ARCHITECTURE.md) (how the system works), [DEPLOYMENT_PATTERNS.md](DEPLOYMENT_PATTERNS.md) (which model for which role), [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) (wiring an agent to Turbohaul), and [TURBOQUANT_FLAGS.md](TURBOQUANT_FLAGS.md) (the production flag doctrine in brief).

Grounded in the manager's own allowlist at commit time; the code (`src/turbohaul/manifest.py`) is the final authority — see the Preface.

---

## Preface — how to read this reference

**`src/turbohaul/manifest.py` is ground truth.** Every accepted flag, its type, its numeric bound, and its enum value set live in that one file — specifically in `SAFE_LLAMA_FLAGS` (the allowlist), `SAFE_LLAMA_FLAG_BOUNDS` (numeric min/max), `SAFE_LLAMA_FLAG_STRING_ENUMS` (string enums), `SAFE_CHAT_TEMPLATE_NAMES`, and `DENIED_FLAGS`. If a flag, bound, or enum value is not in that file, Turbohaul does not accept it — full stop. This document mirrors that file; when the two ever disagree, the file wins and this doc is the bug.

**Adding a flag is a code change, not a YAML edit.** The allowlist is *closed*: an unknown key in a manifest's `llama_server_flags` is rejected at load with `not in the closed allowlist`, never silently forwarded to `llama-server`. Introducing a new flag means editing `SAFE_LLAMA_FLAGS` (plus any bound/enum entry), passing review, and shipping the code. This is deliberate — it is the security boundary against flag-injection RCE. You cannot smuggle a capability in through a manifest.

**The 2-minute mental model.** A manifest is one YAML file per model. It has a handful of top-level identity/sizing fields (Section 1) and one `llama_server_flags` map (Section 2). The map is validated flag-by-flag through a fixed four-stage gauntlet: deny → suffix-guard → allowlist → value-check. Values that survive get encoded into `llama-server`'s argv at spawn time. The single most important operational fact: **most flags only take effect on a cold model spawn.** Editing a manifest for a model that is currently loaded does nothing until that process is torn down and re-forked. Section 2's spawn-argv-vs-request-body table tells you which is which, and how to force the reload.

---

## Section 1 — Manifest anatomy

A manifest is one YAML file per model at `<manifests_path>/<model_tag>.yaml` (production path `/var/lib/turbohaul/manifests/`). It is edited via `PUT /api/manifests/{tag}` (with an `If-Match` ETag) or on disk, and is parsed into the `Manifest` pydantic model, which is `extra="forbid"` — **any unknown top-level key is rejected**, not ignored.

### Field table

| Field | Type | Default | Why it exists | Gotcha |
|---|---|---|---|---|
| `model_tag` | `str` | **required** | Primary key + filename stem. Must match `TAG_RE` = `^[a-z0-9][a-z0-9._-]{0,63}$`. | Lowercase ASCII only, 1–64 chars, must start with `[a-z0-9]`, no `/`, no leading dot/dash, no `..` traversal. Re-validated at *every* path resolution (read/write/delete), not just on create (anti-traversal). An uppercase letter or a slash → `ManifestValidationError`. |
| `display_name` | `str` | `""` | Free-text human label shown in the UI. | Cosmetic; no validation beyond being a string. Typical values are descriptive labels like `'35B MoE parallel:2 RAM-KV MTP'`. |
| `description` | `str` | `""` | Free-text notes (provenance, quant, ticket ref). | Cosmetic only. |
| `gguf_blob_sha256` | `str` | **required** | Content-address of the model blob in the store; ties the manifest to an exact GGUF. | Must `fullmatch` `[0-9a-f]{64}` — **lowercase hex, exactly 64 chars**. A pasted uppercase digest is rejected; lowercase it first. Wrong length or any non-hex char → `ManifestValidationError`. |
| `gguf_size_bytes` | `int ≥ 0` | `0` | Declared blob size (bookkeeping / sanity). | Pydantic `ge=0`; a negative value is a validation error. Not load-bearing for spawn. |
| `context_size` | `int ≥ 1` | `2048` | Manifest-level declared model context length. | **DISTINCT from the `ctx_size` *flag*.** This top-level field is model metadata (`ge=1`, default 2048); `llama_server_flags.ctx_size` is what actually gets passed to `llama-server` as `--ctx-size`. In production they are usually set to the *same* number (e.g. both `250000`), but nothing in the schema forces them to agree — the field and the flag are validated independently. Do not assume setting one sets the other. |
| `expected_vram_bytes` | `int ≥ 0` | `0` | Footprint used by the VRAM-fit pre-check gate before a model is allowed to spawn. | `ge=0`. **A default of `0` effectively disables the VRAM gate for that model** (a zero footprint always "fits"); set a real value for the gate to protect you. Production values are the true device budget, e.g. `22500000000`, `24000000000`. |
| `arch` | `str` | `""` | Declares the model's architecture family so the manager can size a non-standard KV shape correctly. Set `arch: "qwen35"` for a qwen35 hybrid (state-space/SSM + attention); leave it `""` for an ordinary pure-attention model. | The default `""` is byte-identical to prior behaviour (pure attention). Setting `arch: "qwen35"` is now **load-bearing on its own**: it triggers the manager to parse the model's real GGUF attention dims and size the KV cache from them (the dimension-aware path in Section 4), independent of `hybrid_kv_ratio`. |
| `hybrid_kv_ratio` | `float` `0.0`–`1.0` | `1.0` | The fraction of layers that contribute a **growing** per-token KV cache. In a qwen35 hybrid the SSM layers keep a fixed-size recurrent state (not a growing cache), so the model's per-token KV is smaller than a pure-attention model of the same size. **This ratio scales the file-size *fallback* estimate path only** (Section 4). | Bounded `0.0..1.0`. **Default `1.0` = pure attention = byte-identical sizing for every existing model.** For a `qwen35` model whose GGUF dims parse, the dimension-aware path (Section 4) takes over and **ignores `hybrid_kv_ratio`** — it then only matters as the fallback when dims can't be parsed and no `kv_bytes_per_token` override is set. |
| `kv_bytes_per_token` | `float ≥ 1024.0` or unset | *(unset)* | An operator-**measured** effective KV cache cost, in **BYTES per token**, used verbatim by the KV-fit estimate (Section 4) as the highest-precedence override. Derive it from a real measurement, e.g. `measured_marginal_MiB * 1048576 / ctx_size`. | **Units are BYTES/token, NOT KiB** — a measured 13.5 KiB/token is `13824.0`. A **1 KiB/token floor** (`Field ge=1024.0`) rejects a KiB-vs-bytes typo at load. Applied **verbatim**: no quant-scale, no `hybrid_kv_ratio` multiply. Leave **unset** for every existing model (the default) — byte-identical. |
| `revision` | `int ≥ 1` | `1` | The ETag value for optimistic-concurrency writes; server-incremented on every atomic update. | `ge=1`. You do not hand-manage this — `GET` returns it as `ETag: "<revision>"`, you echo it back in `If-Match` on `PUT`, and the server bumps it. Seen climbing in production (e.g. `revision: 21`, `25`) as manifests are re-tuned. A stale `If-Match` → HTTP 412. |
| `llama_server_flags` | `map` | `{}` (empty) | The closed allowlist of `llama-server` spawn flags — see Section 2. | Every key/value is gauntlet-validated. Unknown key → reject. This is where all the real tuning lives. |
| `prompt_template` | `object` | empty `PromptTemplate` | Server-side prompt scaffolding: `{system_default: str, stop_tokens: [str]}`. | Its own `extra="forbid"` model — unknown sub-keys are rejected. `system_default` is a plain system-prompt string (default `""`); `stop_tokens` is a list of stop strings (default `[]`, e.g. `["<|im_end|>", "<|endoftext|>"]`). |

### The `context_size` vs `ctx_size` distinction (the #1 trap)

These are two different things that happen to be spelled almost the same:

- **`context_size`** (top-level Manifest field, default `2048`) — declared model context metadata.
- **`ctx_size`** (a key *inside* `llama_server_flags`, bound `1..2_000_000`) — the value forwarded to `llama-server` as `--ctx-size`, i.e. the KV window the process actually allocates.

Because they are validated by separate code paths, you can set one and forget the other. In every shipped production manifest they are set to the same number by convention (both `16384`, both `250000`, both `500000`, etc.), but that is discipline, not enforcement. When you tune context length, change **both** — and if the model runs `parallel > 1`, remember the cross-field rule in Section 2 keys off the **flag** `ctx_size`, not the top-level field.

### `expected_vram_bytes` and its gate role

`expected_vram_bytes` is the footprint the VRAM-fit pre-check consults before allowing a spawn. Its `ge=0` default of `0` means an un-set value declares a zero-byte footprint, which trivially passes the gate — so the safety check is only as good as the number you put here. Set it to the real device-budget target (production manifests use values like `21500000000`–`29000000000` for 24 GB-class cards); leaving it `0` opts that model out of VRAM protection.

### `revision` / ETag mechanics

`revision` *is* the ETag. The write path (`write_manifest_atomic`) enforces optimistic concurrency:

- **Create** (file does not exist): `If-Match` must be **absent**; the manifest is written as-is.
- **Update** (file exists): `If-Match` is **required**. A *missing* `If-Match` on an existing file raises `ConcurrencyError` (→ HTTP 412) — it does **not** silently overwrite (lost-update guard). A *mismatched* `If-Match` also raises `ConcurrencyError`. On success the server sets `revision = existing.revision + 1`.

So the loop is: `GET` → read `ETag: "<n>"` → edit → `PUT` with `If-Match: "<n>"` → server writes and returns `<n+1>`. Out-of-band disk edits are caught by this same compare-on-write — there is no inotify watcher; concurrency is optimistic, not event-driven.

### `prompt_template` — `{system_default, stop_tokens}`

`prompt_template` is a nested `extra="forbid"` object with exactly two fields:

- `system_default: str` (default `""`) — a default system prompt injected when the request supplies none.
- `stop_tokens: list[str]` (default `[]`) — stop strings, e.g. `["<|im_end|>", "<|endoftext|>"]`.

Most production manifests leave it empty (`system_default: ''`, `stop_tokens: []`) and let the model's chat template handle stops. Any key other than these two inside `prompt_template` is a validation error.

### `arch` and `hybrid_kv_ratio` — hybrid (SSM + attention) model sizing

Two fields describe models whose KV footprint is **not** a plain function of size: hybrids that interleave state-space (SSM) layers with attention layers, such as the `qwen35` hybrid family.

- **`arch`** (`str`, default `""`) — the architecture family. Empty for an ordinary pure-attention model (the default, unchanged). Set `arch: "qwen35"` for a qwen35 hybrid.
- **`hybrid_kv_ratio`** (`float`, default `1.0`, bounded `0.0..1.0`) — the fraction of layers that grow a per-token KV cache. In a qwen35 hybrid the SSM layers hold a **fixed-size recurrent state** rather than a cache that grows with context, so only the attention layers contribute the linear-in-tokens KV term. Setting `hybrid_kv_ratio` to that attention-layer fraction lets the KV-fit estimator scale the per-token KV down, so the VRAM/RAM gate doesn't over-estimate a hybrid's cache (see Section 4 for the exact math).

**The default `1.0` is a no-op.** A ratio of `1.0` means "every layer grows KV" = pure attention = the exact sizing every existing manifest already gets. Lower it only for a real hybrid. The two fields are used as a pair: set `arch` to the hybrid family **and** `hybrid_kv_ratio` to the growing-KV fraction together.

**Nothing new in `llama_server_flags`.** A hybrid model is content-addressed by `gguf_blob_sha256` like any other model, and its weight quant (existing quant types such as `TURBO2_0` and `TQ4_1S` are unchanged) is auto-detected by the engine from the GGUF header field `general.file_type`. There is no new spawn flag to set and nothing added to the allowlist for it. Hybrids also reuse the existing TurboQuant KV-cache types (`turbo2`/`turbo3`/`turbo4` in `cache_type_k`/`cache_type_v`); there is no new KV type. So configuring one is just: point `gguf_blob_sha256` at the blob, set `arch` + `hybrid_kv_ratio`, and carry the usual doctrine flags.

```yaml
# hybrid-model top-level fields (illustrative ratio):
arch: "qwen35"          # SSM + attention hybrid
hybrid_kv_ratio: 0.5    # only ~this fraction of layers grow a per-token KV cache;
                        # SSM layers hold a fixed recurrent state. 1.0 = pure attention (no change).
```

---

## Section 2 — How `llama_server_flags` works

`llama_server_flags` is a **closed allowlist** of ~90 `llama-server` flags. Its security model is a *deny-by-default* gauntlet: a flag must survive four checks, in order, or the whole manifest is rejected. Adding a genuinely new flag is a code change + review (edit `SAFE_LLAMA_FLAGS`), never a YAML edit.

### The validation order (deny → suffix-guard → allowlist → value)

Per flag, `Manifest._flags_allowlist` runs these in exactly this sequence (order matters — the earliest failure wins):

1. **Deny check.** If the key is in `DENIED_FLAGS` (~50 path/URL/credential/RCE flags — `model`, `host`, `port`, `lora*`, `model_url`, `hf_repo*`, `path`, `media_path`, `api_key*`, `ssl_*_file`, `chat_template_file`, `override_kv`, `tools`, `tensor_split`, `samplers`, `grammar`, …) → reject with "explicitly denied". This is the first line and beats everything.
2. **Suffix forward-defense (`_suffix_guard_check`).** If the key *name* matches a dangerous pattern — regex `.*_file$` / `.*_path$` / `.*_dir$` / `.*_url$` / `.*_repo$` / `.*_key$`, or prefix `^hf_` / `^lora` / `^control_vector` / `^lookup_cache_` / `^ssl_` / `^api_key` / `^slot_save_` / `^webui_` / `^docker_` — it is rejected **even if a future edit mistakenly adds it to the allowlist**. This catches new upstream (Tom's Fork) flags that ship a path/credential before anyone denylists them. (`_SUFFIX_GUARD_EXCEPTIONS` is currently empty.)
3. **Allowlist membership.** If the key is not in `SAFE_LLAMA_FLAGS` → reject with "not in the closed allowlist". Unknown = rejected.
4. **Value validation (`_validate_flag_value`).** Type, then enum (if applicable), then numeric bounds. Details below.

After all per-flag checks, one **cross-field** pass runs: `_validate_parallel_ctx` (the `parallel > 1` guard — see below).

### How values are typed, bounded, and enum-checked

`_validate_flag_value` applies, per flag:

- **Type spec** from `SAFE_LLAMA_FLAGS` (e.g. `int`, `bool`, `float`, `str`, or a tuple like `(int, str)` for `n_gpu_layers`).
- **String enums** from `SAFE_LLAMA_FLAG_STRING_ENUMS` — the value must be exactly one of a fixed set (e.g. `numa ∈ {none, distribute, isolate, numactl}`; `split_mode ∈ {none, layer, row, tensor}`; `cache_type_k/v ∈ {f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo2, turbo3, turbo4}`).
- **Numeric bounds** from `SAFE_LLAMA_FLAG_BOUNDS` — inclusive `(min, max)`, DoS-prevention (e.g. `ctx_size 1..2_000_000`, `parallel 1..256`, `top_p 0.0..1.0`, `temp 0.0..10.0`). A value outside range → reject.
- **`int → float` promotion is allowed** (an integer where a float is expected is fine).

**Bool-coercion rejection (load-bearing).** Python treats `bool` as a subclass of `int`, which would let `true`/`false` sneak into an int field (and vice-versa). Turbohaul explicitly blocks both directions: a `bool`-typed flag rejects a non-bool; an `int`-typed flag rejects a `bool` ("Python bool-is-int coerce explicitly rejected"). So `ctx_size: true` and `flash_attn: 1` both fail. Two flags are special-cased because they legitimately accept multiple types:

- **`flash_attn`** — accepts a `bool` *or* a string in `{on, off, auto, enabled, disabled}`. Anything else rejects.
- **`n_gpu_layers`** — accepts an `int` (bounded `-1..999`) *or* the strings `all` / `auto`. A `bool` is explicitly rejected.
- **`chat_template`** — must be a `str`, must **not** contain `{%` or `{{` (SSTI guard), and must be either a built-in name in `SAFE_CHAT_TEMPLATE_NAMES` or a short `^[A-Za-z0-9_.\-]+$` token ≤ 256 chars. Custom Jinja *bodies* are forced down the denied `chat_template_file` path — they cannot be inlined.

### The cross-field `parallel` guard

When `llama_server_flags.parallel > 1`, `_validate_parallel_ctx` enforces three rules in the same validation pass or the manifest is rejected (`parallel ≤ 1` is a no-op, back-compat):

1. **`kv_unified: true` is required** — a unified KV pool keeps cache accounting exact and flat across concurrent slots.
2. **`ctx_size` (the flag) must be evenly divisible by `parallel`** — otherwise the per-slot KV window is silently truncated.
3. **The per-slot window `ctx_size // parallel` must be ≥ `PER_SLOT_CTX_FLOOR` (8192)** — below this, each slot's usable context is too small.

Production example (a 35B MoE `parallel: 2` config): `ctx_size: 250000`, `parallel: 2`, `kv_unified: true` → 250000 % 2 == 0 and 250000 / 2 = 125000 ≥ 8192. All three rules satisfied.

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
| **Request body** (per-call) | `temperature`, `top_p`, `top_k`, `reasoning_budget`, `stop`, `max_tokens`, `keep_alive` | **Hot.** Applied per request through the forwarder; a manifest `PUT` (or per-request override) affects the next request immediately. |

The five doctrine flags (`flash_attn`, `no_context_shift`, `cache_reuse`, `slot_prompt_similarity`, `no_perf`) and all the structural flags (`ctx_size`, `cache_type_k/v`, `parallel`, `n_gpu_layers`, `jinja`, `spec_*`) are **spawn argv** — patching them on a running model is inert until reload.

**Forcing the cold-spawn — three options** (from `docs/TURBOQUANT_FLAGS.md`, discovered 2026-05-19 when a `/proc/<pid>/cmdline` audit showed the old cmdline still bound after a `PUT`):

- **Option A — `keep_alive: 0`.** Send any request to the same manifest tag with body `"keep_alive": 0` (Ollama-style, parsed at `chat_completion.py:parse_keep_alive`). This sets the slot's `IDLE_HOT` window to 0 → the running `llama-server` is torn down at the end of that request; the *next* request cold-spawns with the new flags.
- **Option B — wait for natural `IDLE_HOT` teardown** (`idle_hot.remaining_s → 0`), then the next request triggers a cold-spawn. (This was the option used in the 2026-05-19 discovery; ~3 min wait.)
- **Option C — `docker restart turbohaul`** (nuclear; recovers cleanly but interrupts in-flight requests).

**`/proc/cmdline` verification recipe** — confirm the running process actually carries the flag (not just the manifest):

```bash
# 1. Manifest has the flag (post-PUT):
curl -s http://localhost:11401/api/manifests/<your-model-tag> | jq '.llama_server_flags'

# 2. The running child process reflects it (post-cold-spawn):
docker exec turbohaul bash -c \
  'pgrep -af llama-server | head -1 | awk "{print \$1}" | xargs -I{} cat /proc/{}/cmdline | tr "\0" " "'
# Expect e.g.: --flash-attn on ... --no-context-shift ... --cache-reuse 256 ... --slot-prompt-similarity 0.5 ... --no-perf
```

If `/api/manifests` shows the flag but `/proc/<pid>/cmdline` does not, the running slot is **stale** (the manifest was patched after spawn) — trigger Option A/B/C and re-verify.

### The five doctrine flags — the tradeoffs

New manifests should ship these unless a model-specific reason says otherwise (`docs/TURBOQUANT_FLAGS.md`, locked 2026-05-19):

- **`flash_attn: true`** — required for native FP4 MMQ on Blackwell. Explicit in Turbohaul (was implicit in the old sidecar). Type: `bool | str-enum`. Argv: normalized to `--flash-attn on`. Spawn-argv.
- **`no_context_shift: true`** — avoids the `shift_context` loop bug that stalled long-context inference (standing lock). Type `bool`. Spawn-argv. *Deviate* only for a model that shifts correctly — but the default stays `true` because the loop bug recurs faster than fixes.
- **`cache_reuse: 256`** — enables prefix-cache reuse across requests in the same warm slot; cuts long-tail prefill on follow-ups. Type `int`, bound `0..65536`. Spawn-argv. *Deviate* (omit) only for one-shot batch models with no follow-up traffic — the overhead of leaving it on is negligible.
- **`slot_prompt_similarity: 0.5`** — lets a slot reuse its prefix cache even when the new prompt is not byte-identical (50% similarity threshold); improves `ACTIVE_MATCH` hit rate. Type `float`, bound `0.0..1.0`. Spawn-argv.
- **`no_perf: true`** — suppresses per-request perf logging (less log noise + minor CPU). Type `bool`. Spawn-argv. *Deviate* (`false`) when actively perf-debugging a model to surface per-request timings — flip back after. (Several production benchmark manifests run `no_perf: false` for exactly this reason.)

These compose with the TurboQuant KV flags `cache_type_k` / `cache_type_v` (`turbo3` is the production default; `turbo4` evaluation pending).

---

## Section 3 — Identity & Sizing Flags

These flags set the shape of the process: how much context it holds, where the weights live, how many concurrent slots it serves, and how it threads work. Almost all of them are **spawn-argv** — they are baked into the `llama-server` command line at fork time, so a manifest `PUT` does **not** change a running model. You must trigger a cold-spawn (see the doctrine section for Options A/B/C) and confirm with `/proc/<pid>/cmdline`. Every value/bound/type below is from `manifest.py` (`SAFE_LLAMA_FLAGS`, `SAFE_LLAMA_FLAG_BOUNDS`); production values are from the 30 live manifests in `/var/lib/turbohaul/manifests/`.

### The core sizing set

| Flag | Type | Bound (min, max) | Prod value(s) seen | Apply | Purpose |
|---|---|---|---|---|---|
| `ctx_size` | int | `(1, 2_000_000)` | `8192`, `12288`, `16384`, `32768`, `250000`, `500000` | spawn-argv | The aggregate KV context window `llama-server` allocates. Drives the KV-cache VRAM/RAM estimate directly (see Section 4). With `parallel > 1` this is the **aggregate** across all slots, not per-slot. |
| `n_gpu_layers` | int **or** `"all"`/`"auto"` | int bounded `(-1, 999)`; bool rejected | `999` (every live manifest) | spawn-argv | How many model layers offload to GPU. `999` = "all layers on GPU" (the ceiling; llama.cpp clamps to the real layer count). `-1` also means all upstream; strings `all`/`auto` are accepted but the reference manifests use `999` uniformly. |
| `parallel` | int | `(1, 256)` | `1` (default), `2` | spawn-argv | Number of concurrent same-model slots served by one sidecar. `>1` splits `ctx_size` across slots and pulls in the cross-field rule below. Default `1` (omit for serial serving). |
| `batch_size` | int | `(1, 65536)` | not set in live manifests (engine default) | spawn-argv | Logical prompt batch (`--batch-size` / `-b`): tokens the server groups per prefill decode call. Larger = faster prefill, more transient compute VRAM. Left unset in production (engine default). |
| `ubatch_size` | int | `(1, 65536)` | not set in live manifests | spawn-argv | Physical micro-batch (`--ubatch-size` / `-ub`): the sub-batch actually submitted to the GPU per kernel launch. Must be ≤ `batch_size`. Tunes prefill throughput vs. per-step memory; left unset in production. |
| `keep` | int | `(-1, 65536)` | not set in live manifests | spawn-argv | `--keep`: number of prompt tokens (from the front) to retain when the context is full and old tokens are discarded. `-1` keeps the whole initial prompt. Only relevant when context-shift/truncation is active — production runs `no_context_shift: true`, so it is unused. |
| `n_predict` | int | `(-1, 1_000_000)` | `-1` (MTP/dense manifests) | spawn-argv | `--n-predict` / `-n`: default max tokens to generate when a request doesn't specify. `-1` = unlimited (let the client's `max_tokens`/stop tokens govern). Set to `-1` on most reasoning manifests so the request body controls length. |

**`ctx_size` vs `context_size` — do not confuse them.** `context_size` is a **top-level Manifest field** (`int ≥ 1`, default `2048`) used by the VRAM-fit pre-check; `ctx_size` is the **llama_server_flag** that becomes `--ctx-size` on the child process. In every production manifest they are set to the same value (e.g. both `250000`), but they are distinct keys validated separately. If you copy a manifest, keep them in sync.

### Threading

| Flag | Type | Bound | Prod value | Apply | Purpose |
|---|---|---|---|---|---|
| `threads` | int | `(-1, 256)` | `16` (cpu-moe manifests only) | spawn-argv | `--threads` / `-t`: CPU threads for generation/eval. `-1` = auto (all cores). Only set explicitly on `cpu_moe` manifests (the CPU-MoE 35B configs → `16`) where MoE experts run on CPU and thread count materially affects throughput. |
| `threads_batch` | int | `(-1, 256)` | not set in live manifests | spawn-argv | `--threads-batch` / `-tb`: CPU threads for prompt prefill/batch processing (can differ from generation threads). `-1` = follow `threads`. Unset in production. |
| `threads_http` | int | `(-1, 256)` | not set in live manifests | spawn-argv | `--threads-http`: HTTP server worker threads for the llama-server API layer. `-1` = auto. Unset; the fronting forwarder handles concurrency. |
| `cont_batching` | bool | — | `true` (only on `parallel: 2` manifests) | spawn-argv | `--cont-batching`: continuous batching so multiple in-flight slots interleave decode steps instead of running strictly serially. Effectively mandatory companion to `parallel: 2` — always co-present with it in production (every `parallel: 2` config). |

### THE cross-field parallel rule (load-bearing)

When `parallel > 1`, `_validate_parallel_ctx` in `manifest.py` enforces three conditions in the same validation pass, or the manifest is **rejected** at load (it never reaches spawn). `parallel ≤ 1` is a no-op (back-compat). The three rules:

1. **`kv_unified: true` is REQUIRED.** Without a unified KV pool, `--parallel N`'s per-slot KV accounting diverges from the single count the VRAM gate assumes (that count is only accidentally correct because `--parallel` divides `ctx`). The unified pool keeps the cache exact and flat across concurrent slots — verified: a 35B `parallel: 2` + `kv_unified` adds ~0 extra VRAM. Omitting it raises `ManifestValidationError`.
2. **`ctx_size % parallel == 0`** — `ctx_size` must divide evenly across the N slots. Otherwise `llama-server` silently truncates the per-slot KV window. A non-divisible pair is rejected with a "choose a ctx_size that divides evenly" error.
3. **`ctx_size // parallel ≥ PER_SLOT_CTX_FLOOR` (8192).** Each per-slot window must be at least 8192 tokens or a slot's usable context is too small to be useful. Below the floor → rejected.

**Production proof of the rule satisfied:** a 35B MoE config runs `parallel: 2`, `ctx_size: 500000`, `kv_unified: true`, `cont_batching: true` → `500000 / 2 = 250000` per slot (divisible, well above 8192). Other `parallel: 2` configs use `ctx_size: 250000` → `125000` per slot. Note that a manifest may carry `parallel: 1` **explicitly alongside** `kv_unified: true` + `cont_batching: true` — that is legal because the guard is a no-op at `parallel: 1`; the extra flags are inert but harmless.

---

## Section 4 — KV-Cache & TurboQuant

The KV cache (per-token attention state) grows linearly with `ctx_size` and at long contexts can rival the model weights in size. These flags decide the KV **precision** (TurboQuant compressed types), **where it lives** (VRAM vs. host RAM), and the **checkpoint/reuse** machinery. All are **spawn-argv** except where noted. Enum sets and bounds are from `manifest.py` (`SAFE_LLAMA_FLAG_STRING_ENUMS`, `SAFE_LLAMA_FLAG_BOUNDS`); the KV size-factor table and sizing math are from LLM Wiki 05.

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
- **`turbo3`** — the balanced production default. Common across most 27B dense and 35B-MoE configs. Good compression (0.1875×) at good quality.
- **`turbo2`** — the most aggressive TurboQuant (0.125×). Trades the most precision for the smallest cache; reach for it when VRAM is critically tight. Typical use: a 27B dense/MTP pair (turbo2 to fit 250K ctx dual-card) and GPU-pinned co-residence configs (turbo2 for headroom). A 35B MoE config can be moved turbo3→turbo2 for VRAM headroom.
- **`turbo4`** — trades less precision for a **larger** cache (0.25×, same footprint as 4-bit `q4_0`). Highest-quality of the turbo family; use when you have VRAM to spare and want maximum KV fidelity. Marked "evaluation pending" — no production manifest ships it yet.
- **`f16`** — baseline, no compression. Typical use: manifests that run `flash_attn: false`, so they don't ride the compressed-KV fused path. One notable **mixed** config sets `cache_type_k: f16` + `cache_type_v: turbo2`.

> **The estimator keys off `cache_type_k` ALONE.** The manager reads `cache_type_k` (falling back to `f16` if unset), applies its factor, and **ignores `cache_type_v` entirely** for sizing. In normal use you set both to the same value, so this rarely matters — but a mixed config (K=f16, V=turbo2) is sized as if the whole cache were f16. A *misspelled* type (e.g. `turbo33`) is rejected at manifest validation by the `cache_type_k`/`cache_type_v` string-enum check — it never reaches the estimator; only an *unset* `cache_type_k` falls back to the f16 factor (1.0) for sizing.

`flash_attn: true` is effectively required to use the compressed KV types — they ride the fused-attention path (see Section 5).

### KV placement — VRAM vs. host RAM

| Flag | Type | Bound | Prod value | Apply | Purpose |
|---|---|---|---|---|---|
| `no_kv_offload` | bool | — | `true` (RAM-KV manifests) | spawn-argv | `--no-kv-offload`: pushes the **entire KV cache into host RAM** while weights stay on the GPU. Removes the KV term from the VRAM equation (it's checked against host RAM instead). This is the recipe that fits 250K context on a 24 GB card. The VRAM gate keys on `no_kv_offload: true` **specifically** — a `false` value is omitted from argv and has no effect. |
| `kv_offload` | bool | — | not set in live manifests | spawn-argv | The complementary/inverse toggle (`--kv-offload`, default behavior = KV in VRAM). Redundant with the default; the reference manifests express the offload decision via `no_kv_offload` only. |
| `kv_unified` | bool | — | `true` (all `parallel: 2` manifests + some explicit `parallel: 1`) | spawn-argv | `--kv-unified`: one shared KV pool across all slots instead of one-per-slot. **Required** whenever `parallel > 1` (enforced by the cross-field rule, Section 3). Keeps RAM-resident KV a single shared pool, not duplicated per slot. |
| `cache_ram` | int (MiB) | `(0, 262144)` | `32768` (RAM-KV manifests) | spawn-argv | The host-RAM budget (in **MiB**) reserved for the RAM-resident KV cache when `no_kv_offload: true`. Live pairing is always `no_kv_offload: true` + `cache_ram: 32768` (= 32 GiB), seen on the RAM-KV long-context manifests. |

**The RAM-KV recipe (from live manifests):** to serve a very long context that won't fit KV in VRAM —
```yaml
no_kv_offload: true
cache_ram: 32768        # 32 GiB host-RAM KV budget
ctx_size: 250000
# for concurrency, add:
parallel: 2
kv_unified: true
cont_batching: true
```

### Checkpoint / reuse machinery

| Flag | Type | Bound | Prod value | Apply | Purpose |
|---|---|---|---|---|---|
| `cache_prompt` | bool | — | not set in live manifests | spawn-argv | Cache the processed prompt so an identical prefix isn't re-prefilled. Superseded in production by the `cache_reuse` + `slot_prompt_similarity` doctrine pair (Section 5); left unset. |
| `cache_idle_slots` | bool | — | not set in live manifests | spawn-argv | Keep idle slot KV state resident (don't free the cache when a slot goes idle) so a returning request can warm-reuse it. Unset in production. |
| `ctx_checkpoints` | int | `(0, 1024)` | not set in live manifests | spawn-argv | Number of context checkpoints (`--ctx-checkpoints`) the server keeps for fast restore/rollback of KV state. Unset; the manager handles KV save/restore at its own layer. |
| `checkpoint_every_n_tokens` | int | `(1, 1_000_000)` | not set in live manifests | spawn-argv | Cadence (in tokens) at which a context checkpoint is taken. Companion to `ctx_checkpoints`; unset in production. |

### VRAM/RAM sizing math (from Wiki 05)

The manager estimates fit **without loading the model** — the `check_kv_cache_fit` gate (3rd of 5 safety gates). KV size:

**KV-fit precedence.** The KV estimate is whichever of these applies first:
1. an explicit **measured `kv_bytes_per_token`** override — used verbatim (no quant-scale, no hybrid multiply);
2. **parsed GGUF attention dims** for a `qwen35` model — sized from the real attention layers, and **ignores `hybrid_kv_ratio`** (the layer count already reflects the hybrid fraction);
3. the **legacy file-size heuristic** below — the *only* path that applies `hybrid_kv_ratio`.

The formula below is tier 3, the file-size fallback. Every non-`qwen35` model with no override uses it, byte-identically:

```
gguf_mib          = gguf_size_bytes / (1024*1024)
bytes_per_tok_f16 = (9 * gguf_mib) / 1024            # ≈ 9 KB/token per GiB of model body, at f16
bytes_per_tok     = bytes_per_tok_f16 * quant_factor  # quant_factor from the cache_type_k table
kv_cache_mib      = (bytes_per_tok * ctx_size) / 1024 * hybrid_kv_ratio   # hybrid_kv_ratio applies to THIS fallback path only; 1.0 (default) = no change
```

**Hybrid models (`arch: "qwen35"`).** For a `qwen35` model the manager parses the real GGUF attention dims and sizes `kv_cache_mib` from the actual attention layers (tier 2 above) — this is the accurate path and it does **not** apply `hybrid_kv_ratio`. `hybrid_kv_ratio` matters only on the **file-size fallback** (tier 3): when the GGUF dims can't be parsed and no `kv_bytes_per_token` override is set, the estimator multiplies the per-token KV term by `hybrid_kv_ratio` so a hybrid's fallback `kv_cache_mib` is proportionally smaller than a pure-attention model of the same body size. The default `hybrid_kv_ratio: 1.0` leaves that factor at 1 — every existing pure-attention model is sized exactly as before.

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

where `par_extra = (parallel − 1) × 256 MiB` (the per-slot compute floor). The KV term is **not** multiplied by `parallel` — `ctx_size` is the aggregate window split across slots, and a unified pool shares the RAM-KV rather than duplicating it.

**Worked example — ~20 GiB model @ 128K ctx on a 24 GB card:** f16 KV ≈ 22,500 MiB → needed ≈ 44 GiB ❌; `turbo3` KV ≈ 4,200 MiB → needed ≈ 25.7 GiB (fits with offload, or lower ctx). **Measured `parallel: 2` + RAM-KV fit** (35B-class MoE, `ctx_size: 250000`, `parallel: 2`, `kv_unified: true`, `no_kv_offload: true`, `cont_batching: true`): peak **21,903 MiB VRAM, ~2,084 MiB free**, both slots served with no OOM — a proven fit, not a projection.

---

## Section 5 — Performance & GPU Doctrine

This section covers the five standing doctrine flags (which are on by default in production), the MoE / multi-GPU placement flags, and the performance long-tail. Doctrine and MoE flags are **spawn-argv**. Bounds/enums from `manifest.py`; doctrine from `docs/TURBOQUANT_FLAGS.md`; MoE placement recipes from live manifests + Wiki 05.

### The five doctrine flags

These ship **on by default** on new production manifests unless a model-specific reason requires deviation (`docs/TURBOQUANT_FLAGS.md`, locked 2026-05-19). All five are spawn-argv.

**`flash_attn`** (`bool | str`; str enum `{on, off, auto, enabled, disabled}`; default doctrine `true`). Enables the fused-attention path required for the compressed-KV and low-precision MMQ kernels on Blackwell (native FP4 MMQ). On argv-build a bool is normalized to an explicit `--flash-attn on` / `--flash-attn off` (not a bare flag), so the child process always carries a value — useful when auditing `/proc/<pid>/cmdline`. **When to deviate:** some manifests run `flash_attn: false` (with `f16` KV) — those models don't ride the compressed-KV fused path, so flash-attn is off and KV stays f16. Otherwise leave it `true`; the TurboQuant `turbo*` types depend on it.

**`no_context_shift`** (`bool`; default doctrine `true`). Avoids the `shift_context` loop bug that previously stalled long-context inference (standing lock). **When to deviate:** a model that triggers context-shift correctly *could* run `false`, but the default is `true` because the loop bug recurs faster than fixes. Set on essentially every live manifest.

**`cache_reuse`** (`int`; bound `(0, 65536)`; default doctrine `256`). Enables prefix-cache reuse across requests in the same warm slot — cuts long-tail prefill on follow-ups. **When to deviate:** a pure one-shot batch workload with no follow-up traffic gains nothing, but the overhead of leaving it on is negligible, so it's safe to keep. Live value is uniformly `256`.

**`slot_prompt_similarity`** (`float`; bound `(0.0, 1.0)`; default doctrine `0.5`). Lets a slot reuse the prefix cache even when the prompt isn't byte-identical — a 50% similarity threshold — improving the ACTIVE_MATCH / warm-reuse hit rate. **When to deviate:** raise toward 1.0 to require near-identical prompts (fewer false reuses), lower for looser matching. Live value uniformly `0.5`.

**`no_perf`** (`bool`; doctrine `true`, but **usage is mixed**). Suppresses per-request perf logging — a small CPU + log-noise win. **When to deviate:** a model under active perf debugging wants `no_perf: false` to surface per-request timings (flip back after). In practice it is genuinely split: `true` on the routed serving configs, `false` on the benchmark variants where per-request timings are the point of the run.

### MoE / multi-GPU placement

| Flag | Type | Bound / enum | Prod value(s) | Apply | Purpose |
|---|---|---|---|---|---|
| `cpu_moe` | bool | — | `true` (CPU-MoE 35B configs) | spawn-argv | `-cmoe`: place **all** MoE expert layers on CPU (weights on host, attention on GPU). Frees GPU VRAM at the cost of CPU-bound expert compute. Live pairing: `cpu_moe: true` + `no_mmap: true` + `mlock: true` + `threads: 16`. |
| `n_cpu_moe` | int | `(0, 256)` | `10` (GPU1-pinned config) | spawn-argv | `-ncmoe N`: place the **first N** MoE layers on CPU (a partial-overflow version of `cpu_moe`). Use to shave exactly enough VRAM to fit rather than dumping all experts to CPU. Live: `10` on the GPU1-pinned co-residence variant. |
| `split_mode` | str | enum `{none, layer, row, tensor}` | `layer`, `none` | spawn-argv | `--split-mode`: how to spread the model across multiple GPUs. `layer` = split by layer across both cards (used with `main_gpu: 0` on the dual-card serving configs). `none` = single-card, no split (used on GPU-pinned co-residence variants alongside `main_gpu: 0` or `1`). `row`/`tensor` are allowed but not used in production. |
| `main_gpu` | int | `(0, 16)` | `0`, `1` | spawn-argv | `--main-gpu`: the primary GPU index (holds the non-split tensors / is the pin target when `split_mode: none`). `0` on most; `1` on the GPU1-pinned config (pinned to the second card for co-residence with a GPU0 model). |
| `fit` | str | enum `{on, off}` | not set in live manifests | spawn-argv | Tom's-Fork auto-memory-fit toggle. When `on`, the fork auto-sizes to fit available memory. Unset in production (the manager sizes explicitly via the VRAM gate). |
| `fit_ctx` | int | `(1, 2_000_000)` | not set in live manifests | spawn-argv | Target context for the `fit` auto-sizer. Unset; sizing is explicit via `ctx_size` + the fit gate. |

> **`tensor_split` and `fit_target` are NOT in the allowlist** — both are CSV-string flags with shell-meta risk, intentionally deferred (`DENIED_FLAGS` in `manifest.py`). You cannot pin a per-GPU tensor ratio via manifest today.

**The cpu-moe overflow recipe (from live manifests):**
```yaml
# Full offload of all experts to CPU (CPU-MoE 35B configs):
cpu_moe: true
no_mmap: true      # force-load weights into RAM rather than mmap (see long-tail)
mlock: true        # lock them resident, no swap
threads: 16        # CPU expert compute needs the thread count set

# Partial overflow — shave just enough (GPU1-pinned config):
n_cpu_moe: 10      # first 10 MoE layers to CPU
split_mode: none   # single-card pin
main_gpu: 1
```

### The performance long-tail

Each of these is allowlisted; most are unset in production (engine defaults). Value/bound + one-line purpose:

| Flag | Type | Bound / enum | Prod value | Apply | Purpose |
|---|---|---|---|---|---|
| `mlock` | bool | — | `true` (cpu-moe manifests) | spawn-argv | `--mlock`: lock model pages in RAM so the OS never swaps them out — steadier latency, needs the RAM headroom. Live only on `cpu_moe` manifests. |
| `no_mmap` | bool | — | `true` (cpu-moe + co-residence pins) | spawn-argv | `--no-mmap`: load weights fully into memory instead of memory-mapping the GGUF. Faster steady-state, higher load-time RAM. Live: the CPU-MoE 35B configs and the GPU-pinned co-residence pins. |
| `numa` | str | enum `{none, distribute, isolate, numactl}` | not set in live manifests | spawn-argv | `--numa`: NUMA placement policy for multi-socket hosts. `distribute` spreads threads across nodes; `isolate` pins to one; `numactl` defers to an external `numactl` wrapper. Unset (single-socket host). |
| `swa_full` | bool | — | not set in live manifests | spawn-argv | `--swa-full`: use the full (non-windowed) sliding-window-attention KV cache for SWA models — more VRAM, avoids the windowed-attention approximation. Unset. |
| `warmup` | bool | — | not set in live manifests | spawn-argv | `--warmup` / `--no-warmup`: run a dummy decode at spawn to warm kernels/allocations before serving. Unset (engine default). |
| `check_tensors` | bool | — | not set in live manifests | spawn-argv | `--check-tensors`: validate tensor data on load (catches corrupt GGUF) at a load-time cost. Unset — blobs are already SHA256-gated by the manifest. |
| `repack` | bool | — | not set in live manifests | spawn-argv | Repack weights into a CPU-optimized layout at load (throughput win for CPU-heavy paths). Unset. |
| `op_offload` | bool | — | not set in live manifests | spawn-argv | Offload individual ops to the GPU where beneficial (fine-grained op placement). Unset (engine default). |
| `no_host` | bool | — | not set in live manifests | spawn-argv | `--no-host`: disable the host (CPU) buffer path, forcing device-side buffers. Unset. |
| `direct_io` | bool | — | not set in live manifests | spawn-argv | `--direct-io`: use direct/unbuffered I/O for GGUF reads (bypass page cache) — helps on fast NVMe, hurts on re-reads. Unset. |
| `sleep_idle_seconds` | int | `(-1, 86400)` | `-1` (co-residence pins) | spawn-argv | Idle timeout before the sidecar sleeps/tears down. `-1` = never sleep (pin the process resident). Live: `-1` on the GPU0-pinned and GPU1-pinned configs so the co-resident pair stays hot. |

> **Boolean encoding (all bool flags above):** `True` → bare `--flag`; `False` → **omitted entirely** (not `--flag false`). So a `false` boolean has no effect on the spawned process — it's as if the flag were absent. The one exception is `flash_attn`, which always emits an explicit `on`/`off` value.

---

## Section 6 — Chat Template & Reasoning

These flags control how `llama-server` turns the OpenAI/Ollama request into the model's on-the-wire prompt, and how a thinking model's reasoning is parsed and budgeted. Every flag here is **spawn-argv** (it becomes part of the `llama-server` command line) with the single exception of `reasoning_budget`, which is **request-body-hot** — the forwarder can apply it per call, so it is the one knob in this section you can tune without a cold-spawn. Changing any of the others requires a cold-spawn (Option A `keep_alive: 0` / Option B natural idle teardown / Option C container restart) to take effect on a running slot.

### 6.1 `jinja` — load-bearing, keep it on

```yaml
jinja: true
```

`jinja` is a `bool` (allowlist type `bool`; argv `True` → bare `--jinja`, `False` → omitted). It is the single most important flag in this section and appears set to `true` in **every** live production manifest (all 37 checked).

`--jinja` switches `llama-server` from its legacy hard-coded prompt formatter to the model's own bundled Jinja chat template. Two capabilities depend on it, and both silently break without it:

1. **Tool calls.** Structured `tools` / `tool_choice` only work when the server renders the model's Jinja template, because that template is what emits the tool-call grammar the parser keys off. Without `jinja: true`, tool advertising is a no-op and text-JSON tool-call recovery cannot fire (recovery also requires the request to have advertised `tools`).
2. **Preserved thinking.** Thinking models only preserve `<think>…</think>` blocks and the structured `reasoning_content` field under `--jinja`. Turn it off and reasoning either merges into content or is dropped, depending on the model.

**Doctrine:** if you copy a manifest, keep `jinja: true`. It is treated as load-bearing across the manifest recipes. There is no production reason to run a chat model with it off.

### 6.2 `chat_template` — built-in name or bounded token, never a Jinja body

`chat_template` is a `str`, but it is the most heavily gated flag in the allowlist because an inlined Jinja template body is a server-side-template-injection (SSTI) vector. It is **not present in any live production manifest** — production relies on `jinja: true` selecting the model's own bundled template, so you rarely set `chat_template` at all. Set it only to *override* the auto-selected template with a specific built-in.

Validation (from `manifest.py` `_validate_flag_value` / `_check_jinja_injection`) runs three gates in order:

1. **SSTI guard (hard reject).** If the value contains `{%` or `{{` it is rejected outright — those are Jinja constructs that a non-sandboxed Jinja env could exploit to read the filesystem. A genuine custom Jinja template must go through the *denied* `chat_template_file` path, which cannot be smuggled into a manifest at all.
2. **Built-in enum match.** If the value is one of the `SAFE_CHAT_TEMPLATE_NAMES` it is accepted immediately. The set is the subset of llama.cpp's bundled templates the allowlist recognizes:

   `chatml`, `llama2`, `llama3`, `llama3.1`, `llama3.2`, `llama3.3`, `gemma`, `gemma2`, `gemma3`, `gemma4`, `mistral`, `mistral-v1`, `mistral-v3`, `mistral-v3-tekken`, `mistral-v7`, `phi3`, `phi4`, `deepseek`, `deepseek2`, `deepseek-r1`, `qwen`, `qwen2`, `qwen2.5`, `qwen3`, `qwen3.5`, `qwen3.6`, `command-r`, `command-r-plus`, `vicuna`, `alpaca`, `zephyr`, `chatglm3`, `chatglm4`, `openchat`, `orion`, `yi`, `monarch`, `smollm`, `minicpm`, `exaone3`, `rwkv-world`, `granite`, `qwen3-thinking`, `qwq`, `default` (45 names).
3. **Bounded plain-token fallback.** If the value is not a known built-in name, it is accepted only if it is a plain identifier of **≤ 256 chars** matching `^[A-Za-z0-9_.\-]+$` (alphanumeric plus `. _ -`). Anything longer, or containing any other character, is rejected. This lets a newly-shipped built-in name that isn't yet in the enum through, while structurally forbidding a template body (which would need spaces, braces, or `>256` chars).

| Property | Value |
|---|---|
| Type | `str` (special-cased) |
| Accepted | built-in enum name, OR `≤256`-char `^[A-Za-z0-9_.\-]+$` token |
| Hard-rejected | any value containing `{%` or `{{` (SSTI guard) |
| Layer | spawn-argv (cold-spawn to apply) |
| Seen in production | none — production uses `jinja: true` + the model's bundled template |

### 6.3 `skip_chat_parsing`, `special`, `spm_infill` — parsing toggles

All three are `bool` (argv `True` → bare `--flag`, `False` → omitted). **None appears in any live production manifest** — they are available for niche cases, not part of the standard doctrine. Their semantics map directly to the corresponding llama.cpp / `llama-server` options:

| Flag | Type | What it maps to in llama.cpp | When you'd use it |
|---|---|---|---|
| `skip_chat_parsing` | `bool` | Skips server-side parsing of the assistant output back into structured chat fields (`--no-parse-special`-adjacent behavior on the response path) — the raw generated text is returned without the tool-call / reasoning extraction pass. | Debugging what the model literally emitted, or a client that wants to do its own parsing. Leave off in normal use — it defeats tool-call recovery and reasoning-block extraction. |
| `special` | `bool` | Renders/allows **special tokens** to be emitted in output rather than being suppressed (`--special`). | Rare — inspecting a model's control tokens, or a downstream consumer that needs the raw special tokens in the stream. |
| `spm_infill` | `bool` | Selects **SentencePiece-style infill** token ordering for fill-in-the-middle (FIM) completions (`--spm-infill`). Only meaningful for code/infill models that use the SPM prefix/suffix convention. | A FIM-capable model whose infill ordering is the SentencePiece variant. No effect on ordinary chat models. |

Because none of these is in a shipped manifest, treat them as "known-to-llama.cpp, off by default" — set one only if you have a specific model behavior you are matching, and cold-spawn to apply.

### 6.4 Reasoning — `reasoning`, `reasoning_format`, `reasoning_budget`

This trio governs thinking models. In production, `reasoning: auto` is set on essentially every reasoning-capable manifest; `reasoning_format` and `reasoning_budget` are set alongside it.

**`reasoning`** — `str` enum, allowed values `{on, off, auto}` (from `SAFE_LLAMA_FLAG_STRING_ENUMS`). Spawn-argv.

- `on` — force the reasoning path (always parse/emit a thinking block).
- `off` — disable it.
- `auto` — let `llama-server` decide from the model's template/metadata. **This is the production default** (`reasoning: auto` on every thinking manifest).

**`reasoning_format`** — `str` enum, allowed values `{none, deepseek, deepseek-legacy, auto}`. Spawn-argv. Selects how the thinking segment is delimited/extracted so it can be surfaced as structured `reasoning_content` (and, on the non-stream path, wrapped as `<think>…</think>` in `message.content`).

| Value | Meaning | Production usage |
|---|---|---|
| `deepseek-legacy` | The older `<think>` delimiter convention. | The dominant production value — set on most 35B MoE / MTP thinking configs. |
| `auto` | Let the server infer the format from the model. | Set on the 27B dense/MTP configs. |
| `deepseek` | The current reasoning-delimiter convention. | Available; not observed in the current live set. |
| `none` | No reasoning-format extraction. | Available; not observed in the current live set. |

Note that many manifests set `reasoning: auto` and omit `reasoning_format` entirely — the server then infers formatting itself.

**`reasoning_budget`** — `int`, bounds `(-1, 1_000_000)`. **This is the one request-body-hot flag in the section** — `reasoning_budget` is among the per-request forwarded knobs (alongside `temperature`, `top_p`, `max_tokens`), so a manifest change to it affects the next request without a cold-spawn, and a client may also send it per call. It caps how many tokens the model may spend inside its thinking block before it must produce the answer (the "preserved-thinking depth" knob).

Semantics:

- `-1` — unbounded thinking (no cap).
- `0` — no thinking budget (effectively suppresses the reasoning span).
- `N` (positive) — allow up to N reasoning tokens.

**Production convention:** set `reasoning_budget` to **about half of `max_tokens`** — "half-of-max_tokens for half-budget," annotated directly in several manifests (`reasoning_budget 8192 … half-of-max_tokens`). This leaves the other half of the token budget for the actual answer, so a thinking model doesn't exhaust its window inside `<think>`. Observed values:

| `reasoning_budget` | Where it's used |
|---|---|
| `8192` | the majority of manifests (≈16K max_tokens configs) |
| `9192` | the 27B dense/MTP configs (≈18K max_tokens) |
| `10192` | the largest 35B MoE reasoning config (≈20K max_tokens) |
| `4096` | a smaller ~14B reasoning model |

Related caution from the client recipes: keep the request's `max_tokens` **≥ 2000** for thinking-model agent loops, or the model can exhaust a small budget entirely inside its thinking block and never emit an answer.

---

## Section 7 — MoE + MTP / Speculative Decoding

Speculative decoding runs a cheap *draft* that proposes several tokens per step, which the main model then verifies in a single forward pass. When the draft is accurate this raises tokens/sec with **no quality loss** — the main model still makes every final decision. Turbohaul's allowlist exposes exactly one speculative family: **draft-MTP** (multi-token prediction). Every flag in this section is **spawn-argv** (it becomes part of the `llama-server` command line), so applying or changing any of them requires a cold-spawn. They compose with the TurboQuant KV path (`cache_type_k`/`cache_type_v` = `turbo2/3/4`) and require `flash_attn` on — the same fused-attention path the compressed-KV kernels ride.

### 7.1 `spec_type` — the family selector

```yaml
spec_type: draft-mtp
```

`spec_type` is a `str` enum, and the allowlist accepts **only** `draft-mtp` (`SAFE_LLAMA_FLAG_STRING_ENUMS["spec_type"] = {"draft-mtp"}`). The comment in `manifest.py` notes that llama.cpp's PR #22673 enum also defines `draft` and `draft-eagle3`, but Turbohaul deliberately enables only `draft-mtp`. Any other value is rejected at manifest validation.

`draft-mtp` selects multi-token-prediction speculative decode, which uses the model's own bundled **nextn / MTP head** as the drafter rather than a separate draft model. This is why there is no `model_draft` flag here — that flag is in `DENIED_FLAGS` (it takes an arbitrary GGUF path). MTP needs a GGUF that was built with the nextn head (models that ship a multi-token-prediction head); on a model without it, `spec_type: draft-mtp` has nothing to draft from.

`spec_type` appears on roughly half the production manifests — every `-mtp` variant plus the dense 27B configs and the benchmark draft tiers. The paired `-nomtp` manifests are the exact same models with `spec_type` and its sub-knobs **omitted** — they exist precisely to A/B MTP on vs off.

### 7.2 `spec_draft_*` sub-knobs

All draft-tuning knobs only take effect when `spec_type: draft-mtp` is set. Types and bounds are exact from `SAFE_LLAMA_FLAGS` + `SAFE_LLAMA_FLAG_BOUNDS`:

| Flag | Type | Bound | What it maps to in llama.cpp | Production value |
|---|---|---|---|---|
| `spec_draft_n_max` | `int` | `(0, 64)` | Max draft tokens proposed per step (`--draft-max` / `-n_draft`-class). MTP commonly ~2–3. Higher = more speculation per step (bigger win when the draft is accepted, bigger wasted-compute cost when rejected). | `2` on 27b-class configs; `3` on the 35b-MoE / MTP configs |
| `spec_draft_n_min` | `int` | `(0, 64)` | Min draft tokens per step (`--draft-min`). Floors how many tokens the drafter must propose before verification. | not set in the live manifests (server default) |
| `spec_draft_p_min` | `float` | `(0.0, 1.0)` | Minimum probability to keep drafting (`--draft-p-min`). The drafter stops proposing once its confidence falls below this — a higher value drafts fewer, higher-confidence tokens. | not set in the live manifests (wiki example uses `0.6`) |
| `spec_draft_p_split` | `float` | `(0.0, 1.0)` | Draft-tree split-probability threshold (`--draft-p-split`) — governs when the speculative tree branches. | not set in the live manifests |
| `spec_draft_ngl` | `int` | `(-1, 999)` | Draft-model GPU layers (`--gpu-layers-draft` / `-ngld`). For bundled MTP the head normally lives on the same device as the main model, so this is rarely needed. | not set in the live manifests |
| `spec_draft_backend_sampling` | `bool` | — | Use backend-side sampling for the draft path (argv `True` → bare flag, `False` → omitted). | not set in the live manifests |

**In practice**, production sets only two of these: `spec_type: draft-mtp` and `spec_draft_n_max` (2 or 3). The remaining sub-knobs fall through to the server defaults. `n_max` splits cleanly by model size in the live set: the 27b-class models draft 2 tokens/step, the 35b-MoE and the larger draft-tier configs draft 3.

### 7.3 `n_rs_seq` relationship

When `draft-mtp` is on, the engine's restored-state sequence count `n_rs_seq` equals `spec_draft_n_max`. In other words, `spec_draft_n_max: 2` implies `n_rs_seq = 2` and `spec_draft_n_max: 3` implies `n_rs_seq = 3`. This matters for KV-state accounting: an MTP-enabled slot tracks `n_max` speculative sequences, so the restored/checkpointed KV state carries that many parallel sequence lanes. Keep this in mind when reasoning about KV-reuse and restore behavior on an MTP slot — the sequence bookkeeping is driven by `spec_draft_n_max`, not by `parallel`.

### 7.4 When MTP helps vs hurts, and how it composes

MTP is a **workload-dependent** win, not a free one:

- **Helps most** on predictable, low-entropy continuations: structured output, code, JSON, repetitive formatting. There the draft's proposed tokens are frequently accepted, so several tokens land per verification pass and tokens/sec climbs.
- **Helps least (can even hurt)** on high-entropy creative text: draft acceptance is low, so most speculative tokens are rejected and the extra draft compute is wasted. This is the tradeoff `spec_draft_n_max` tunes — a higher `n_max` amplifies both the upside (when accepted) and the wasted work (when rejected). The `-nomtp` manifest pairs exist so you can measure this per model rather than assume it.

**Composition with the rest of the stack:**

- **TurboQuant KV** (`cache_type_k`/`cache_type_v` = `turbo2/3/4`): MTP composes cleanly with compressed KV — the canonical example runs `spec_type: draft-mtp` alongside `cache_type_k: turbo3` / `cache_type_v: turbo3`.
- **`flash_attn`**: required on — flash-attention provides the fused kernel path that both the compressed-KV types and the MTP verify pass depend on. Every MTP manifest in production has `flash_attn: true`.
- **Turbo KV + `flash_attn` + MTP together** is the standard high-throughput production recipe (the MTP variants all stack turbo3 KV + `flash_attn: true` + `spec_type: draft-mtp`).

A minimal MTP manifest fragment (matching the shipped pattern):

```yaml
llama_server_flags:
  flash_attn: true
  cache_type_k: turbo3
  cache_type_v: turbo3
  spec_type: draft-mtp
  spec_draft_n_max: 3      # 2 on 27b-class, 3 on 35b-MoE / d-tier families
```

---

## Section 8 — Context extension: RoPE / YaRN

These nine flags change how the model's **positional encoding** is scaled so `llama-server` can run a context window *longer than the length the model was trained on*. They are all **spawn-argv** — baked into the `llama-server` command line at process fork, so a change to any of them requires a **cold-spawn** to take effect (the running slot keeps the old cmdline; see the doctrine note at the end of this section). None of them are per-request knobs.

### When (and why) you'd touch these at all

A GGUF is trained at some native context length (say 32K). If you set `ctx_size` beyond that native length, the RoPE frequencies the model learned no longer line up with the token positions it now sees, and quality degrades — the model gets "lost" past its trained horizon. RoPE scaling and YaRN are the two techniques that **rescale the rotary frequencies** so the model degrades gracefully instead of falling off a cliff.

**The honest default: leave all nine unset.** Modern GGUFs bake their correct RoPE parameters into the file's metadata, and `llama-server` reads them automatically. You only reach for these flags when you are deliberately pushing a model past its trained window and the metadata alone isn't giving you what you want. In the reference manifests, **none of the production YAMLs set any RoPE/YaRN flag** — the long-context configs (e.g. a config at `ctx_size: 250000`) rely on the model's baked metadata plus KV-in-RAM (`no_kv_offload`), not manual RoPE overrides. Treat this whole section as an advanced escape hatch.

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

> **Almost every flag in this section is REQUEST-BODY-HOT — do not bake it into a manifest.** llama.cpp accepts these as per-request parameters on the completions/chat API. The Turbohaul forwarder passes `temperature`, `top_p`, `top_k`, `stop`, `max_tokens` and their siblings straight through from each client call (per the TurboQuant flag doctrine's spawn-vs-request table and the client-setup recipes). If you put a sampling flag in `llama_server_flags`, all you are doing is **changing the server-side default** for that sampler — and because `llama_server_flags` is spawn-argv, that default only changes on a **cold-spawn**, and any client request can still override it per-call. So the normal path is: set sampling per-request from the client; touch the manifest **only** when you genuinely want to move the *default* (and accept a cold-spawn to do it). This is why the live production manifests carry essentially none of these — sampling is left to the caller. The bound column below is still enforced if you *do* set one in a manifest.

**Column key:** every row is **request-hot** (overridable per-call). The manifest bound applies only when you choose to set it as a spawn-time default. Bounds are from `SAFE_LLAMA_FLAG_BOUNDS`.

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

All request-hot; all off by default. Use one family deliberately — stacking many at once interacts unpredictably.

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
| `adaptive_target` | `-1.0`–`100.0` | Target for llama.cpp's adaptive sampler (a self-tuning sampler); `-1.0` = default/off sentinel. Semantics track llama.cpp's `--adaptive-*` implementation — set only if you're deliberately using that sampler. |
| `adaptive_decay` | `0.0`–`1.0` | Decay rate (0–1 factor) of the adaptive sampler's internal state. Pairs with `adaptive_target`. |

**Practical guidance.** For everyday use, set nothing here at the manifest level and let clients send `temperature` + one truncation sampler (`top_p` **or** `min_p`) plus a mild `repeat_penalty` per request. Reach for Mirostat, XTC, DRY, dynamic-temp, or the adaptive sampler only for a specific behavior problem (Mirostat for stable perplexity, DRY for stubborn loop-repetition, XTC/dynatemp for creative diversity) — and pick **one** family at a time. If you truly want a model to *default* to a non-standard sampler for every caller, that's the one legitimate reason to write a sampling flag into `llama_server_flags` (and then cold-spawn).

---

## Section 10 — Server toggles & debug

Two groups of process-level switches: **server toggles** that turn HTTP endpoints and modes on/off, and **debug/logging** switches that control `llama-server`'s console output. **All of these are spawn-argv** — they're part of the `llama-server` command line, so a change requires a **cold-spawn** to take effect. None are per-request. In the live production manifests, none of these are currently set (the models run with llama.cpp's defaults) — they're here for when you need to expose an endpoint or debug a spawn.

### Server toggles

| Flag | Type / enum (manifest.py) | What it exposes / does | When to use |
|---|---|---|---|
| `metrics` | bool | Enables the `/metrics` Prometheus endpoint on `llama-server` (per-slot token throughput, timings, etc.). | Turn on when you want to scrape the sidecar directly for observability. Off = one less exposed endpoint. |
| `slots` | bool | Enables the `/slots` endpoint that reports per-slot state (which is what the manager's `LiveSlotsPoller` reads). | Needed if you want raw slot introspection on that sidecar. Note the manager's monitor polls `/slots` at the manager layer; this flag is the server-side toggle for the endpoint itself. |
| `props` | bool | Enables `/props` (model properties + the loaded chat template) served by `llama-server`. | Handy for confirming which template/props the server actually loaded. |
| `embeddings` | bool | Puts the server in / enables **embeddings** mode so `/embedding` (and `/v1/embeddings`) return vectors. | Turn on only for an embedding model. On a chat model it changes pooling behavior and is usually wrong — leave off for generative models. |
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
| `log_verbosity` | int, bound `0`–`4` | Numeric log-level threshold. `0` = quietest, `4` = most verbose. Finer-grained than the boolean `verbose`. |

**Relationship to the doctrine `no_perf` flag.** Note that per-request performance timings are governed by the separate `no_perf` doctrine flag (Section covering the five doctrine flags), not by these debug toggles — production manifests set `no_perf: true` to suppress that specific per-request perf spam while leaving normal logging alone. If you're perf-debugging a model, flip `no_perf: false` (and cold-spawn) rather than cranking `verbose`/`log_verbosity`, which mostly add load/spawn noise, not per-request timings.

**Practical guidance.** Leave the whole section unset for normal generative serving. Enable a server toggle only to expose the specific endpoint you need (`embeddings`/`reranking`/`pooling` for the right model class; `metrics`/`slots`/`props` for observability). Reach for the debug switches only during a spawn investigation, and remember every one of them needs a cold-spawn to bind — patch the manifest, trigger teardown (doctrine Option A `keep_alive:0` / Option B natural idle-hot teardown / Option C `docker restart`), then confirm via `/proc/<pid>/cmdline`.

---

## Section 11 — Denied Flags & Why (Security Appendix)

`llama_server_flags` is a **closed allowlist**, but the allowlist is only half the defense. Before a key is even checked for membership, it runs a **deny gauntlet** in `manifest.py` (`Manifest._flags_allowlist` → per-key order): (1) explicit `DENIED_FLAGS` set → reject; (2) suffix-pattern forward-defense (`_suffix_guard_check`) → reject; (3) allowlist membership → reject if absent; (4) value/type/enum/bounds. A flag that clears (1)+(2) but is not in the allowlist still dies at (3). This section explains **why** whole classes of `llama-server` flags are forbidden, and gives the do-not-try list so you don't waste a `PUT` cycle discovering a `412`/validation error the hard way.

The governing principle: **the manifest describes a model, not the machine.** Anything that would let a YAML file name a filesystem path, a credential, a network endpoint, or a code-execution primitive is the *manager's* concern (boot config, host paths, env-held tokens) — never a per-model manifest. A manifest is attacker-reachable via `PUT /api/manifests/{tag}`; the boot config is not. So the trust boundary is drawn exactly at "can this value point at something outside the model."

### 11.1 The explicit `DENIED_FLAGS` set (52 keys) — by class

Every one of these is hard-rejected with `ManifestValidationError("... is explicitly denied (path-traversal/RCE class)")`. Grouped by *why*:

| Class | Flags | Why forbidden |
|---|---|---|
| **Direct RCE** | `tools` | `tools` enables llama-server's server-side tool interface (`exec_shell_command` / `write_file` / `edit_file`). A manifest that could set it is a remote-code-execution primitive. Never negotiable. |
| **Arbitrary file READ (path-bearing)** | `path`, `media_path`, `models_dir`, `models_preset`, `model`, `alias`, `model_draft`, `model_vocoder`, `webui_config_file`, `grammar_file`, `json_schema_file`, `chat_template_file`, `in_prefix_file`, `in_suffix_file`, `cache_prompt_file`, `slot_save_path`, `log_file`, `api_key_file`, `ssl_key_file`, `ssl_cert_file`, `lookup_cache_static`, `lookup_cache_dynamic`, `control_vector`, `control_vector_scaled`, `binary_override` | Each takes a filesystem path. `path`/`media_path` are CRITICAL — they set llama-server's static-file serving root, so an attacker could point it at `/etc` and exfiltrate host secrets over HTTP. `ssl_*_file` leaks PEM private keys. `model`/`models_dir`/`binary_override` let a manifest load an arbitrary GGUF or swap the binary. The manager owns model paths (blob store + manifest resolution); a manifest may only *content-address* a blob via `gguf_blob_sha256`, never name a path. |
| **SSRF + remote fetch** | `model_url`, `model_url_draft`, `hf_repo`, `hf_repo_draft`, `hf_file`, `hf_repo_v`, `hf_file_v`, `docker_repo`, `webui_mcp_proxy` | These make llama-server fetch over the network at spawn — attacker-controlled URL = SSRF (probe internal services) plus arbitrary-download RCE (pull a poisoned GGUF). `webui_mcp_proxy` is a CORS-bypass / SSRF pivot per Tom's Fork README. Model provenance is the manager's job (the `pull` subsystem with an HF host-allowlist and HTTPS-only), not a per-model flag. |
| **Credential injection** | `api_key`, `hf_token` | A manifest that sets an API key or HF token injects/rotates a credential the operator didn't authorize (and would log it in plaintext YAML). Tokens live in env vars named by the boot config (`pull.hf_api_key_env`), never inline. |
| **Network bind / topology** | `host`, `port`, `rpc` | These control *where the process listens* and RPC backend wiring — the manager assigns ports (`default_port_base`) and binds; a manifest that could rebind is a takeover/pivot primitive. |
| **KV / weight override** | `override_kv` | Rewrites GGUF metadata at load (rope config, arch fields, etc.) — a smuggle-path to change model behavior or trigger loader bugs. |
| **LoRA / adapter injection** | `lora`, `lora_base`, `lora_scaled`, `mmproj` | Load arbitrary adapter/projector weights from a path — same arbitrary-file-load class as `model`, plus behavior-modification. |
| **Deferred — parser needed, not a hard "never"** | `grammar` (inline BNF), `tensor_split` (CSV `N,N,N`), `fit_target` (CSV `MiB,MiB`), `samplers` (semicolon list), `dry_sequence_breaker` (str list), `chat_template_kwargs` (JSON string), `reasoning_budget_message` (mid-stream str) | These are *value-only* and not inherently path/RCE-bearing, but each carries a string with shell-meta / injection / recursive-scalar risk that the current validator can't safely parse. Denied until a dedicated pre-validator lands. If you need one, it's a code change + review — do not try to slip it into a manifest. |

### 11.2 The suffix forward-defense — why it exists

Even if a *new* dangerous flag ships in a future Tom's Fork pull and nobody remembers to add it to `DENIED_FLAGS`, `_suffix_guard_check` rejects it by **name shape** before it can ever be enumerated. Any key matching one of these patterns is rejected (unless whitelisted in `_SUFFIX_GUARD_EXCEPTIONS`, which is currently empty):

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

This is a **belt-and-suspenders** measure: the deny list is the known-bad list; the suffix guard is the "shape of bad" list. A flag has to survive **both** to reach the allowlist check. This is why you cannot introduce, say, a hypothetical `preset_dir` or `steering_url` even though neither is spelled out in `DENIED_FLAGS` — the suffix guard eats it.

### 11.3 The do-not-try list (fast reference)

If you find yourself wanting any of the below in a manifest, stop — it will be rejected, and the capability lives elsewhere:

- **A file path of any kind** (model, LoRA, grammar, chat template, log, SSL, KV save, static dir) → **denied.** Model blobs are content-addressed by `gguf_blob_sha256`; custom Jinja templates go through the manager's (denied-here) `chat_template_file` path only under operator control; logs/KV-dumps are the manager's.
- **A URL or `hf_*` / `docker_*` fetch** → **denied (SSRF/RCE).** Use the manager's `pull` subsystem (host-allowlisted, HTTPS-only).
- **An API key, HF token, or SSL cert/key** → **denied.** Credentials are env-var-named in boot config, never inline.
- **`host` / `port` / `rpc`** → **denied.** The manager assigns and binds; see boot config `server.*` + `default_port_base`.
- **`tools`** → **denied (direct RCE).** No exception.
- **`override_kv`** → **denied.** GGUF metadata is fixed at build time.
- **`tensor_split` / `fit_target` / `samplers` / `grammar` (inline) / `dry_sequence_breaker` / `chat_template_kwargs`** → **denied *for now*** (unsafe string parse). These are on the roadmap behind dedicated validators; don't hand-edit them in.
- **`chat_template` containing `{%` or `{{`** → **rejected** by `_check_jinja_injection` (SSTI guard). Use a built-in template *name* from `SAFE_CHAT_TEMPLATE_NAMES` (45 options) or a short `^[A-Za-z0-9_.\-]+$` token ≤256 chars. Inline Jinja bodies are forced down the denied file path by design.
- **Any brand-new flag whose name ends in `_file/_path/_dir/_url/_repo/_key` or starts with `hf_/lora/ssl_/api_key/control_vector/lookup_cache_/slot_save_/webui_/docker_`** → **rejected by the suffix guard** before allowlist lookup, even if it looks harmless.

Adding *any* flag to the allowlist is a **code change + review**, never a YAML edit — that's the whole point of the closed allowlist.

---

## Section 12 — Worked Per-Model Configs (Annotated Real Manifests)

These are **representative production manifests** pulled from `docker exec turbohaul cat /var/lib/turbohaul/manifests/*.yaml`. Each is a real serving config; the annotations explain *why* each setting is what it is. Every top-level field and flag is validated by `manifest.py` before it can be written. Note the common doctrine baseline shared by nearly all of them (the five TurboQuant flags + `jinja: true` + `reasoning_budget`) — the per-config *deltas* are what carry the engineering intent.

### 12.1 Dense 27B, KV-in-VRAM, single card

The simplest real production shape: a dense ~27B model with a small context and compressed KV entirely in VRAM. This is the "fast VRAM variant" — greedy output identical to the RAM-KV sibling but faster because there's no host-RAM KV round-trip.

```yaml
model_tag: dense-27b-ivram
gguf_blob_sha256: 818d68223be4d8518dac0b3b5604dde633cbbcbae1f491d842a3e26711c6606d
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
  no_context_shift: true         # avoid shift_context loop bug (standing lock). doctrine flag.
  cache_reuse: 256               # prefix-cache reuse across warm requests. doctrine flag.
  slot_prompt_similarity: 0.5    # 50% prefix-similarity threshold for cache hit. float 0..1. doctrine.
  no_perf: false                 # per-request perf timings LEFT ON here (this is an eval variant).
  jinja: true                    # LOAD-BEARING: tool_calls + <think> preservation need Jinja branch.
  reasoning_budget: 8192         # preserved-thinking depth ~half of a 16K max_tokens. int, request-hot.
```

Why these choices: `turbo3`/`turbo3` KV at 16K fits comfortably in a 24 GB card with weights, so no `no_kv_offload` is needed — everything stays in VRAM for speed. `no_perf: false` is the tell that this is a benchmark/eval variant (production traffic variants set it `true` to cut log noise). No `spec_type` → MTP speculative decode is off in this "no-MTP" sibling.

### 12.2 Dense 27B, MoE-less, RAM-KV for 250K context, GPU-pinned, layer-split

This is the flagship long-context dense config: **250K context** achieved by pushing the KV cache to host RAM (`no_kv_offload`), weights staying on GPU, plus MTP speculative decode and dual-card layer split.

```yaml
model_tag: dense-27b-250k
context_size: 250000
expected_vram_bytes: 24000000000    # full 24 GB budget claimed
revision: 25                         # heavily iterated (ETag = 25)
llama_server_flags:
  parallel: 1                        # single slot — no concurrency here. bound 1..256.
  ctx_size: 250000                   # 250K window. Only reachable because KV is in RAM (below).
  n_gpu_layers: 999                  # all weights on GPU.
  split_mode: layer                  # split model layers across BOTH cards. enum none/layer/row/tensor.
  main_gpu: 0                        # card 0 is primary for the split. int, bound 0..16.
  cache_type_k: turbo3
  cache_type_v: turbo3               # turbo3 KV keeps the 250K RAM-KV footprint ~10.6 GB (per description).
  n_predict: -1
  reasoning: auto
  reasoning_budget: 9192             # ~half of a ~18K max_tokens budget (half-of-max rule).
  jinja: true
  flash_attn: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true                      # production variant — perf logging OFF.
  spec_type: draft-mtp               # MTP speculative decode. enum {draft-mtp} only. spawn-argv.
  spec_draft_n_max: 2                # draft up to 2 tokens/step. bound 0..64.
  reasoning_format: auto             # thinking-block parse format auto. enum none/deepseek/deepseek-legacy/auto.
```

Why: the description documents "250K ctx via `--no-kv-offload` (KV in system RAM, weights on GPU). ~18.9GB VRAM / ~10.6GB RAM KV." The engine is **all-card-or-all-RAM** for KV — there is no tiered VRAM-first overflow — so to exceed what fits in VRAM you move the *entire* KV to RAM. `split_mode: layer` + `main_gpu: 0` spreads the weights across both 24 GB cards so the model itself fits with headroom. `spec_type: draft-mtp` with `spec_draft_n_max: 2` uses the GGUF's baked next-token-prediction head to draft 2 tokens per step (this blob has an MTP head). This is the canonical "long-context reasoning workhorse" shape.

> Note: `no_kv_offload`/`cache_ram` live on the RAM-KV benchmark variants; this particular dense-27B config reaches 250K via layer-split across both cards instead. Compare with §12.5 for the explicit `no_kv_offload` + `cache_ram` RAM-KV shape.

### 12.3 35B MoE, `parallel: 2` concurrent slots, RAM-KV, unified KV

The concurrency flagship: **one** 35B-A3B MoE resident, serving **two** same-model requests at once on a single card, with a **500K aggregate** context (250K per slot) held in host RAM.

```yaml
model_tag: moe-35b-parallel2
context_size: 500000
expected_vram_bytes: 24000000000
revision: 19
llama_server_flags:
  parallel: 2                    # 2 concurrent slots for THIS one model. Triggers the cross-field guard.
  cont_batching: true            # continuous batching so the 2 slots interleave. REQUIRED for real concurrency.
  kv_unified: true               # REQUIRED when parallel>1 — one unified KV pool (guard rejects without it).
  ctx_size: 500000               # aggregate. 500000 % 2 == 0 AND 250000/slot ≥ 8192 floor → passes guard.
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
  spec_draft_n_max: 3            # MoE draws 3 draft tokens/step (vs 2 for the dense 27B).
```

Why: this config exercises the **`_validate_parallel_ctx` cross-field rule** in full. `parallel: 2` mandates `kv_unified: true` (a unified pool keeps KV accounting exact and flat across slots — measured to add ~0 VRAM); `ctx_size` must be evenly divisible by `parallel` (500000/2 = 250000, clean); and the per-slot window (250000) must clear `PER_SLOT_CTX_FLOOR` = 8192. All three pass. `cont_batching: true` is what actually interleaves the two slots' decode steps. Because a two-slot KV in VRAM wouldn't fit for a 35B MoE, the KV is RAM-backed (the measured `21,903 MiB` / `~2,084 MiB`-free fit). `spec_draft_n_max: 3` is the MoE's larger draft depth. `reasoning_format: deepseek-legacy` matches this finetune's thinking-tag convention.

### 12.4 35B MoE, CPU-MoE overflow

When even RAM-KV isn't enough headroom, the MoE *expert layers themselves* are pushed to CPU. This is the "cpu-moe overflow" shape: keep attention + shared layers on GPU, run the sparse expert FFNs on CPU.

```yaml
model_tag: moe-35b-cpumoe
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

Why: `cpu_moe: true` (maps to `-cmoe`) moves the entire MoE expert stack off the GPU, which is why `expected_vram_bytes` drops to ~11 GB versus ~22 GB for the all-GPU siblings. Once experts run on CPU, three companion flags matter: `no_mmap: true` loads all weights eagerly (avoids page-fault stalls mid-inference), `mlock: true` pins them so they never swap, and `threads: 16` gives the CPU expert-matmul enough parallelism. The `-ncmoe N` variant (`n_cpu_moe`, seen on the GPU1-pinned config below) is the *partial* form — offload only N expert layers to CPU instead of all — used when you want to trade a little VRAM for speed rather than dump the whole expert stack.

### 12.5 GPU-pinned co-residence pair

The co-residence shape runs **two different models simultaneously**, each pinned to its own physical card (`split_mode: none` + a fixed `main_gpu`), so neither model's layers cross the PCIe bus. This is a deliberate departure from the single-model-resident norm.

**GPU0-pinned dense 27B:**
```yaml
model_tag: dense-27b-gpu0
context_size: 250000
expected_vram_bytes: 22500000000   # ~22.4 GB claimed on GPU0
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
model_tag: moe-35b-gpu1
context_size: 250000
expected_vram_bytes: 20500000000   # ~19.4 GB on GPU1
revision: 3
llama_server_flags:
  parallel: 2
  cont_batching: true
  kv_unified: true                 # parallel:2 → unified KV required (guard).
  ctx_size: 500000                 # 250K per slot.
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

Why the pair works: `split_mode: none` + distinct `main_gpu` (0 vs 1) guarantees each model lives on exactly one card, so they don't contend for the same VRAM or cross PCIe. `sleep_idle_seconds: -1` disables the idle-teardown that would otherwise free the slot — for a co-residence pair you *want* both pinned and hot. Both use `turbo2` KV (the most aggressive quant, ~0.125× f16) to fit 250K windows in the per-card budget. The 35B on GPU1 additionally uses `n_cpu_moe: 10` (partial CPU offload of 10 expert layers) to claw back enough VRAM on the smaller effective budget, and runs `parallel: 2` for its own two-slot concurrency. Note these two are minimal flag sets — they're the exact validated standalone per-card flags (no MTP, no `no_context_shift`/`cache_reuse` doctrine tail) preserved verbatim for co-residence.

### 12.6 Thinking model with explicit reasoning budget (turbo2 headroom variant)

The reasoning-heavy variant shows the interplay of `reasoning`, `reasoning_format`, `reasoning_budget`, and mixed K/V cache quants for VRAM headroom.

```yaml
model_tag: moe-35b-reasoning
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
  reasoning_budget: 10192          # LARGEST budget shown here — ~half a ~20K max_tokens.
  spec_type: draft-mtp
  spec_draft_n_max: 3
```

Why: this is a documented **VRAM-headroom** tuning (a `turbo3→turbo2` KV change for VRAM headroom, plus a Q4_K_M→Q4_K_S blob swap that left "~2GB free at full ctx"). The standout is the **asymmetric KV quant**: `cache_type_k: f16` keeps the key cache at full precision (keys are more sensitive to quantization error) while `cache_type_v: turbo2` aggressively compresses the value cache — a quality/VRAM trade you can make per-side because K and V are independent flags. `reasoning_budget: 10192` is the largest preserved-thinking budget shown here, following the "half-of-max_tokens" rule for a ~20K generation cap. `reasoning: auto` lets the engine detect thinking blocks; `reasoning_format: deepseek-legacy` matches the finetune's `<think>` tag convention. Unlike the concurrency config in §12.3, this one is `parallel: 1` — a single deep-reasoning slot rather than two shallow ones.

### 12.7 What to copy from these

- **Always carry `jinja: true`** — every production manifest has it; tool calls and `<think>` preservation break without it.
- **The five doctrine flags** (`flash_attn`, `no_context_shift`, `cache_reuse: 256`, `slot_prompt_similarity: 0.5`, `no_perf`) are the shared baseline — flip `no_perf: false` only for eval/debug variants.
- **KV placement is the big lever:** KV-in-VRAM (`turbo2`/`turbo3`, no offload) = fast, small context; `no_kv_offload: true` + `cache_ram: N` = long context bounded by host RAM instead of 24 GB VRAM.
- **`parallel > 1` is a package:** it *always* needs `kv_unified: true` + `cont_batching: true` + an evenly-divisible `ctx_size` with per-slot ≥ 8192, or the manifest is rejected at load.
- **MoE VRAM ladder:** all-GPU → `n_cpu_moe: N` (partial) → `cpu_moe: true` (all experts on CPU, + `no_mmap`+`mlock`+`threads`).

---

## Appendix A — Full flag index

Every key in `SAFE_LLAMA_FLAGS` (105 entries), with category, accepted type, numeric bound or string enum from `manifest.py`, whether it lands as **spawn-argv** (any value set in `llama_server_flags` is baked into the `llama-server` command line by `flags_to_argv`, so it needs a **cold-spawn** to take effect) or is also usable **request-body-hot** (sampling/reasoning knobs the forwarder can apply per-call), a representative default seen in production manifests, and a one-line purpose.

**Spawn-argv vs request-body:** *Everything in a manifest is spawn-argv* — a `PUT` to `llama_server_flags` only affects the **next cold-spawn** (see the doctrine doc's Option A/B/C to force one). The "request-body-hot" tag marks flags that *also* correspond to per-request body fields (`temperature`, `top_p`, `top_k`, `reasoning_budget`, `n_predict`/`max_tokens`, `seed`, penalties, etc.) which the forwarder applies live without a respawn. Booleans encode as `--flag` when `true`, **omitted** when `false` (except `flash_attn`, which always emits `--flash-attn on|off`).

**Encoding notes:** `flash_attn` bool→`--flash-attn on/off`; all other bools true→`--flag`, false→omitted; ints/floats/strings→`--flag <value>`. `bool↔int` coercion is rejected; `int→float` promotion allowed.

### A.1 Performance + memory layout

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `ctx_size` | int | 1..2,000,000 | spawn-argv | 250000 / 16384 | Context window (KV size driver). |
| `n_gpu_layers` | int \| str | -1..999, or `all`/`auto` | spawn-argv | 999 | Layers on GPU (999 = all). |
| `threads` | int | -1..256 | spawn-argv | 16 (cpu-moe) | CPU threads for generation. |
| `threads_batch` | int | -1..256 | spawn-argv | — | CPU threads for prompt batch. |
| `threads_http` | int | -1..256 | spawn-argv | — | HTTP server worker threads. |
| `parallel` | int | 1..256 | spawn-argv | 1 / 2 | Concurrent slots for one model (>1 triggers cross-field guard). |
| `batch_size` | int | 1..65536 | spawn-argv | — | Logical prompt batch (`-b`). |
| `ubatch_size` | int | 1..65536 | spawn-argv | — | Physical micro-batch (`-ub`). |
| `n_predict` | int | -1..1,000,000 | spawn-argv | -1 | Server-side gen cap (-1 = client-driven). |
| `keep` | int | -1..65536 | spawn-argv | — | Tokens kept when context truncates. |
| `flash_attn` | bool \| str | `on/off/auto/enabled/disabled` | spawn-argv | true | FlashAttention (FP4 MMQ on Blackwell). Always emits explicit `on/off`. |
| `mlock` | bool | — | spawn-argv | true (cpu-moe) | Pin weights in RAM (no swap). |
| `no_mmap` | bool | — | spawn-argv | true (cpu-moe/pinned) | Disable mmap; eager full load. |
| `numa` | str | `none/distribute/isolate/numactl` | spawn-argv | — | NUMA memory policy. |
| `swa_full` | bool | — | spawn-argv | — | Force full-size sliding-window-attention KV (no SWA shrink). |
| `no_perf` | bool | — | spawn-argv | true (prod) / false (eval) | Suppress per-request perf logging. Doctrine flag. |
| `sleep_idle_seconds` | int | -1..86400 | spawn-argv | -1 (co-resident) | Idle-teardown timer; -1 = never sleep. |
| `cache_reuse` | int | 0..65536 | spawn-argv | 256 | Prefix-cache reuse across warm requests. Doctrine flag. |
| `no_context_shift` | bool | — | spawn-argv | true | Disable context-shift loop (standing lock). Doctrine flag. |
| `slot_prompt_similarity` | float | 0.0..1.0 | spawn-argv | 0.5 | Prefix-similarity threshold for slot cache reuse. Doctrine flag. |
| `warmup` | bool | — | spawn-argv | — | Run a warmup pass at load. |
| `check_tensors` | bool | — | spawn-argv | — | Validate tensor data on load. |
| `repack` | bool | — | spawn-argv | — | Repack quantized weights for the target CPU/GPU layout. |
| `op_offload` | bool | — | spawn-argv | — | Offload eligible ops to GPU (llama.cpp `--op-offload`). |
| `no_host` | bool | — | spawn-argv | — | Disable host-buffer fallback (maps to llama.cpp `--no-host`). |
| `direct_io` | bool | — | spawn-argv | — | Use direct I/O for model file reads. |
| `cont_batching` | bool | — | spawn-argv | true (parallel>1) | Continuous batching; required for real `parallel>1` interleaving. |

### A.2 KV-cache

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `cache_type_k` | str | `f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1/turbo2/turbo3/turbo4` | spawn-argv | turbo3 / turbo2 / f16 | K-cache quant type. turbo2≈0.125×, turbo3≈0.1875×, turbo4≈0.25× f16 size. |
| `cache_type_v` | str | (same enum as K) | spawn-argv | turbo3 / turbo2 | V-cache quant type. Set independently from K (asymmetric OK). |
| `kv_offload` | bool | — | spawn-argv | — | Explicitly enable KV offload to GPU (llama.cpp default-on). |
| `no_kv_offload` | bool | — | spawn-argv | true (RAM-KV) | Put KV cache in host RAM, not VRAM — enables huge contexts bounded by RAM. |
| `kv_unified` | bool | — | spawn-argv | true (parallel>1) | Single unified KV pool across slots. REQUIRED when `parallel>1`. |
| `cache_idle_slots` | bool | — | spawn-argv | — | Keep idle slots' KV cached rather than freeing. |
| `cache_prompt` | bool | — | spawn-argv | — | Cache the prompt tokens for reuse (llama.cpp `--cache-prompt`). |
| `cache_ram` | int | 0..262144 (MiB) | spawn-argv | 32768 | Host-RAM KV cache budget in MiB (pairs with `no_kv_offload`). |
| `ctx_checkpoints` | int | 0..1024 | spawn-argv | — | Number of context checkpoints retained for fast restore. |
| `checkpoint_every_n_tokens` | int | 1..1,000,000 | spawn-argv | — | Checkpoint cadence in tokens. |

### A.3 Context / RoPE / YaRN

No production manifest sets these (models use native context); they exist for context-extension tuning. All spawn-argv.

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
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

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `cpu_moe` | bool | — | spawn-argv | true (D3 variants) | Put ALL MoE expert layers on CPU (`-cmoe`). Big VRAM drop. |
| `n_cpu_moe` | int | 0..256 | spawn-argv | 10 (gpu1) | Offload N expert layers to CPU (`-ncmoe`); partial form of `cpu_moe`. |
| `split_mode` | str | `none/layer/row/tensor` | spawn-argv | layer / none | Multi-GPU split strategy. `none` = pin to one card; `layer` = split layers across cards. |
| `main_gpu` | int | 0..16 | spawn-argv | 0 / 1 | Primary GPU for the split / pin target. |
| `fit` | str | `on/off` | spawn-argv | — | Tom's Fork auto-memory-fit toggle. |
| `fit_ctx` | int | 1..2,000,000 | spawn-argv | — | Target context for the auto-fit solver. |

### A.5 Sampling

All also **request-body-hot** — the forwarder applies these per-call; a manifest value is the default the cold-spawn bakes in.

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `temp` | float | 0.0..10.0 | argv + body | — | Sampling temperature. |
| `top_k` | int | 0..10000 | argv + body | — | Top-K sampling cutoff. |
| `top_p` | float | 0.0..1.0 | argv + body | — | Nucleus (top-P) sampling. |
| `min_p` | float | 0.0..1.0 | argv + body | — | Min-P sampling floor. |
| `typical_p` | float | 0.0..1.0 | argv + body | — | Locally-typical sampling (Ollama parity). |
| `top_n_sigma` | float | -1.0..100.0 | argv + body | — | Top-nσ logit-std sampling cutoff (-1 = off). |
| `repeat_penalty` | float | 0.0..10.0 | argv + body | — | Repetition penalty. |
| `repeat_last_n` | int | -1..65536 | argv + body | — | Window for repeat penalty (-1 = ctx). |
| `presence_penalty` | float | -10.0..10.0 | argv + body | — | Presence penalty (Ollama parity). |
| `frequency_penalty` | float | -10.0..10.0 | argv + body | — | Frequency penalty (Ollama parity). |
| `seed` | int | -1..2⁶³-1 | argv + body | — | RNG seed (-1 = random). |
| `mirostat` | int | 0..2 | argv + body | — | Mirostat mode (0 off, 1 v1, 2 v2). |
| `mirostat_lr` | float | 0.0..1.0 | argv + body | — | Mirostat learning rate (η). |
| `mirostat_ent` | float | 0.0..100.0 | argv + body | — | Mirostat target entropy (τ). |
| `xtc_probability` | float | 0.0..1.0 | argv + body | — | XTC (exclude-top-choices) trigger probability. |
| `xtc_threshold` | float | 0.0..1.0 | argv + body | — | XTC logit threshold. |
| `dynatemp_range` | float | 0.0..10.0 | argv + body | — | Dynamic-temperature range. |
| `dynatemp_exp` | float | 0.0..10.0 | argv + body | — | Dynamic-temperature exponent. |
| `dry_multiplier` | float | 0.0..10.0 | argv + body | — | DRY repetition-penalty multiplier. |
| `dry_base` | float | 1.0..10.0 | argv + body | — | DRY penalty base. |
| `dry_allowed_length` | int | 0..65536 | argv + body | — | DRY max allowed repeat length before penalizing. |
| `dry_penalty_last_n` | int | -1..65536 | argv + body | — | DRY lookback window (-1 = ctx). |
| `adaptive_target` | float | -1.0..100.0 | argv + body | — | Adaptive-sampling target (maps to Tom's Fork adaptive knob; -1 = off). |
| `adaptive_decay` | float | 0.0..1.0 | argv + body | — | Adaptive-sampling decay rate. |
| `ignore_eos` | bool | — | argv + body | — | Ignore EOS token (keep generating). |

### A.6 Chat / template

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `chat_template` | str | built-in name (45-set) OR `^[A-Za-z0-9_.\-]+$` ≤256 chars; no `{%`/`{{` | spawn-argv | — | Chat template by name. Jinja bodies rejected (SSTI guard) → use `chat_template_file` (denied path, operator-only). |
| `jinja` | bool | — | spawn-argv | true | Use Jinja chat-template branch. LOAD-BEARING for tool_calls + `<think>` preservation. |
| `skip_chat_parsing` | bool | — | spawn-argv | — | Bypass server-side chat parsing (raw prompt). |
| `special` | bool | — | spawn-argv | — | Render special/control tokens in output. |
| `spm_infill` | bool | — | spawn-argv | — | SentencePiece infill token ordering (FIM). |

### A.7 Reasoning

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `reasoning_format` | str | `none/deepseek/deepseek-legacy/auto` | spawn-argv | auto / deepseek-legacy | Thinking-block parse format. |
| `reasoning` | str | `on/off/auto` | spawn-argv | auto | Enable/detect reasoning mode. |
| `reasoning_budget` | int | -1..1,000,000 | argv + body | 8192 / 9192 / 10192 | Preserved-thinking depth (-1 unlimited, 0 off). Live convention ≈ half of max_tokens. |

### A.8 Speculative decoding / MTP

All spawn-argv; require a GGUF with a baked next-token-prediction (nextn/MTP) head.

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `spec_type` | str | `draft-mtp` only | spawn-argv | draft-mtp | Speculative-decode mode (only MTP enabled). |
| `spec_draft_n_max` | int | 0..64 | spawn-argv | 2 (dense) / 3 (MoE) | Max draft tokens per step. |
| `spec_draft_n_min` | int | 0..64 | spawn-argv | — | Min draft tokens per step. |
| `spec_draft_p_min` | float | 0.0..1.0 | spawn-argv | — | Min probability to continue drafting. |
| `spec_draft_p_split` | float | 0.0..1.0 | spawn-argv | — | Draft split-probability threshold. |
| `spec_draft_ngl` | int | -1..999 | spawn-argv | — | Draft-head GPU layers (usually same device). |
| `spec_draft_backend_sampling` | bool | — | spawn-argv | — | Backend-side sampling for the draft path. |

### A.9 Server toggles

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `metrics` | bool | — | spawn-argv | — | Enable Prometheus `/metrics` endpoint. |
| `slots` | bool | — | spawn-argv | — | Enable `/slots` introspection endpoint. |
| `props` | bool | — | spawn-argv | — | Enable `/props` endpoint. |
| `embeddings` | bool | — | spawn-argv | — | Enable embeddings endpoint (embedding mode). |
| `reranking` | bool | — | spawn-argv | — | Enable rerank endpoint. |
| `pooling` | str | `none/mean/cls/last/rank` | spawn-argv | — | Embedding pooling strategy. |
| `offline` | bool | — | spawn-argv | — | Offline mode (no network fetches). |

### A.10 Debug

| Flag | Type | Bound / enum | Argv/body | Default seen | Purpose |
|---|---|---|---|---|---|
| `verbose` | bool | — | spawn-argv | — | Verbose server logging. |
| `log_disable` | bool | — | spawn-argv | — | Disable logging entirely. |
| `log_colors` | str | `on/off/auto` | spawn-argv | — | ANSI color in logs. |
| `log_prefix` | bool | — | spawn-argv | — | Prefix log lines with level/source. |
| `log_timestamps` | bool | — | spawn-argv | — | Timestamp log lines. |
| `log_verbosity` | int | 0..4 | spawn-argv | — | Log verbosity level. |

> **Coverage note:** this table is the complete `SAFE_LLAMA_FLAGS` allowlist (105 keys) as of `origin/main:src/turbohaul/manifest.py`. 58 keys carry numeric `SAFE_LLAMA_FLAG_BOUNDS`; 11 carry `SAFE_LLAMA_FLAG_STRING_ENUMS`; `flash_attn`, `n_gpu_layers`, and `chat_template` are special-cased. Any key **not** in this table is rejected at manifest load. For the forbidden classes and the suffix forward-defense, see §11.
