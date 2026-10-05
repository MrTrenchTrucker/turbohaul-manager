# TurboQuant Flag Doctrine

**Status:** Recommended defaults for new manifests. Flag changes are applied to a manifest via `PUT /api/manifests/{tag}` with ETag/If-Match atomic concurrency, and take effect at the next cold-spawn; verify them on the running process via `/proc/<pid>/cmdline`.

This document defines the canonical TurboQuant flag set Turbohaul-Manager uses for new manifests targeting the TurboQuant llama.cpp fork vendored in `engine/llama-cpp-turboquant/`. New manifests should ship with these defaults unless a model-specific reason requires deviation.

---

## The five doctrine flags

| Flag | Value | Why |
|---|---|---|
| `flash_attn` | `true` | Needed by the turbo cache types (the engine enables it automatically if it is off) and required for any quantized V cache; written explicitly in manifests. |
| `no_context_shift` | `true` | Turns off the engine's context shifting (`--no-context-shift`). |
| `cache_reuse` | `256` | Minimum chunk size (in tokens) the engine will try to reuse from the cache via KV shifting; a non-zero value enables reuse of cached prompt chunks across requests, which can cut prefill on follow-ups. |
| `slot_prompt_similarity` | `0.5` | How closely a request's prompt must match a slot's cached prompt for the engine to reuse that slot (`0.5` = 50% similarity; `0.0` disables the check). Lets a slot be reused when the prompt is not byte-identical. |
| `no_perf` | `true` | Disables libllama's internal performance timings (`--no-perf`). |

These complement the existing `cache_type_k` / `cache_type_v` TurboQuant cache flags (`turbo3` is the default).

## Why a turbo KV cache type is the default

The KV cache cost isn't just about total size — it's about what a setting trades away to get
there. The table below shows KV cache size relative to full 16-bit precision:

| Setting | Size relative to 16-bit | Effective width |
|---|---|---|
| `f16` | 1.0 | 16-bit |
| `q8_0` | 0.5 | 8-bit |
| `turbo4` | 0.25 | 4-bit |
| `turbo3` | 0.1875 | 3-bit |
| `turbo2` | 0.125 | 2-bit |

The turbo types aren't a general-purpose quantization applied to KV data after the fact — they're
designed specifically for it, which is why they are much smaller than the conventional quantizations: a
2-bit turbo cache is **four times smaller than `q8_0`**. That's why a turbo type is the sensible
default over `q8_0`.

A smaller turbo number isn't automatically better, though. The turbo levels trade size against fidelity, and the
right default is a balance point, not the smallest available number — pick based on how much of
that trade your deployment can afford, rather than reflexively taking the smallest cache.

**Default:** `turbo3`, for both `cache_type_k` and `cache_type_v`.

## Spawn-time vs request-time — the critical distinction

| Layer | Examples | Reload behavior |
|---|---|---|
| **Spawn argv** (process-fork) | `flash_attn`, `no_context_shift`, `cache_reuse`, `slot_prompt_similarity`, `no_perf`, `ctx_size`, `cache_type_k`, `cache_type_v`, `n_gpu_layers`, `jinja` | **COLD-SPAWN ONLY** — manifest PUT does not affect running `llama-server`. Old cmdline persists until process exits. |
| **Request body** (per-call) | `temperature`, `top_p`, `reasoning_budget`, `top_k`, `max_tokens` | **Hot** — sent in each request body and forwarded per request; no respawn needed. |

The five doctrine flags above are **spawn argv**. Patching them on a running model requires one of:

- **Option A** — send any request to the same manifest tag with body `"keep_alive": 0` (Ollama-style; parsed by `parse_keep_alive` in `src/turbohaul/api/chat_completion.py`). Sets the slot's `IDLE_HOT` window to 0 → the running `llama-server` is torn down at the end of that request (unless the next queued request is for the same model, in which case it is held warm for up to 30 s so that request can reuse it); the next request triggers a cold-spawn.
- **Option B** — wait for natural `IDLE_HOT` teardown (`idle_hot.remaining_s → 0`), then next request triggers cold-spawn.
- **Option C** — `docker restart turbohaul-manager` (nuclear; recovers cleanly but interrupts in-flight requests).

A manifest PUT does not touch an already-running engine: after a PUT, the `/proc/<pid>/cmdline` of a model spawned before it still shows the old flags until the next cold-spawn.

## Verification recipe

```bash
# 1. Confirm manifest has the flag (post-PUT)
curl -s http://<turbohaul-host>:11401/api/manifests/<tag> | jq '.llama_server_flags'

# 2. Confirm /proc/<pid>/cmdline reflects the flag (post-cold-spawn)
docker exec turbohaul-manager bash -c \
    'pgrep -af llama-server | head -1 | awk "{print \$1}" | xargs -I{} cat /proc/{}/cmdline | tr "\0" " "'

# Expect to see: --flash-attn ... --no-context-shift ... --cache-reuse 256 ... --slot-prompt-similarity 0.5 ... --no-perf
```

If `/api/manifests` shows the flag but `/proc/<pid>/cmdline` does not — the running slot is stale (manifest was patched after spawn). Trigger Option A/B/C and re-verify.

## Patching recipe (one model)

```bash
TAG=my-model-27b
BASE=http://<turbohaul-host>:11401

# 1. GET current manifest + ETag
ETAG=$(curl -s -D - -o /dev/null ${BASE}/api/manifests/${TAG} | grep -i etag | awk '{print $2}' | tr -d '\r\n"')
curl -s ${BASE}/api/manifests/${TAG} > /tmp/manifest.json

# 2. Add the five flags to llama_server_flags (jq merges in memory; the GET adds a read-only
#    cache_reuse_inert_mmproj marker that a PUT rejects, so drop it)
jq 'del(.cache_reuse_inert_mmproj) | .llama_server_flags += {
    flash_attn: true,
    no_context_shift: true,
    cache_reuse: 256,
    slot_prompt_similarity: 0.5,
    no_perf: true
}' /tmp/manifest.json > /tmp/manifest.patched.json

# 3. PUT with If-Match (atomic ETag concurrency)
curl -s -X PUT ${BASE}/api/manifests/${TAG} \
    -H "If-Match: \"${ETAG}\"" \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/manifest.patched.json

# 4. Trigger cold-spawn (Option B = wait idle_hot teardown, then send a real request)
# 5. Verify via /proc/<pid>/cmdline (see Verification recipe above)
```

## When to deviate

- A model that handles context shifting correctly may benefit from `no_context_shift: false` — the recommended value is `true` (context shifting off).
- A model with no follow-up traffic pattern (one-shot batch use) gains nothing from `cache_reuse` + `slot_prompt_similarity` — safe to omit.
- A model under active perf debugging may want `no_perf: false` to surface the engine's performance timings — flip back after debug.

## See also

- [TURBOQUANT_KV_QUANTIZATION.md](./TURBOQUANT_KV_QUANTIZATION.md) — what the `turbo2`/`turbo3`/`turbo4`
  cache types are, and the high-GQA case where the engine overrides a declared K type.
- [MULTI_AGENT_SHARING.md](./MULTI_AGENT_SHARING.md) — multi-agent serialization context.
- [PERSISTENCE_CHECKLIST.md](./PERSISTENCE_CHECKLIST.md) — manifest persistence (manifests live in `/var/lib/turbohaul/manifests/` as YAML; the checklist covers the state volume that holds them).
- `src/turbohaul/manifest.py` `SAFE_LLAMA_FLAGS` — the in-code allowlist of accepted flags (108 total). See [MODEL_CONFIG_REFERENCE.md](./MODEL_CONFIG_REFERENCE.md) Appendix A for the full indexed table.
- [MODELS_AND_MANIFESTS.md](./MODELS_AND_MANIFESTS.md) — the guide to adding and configuring a model, from the backend and the dashboard.
