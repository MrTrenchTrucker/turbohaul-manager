# TurboQuant Flag Doctrine

**Status:** Locked in production manifests 2026-05-19. Applied to production manifests via `PUT /api/manifests/{tag}` with ETag/If-Match atomic concurrency. Verified live via `/proc/<pid>/cmdline` after cold-spawn.

This document defines the canonical TurboQuant flag set Turbohaul-Manager uses for new manifests targeting Tom's TurboQuant llama.cpp fork. New manifests should ship with these defaults unless a model-specific reason requires deviation.

---

## The five doctrine flags

| Flag | Value | Why |
|---|---|---|
| `flash_attn` | `true` | Required for native FP4 MMQ on Blackwell. Was implicit in the old Turboquant sidecar; explicit in Turbohaul manifests. |
| `no_context_shift` | `true` | Avoids the `shift_context` loop bug that previously stalled long-context inference. Standing default. |
| `cache_reuse` | `256` | Enables prefix-cache reuse across requests in the same warm slot. Cuts long-tail prefill on follow-ups. |
| `slot_prompt_similarity` | `0.5` | Allows the slot to reuse prefix cache even when the prompt is not byte-identical (50% similarity threshold). Improves ACTIVE_MATCH hit rate. |
| `no_perf` | `true` | Suppresses per-request perf logging — reduces server-side log noise + minor CPU. |

These complement the existing `cache_type_k` / `cache_type_v` TurboQuant cache flags (`turbo3` default; `turbo4` evaluation pending). The TurboQuant KV-cache flags also compose with MTP speculative decoding (`--spec-type draft-mtp`) in the same llama.cpp binary.

## Spawn-time vs request-time — the critical distinction

| Layer | Examples | Reload behavior |
|---|---|---|
| **Spawn argv** (process-fork) | `flash_attn`, `no_context_shift`, `cache_reuse`, `slot_prompt_similarity`, `no_perf`, `ctx_size`, `cache_type_k`, `cache_type_v`, `n_gpu_layers`, `jinja` | **COLD-SPAWN ONLY** — manifest PUT does not affect running `llama-server`. Old cmdline persists until process exits. |
| **Request body** (per-call) | `temperature`, `top_p`, `reasoning_budget`, `top_k`, `stop`, `max_tokens` | **Hot** — applied per request through the forwarder. Manifest PUT changes affect the next request. |

The five doctrine flags above are **spawn argv**. Patching them on a running model requires one of:

- **Option A** — send any request to the same manifest tag with body `"keep_alive": 0` (Ollama-style; parsed at `chat_completion.py:parse_keep_alive`). Sets the slot's `IDLE_HOT` window to 0 → the running `llama-server` is torn down at the end of that request; the next request triggers a cold-spawn.
- **Option B** — wait for natural `IDLE_HOT` teardown (`idle_hot.remaining_s → 0`), then next request triggers cold-spawn.
- **Option C** — `docker restart <container>` (nuclear; recovers cleanly but interrupts in-flight requests).

Discovered 2026-05-19 post-PUT when `/proc/<pid>/cmdline` audit showed the old cmdline still bound on a model spawned before the manifest PUT. Option B chosen (waited ~3 min for natural teardown); next request spawned with the new flag set verified live.

## Verification recipe

```bash
# 1. Confirm manifest has the flag (post-PUT)
curl -s http://localhost:11401/api/manifests/qwen3.6-27b-dense | jq '.llama_server_flags'

# 2. Confirm /proc/<pid>/cmdline reflects the flag (post-cold-spawn)
docker exec <container> bash -c \
    'pgrep -af llama-server | head -1 | awk "{print \$1}" | xargs -I{} cat /proc/{}/cmdline | tr "\0" " "'

# Expect to see: --flash-attn ... --no-context-shift ... --cache-reuse 256 ... --slot-prompt-similarity 0.5 ... --no-perf
```

If `/api/manifests` shows the flag but `/proc/<pid>/cmdline` does not — the running slot is stale (manifest was patched after spawn). Trigger Option A/B/C and re-verify.

## Patching recipe (one model)

```bash
TAG=qwen3.6-27b-dense
BASE=http://localhost:11401

# 1. GET current manifest + ETag
ETAG=$(curl -sI ${BASE}/api/manifests/${TAG} | grep -i etag | awk '{print $2}' | tr -d '\r\n"')
curl -s ${BASE}/api/manifests/${TAG} > /tmp/manifest.json

# 2. Add the five flags to llama_server_flags (jq -e merges atomically in-memory)
jq '.llama_server_flags += {
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

- A model that triggers `shift_context` correctly may benefit from `no_context_shift: false` — but the default is `true` because the loop bug recurs faster than fixes.
- A model with no follow-up traffic pattern (one-shot batch use) gains nothing from `cache_reuse` + `slot_prompt_similarity` — safe to omit, but the small overhead from leaving them on is negligible.
- A model under active perf debugging may want `no_perf: false` to surface per-request timings — flip back after debug.

## See also

- [MULTI_AGENT_SHARING.md](./MULTI_AGENT_SHARING.md) — multi-agent serialization context.
- [PERSISTENCE_CHECKLIST.md](./PERSISTENCE_CHECKLIST.md) — manifest persistence (manifests live in `/var/lib/turbohaul/manifests/` YAML — see persistence audit for current bind-mount status).
- `src/turbohaul/manifest.py` `SAFE_LLAMA_FLAGS` — the in-code allowlist of accepted flags (~80 total).
