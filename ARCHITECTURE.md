# Turbohaul-Manager — Architecture & Design (v0.1 lock)

**Status:** Design locked 2026-05-16. Phase 0 done. Phase 1 in flight (this doc).
**Owner:** PL Claude (claude-pl) — Cmdr MrTrench directive.
**Forgejo:** http://10.244.136.121:3030/matthew/turbohaul-manager
**RC:** RC-TURBOHAUL-V1 (to be filed alongside this doc).

---

## 1. Mission

A **standalone HTTP inference server** that:
- Mimics the Ollama API surface (`/api/generate`, `/api/chat`, `/v1/chat/completions`, `/api/pull`, `/api/tags`) so any Ollama-aware client can swap us in transparently.
- Uses **Tom's Fork TurboQuant llama.cpp** (`github.com/TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`, MIT) as its inference backend — `llama-server` subprocess per active sidecar.
- Provides **BYOM** (Bring-Your-Own-Model) blob storage. Pull from Ollama registry / HuggingFace / arbitrary URL / local-file import.
- Provides a **FIFO request queue with grace period + idle hot-load** that solves the cross-process sidecar race the original manager exhibited.
- Plays in fleet (Logbook / GEAR / advisors) and outside-fleet (Open WebUI, other Ollama-aware tools) — purely BYOI from the consumer side.

## 2. Lineage

- **Replaces** `/mnt/AppData_SSD/Apps/sidecar-manager` (snapshot at Forgejo `matthew/sidecar-manager-archive`, commit `9eda7fa`). The old manager used preempt-based `/ensure` with a single asyncio.Lock — IN-PROCESS-only, races with concurrent callers (Secretary vs gear-review) across processes when one's `/ensure→chat-completion` interleaves with another's `/ensure`. Documented case: 2026-05-16 Wave 4 E2E hung 240s on this race.
- **Adopts shape from** Ollama (`github.com/ollama/ollama`, MIT, Forgejo mirror `matthew/ollama-mirror`).
- **Embeds binary from** Tom's Fork TurboQuant (Forgejo mirror `matthew/llamacpp-turboquant-mirror`).
- **Lives at** Forgejo `matthew/turbohaul-manager` (new repo, no license — proprietary OR open-source TBD).

## 3. Non-goals (v1)

- Not pulling the full Ollama registry catalog upfront. BYOM only.
- Not embedding llama.cpp via libllama bindings. **Subprocess llama-server only.**
- Not horizontal-scaling across machines. Single-host, single Python process.
- Not auth/authz/TLS. Assumes trusted network (ZeroTier + LAN). Future.
- Not Ollama Modelfile DSL. Just direct GGUF + per-model yaml.

## 4. Definitions

- **Sidecar** — one `llama-server` subprocess running a single model with explicit flags from its per-model manifest yaml.
- **Slot** — a queue position holding `{slot_id, model_tag, prompt, context, thread_id, status}`. Cold until activation.
- **Active sidecar** — the slot currently loaded into VRAM, running an inference request.
- **Grace period** — 60s (configurable) window after slot completion where the model stays loaded for `thread_id`-matched follow-ups. Follow-up during grace → instant. Follow-up after grace → re-queued at FIFO tail.
- **Idle hot-load** — 300s (configurable) window after the entire queue drains where the last model stays warm. Fresh request with that same model → instant. Fresh request with different model → swap (subprocess kill + new spawn).

## 5. Phase plan

| Phase | Scope | Status |
|---|---|---|
| 0 | Forgejo prep (Tom's Fork + Ollama + sidecar-manager-archive mirrors) + license audit | ✓ DONE 2026-05-16 |
| 1 | Architecture doc (this) + RBSRS critique (Architecture Brainstormer #12 + Devil's Advocate #13 + Failure Predictor #16 + Security Reviewer) | IN_PROGRESS — this doc commit + RBSRS pass pending |
| 2 | Core queue + slot manager + grace + idle hot-load + per-model manifest yaml + subprocess management. Pytest unit tests. Port 11401. | pending |
| 3 | Ollama-compat API + OpenAI-compat surface + `/status` + WebSocket `/ws/state` | pending |
| 4 | Blob store (sha256 content-addressed) + pull endpoints (Ollama registry / HF / URL / local) + delete | pending |
| 5 | Frontend (React+Vite, mounted by FastAPI at `/ui/*`, WebSocket-live, Config view with yaml editor) | pending |
| 6 | Dockerfile + docker-compose + smoke E2E (Secretary doc-to-ratecon via new manager) + ship to Forgejo + image tarball | pending |

## 6. State machine per request

```
RECEIVED  ──→  staging queue has room? ──no──┐
   │                                          │
   │ yes                                      │
   ▼                                  ACCEPT_BUFFER (FIFO, unbounded)
STAGED  ←─────────────────────────────────────┘
   │ slot allocated, model+prompt tagged, NOT loaded
   ▼
LOADING  ──→  subprocess llama-server spawned with manifest flags
   │ poll /health until 200
   ▼
ACTIVE  ──→  request streaming/responding to client
   │ chat-completion finish
   ▼
GRACE  ──→  60s window, thread_id-owned, model stays loaded
   │
   ├─→ follow-up with matching thread_id within 60s  ──→  back to ACTIVE
   │
   ▼ grace expires (no matching follow-up)
POPPED  ──→  SIGTERM llama-server, wait 5s, SIGKILL if alive, free VRAM
   │
   ├─→ next FIFO item in queue?  ──yes──→  back to STAGED
   │
   ▼ queue empty
IDLE_HOT  ──→  300s window, model stays loaded, no thread_id lock
   │
   ├─→ new request for same model_tag within 300s  ──→  back to ACTIVE
   │
   ├─→ new request for DIFFERENT model_tag within 300s  ──→  POPPED, swap, STAGED
   │
   ▼ idle expires
COLD  ──→  subprocess exited, slot gone, GPU 0 MiB
```

## 7. Config — top-level (yaml + env)

**Path:** `/etc/turbohaul/turbohaul.yaml` (host canonical), `/app/turbohaul.yaml` (container). Override via `$TURBOHAUL_CONFIG_YAML` env.

```yaml
# /etc/turbohaul/turbohaul.yaml
server:
  port: 11401                  # env: TURBOHAUL_PORT
  host: "0.0.0.0"

queue:
  max_parallel_sidecars: 1     # env: TURBOHAUL_MAX_PARALLEL  (CONTAK-01 = 1; bigger hosts can run N)
  staging_queue_depth: 100     # env: TURBOHAUL_STAGING_DEPTH (configurable to 1..N)
  acceptance_buffer_max: 10000 # env: TURBOHAUL_ACCEPT_MAX    (back-pressure ceiling for receive)
  grace_seconds: 60            # env: TURBOHAUL_GRACE_S
  idle_hot_load_seconds: 300   # env: TURBOHAUL_IDLE_HOT_S

storage:
  blob_store_path: /var/lib/turbohaul/blobs
  manifests_path: /var/lib/turbohaul/manifests
  state_db_path: /var/lib/turbohaul/state.sqlite

runtime:
  llama_server_binary: /opt/turboquant/build/bin/llama-server
  default_port_base: 11500     # llama-server child ports allocated from here

pull:
  hf_api_key_env: HF_API_KEY          # set externally; turbohaul reads at /api/pull-hf time
  ollama_registry: https://registry.ollama.ai
  pull_concurrency: 2                  # simultaneous downloads
  pull_chunk_size_mb: 64

ui:
  enabled: true
  static_path: /opt/turbohaul/ui_dist  # built FE artifacts mounted at /ui/*
```

All keys overridable by env var. Env beats yaml. Env vars use UPPER_SNAKE with `TURBOHAUL_` prefix.

## 8. Per-model manifest yaml (yaml only, NOT env vars)

**Path:** `/var/lib/turbohaul/manifests/<model_tag>.yaml`. ONE FILE PER MODEL. Edit via FE Config view (PUT `/api/manifests/{tag}`) OR direct edit on disk. Hot-reloaded by manager on next slot stage.

**Example — Qwen3.6-35B-A3B MoE Q4:**

```yaml
model_tag: qwen3.6-35b-moe
display_name: "Qwen 3.6 35B-A3B MoE Q4"
description: "Active 3B / total 35B sparse MoE. Q4 quant. KV-cache turbo4."
gguf_blob_sha256: 1a2b3c4d...              # points into blob store
gguf_size_bytes: 22000000000               # ~21 GB
context_size: 131072
expected_vram_bytes: 22500000000           # for VRAM-fit gating

llama_server_flags:
  ctx_size: 131072
  n_gpu_layers: 999
  cache_type_k: turbo4
  cache_type_v: turbo4
  flash_attn: 1
  threads: 8
  parallel: 1
  mlock: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.50
  no_perf: true
  sleep_idle_seconds: 300
  chat_template: peg-native
  jinja: true
  reasoning_format: deepseek
  # MoE-specific
  n_cpu_moe: false                          # Cmdr's example — control MoE expert RAM offload per-model
  # any flag llama-server accepts is allowed here

prompt_template:
  system_default: ""
  stop_tokens: ["<|im_end|>", "<|endoftext|>"]
```

**Schema is open:** any key in `llama_server_flags` is passed to `llama-server` as a CLI flag (`--<key-kebab-case>` with value). Unknown keys log a warning but don't fail. Flag mapping (yaml `snake_case` → CLI `kebab-case`) is deterministic; e.g., `n_cpu_moe: false` → `--n-cpu-moe false`. Boolean true with no value → `--<flag>` (no value). Booleans false → flag omitted entirely.

## 9. API surface (Ollama-compat + OpenAI-compat + extensions)

| Method + Path | Ollama-shape? | OpenAI-shape? | Purpose |
|---|---|---|---|
| `POST /api/generate` | ✓ | | Single-turn completion (Ollama-compatible) |
| `POST /api/chat` | ✓ | | Multi-turn chat with `thread_id` support |
| `POST /v1/chat/completions` | | ✓ | OpenAI-compatible (LiteLLM clients) |
| `POST /v1/completions` | | ✓ | OpenAI-compatible |
| `GET /api/tags` | ✓ | | List installed models from blob store |
| `GET /api/show` | ✓ | | Model details (manifest) |
| `GET /api/version` | ✓ | | Version info |
| `POST /api/pull` | ✓ | | Pull from Ollama registry |
| `POST /api/pull-hf` | (ext) | | Pull from HuggingFace (HF_API_KEY env) |
| `POST /api/pull-url` | (ext) | | Pull from arbitrary URL |
| `POST /api/import` | (ext) | | Import local GGUF file path |
| `DELETE /api/delete` | ✓ | | Remove model from blob store |
| `GET /status` | (ext) | | Queue depth, active sidecar, grace state, idle TTL, parallel-slots-in-use |
| `GET /api/manifests/{tag}` | (ext) | | Get per-model yaml |
| `PUT /api/manifests/{tag}` | (ext) | | Write per-model yaml (FE edit path) |
| `GET /api/config` | (ext) | | Get top-level yaml |
| `PUT /api/config` | (ext) | | Write top-level yaml (FE edit path; some fields require restart, flagged in response) |
| `WS /ws/state` | (ext) | | WebSocket — streams state transitions, queue updates, llama-server stderr |

**`thread_id` semantic (cross-cutting):**
- All POST endpoints accept optional `thread_id` (string, client-supplied). If absent on first request, manager generates UUID + returns in response.
- Follow-up requests within 60s grace with matching `thread_id` AND matching `model_tag` → instant on warm slot.
- Follow-up after grace expired → re-queued at FIFO tail like a fresh request.

## 10. Subprocess management

- Slot activation: `subprocess.Popen(['/opt/turboquant/build/bin/llama-server', '--port', str(slot_port), '-m', gguf_path, '--host', '127.0.0.1', *flag_args])`
- Each sidecar gets a unique port from `runtime.default_port_base` (e.g., 11500, 11501, ...) — only `max_parallel_sidecars` ever active.
- Health check: poll `http://127.0.0.1:<slot_port>/health` every 2s after spawn; ACTIVE when 200. Timeout 600s for cold load.
- Slot pop: `proc.terminate()` (SIGTERM) → wait 5s → `proc.kill()` (SIGKILL) if still alive → join. Free VRAM before next stage.
- Stdout/stderr captured to ring buffer per slot (last 1000 lines) for FE Logs view.

## 11. Front-end (matches BE EXACTLY)

**Stack:** React + Vite (matches Logbook fleet pattern). TypeScript. Tailwind for styling.
**Mount:** FastAPI serves built artifacts as static at `/ui/*` (single-port deploy, no nginx).
**Live:** WebSocket `/ws/state` drives all real-time views.

**Both BE and FE edit yaml** — FE goes through `PUT /api/config` + `PUT /api/manifests/{tag}` (BE owns disk writes; FE never writes filesystem directly). FE shows monaco-editor or codemirror for yaml with schema validation client-side + server-side. Some fields require restart to take effect — server response flags `restart_required: bool` and FE shows a banner "restart manager to apply".

**Views:**

| View | Path | Purpose |
|---|---|---|
| Dashboard | `/ui/` | Active sidecar (tokens/sec, n_decoded/n_predict, current request preview), queue depth gauge, recent throughput chart |
| Queue | `/ui/queue` | Full FIFO list with positions, model_tags, status (STAGED/LOADING/ACTIVE/GRACE), thread_ids, ETAs |
| Blob | `/ui/blob` | Installed models w/ sizes; Pull (Ollama/HF/URL); Import (local file); Delete |
| Config | `/ui/config` | Edit main yaml + edit per-model yamls; in-browser yaml editor; save → BE writes; restart-required fields flagged |
| Logs | `/ui/logs/{slot_port}` | Tail llama-server stderr for active slot |
| Settings | `/ui/settings` | About, version, links to LICENSE + THIRD_PARTY_LICENSES |

## 12. Storage layout

```
/var/lib/turbohaul/
├── blobs/
│   └── sha256/
│       ├── 1a/
│       │   └── 1a2b3c4d... (large file, no extension)
│       └── ...
├── manifests/
│   ├── qwen3.6-35b-moe.yaml
│   ├── qwen3.6-27b-dense.yaml
│   └── ...
├── conf/
│   └── turbohaul.yaml  (symlinked from /etc/turbohaul/turbohaul.yaml)
└── state.sqlite          (queue snapshot for cold-recovery, slot history)
```

## 13. Migration

- Phase 6 ships parallel: new manager on **port 11401**, old manager keeps running on 11400.
- Fleet stays on 11400 until soak ≥1 week (or Cmdr's call sooner).
- Cutover: update `secretary_providers.yaml` `pre_call_ensure.url` from `:11400/ensure` to `:11401/api/ensure-equiv` OR remove pre-call-ensure entirely (new manager queues internally — no `/ensure` needed; client just POSTs `/v1/chat/completions` and waits FIFO).
- Logbook + GEAR + advisor + any LiteLLM route table — point at 11401 once soaked.
- BYOI: all consumers point to env-var URL, not hard-coded.

## 14. Licensing & attribution

- Tom's Fork TurboQuant llama.cpp = **MIT** (verified on `feature/turboquant-kv-cache:LICENSE`, blob `e7dca554`, same ggml authors as upstream)
- Upstream llama.cpp (`ggml-org/llama.cpp`) = **MIT**
- Ollama (`ollama/ollama`) = **MIT** (we mimic API shape only, no source vendored)
- Our Turbohaul-Manager = **no current license** (proprietary or open-source — TBD by Cmdr)

**Required MODs from license audit (RBSRS 8.5/10):**
1. Ship `THIRD_PARTY_LICENSES` file in Docker image containing upstream MIT verbatim (covers both upstream + Tom's Fork — textually identical).
2. README attribution line: `"Inference backend: llama-server built from Tom's TurboQuant fork of llama.cpp (MIT). Ollama-compatible HTTP API surface."`
3. Trademark hygiene: use "Ollama-compatible" only (nominative fair use). Don't use "Ollama" or "Llama" wordmarks in product naming.
4. Optional: courtesy email to TheTom (the fork author) on first ship.

**Risk flags from audit:**
- Meta Llama AUP — if `/api/pull-hf` surfaces Llama-family weights, the 700M-MAU Llama Community License threshold applies. We're under the threshold; not distributing weights. LOW.

## 15. Open follow-ons / risks

- **VRAM-fit gating** — manifest declares `expected_vram_bytes`; manager refuses to stage a sidecar whose declared VRAM exceeds host capacity. Not strict-required for v1 but useful safety. **PHASE 4 deliverable.**
- **Failure modes during LOADING** — llama-server crashes during health-poll. Need: bounded retry (2 attempts), error to client, recycle slot, requeue. **PHASE 2 deliverable.**
- **Manifest write atomic** — `PUT /api/manifests/{tag}` must do atomic write (tempfile + rename) to avoid half-written yaml. **PHASE 3 deliverable.**
- **Cmdr's "1-min grace + 5-min idle hot-load" defaults** — these are seconds, easily adjustable per-env-var. Solid for fleet use (Secretary follow-ups, gear-review per-gear ensure). May need tuning per-deployment.
- **Per-model VRAM concurrency** — different models with combined VRAM under host capacity COULD run concurrent (e.g., a 7B + a 13B on 24GB). v1 = 1-at-a-time (max_parallel_sidecars=1 on CONTAK-01). Beefier hosts can increase. Tagging which slots can co-locate = future.
- **Cmdr's BYOI fleet-wide directive** — Logbook + GEAR + advisor all already config-driven. Audit at Phase 6 to confirm no hard-coded `192.168.50.128:11400` references.

## 16. RBSRS sub-agent records

| Phase | Domain | Role | Mode | Outcome | Verdict |
|---|---|---|---|---|---|
| 0 | license-audit-inference-stack | Documentation Reader (#4) as Software License Verification Specialist | red-hat | 8.5/10 | GO-WITH-MOD |
| 1 | architecture-design-inference-server | Architecture Brainstormer (#12) | human-thinker | pending | — |
| 1 | architecture-design-inference-server | Devil's Advocate (#13) | red-hat | pending | — |
| 1 | architecture-design-inference-server | Failure Predictor (#16) | red-hat | pending | — |
| 1 | security-review-public-inference | Security Reviewer | red-hat | pending | — |

## 17. References

- Forgejo repos:
  - http://10.244.136.121:3030/matthew/turbohaul-manager (THIS PROJECT)
  - http://10.244.136.121:3030/matthew/llamacpp-turboquant-mirror (Tom's Fork analysis copy)
  - http://10.244.136.121:3030/matthew/ollama-mirror (Ollama analysis copy)
  - http://10.244.136.121:3030/matthew/sidecar-manager-archive (old manager frozen snapshot)
- Upstream:
  - https://github.com/TheTom/llama-cpp-turboquant — Tom's Fork TurboQuant llama.cpp (branch `feature/turboquant-kv-cache`)
  - https://github.com/ollama/ollama — Ollama
  - https://github.com/ggml-org/llama.cpp — Upstream llama.cpp
- Internal docs:
  - SOP_Schema_Commit_Gate.md — applies to auth/governance edits (most of this build is non-auth)
  - SOP_Deploy_Doctrine.md — uvicorn-reload patterns
  - SOP_Sidecar_Idle_Unload_Watchdog.md — relevant prior art
  - feedback_sidecar_manager_queueing_via_asyncio_lock_intended_20260515.md — single-asyncio-Lock pattern (Cmdr-endorsed for current manager; supersedes for Turbohaul)
  - project_wave4_orphan_thread_fallback_20260516.md — the Wave 4 ship that triggered the cross-process race surfacing
- Project name: **Turbohaul-Manager** (Cmdr-approved 2026-05-16).

---

**End of v0.1 design lock.** Next: RBSRS Phase 1 critique → revisions → Phase 2 implementation start.
