# Turbohaul-Manager — Architecture & Design (v0.2)

**Status:** v0.2 design locked 2026-05-16 post architecture critique. Phase 0 ✓. Phase 1 ✓.
**Lineage:** v0.1 (commit d8ceb16) → v0.2 (THIS DOC, post-critique synthesis)
**Home:** https://github.com/MrTrenchTrucker/turbohaul-manager

**Design mandate:** *"Go the no-tech-debt route. A one-stop-shop for local inference using TurboQuant sidecars."*

v0.2 honors this by baking ALL critique must-fixes into the FOUNDATION, not deferred sections. No "Phase X will fix it later" patterns. Production-quality supervision, security hardening, and operational discipline land in Phase 2 from line one.

---

## 1. Mission

A **standalone HTTP inference server** that:
- Mimics the Ollama API surface (`/api/generate`, `/api/chat`, `/v1/chat/completions`, `/api/pull`, `/api/tags`) so any Ollama-aware client can swap us in transparently.
- Uses **the TurboQuant llama.cpp fork** (`github.com/TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`, MIT, pinned SHA `<TBD-Phase-2>`) as its inference backend — supervised `llama-server` subprocess per active sidecar.
- Provides **BYOM** (Bring-Your-Own-Model) blob storage. Pull from Ollama registry / HuggingFace (allowlist-pinned) / vetted URL / local-staging import.
- Provides a **FIFO request queue with grace + idle hot-load** that solves the documented cross-process sidecar race the original manager exhibited.
- One-stop-shop for **internal-network** local inference (multiple agent and service consumers). BYOI from consumer side via config-driven URL.

**v0.2 SCOPE NARROWING:** v1 ships for **trusted internal-network use only**. External use (Open WebUI users, untrusted clients) is **explicitly out-of-scope** for v1 and deferred to a v2 hardened build that adds bearer-auth + WS-scoping + TLS termination. This closes the §1-vs-§3 lane mismatch flagged in security review.

## 2. Lineage

- **Replaces** an earlier sidecar-manager. The old manager used preempt-based `/ensure` with a single in-process `asyncio.Lock` — IN-PROCESS-only, races with concurrent callers across processes. Documented case: an E2E run hung 240s on this race.
- **Adopts shape from** Ollama (`github.com/ollama/ollama`, MIT) — API surface only, no source vendored.
- **Embeds binary from** the TurboQuant llama.cpp fork, built in Dockerfile build-stage from pinned SHA.
- **Lives at** https://github.com/MrTrenchTrucker/turbohaul-manager.

## 3. Non-goals (v1)

- Not pulling the full Ollama registry catalog upfront. BYOM only.
- Not embedding llama.cpp via libllama bindings. **Supervised `llama-server` subprocess only.**
- Not horizontal-scaling across machines. Single-host, single Python process, **singleton-per-GPU enforced** (§3.1).
- **Not external-user-facing.** v1 targets trusted internal-network use. Open WebUI integration deferred to v2.
- Not Ollama Modelfile DSL. Just direct GGUF + per-model yaml.

### 3.1 Singleton Invariant (NEW v0.2)

**Load-bearing:** Turbohaul-Manager MUST be the **only writer** to GPU 0 on a given host. This is the entire reason this rebuild was justified. Promoted from non-goal to invariant.

Enforcement (Phase 2 baseline):
- At boot, acquire exclusive `fcntl.flock` on `/var/lib/turbohaul/state.sqlite`. If held by another process, refuse to start with explicit error pointing to the existing PID.
- At boot, scan `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` on GPU 0. If foreign `llama-server` processes are present AND not in our state.sqlite reconciliation map, refuse to start (or, with `--adopt-orphans` flag, kill them after a high-severity warning logged).
- A boot-time orphan reaper scans for `llama-server` children with `parent=1` and port in `runtime.default_port_base` range; reconciles against state.sqlite slot history; kills any unmatched (`SIGTERM` → 5s → `SIGKILL`).

### 3.2 Threat Model & Trust Boundary (NEW v0.2)

- **In-scope:** a trusted internal network (private mesh + LAN). All callers are known internal agents and services.
- **Out-of-scope for v1:** external untrusted users, Open WebUI multi-user deployments, public internet exposure.
- **Bind invariant:** `server.host` defaults to `127.0.0.1`. Operator can explicitly bind to a private-network interface IP via config; binding to `0.0.0.0` requires `--allow-public-bind` CLI flag AND logs a high-severity warning. **Never `0.0.0.0` by default.**
- **Auth posture:** v1 trusts the internal network. v2 will require bearer-auth on all mutating endpoints. v0.2 doc spec lays the groundwork so v2 can drop in without a structural rewrite.

## 4. Definitions & Initial Timing Constants

- **Sidecar** — one supervised `llama-server` subprocess running a single model with explicit flags from its per-model manifest yaml.
- **Slot** — a queue position holding `{slot_id, model_tag, prompt, context, thread_id, status}`. Cold until activation.
- **Active sidecar** — the slot currently loaded into VRAM, running an inference request.
- **Grace period** — window after slot completion where the model stays loaded for `thread_id`-matched follow-ups.
- **Idle hot-load** — window after the entire queue drains where the last model stays warm for any same-model-tag fresh request.

**Initial timing defaults (v0.2):**

| Constant | v0.1 spec | v0.2 initial default | Rationale |
|---|---|---|---|
| `grace_seconds` | 60 | **30** | Conservative. We ship 30 and instrument; if traffic data justifies 60, bump in v0.3 config-only change. |
| `idle_hot_load_seconds` | 300 | **120** | Conservative. We ship 120 and instrument; same rationale. |
| `max_grace_extensions` | (not specified) | **5** | Starvation cap — after 5 consecutive grace-renewals on the same thread, force re-queue at FIFO tail. Prevents a single noisy client from starving other clients. |
| `loading_health_timeout_s` | 600 | **600** | Unchanged — cold load of 21GB GGUF can take 5-8min legitimately. |
| `drained_sigterm_window_s` | 5 | **15 (active slot) / 5 (cold)** | See §10 — 5s on a mid-decode 21GB sidecar leaves CUDA allocator stuck. |

**Phase 2 deliverable:** instrument the current manager for 1 week to log time-between-requests-same-thread + time-between-requests-same-model-diff-thread + model-swap-frequency + request-burst-size. Use those measurements to tune defaults in v0.3 (config-only PR, no code change).

## 5. Phase Plan (v0.2)

| Phase | Scope | Status | Time est |
|---|---|---|---|
| 0 | repo prep (analysis mirrors + new repo) + license audit (all MIT) | ✓ DONE 2026-05-16 | done |
| 1 | Architecture v0.1 doc + design critique + v0.2 synthesis (this doc) | ✓ DONE 2026-05-16 | done |
| 2 | **Core queue + slot manager + ALL supervision discipline baked in from line one:** state machine + subprocess setsid/killpg + orphan reaper + flag allowlist + tag validation + atomic manifest writes + VRAM-fit pre-check + ETag/If-Match yaml writes + singleton invariant + WS redaction. Pytest unit + integration tests. Port 11401. Bind 127.0.0.1 + private-network IP. | pending | **4-6 days** (was 3-5, +1-2 for baked-in supervision) |
| 3 | Ollama-compat + OpenAI-compat API surface + `/status` + WebSocket `/ws/state` (redacted) + GET/PUT /api/manifests + GET/PUT /api/config (split mutable/boot fields) + thread_id prefix-hash fallback for naive Ollama clients | pending | 2-3 days |
| 4 | Blob store (content-addressed + read-only post-rename) + pull endpoints with SSRF guard + import_allowed_root + GGUF-magic check + re-verify-on-stage | pending | 3-5 days |
| 5 | Frontend React+Vite + CSP + monaco/codemirror sandboxed + WebSocket-live + text-only manifest rendering | pending | 3-4 days |
| 6 | Dockerfile (pinned the TurboQuant llama.cpp fork SHA in build-stage) + docker-compose + THIRD_PARTY_LICENSES + README attribution + Phase 6a **shadow mode** soak + cutover runbook + smoke E2E | pending | 1-2 days |
| 6a | Shadow-mode soak (parallel 11401/11400, traffic teed, delta-logged) — 24-48h before any cutover. Success criteria + rollback runbook documented in §13. | pending | 2-3 days within Phase 6 |

Total: ~3-4 weeks calendar (v0.1 said 2-3 weeks; +1 week for hardening). Worth it for no-tech-debt.

## 6. State Machine per Request (v0.2 — expanded)

```
RECEIVED  ──→  acceptance buffer has room? ──no──┐ (back-pressure 503 retry-after)
   │                                              │
   │ yes                                          │
   ▼                                  ACCEPT_BUFFER (FIFO, bounded by acceptance_buffer_max)
STAGED  ←─────────────────────────────────────────┘
   │ slot allocated, model+prompt tagged, NOT loaded; VRAM-fit pre-check passes
   ▼
LOADING  ──→  subprocess spawned (setsid, supervised), poll /health
   │ ├─→ health 200 within loading_health_timeout_s ──→ ACTIVE
   │ └─→ timeout OR crash ──→ LOADING-FAIL (bounded retry, max 2 attempts; on exhaust, 500 to client + recycle slot)
   ▼
ACTIVE  ──→  request streaming/responding to client
   │ chat-completion finish
   ▼
GRACE  ──→  grace_seconds window, thread_id-owned, model stays loaded
   │
   ├─→ follow-up with matching thread_id within window ──→ GRACE-BUSY (NEW)
   │
   └─→ no matching follow-up within window ──→ POPPED

GRACE-BUSY (NEW v0.2)  ──→  matched follow-up running on the warm slot
   │
   ├─→ further matched follow-ups arrive: queue HEAD-OF-LINE FIFO behind current
   │
   ├─→ extension count <= max_grace_extensions ──→ on completion, back to GRACE
   │
   └─→ extension count > max_grace_extensions ──→ on completion, POPPED (force fairness)

ACTIVE-MATCH (NEW v0.2)  ──→  matched-thread arrival mid-stream
   │
   └─→ queues at FIFO HEAD (not tail) with bounded wait. Document: order on warm slot

POPPED  ──→  drained-SIGTERM (15s for ACTIVE/GRACE/GRACE-BUSY, 5s for IDLE_HOT/cold), then SIGKILL on process group via killpg, then VRAM-clear verification (nvidia-smi memory.used drops below threshold OR 30s timeout → manager-level alert)
   │
   ├─→ next FIFO item in queue?  ──yes──→  STAGED
   │
   ▼ queue empty
IDLE_HOT  ──→  idle_hot_load_seconds window, model stays loaded
   │
   ├─→ new request for same model_tag ──→ ACTIVE
   │
   ├─→ new request for DIFFERENT model_tag ──→ POPPED, swap, STAGED
   │
   ▼ idle expires
COLD  ──→  subprocess exited, slot gone, GPU 0 MiB

COLD-RECOVERY (NEW v0.2)  ──→  on manager boot, scan for orphaned llama-server processes (parent=1, port in default_port_base range), reconcile against state.sqlite slot history. Unmatched orphans: SIGTERM 5s → SIGKILL. Then proceed to accept traffic.
```

**Failure handling:**
- LOADING-FAIL retry: 2 attempts. On exhaustion, slot recycled + 500 returned to client + audit event `slot_loading_exhausted`.
- VRAM-clear timeout in POPPED: if `nvidia-smi memory.used` doesn't drop below threshold within 30s, manager logs `vram_clear_failed`, does NOT auto-restage (would compound failure), and surfaces alert. Operator action required.
- thread_id is a **routing hint, not auth**. Matching key = `(thread_id, source_ip)` pair to bound forge-attack surface. Optional v2 HMAC-signed thread_ids.

## 7. Config — Top-Level (v0.2 — split BootConfig vs RuntimeConfig)

**Path:** `/etc/turbohaul/turbohaul.yaml` (host canonical), `/app/turbohaul.yaml` (container). Override via `$TURBOHAUL_CONFIG_YAML` env.

```yaml
# /etc/turbohaul/turbohaul.yaml

# === BOOT CONFIG (read-only at runtime; restart required to change) ===
server:
  host: "127.0.0.1"           # env: TURBOHAUL_HOST  (NEVER 0.0.0.0 default — see §3.2)
  port: 11401                  # env: TURBOHAUL_PORT
  allow_public_bind: false     # if true, allows 0.0.0.0 (only with --allow-public-bind CLI flag too)

storage:
  blob_store_path: /var/lib/turbohaul/blobs
  manifests_path: /var/lib/turbohaul/manifests
  import_allowed_root: /var/lib/turbohaul/import-staging  # /api/import sandboxed here only
  state_db_path: /var/lib/turbohaul/state.sqlite

runtime:
  llama_server_binary: /opt/turboquant/build/bin/llama-server
  llama_server_binary_sha256: <PINNED-IN-PHASE-2>   # verified at boot; mismatch = refuse to start
  default_port_base: 11500     # llama-server child ports allocated from here

ui:
  enabled: true
  static_path: /opt/turbohaul/ui_dist

# === RUNTIME CONFIG (mutable via PUT /api/config; takes effect on next slot stage or per-field) ===
queue:
  max_parallel_sidecars: 1     # env: TURBOHAUL_MAX_PARALLEL
  staging_queue_depth: 100     # env: TURBOHAUL_STAGING_DEPTH
  acceptance_buffer_max: 10000 # env: TURBOHAUL_ACCEPT_MAX
  grace_seconds: 30            # env: TURBOHAUL_GRACE_S
  idle_hot_load_seconds: 120   # env: TURBOHAUL_IDLE_HOT_S
  max_grace_extensions: 5      # env: TURBOHAUL_MAX_GRACE_EXT
  loading_health_timeout_s: 600
  drained_sigterm_window_active_s: 15
  drained_sigterm_window_cold_s: 5

pull:
  hf_api_key_env: HF_API_KEY                            # name of env var holding key (NOT value)
  hf_host_allowlist: ["huggingface.co", "hf.co"]        # exact hosts + subdomains only
  pull_url_https_only: true
  pull_concurrency: 2
  pull_chunk_size_mb: 64
  per_stream_max_bytes: 107374182400                    # 100 GB hard ceiling per pull stream
```

### 7.1 Config Write Protection (NEW v0.2)

- `PUT /api/config` accepts ONLY runtime-mutable fields (anything under `queue:`, `pull:`).
- Boot fields (`server.*`, `storage.*`, `runtime.*`, `ui.*`) attempted via PUT return **HTTP 403** with `{"detail":"field <name> is boot-only; restart manager to change"}`.
- Pydantic enforcement: two layers — `BootConfig` (frozen on load, no setters exposed) vs `RuntimeConfig` (mutable, PUT-able). Schema validation REJECTS unknown keys (no silent acceptance).
- `runtime.llama_server_binary` + `runtime.llama_server_binary_sha256` verified at boot via `sha256sum` — mismatch refuses to start. Prevents config-driven binary swap attack.

## 8. Per-Model Manifest Yaml (v0.2 — ALLOWLIST schema)

**Path:** `/var/lib/turbohaul/manifests/<model_tag>.yaml`. ONE FILE PER MODEL. Edit via FE Config view (PUT `/api/manifests/{tag}` with `If-Match` ETag) OR direct edit on disk (with inotify watchdog alert; see §8.2). Hot-reloaded on next slot stage.

**Example — Qwen3.6-35B-A3B MoE Q4:**

```yaml
model_tag: qwen3.6-35b-moe
display_name: "Qwen 3.6 35B-A3B MoE Q4"            # SANITIZED-AS-TEXT-ONLY in FE rendering — see §11.2
description: "Active 3B / total 35B sparse MoE. Q4 quant. KV-cache turbo4."
gguf_blob_sha256: 1a2b3c4d...                       # must match a blob in /var/lib/turbohaul/blobs/sha256/
gguf_size_bytes: 22000000000
context_size: 131072
expected_vram_bytes: 22500000000                    # MANDATORY (Phase 2 — was Phase 4 in v0.1)
revision: 1                                          # auto-incremented on every write; powers ETag/If-Match

llama_server_flags:                                 # CLOSED ALLOWLIST per v0.2 §8.1
  # Performance + memory layout
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
  # Chat template
  chat_template: peg-native
  jinja: true
  reasoning_format: deepseek
  # MoE-specific
  n_cpu_moe: false

prompt_template:
  system_default: ""
  stop_tokens: ["<|im_end|>", "<|endoftext|>"]
```

### 8.1 Tag Validation + Manifest Write Safety (NEW v0.2)

- `model_tag` MUST match regex `^[a-z0-9][a-z0-9._-]{0,63}$`. Case-insensitive lowercase ASCII only. No `..`, no `/`, no `\`, no NUL, no leading dot, max 64 chars.
- All manifest reads/writes pass through `os.path.realpath()` and assert resulting path is within `manifests_path`. Symlink escape = 400 Bad Request.
- Directories created `0o700`, files `0o600`.
- **llama_server_flags is a CLOSED ALLOWLIST.** Unknown keys REJECTED (400, no silent acceptance). Path-bearing flags **explicitly denied**: `mmproj`, `lora`, `lora-base`, `lora-scaled`, `grammar-file`, `json-schema-file`, `log-file`, `slot-save-path`, `chat-template-file`, `in-prefix-file`, `in-suffix-file`, `hf-token`, `override-kv`, `cache-prompt-file`, `binary-override` (any flag taking a filesystem path).
- The allowlist lives in `app/turbohaul/manifest_schema.py` Pydantic model with `Extra.forbid`. Adding a new flag requires a code change + PR, not a yaml edit.

### 8.2 Atomic Manifest Writes + Concurrency (NEW v0.2)

- All writes use tempfile-in-same-dir + `fsync(file)` + `rename` + `fsync(dir)` to be POSIX-atomic on ext4. **Promoted from Phase 3 to Phase 2 baseline.**
- Concurrency: PUT requires `If-Match: "<revision>"` header matching current on-disk revision. Mismatch returns **HTTP 412 Precondition Failed** with current yaml body for client-side merge. FE on 412 shows "manifest changed on disk since you opened it — reload to re-apply" banner. (Cross-ref §9 PUT semantics.)
- Out-of-band disk edits (an operator SSHes in and edits yaml directly) trigger an inotify watchdog that increments `revision` automatically + writes audit event. FE shows "manifest changed on disk by external editor" notice on next GET. Hot-reload still applies on next stage.

## 9. API Surface (v0.2 — runtime-mutable PUT semantics + thread_id prefix-hash fallback)

| Method + Path | Ollama-shape? | OpenAI-shape? | Purpose | v0.2 changes |
|---|---|---|---|---|
| `POST /api/generate` | ✓ | | Single-turn (Ollama) | Optional thread_id; auto-derive from prompt-prefix-hash for naive Ollama clients |
| `POST /api/chat` | ✓ | | Multi-turn w/ thread_id | Same auto-derive fallback |
| `POST /v1/chat/completions` | | ✓ | OpenAI | Same |
| `POST /v1/completions` | | ✓ | OpenAI | Same |
| `GET /api/tags` | ✓ | | List models | |
| `GET /api/show` | ✓ | | Model details | Returns sanitized manifest (display_name + description AS TEXT — no HTML; see §11.2) |
| `GET /api/version` | ✓ | | Version | Returns `{"version": "1.0.0", "backend": "turboquant-llama-cpp@<pinned-sha>", "api_compat": "ollama-superset", "User-Agent": "Turbohaul-Manager/1.0.0 (Ollama-compatible)"}` |
| `POST /api/pull` | ✓ | | Pull from Ollama registry | Subject to §9.1 SSRF guard |
| `POST /api/pull-hf` | (ext) | | HuggingFace pull | §9.1 — hostname must match `hf_host_allowlist` |
| `POST /api/pull-url` | (ext) | | Arbitrary URL pull | §9.1 — https-only + DNS-pin + private-IP blocklist |
| `POST /api/import` | (ext) | | Local-file import | §9.2 — path must be under `import_allowed_root`; GGUF magic check |
| `DELETE /api/delete` | ✓ | | Remove model | |
| `GET /status` | (ext) | | Queue + active + grace + idle state | Schema in §9.3 |
| `GET /api/manifests/{tag}` | (ext) | | Get per-model yaml | Returns `ETag: "<revision>"` header |
| `PUT /api/manifests/{tag}` | (ext) | | Write per-model yaml | Requires `If-Match: "<revision>"`; 412 on mismatch; flags allowlist enforced |
| `GET /api/config` | (ext) | | Get top-level yaml | Returns only fields the requester is allowed to see (BootConfig is read-only-visible) |
| `PUT /api/config` | (ext) | | Write runtime fields only | Boot fields → 403; unknown keys → 400 |
| `WS /ws/state` | (ext) | | State events (REDACTED) | §11.1 — no prompts, no responses, no stderr |
| `GET /api/logs/{slot_port}` | (ext) | | Admin-only stderr ring buffer | Separate from /ws/state; future v2 auth-gated |

**`thread_id` semantic (v0.2):**
- All POST endpoints accept optional `thread_id` (string, client-supplied). Well-behaved clients supply an explicit thread_id.
- **For naive Ollama clients** (Open WebUI, generic Ollama-aware tools) that don't supply thread_id: manager auto-derives thread_id from SHA-256 hash of first N tokens of prompt + model_tag. Same-prefix follow-ups → same thread_id → warm slot. Different prompt → new thread_id → fresh queue entry. This makes the "Ollama compat" claim real, not aspirational.

### 9.1 Pull-URL Safety (NEW v0.2)

- URL scheme: **`https://` only**. `http`, `file`, `ftp`, `gopher`, `dict`, `ldap`, `data` REJECTED with 400.
- Hostname resolved ONCE up front; resulting IP MUST NOT be in:
  - RFC1918 (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
  - Loopback (`127.0.0.0/8`)
  - Link-local (`169.254.0.0/16`)
  - IPv6 link-local (`fe80::/10`), unique-local (`fc00::/7`)
  - Private overlay-network range (`10.244.0.0/16`)
  - NAT64 (`64:ff9b::/96`) — bypass class
  - IPv4-compat IPv6 (`::/96`) — bypass class
  - IPv4-mapped IPv6 (`::ffff:0:0/96`)
- Connection MUST verify peer IP matches the pre-resolved IP (defense against DNS rebinding).
- Redirects: NOT followed by default. If `follow_redirects: true` is set in request body, each redirect URL re-validated through the same pipeline.
- `HF_API_KEY` injected as `Authorization: Bearer <key>` ONLY when hostname matches `pull.hf_host_allowlist` (default `huggingface.co`, `hf.co`). For any other host, header omitted.
- Per-stream byte ceiling (`pull.per_stream_max_bytes`, default 100 GB). Stream closed + tempfile deleted on ceiling.
- All pull requests audit-logged to `state.sqlite` (`{requester, url, resolved_ip, status, bytes, sha256, started, finished}`).

### 9.2 Import Safety (NEW v0.2)

- `POST /api/import { "path": "..." }` accepts paths ONLY under `storage.import_allowed_root` (default `/var/lib/turbohaul/import-staging/`).
- Path passes through `os.path.realpath()` and asserted under `import_allowed_root`. Symlinks rejected (file is opened with `O_NOFOLLOW`).
- Explicit denylist: `/proc/`, `/sys/`, `/dev/`, `/etc/`, `/root/`, `/var/run/`, anywhere outside `import_allowed_root`.
- First 4 bytes of file MUST be `GGUF` (magic check). Reject non-GGUF early.
- Streamed copy with per-blob size cap (`manifest.gguf_size_bytes + 1MB` if manifest known, else 100 GB hard ceiling).

### 9.3 `/status` Response Schema (NEW v0.2)

```json
{
  "queue": {
    "acceptance_buffer_depth": 0,
    "staging_queue_depth": 3,
    "staging_queue_max": 100
  },
  "active": {
    "slot_id": "slot-abc",
    "model_tag": "qwen3.6-35b-moe",
    "state": "ACTIVE",
    "thread_id": "<sanitized — first 8 chars only>",
    "tokens_per_sec": 17.94,
    "n_decoded": 1234,
    "n_predict": 16384
  },
  "grace": {
    "remaining_s": 22,
    "extension_count": 1,
    "max_extensions": 5
  },
  "idle_hot": null,
  "parallel_slots": {
    "used": 1,
    "max": 1
  },
  "host": {
    "vram_used_mib": 22480,
    "vram_total_mib": 24576,
    "uptime_s": 86400
  }
}
```

## 10. Subprocess Management (v0.2 — REWRITE)

Production-quality supervision baked in from line one (Phase 2):

**Spawn:**
- `subprocess.Popen([binary, '--port', str(slot_port), '-m', gguf_path, '--host', '127.0.0.1', *flag_args], start_new_session=True, env={...sanitized env...})`
- `start_new_session=True` (Linux `setsid()`) puts the child in its own process group → enables clean group teardown.
- argv built from list, never shell-string. Each flag value rejected if it contains `\0`, `\n`, leading `-` (unless explicit allowlist entry), shell metacharacters.

**Health-poll:**
- Poll `http://127.0.0.1:<slot_port>/health` every 2s after spawn. ACTIVE on 200.
- Cold-load timeout `loading_health_timeout_s` (default 600s). On timeout, LOADING-FAIL retry path (§6).
- the TurboQuant llama.cpp fork health-contract drift defense: response shape verified — must include expected JSON fields. Schema breakage triggers `loading_health_schema_mismatch` audit event.

**Pop (drained-SIGTERM):**
- For ACTIVE / GRACE / GRACE-BUSY slots: poll `/health?drained=1` (or wait for in-flight stream to complete with bounded `drain_max_s` of 10s); then SIGTERM the process group via `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)`; wait `drained_sigterm_window_active_s` (default 15s); SIGKILL group on timeout.
- For cold / IDLE_HOT slots: immediate SIGTERM via killpg + `drained_sigterm_window_cold_s` (default 5s) wait + SIGKILL.
- **VRAM-clear verification:** poll `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` until `memory.used` drops by at least 90% of `expected_vram_bytes`/MiB OR 30s timeout. Timeout = `vram_clear_failed` audit event; do NOT auto-restage.

**Stdout/stderr:**
- Captured via `asyncio.subprocess` pipes to per-slot ring buffer (last 1000 lines each).
- **Never broadcast to /ws/state.** Available only via `GET /api/logs/{slot_port}` (admin-only future v2).

**Boot-time reconciliation (COLD-RECOVERY in §6):**
- Scan `/proc/*/status` for `llama-server` processes with `PPid: 1` (orphaned to init).
- Scan `nvidia-smi --query-compute-apps=pid` for foreign GPU compute processes.
- For each orphan in `default_port_base` range OR holding GPU memory: cross-reference state.sqlite. If state.sqlite indicates a clean shutdown for that slot OR no record, kill the orphan (SIGTERM 5s → SIGKILL).
- After reconciliation, manager opens flock on state.sqlite and begins accepting traffic.

**Process supervision under FastAPI reload:**
- uvicorn `--reload` is BANNED in production. Production uses `docker restart` for code changes.
- On manager SIGTERM (graceful shutdown): atexit handler + signal handler iterates child slots, calls drained-SIGTERM on each, waits up to 60s for VRAM clear, then exits.

## 11. Front-End (v0.2 — matches BE; CSP + sandboxing added)

**Stack:** React + Vite + TypeScript + Tailwind. Single-port deploy (FastAPI static mount `/ui/*`). WebSocket-live state.

**Both BE and FE edit yaml** — FE goes through `PUT /api/config` + `PUT /api/manifests/{tag}` with `If-Match` ETag. BE owns disk writes. FE never writes filesystem directly. Concurrent-write protection per §8.2.

**Views:**

| View | Path | Purpose |
|---|---|---|
| Dashboard | `/ui/` | Active sidecar (tokens/sec, n_decoded/n_predict, thread_id-first-8), queue depth gauge, throughput chart |
| Queue | `/ui/queue` | FIFO list with positions, model_tags, status, thread_id-first-8, ETAs |
| Blob | `/ui/blob` | Installed models w/ sizes; Pull (Ollama/HF/URL) — submit form goes through §9.1; Import (path under import_allowed_root); Delete |
| Config | `/ui/config` | Main yaml editor + per-model yaml editor (monaco-editor sandboxed). PUT goes through §7.1 / §8.1. ETag-aware UX (412 = "reload + re-apply" banner). Restart-required field flagging. |
| Logs | `/ui/logs/{slot_port}` | Admin-only stderr tail via GET /api/logs/{slot_port} |
| Settings | `/ui/settings` | About, version, links to LICENSE + THIRD_PARTY_LICENSES |

### 11.1 WebSocket Redaction Policy (NEW v0.2)

- `/ws/state` broadcasts **system-level events only**:
  - `queue_change` — depth only, no thread_ids, no model_tags of pending
  - `slot_transition` — slot_id + state transition (STAGED→LOADING→ACTIVE→GRACE→POPPED) + model_tag (visible — non-secret) + thread_id-first-8 (sanitized)
  - `model_pull_progress` — pull task_id + bytes_done/total + status
  - `manager_health` — vram_used_mib, vram_total_mib, parallel_slots_used
- **NEVER on /ws/state:** prompt text, response text, llama-server stderr lines, full thread_ids, full GGUF paths, HF_API_KEY-related events, audit events containing IP addresses.
- Connection scope: caller may subscribe to all system events (default) OR scope to a specific thread_id by supplying it at upgrade-time AND a shared-secret in header (`X-Turbohaul-WS-Token`). Scoped subscriptions still get redacted content.
- llama-server stderr ring buffer is **admin-only** — separate `GET /api/logs/{slot_port}` endpoint (future v2 auth-gated). NOT broadcast.

### 11.2 FE Rendering Policy (NEW v0.2)

- ALL manifest-sourced strings rendered as **text content only** (React `{value}` — never `dangerouslySetInnerHTML`).
- `display_name`, `description`, `chat_template`, `model_tag`, `prompt_template.system_default` — TEXT, never HTML/JSX/Markdown-with-HTML.
- **CSP header on /ui/* responses:**
  ```
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws://localhost:11401 ws://127.0.0.1:11401; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'
  ```
  (no `unsafe-eval`, no external CDN, no inline scripts.)
- monaco-editor / codemirror **self-hosted** (no CDN). Subresource Integrity hashes pinned. Editor configured `readOnly: false` for editing but no `eval` extensions enabled.
- SPA wildcard fallback returns `index.html` ONLY for paths matching `^/ui/[^?#]*$` — paths with `..`, `?`, `#`, or matching `^/api/`, `^/ws/`, `^/status` are NOT SPA-fallback'd.
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin` headers on /ui/* responses.

### 11.3 Network-Perimeter Security Model

Turbohaul-Manager operates under a **network-perimeter trust** model: every HTTP endpoint surfaced by the manager (`/api/config`, `/api/chat`, `/v1/chat/completions`, `/api/pull-*`, `/api/import`, `/v1/logging`, and the `/ws/state` WebSocket) runs **without app-layer authentication**. The trust boundary is the bind address — production deployments bind to `127.0.0.1:11401` or to a private-network interface only, and the operator is responsible for ensuring nothing outside that perimeter reaches the listener. Adding asymmetric auth on a single read-only surface (e.g., `/v1/logging`) without also gating higher-sensitivity write surfaces (`/api/config` PUT, `/api/pull-url`) would be strictly worse than the current uniform posture — an attacker reaching the listener already commands higher-impact surfaces. The redaction discipline documented in §11.1 (no prompts/responses/PII in audit payloads or `/ws/state` events) is the load-bearing protection against accidental data exposure; the `EventBus.REDACTED_KEYS` denylist is a tripwire, not the primary defense. Future v2 deployments that need cross-perimeter access SHALL introduce an external reverse-proxy (Caddy/nginx) with bearer-token auth applied uniformly to all paths, not piecemeal in-app gating.

## 12. Storage Layout

```
/var/lib/turbohaul/
├── blobs/
│   └── sha256/
│       ├── incoming/                 # tempfiles during /api/pull-* downloads
│       │   └── <random>.tmp
│       └── <ab>/                     # 2-char prefix shard
│           └── <full-sha256-hash>    # 0o400, immutable after rename
├── manifests/
│   ├── qwen3.6-35b-moe.yaml          # 0o600
│   └── ...
├── import-staging/                    # NEW v0.2 — sandbox for /api/import (0o700)
├── conf/
│   └── turbohaul.yaml                 # symlinked from /etc/turbohaul/turbohaul.yaml
└── state.sqlite                        # 0o600, flock'd for singleton
```

### 12.1 Blob Lifecycle (NEW v0.2)

1. **Pull stream:** `POST /api/pull-*` streams response body to `blobs/sha256/incoming/<random>.tmp` with per-stream byte ceiling enforced.
2. **Post-disk hash verify:** on stream completion, compute sha256. If matches `gguf_blob_sha256` (when caller supplied) OR the just-computed hash is the new canonical: proceed. Mismatch: delete tempfile, return 400.
3. **Atomic rename:** `os.rename(tempfile, blobs/sha256/<ab>/<full-hash>)` (atomic on ext4).
4. **Immutable lock:** `chmod 0o400` on final blob; on Linux, optionally `chattr +i` (immutable) for tamper-evident defense.
5. **Re-verify on stage:** every time a sidecar stages from a blob (§6 STAGED transition), the manager re-hashes the blob OR (perf-optimized) cross-checks against state.sqlite's last-verified-mtime + sha256. Mismatch = `blob_integrity_failed` audit + refuse to stage + alert. Defends against TOCTOU blob swap.
6. **Disk-quota:** `blob_store_path` filesystem MUST be on a dedicated mount with quota; Phase 2 audit confirms separate fs.

## 13. Migration (v0.2 — REWRITE)

### 13.1 Soak Strategy (Phase 6a — shadow mode)

- Phase 6 ships turbohaul-manager on **port 11401, bound to 127.0.0.1 + private-network IP**. Old manager keeps running on 11400 unchanged.
- **Phase 6a (NEW v0.2):** SHADOW MODE — turbohaul has a `mirror_from_port: 11400` runtime config. When set, turbohaul subscribes to 11400 incoming traffic via a small tee proxy; runs every request through its own queue; discards turbohaul's response (returns 11400's response to client); logs delta to `state.sqlite` (latency, output-token-count, first-token-latency, any errors).
- Shadow mode runs **24-48 hours** before any real cutover. Operator reviews delta logs.

### 13.2 Soak-Pass Criteria (must-pass before cutover)

- Zero unrecovered 5xx for **7 consecutive days**.
- p95 first-token-latency within **+10%** of 11400 baseline.
- Zero orphaned subprocess events (per state.sqlite reconciliation log).
- Zero VRAM-leak events after **100+ slot pops**.
- Zero `vram_clear_failed`, `blob_integrity_failed`, `loading_health_schema_mismatch` audit events.

### 13.3 Cutover

- Cutover edits the consumer provider config's `pre_call_ensure.url` from `:11400/ensure` to NOTHING (turbohaul queues internally — no /ensure pre-call needed). Downstream consumer route tables update to env-var-driven URL pointing at 11401.
- **Hard cutover deadline:** Day-14 from Phase 6 ship. If turbohaul isn't primary by Day-14, treat as a failed migration and roll back. No indefinite parallel-deploy.

### 13.4 Rollback Runbook

- Single command revert: `sed -i 's|11401|11400|g' /etc/consumer_providers.yaml && docker restart <consumer-services>`. (Phase 6 deliverable: actual script `/opt/turbohaul/rollback.sh`.)
- Verification probe: `curl http://localhost:11400/health` returns 200 + recent request lands on old manager (verify via old manager logs).
- Turbohaul-manager continues running on 11401 for post-mortem inspection. Drain queue gracefully (`POST /api/admin/drain` — Phase 6 endpoint).

## 14. Licensing & Attribution (unchanged from v0.1)

- the TurboQuant llama.cpp fork = **MIT** (verified on `feature/turboquant-kv-cache:LICENSE`, blob `e7dca554`)
- Upstream llama.cpp = **MIT**
- Ollama = **MIT** (API shape only, no source vendored)
- Our Turbohaul-Manager = **MIT** (open-source)

**MODs from the license audit:**
1. `THIRD_PARTY_LICENSES` file in Docker image with upstream MIT verbatim
2. README attribution: `"Inference backend: llama-server built from Tom's TurboQuant fork of llama.cpp (MIT). Ollama-compatible HTTP API surface."`
3. Trademark hygiene: "Ollama-compatible" only (nominative fair use)
4. Optional: courtesy email to TheTom on first ship

## 15. Closed Risks (was "Open follow-ons" in v0.1 — now resolved in v0.2)

| v0.1 open item | v0.2 status |
|---|---|
| VRAM-fit gating deferred to Phase 4 | ✓ MOVED to Phase 2 baseline + `expected_vram_bytes` MANDATORY in manifest |
| LOADING failure retry undefined | ✓ §6 LOADING-FAIL bounded retry (2 attempts) |
| Manifest write atomicity deferred to Phase 3 | ✓ §8.2 — MOVED to Phase 2 + ETag/If-Match concurrency |
| "1-min grace + 5-min idle hot-load defaults" untested numbers | ✓ §4 — conservative 30s/120s initial; instrument current manager 1 week → tune in v0.3 |
| Per-model VRAM concurrency (multi-model on >24GB hosts) | Future v1.1 — out of scope for v1 ship |
| BYOI consumer audit (hard-coded 11400 references) | Phase 6 deliverable; cross-check downstream consumer configs |

## 16. References

- Upstream:
  - https://github.com/TheTom/llama-cpp-turboquant — the TurboQuant llama.cpp fork (branch `feature/turboquant-kv-cache`)
  - https://github.com/ollama/ollama — Ollama
  - https://github.com/ggml-org/llama.cpp — Upstream llama.cpp

## 17. Alternatives Considered & Rejected

Two scope-reduction alternatives were raised during design review and rejected in favor of the full Turbohaul-Manager scope per the "no tech debt + one-stop-shop" mandate. Documented here for design transparency.

### 17.1 Alternative A — `lockd-proxy` (REJECTED)

- **Description:** 200-LoC FastAPI app on port 11401 holding `Semaphore(1)`, forwards every call to existing 11400 manager. Cross-process race solved via single-coordinator pattern.
- **Pros:** 1-2 day build vs 3-4 weeks; no new API surface; no FE; no blob store; no auth hardening; minimal code review.
- **Cons:**
  - **Tech-debt path** — leaves the old sidecar-manager in production indefinitely.
  - Does NOT deliver one-stop-shop local inference. No Ollama compat for future 3rd-party tools.
  - No BYOM blob store. Still depends on host-side model management.
  - No FE observability for queue/sidecar/blob/config.
  - No BYOI capability.
  - Patch-on-patch on the old manager that the doc explicitly says we're moving past.
- **Verdict:** Rejected. lockd-proxy is the tech-debt route.

### 17.2 Alternative B — libllama Python bindings (single-process model loading) (REJECTED)

- **Description:** Use llama-cpp-python (libllama) bindings to load models in the FastAPI process directly. No subprocess.
- **Pros:** No subprocess complexity, no orphan reaping, no SIGTERM/SIGKILL timing concerns.
- **Cons:**
  - Single CUDA context — model swap requires process restart (or unsafe in-process CUDA teardown).
  - Bindings are less mature than `llama-server` HTTP API.
  - Loading a 21GB model blocks the FastAPI event loop unless threaded.
  - the TurboQuant llama.cpp fork's innovations are in `llama-server` flags — bindings may lag.
  - Process isolation is genuinely valuable for crash containment.
- **Verdict:** Rejected. Subprocess-per-slot with proper supervision (§3.1 + §10) is the right shape. Critiques validated this — they flagged supervision GAPS, not the subprocess pattern itself.

### 17.3 Alternative C — `lockd-proxy + Turbohaul concurrent (HYBRID)` (REJECTED)

- **Description:** Build lockd-proxy NOW for immediate race relief; build Turbohaul-Manager in parallel over 2-3 weeks; cut over to Turbohaul once shipped.
- **Pros:** Immediate relief on existing race; parallel investment.
- **Cons:**
  - "no tech debt" — lockd-proxy IS the tech debt. Splitting investment makes both deliverables worse.
  - Two systems to maintain during hybrid period.
  - Phase 6a shadow mode in Turbohaul already provides "Turbohaul running parallel without disruption" semantics — that IS the soak strategy.
- **Verdict:** Rejected. v0.2 §13.1 shadow mode covers the "parallel-without-cutover" use case more cleanly.

---

**End of v0.2 design lock.** All Phase 1 findings addressed (doc revisions + implementation-baseline shifts). Ready for Phase 2 implementation start (4-6 days estimate, no tech debt).
