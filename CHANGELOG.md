# Changelog

All notable changes to Turbohaul-Manager are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

GitHub: `https://github.com/MrTrenchTrucker/turbohaul-manager`

---

## [0.4.0] — 2026-06-21

### Highlights

- **Live inference monitor** — a built-in, real-time view of the active generation: tokens/sec, prompt/decode progress, and a live output stream, independent of any external client or logger. The `/status` response gains a `generation` block (sampled ~1 Hz from the running sidecar) and a new Server-Sent-Events endpoint `GET /ui/live/output/stream` tees the model's output as it is produced. Surfaced in the frontend as a Live / Dashboard view.
- **Per-model concurrent dispatch** — each loaded model now serves according to its own concurrency setting. A main chat model can run strictly in series (one request at a time) while a sub-agent model serves up to *N* requests **concurrently** (`parallel: N`) on a single warm `llama-server` sidecar, via a fan-out that admits up to *N* riders with a drain-before-swap guard. The single-model-resident invariant is preserved — concurrency is *within* the active model, never across models.
- **KV-cache offload to host RAM** — with `no_kv_offload`, model weights stay on the GPU while the KV cache lives in system RAM. This lets a large-context model (e.g. 250K tokens) run on a 24 GB card and — combined with `kv_unified: true` — **serve requests in parallel while offloaded** (`parallel: 2` measured to fit and serve concurrently on a 24 GB card). See [docs/KV_CACHE_OFFLOADING.md](docs/KV_CACHE_OFFLOADING.md) (and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
- **Live-panel stability + tool-call display** — the live inference panel no longer flickers between "generating" and "idle" during bursty multi-turn generation (a short grace-hold smooths the phase). Tool calls now render correctly in the live output, including models that emit tool-call arguments as a structured object instead of a JSON string.
- **Faster failure recovery** — a model load that crashes now fails its slot in ~2 s instead of waiting out the full load-health timeout, so a bad load no longer wedges the queue behind it.

### Added

- `src/turbohaul/live_monitor.py` + `src/turbohaul/api/live_stream.py` — the live inference monitor (1 Hz slots poller, live-output buffer, `/ui/live/output/stream` SSE endpoint), plus the `LiveInference` view and `useLiveOutput` hook in the frontend.
- Per-model concurrent dispatch — fan-out rider admission in the manager, model-affinity batching in the queue (prefer the already-warm model when selecting the next request), and a parallel-aware safety gate.
- [docs/KV_CACHE_OFFLOADING.md](docs/KV_CACHE_OFFLOADING.md) — a dedicated KV-cache offload explainer: the mechanism, the VRAM-vs-RAM trade-off, the sizing math, parallel-while-offloaded, and the decode perf cost (~2.5× on the measured 35B-class example).

### Changed

- The VRAM-fit safety gate is parallel-aware: it accounts for per-slot compute when a model is configured for concurrent serving, and refuses conservatively when it cannot measure free VRAM under `parallel > 1`.
- Manifest validation cross-checks `parallel`, `kv_unified`, and context size (each slot's context window must clear a minimum, and the total context must divide evenly across the slots).

> Version note: proposed as a minor bump (0.4.0) for the new live-monitor and per-model-concurrency features.

---

## [0.3.0] — 2026-05-28

### Highlights

- **MTP speculative decoding composed with TurboQuant KV quantization** — MTP speculative decoding (`--spec-type draft-mtp`) composes with TurboQuant turbo2/turbo3/turbo4 KV-cache quantization in a single `llama.cpp` binary, so faster decode and a quantized KV-cache footprint coexist.

---

## [v0.2.3] — 2026-05-19

### Highlights

- **Tool-call recovery for jinja-templated GGUFs** — transparent post-processor restores structured `tool_calls` when `llama-server` jinja templates (notably Qwen3-family) emit calls as JSON text inside `message.content` instead of populating the structured field. See [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).
- **`/v1/chat/completions` tools-field forwarding fixed** — previously the OpenAI endpoint silently dropped `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` from `client_meta`, making the recovery layer unreachable on the OpenAI surface. Now mirrors the `/api/chat` Ollama pattern.
- **Multi-agent GPU sharing** — three clients (one 27b chat client plus two advisor clients) serialize cleanly through one Blackwell-class GPU across a 27b -> 35b -> 27b model-swap exercise. See [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md).
- **Persistence migration** to a host bind-mount for `/var/lib/turbohaul`. State + manifests + blobs now survive `docker rm` and container-layer corruption. A new image tag bakes the runtime updates into the image layer.
- **`response_format` validator** — `json_object` pass-through plus `json_schema` FULL validate + retry + thinking-strip + security mods.
- **`/v1/embeddings`** — llama-server embeddings passthrough.
- **`/v1/logging`** — paginated audit-event endpoint, 20K-token envelope budget, recursive REDACTED scrub.
- **Logs tab + Schema editor in the frontend**.
- **Client-disconnect queue eviction** + background terminal-park sweeper.

### Added

- `src/turbohaul/api/tool_call_recovery.py` — `maybe_recover_tool_calls` post-processor (287 LoC) handling OpenAI canonical `{"name":..,"arguments":..}` shape + Qwen `<tool_call>...</tool_call>` XML wrapper. Reasoning-guard (only scans content after `</think>`), parallel-call support (finditer + brace-balancer for nested args), idempotent skip when upstream populates `tool_calls`, name-allowlist gate against hallucinated tool names.
- `tests/test_tool_call_recovery.py` — 12 test functions / 18 sub-cases / 18-18 GREEN on host pytest.
- `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` keys added to the `/v1/chat/completions` `client_meta` dict in `chat_completion.py`.
- `docs/TOOL_CALL_HANDLING.md` — user-facing doc covering the two wire paths, the recovery post-processor mechanism, the closure-fix history, and testing. (this release)
- `docs/MULTI_AGENT_SHARING.md` — multi-agent serialization architecture + worked example.
- `docs/TURBOQUANT_FLAGS.md` — flag doctrine for production manifests, spawn-vs-request distinction, patching + verification recipes.
- `CHANGELOG.md` — this file. (this release)
- response_format validator — `json_object` MVP + `json_schema` FULL with validate + retry + thinking-strip.
- `/v1/embeddings` BE endpoint.
- `/v1/logging` paginated audit-event endpoint.
- Logs tab in the React frontend — paginated audit feed with REDACTED banner + auto-refresh.
- Schema editor + `responseFormatValidator` in the frontend.
- Client-disconnect queue eviction — slot gets evicted when client closes the connection mid-flight.
- Periodic terminal-park sweeper background task — sync finalize for STAGED + pid=NULL rows older than 24h via off-hot-path DB session.
- Ollama tool-call compat batch on `/api/chat` — `tool_calls` passthrough + done_reason map + lenient JSON fallback on malformed args + `MAX_TOOL_ARG_CHARS = 262144` cap.
- TurboQuant cache types `turbo2` / `turbo3` / `turbo4` allowed on KV cache.
- `audit_db_session` connection pool + `_audit_async` wrapper.
- ACTIVE_MATCH-streaming integration test.

### Fixed

- `/v1/chat/completions` silently dropping `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` from `client_meta` toward `llama-server` and into the recovery post-processor.
- Doc corrections:
  - `/api/admin/unload` claim replaced with the three real cold-spawn paths (Option A `keep_alive: 0` per-request body, Option B natural IDLE_HOT teardown, Option C `docker restart`). The `/api/admin/unload` endpoint does not exist.
  - Multi-agent claim sharpened to "multiplexed serialization" rather than "concurrent execution" — Turbohaul time-slices on a single GPU slot, not parallel tensor execution.

### Changed

- Image tag bumps: `turbohaul-manager:v0.2.2` -> `v0.2.3` references in README + AI_AGENT_SETUP recipes.
- Bind-mount migration baked into the persistent image (`v0.2.3` CUDA bind-mount variant).
- Auto-recovery script updated to reference the new tag.

### Known issues / limitations

- `jinja: true` in the model's manifest is still required for any tool-call work. Tool-call recovery (above) catches the case where jinja + Qwen3 emits as text-JSON; it does not synthesize calls when the model never emits anything tool-call-shaped.
- Multi-residency (two models in VRAM simultaneously) is not supported in v0.2.x. Single-slot serialization is the v0.2 invariant; multi-residency is a v0.3 roadmap item.
- `--reload` uvicorn mode is banned in production (it can reload code before a migration is applied). Production uses `docker restart turbohaul` for code changes.
- `image-vs-patches` debt: prior v0.2.x runtime updates were applied as `docker cp` overlays on the running container rather than baked into a new image. v0.2.3 closes this by baking the changes into the `v0.2.3` CUDA bind-mount image. Going forward, any non-trivial production deploy MUST `docker commit` + update auto-recovery references, OR rebuild from `Dockerfile.cuda` against the current source tree.

### Upgrade path

```bash
# Pull the new image (tag may differ depending on registry mirror)
docker pull ghcr.io/MrTrenchTrucker/turbohaul-manager:v0.2.3

# Stop + remove the old container (state survives because of the bind-mount)
docker stop turbohaul
docker rm turbohaul

# Run the new container with the canonical bind-mount layout
docker run -d --name turbohaul \
    --restart unless-stopped \
    --runtime nvidia --gpus all \
    -p 11401:11401 \
    -p 11434:11434 \
    -v /var/lib/turbohaul:/var/lib/turbohaul \
    -e TURBOHAUL_IDLE_HOT_SECONDS=600 \
    -e TURBOHAUL_GRACE_SECONDS=30 \
    ghcr.io/MrTrenchTrucker/turbohaul-manager:v0.2.3
```

Existing state (`state.sqlite`, `manifests/*.yaml`, `blobs/sha256/*`) is preserved through the bind-mount. First request to a new model may cold-load 30 to 60 seconds; subsequent same-thread follow-ups within the grace + IDLE_HOT windows reuse the warm slot.

---

## [v0.2.2] — earlier in May 2026

Initial public ship at `https://github.com/MrTrenchTrucker/turbohaul-manager`. See the git history at the `v0.2.2` tag for the full change set. v0.2.2 included the full management plane + CUDA Dockerfile + v0.2.1 bug-sweep waves.

---

## Contributors to this release

See [CONTRIBUTORS.md](CONTRIBUTORS.md). The tool-call recovery work and the `/v1/chat/completions` closure-fix, the endpoint batch, doc review, dependency-graph alignment, and release prep were contributed by the maintainers.
