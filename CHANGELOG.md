# Changelog

All notable changes to Turbohaul-Manager are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

GitHub: `https://github.com/MrTrenchTrucker/turbohaul-manager`

---

## [0.3.0] - 2026-05-28

### Highlights

- **MTP speculative decoding composed with TurboQuant KV quantization** — MTP speculative decoding (`--spec-type draft-mtp`) composes with TurboQuant turbo2/3/4 KV cache quantization in a single llama.cpp binary, so faster decode and a quantized KV-cache footprint coexist.

## [v0.2.3] — 2026-05-19

### Highlights

- **Tool-call recovery for jinja-templated GGUFs** — transparent post-processor restores structured `tool_calls` when `llama-server` jinja templates (notably Qwen3-family) emit calls as JSON text inside `message.content` instead of populating the structured field. See [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).
- **`/v1/chat/completions` tools-field forwarding fixed** — the OpenAI endpoint previously dropped `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` from `client_meta`, making the recovery layer unreachable on the OpenAI surface. Now mirrors the `/api/chat` Ollama pattern.
- **Multi-agent GPU sharing proven** — three concurrent agents serialized cleanly through one Blackwell card across a 27b -> 35b -> 27b model-swap smoke. See [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md).
- **Persistence migration** to bind-mount `/var/lib/turbohaul`. State + manifests + blobs now survive `docker rm` and container-layer corruption. See [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md).
- **`response_format` validator** — `json_object` pass-through (commit `e0f5349`) plus `json_schema` FULL validate + retry + thinking-strip + security hardening (commit `93f294f`).
- **`/v1/embeddings`** (commit `de29292`) — llama-server embeddings passthrough.
- **`/v1/logging`** (commit `411f112`) — paginated audit-event endpoint, 20K-token envelope budget, recursive REDACTED scrub.
- **Logs tab + Schema editor in the frontend** — (commits `f3a49fb` + `26a5c46`).
- **Client-disconnect queue eviction** (commit `d7bd407`) + background terminal-park sweeper (commit `9e40f4c`).

### Added

- `src/turbohaul/api/tool_call_recovery.py` — `maybe_recover_tool_calls` post-processor handling OpenAI canonical `{"name":..,"arguments":..}` shape + Qwen `<tool_call>...</tool_call>` XML wrapper. Reasoning-guard (only scans content after `</think>`), parallel-call support (finditer + brace-balancer for nested args), idempotent skip when upstream populates `tool_calls`, name-allowlist gate against hallucinated tool names. (`5ce0f30`)
- `tests/test_tool_call_recovery.py` — recovery layer test coverage. (`5ce0f30`)
- `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` keys added to the `/v1/chat/completions` `client_meta` dict. (`91696d3`)
- `docs/TOOL_CALL_HANDLING.md` — user-facing doc covering the two wire paths, the recovery post-processor mechanism, the closure-fix history, and testing. (this release)
- `docs/MULTI_AGENT_SHARING.md` — multi-agent serialization architecture + proof. (`b697a43`)
- `docs/TURBOQUANT_FLAGS.md` — TurboQuant flag doctrine for production manifests, spawn-vs-request distinction, patching + verification recipes. (`b697a43`)
- `docs/PERSISTENCE_CHECKLIST.md` — deployment persistence checklist, full migration log, image-vs-patches debt finding. (`67767b0`)
- `CHANGELOG.md` — this file. (this release)
- response_format validator — `json_object` MVP (`e0f5349`) + `json_schema` FULL with validate + retry + thinking-strip (`93f294f`).
- `/v1/embeddings` BE endpoint (`de29292`).
- `/v1/logging` paginated audit-event endpoint (`411f112`).
- Logs tab in the React frontend (`f3a49fb`) — paginated audit feed with REDACTED banner + auto-refresh.
- Schema editor + `responseFormatValidator` in the frontend (`26a5c46`).
- Client-disconnect queue eviction (`d7bd407`) — slot gets evicted when client closes the connection mid-flight.
- Periodic terminal-park sweeper background task (`9e40f4c`) — sync finalize for STAGED + pid=NULL rows older than 24h via off-hot-path DB session.
- Ollama tool-call compat batch on `/api/chat` (`f8cde11`) — `tool_calls` passthrough + done_reason map + lenient JSON fallback on malformed args + `MAX_TOOL_ARG_CHARS = 262144` cap.
- TurboQuant cache types `turbo2` / `turbo3` / `turbo4` allowed on KV cache (`8cd5b4e`).
- `audit_db_session` connection pool + `_audit_async` wrapper (`9901db0`).
- ACTIVE_MATCH-streaming integration test (`3efb74f`).

### Fixed

- `/v1/chat/completions` silently dropping `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` from `client_meta` toward `llama-server` and into the recovery post-processor (`91696d3`).
- Doc errors caught by pre-release audit (`e2398e5`):
  - `/api/admin/unload` claim replaced with the three real cold-spawn paths (Option A `keep_alive: 0` per-request body, Option B natural IDLE_HOT teardown, Option C `docker restart`). The `/api/admin/unload` endpoint does not exist.
  - Multi-agent claim sharpened to "multiplexed serialization" rather than "concurrent execution" — Turbohaul time-slices on a single GPU slot, not parallel tensor execution.

### Changed

- Image tag bumps: `turbohaul-manager:v0.2.2` -> `v0.2.3` references in README + AI_AGENT_SETUP recipes.
- Bind-mount migration baked into the persistent image.

### Known issues / limitations

- `jinja: true` in the model's manifest is still required for any tool-call work. Tool-call recovery (above) catches the case where jinja + Qwen3 emits as text-JSON; it does not synthesize calls when the model never emits anything tool-call-shaped.
- Multi-residency (two models in VRAM simultaneously) is not supported in v0.2.x. Single-slot serialization is the v0.2 invariant; multi-residency is a v0.3 roadmap item.
- `--reload` uvicorn mode is banned in production. Production uses `docker restart` for code changes.
- `image-vs-patches` debt: prior v0.2.x runtime updates were applied as `docker cp` overlays on the running container rather than baked into a new image. v0.2.3 closes this via the bind-mount migration. Going forward, any non-trivial production deploy MUST `docker commit` + re-save tarball + update auto-recovery references, OR rebuild from `Dockerfile.cuda` against current dev-tree HEAD.

### Upgrade path

```bash
# Build the new image (no registry image is published; build locally per README)
docker build -f Dockerfile.cuda -t turbohaul-manager:v0.2.3 .

# Stop + remove the old container (state survives because of the bind-mount)
docker stop turbohaul-demo
docker rm turbohaul-demo

# Run the new container with the canonical bind-mount layout
docker run -d --name turbohaul-demo \
    --restart unless-stopped \
    --runtime nvidia --gpus all \
    -p 11401:11401 \
    -p 11434:11434 \
    -v /var/lib/turbohaul:/var/lib/turbohaul \
    -e TURBOHAUL_IDLE_HOT_SECONDS=600 \
    -e TURBOHAUL_GRACE_SECONDS=30 \
    turbohaul-manager:v0.2.3
```

Existing state (`state.sqlite`, `manifests/*.yaml`, `blobs/sha256/*`) is preserved through the bind-mount. First request to a new model may cold-load 30 to 60 seconds; subsequent same-thread follow-ups within the grace + IDLE_HOT windows reuse the warm slot.

---

## [v0.2.2] — earlier in May 2026

Initial public ship at `https://github.com/MrTrenchTrucker/turbohaul-manager`. See git history pre-`7b1bc51` for the v0.2.2 commit set. v0.2.2 included Phase 0-6 management plane + CUDA Dockerfile + v0.2.1 bug-sweep waves.

---

## Contributors to this release

See [CONTRIBUTORS.md](CONTRIBUTORS.md). Built with an AI-augmented multi-agent build pipeline; doc audit and release prep by the project lead.
