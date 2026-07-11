# Changelog

All notable changes to Turbohaul-Manager are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

GitHub (`https://github.com/MrTrenchTrucker/turbohaul-manager`) is the canonical public repository.

---

## [v0.6.0] - 2026-07-10

_Covers changes since the v0.5.0-public baseline (2026-06-28). Versions v0.3.x–v0.5.x were internal iterations consolidated into that baseline._

### Highlights

- **Multi-agent sessions are now first-class — one GPU, many agents, zero context bleed.** Every sub-agent a client spawns gets its own distinct session ID, so its KV cache lives in an isolated bin that no other agent — sibling sub-agent, curator, or the main agent — can read, overwrite, or muddy. Previously, concurrent agents sharing a single GPU could easily cross-contaminate each other's context; keying every bin to the agent's identity makes that structurally impossible. See [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md).

- **Precomputed KV cache makes returning to a long context nearly free.** The main agent keeps its context warm in VRAM between tool calls and restores its full precomputed prefix when it comes back from a batch of sub-agent work — instead of re-reading the whole conversation from scratch. Measured in validation: 96% of a ~105,000-token context was reused as a common prefix, and a 154,647-token context was restored with just a 29-token prefill — 629× less prefill work than recomputing it. Sub-agents and the curator get the same warm reuse while they run.

- **KV cache now survives a model swap.** When the engine has to unload one model to load another, Turbohaul saves the returning session's slot state to disk and restores it on the way back, so swapping models no longer forces an expensive re-prefill of everything that was already computed. Works whether the box is juggling several models or running just one. (#79, #82)

- **Disposable roles clean up after themselves.** Sub-agents, the curator, and compression passes are transient by design, so when one finishes its job its KV is thrown away by contract — enforced both on each turn and again when the model unloads — and can never survive to pollute later work. The main agent always keeps its cache; the throwaway roles never do.

- **A model-swap-safe storage tier keeps caches around only as long as they're useful.** Returning sessions restore instantly from system RAM (a tmpfs tier), and their caches are copied down to SSD so they outlive a full unload. Automatic cleanup keeps both tiers bounded — caches older than 6 hours are dropped, the SSD tier is capped at 40 GiB and the RAM tier at 20 GiB, and a mid-copy cache is never deleted out from under itself. An opt-in purge (off by default) can also drop any cache built for a different engine or model.

- **The role of every request is now declared explicitly, and the dashboard shows the truth.** A client labels each request as the main agent, a sub-agent, the curator, or a compression pass, and those labels drive every save/restore decision — so a curator turn, for example, can ride the main agent's context read-only but can never overwrite it. An unlabeled request falls back to the old behavior, byte-for-byte, so existing callers are unaffected. The dashboard was rebuilt to match: it now shows a live prefill-progress bar, whether a model is genuinely busy versus idle, an accurate grace-timer, and a per-model indicator confirming the model is resident and its cache was restored.

- **Engine hardening — the KV-save path no longer takes the whole engine down with it.** Earlier builds could poison a saved cache with mid-operation engine state and then crash silently on the next restore, because the engine's abort output was being discarded. This release vendors the engine so its exact build is pinned and reproducible, captures that abort output instead of throwing it away, automatically quarantines any cache bin implicated in three strikes (preserving it as evidence), detects and recovers a dead engine instead of hanging on it, and surfaces a clear "engine stalled" state on the dashboard. The net effect: the crash-and-hang loops previously seen during multi-agent bursts no longer occur.

- **Everything ships in one repository — clone it and run, fully offline.** The whole system is now self-contained in a single repo: the inference engine's source, the Python dependencies as wheels, the frontend (source plus a prebuilt bundle), and a ready-to-run prebuilt image. There's no external registry, package index, or second repository to fetch — clone the repo and either run the prebuilt image or build from source with the network switched off. All vendored dependencies are permissively licensed and their licenses are recorded in-repo. See the README's *Offline use* section.

### Added

- Per-agent, single-copy KV caching — each session-and-role pair gets its own cache bin, fingerprinted so sub-agent and curator bins never collide.
- Two-tier cache storage — a fast RAM tier for instant restore plus an SSD tier that survives a full model unload, with caches copied down atomically.
- Automatic cache cleanup — drops caches past a 6-hour age, caps the SSD tier at 40 GiB and the RAM tier at 20 GiB, keeps at most 100 RAM-tier files, and never deletes a cache younger than 60 seconds (to avoid racing an in-flight copy). An opt-in purge (off by default) removes caches built for a mismatched engine or model.
- Per-request control over whether a throwaway role saves its cache, set by the calling client with no environment variables needed.
- A per-model status indicator on the dashboard confirming the model is resident and its cache was restored.
- Dead-engine recovery — a proactive liveness sweep plus a check on the serving path so the manager re-spawns a silently-dead engine instead of hanging on it.
- The engine sidecar's output is now captured to a file rather than discarded, so a crash leaves a diagnosable trace.
- Three-strikes cache quarantine — a bin implicated in repeated failures is moved aside automatically and preserved as evidence.
- Seam-integrity repair — on returning from a sub-agent wave the manager rebuilds the main agent's cache tip cleanly, ending the abort chain that used to crash the wave return.
- Honest dashboard telemetry — the grace clock only appears when the model is actually idle, a prefill pill and progress bar reflect real work, a "busy" state shows while the model is serving, and a red "no telemetry" warning appears after 120 seconds of silence.
- A settings control for the SSD cache ceiling, with live usage and headroom shown in the UI.
- Reasoning blocks are now preserved on tool-call turns, fixing a context divergence that used to appear partway through long sessions.
- Live-output pane now auto-follows new tokens only while you're scrolled to the bottom — scroll up to read and it stops yanking you back down.
- An idempotent completion cache with single-flight de-duplication, so a retried request can't double-run.
- KV reuse now works when the box serves several models or several context windows at once — every save/restore decision routes through one unified policy, and a multi-model path that previously never restored its cache now does.
- Fully self-contained, offline build — the engine source, Python wheels, and frontend (with a committed build) are all vendored in-repo, and `Dockerfile.engine-src` builds the entire stack with no PyPI, npm, or external git access.
- A prebuilt, runnable image is published to the GitHub Container Registry — run `docker pull ghcr.io/mrtrenchtrucker/turbohaul-manager:v0.6.0`, or build offline from source with `Dockerfile.engine-src`.

### Fixed

- **Incoming requests lost their role labels.** When a warm, idle model inherited a new request, it was overwriting that request's fresh role labels with the previous occupant's cached ones — so a sub-agent could be misfiled as the main agent. The new request's labels now always win.
- **Silent crash loops on wave return.** Caches saved mid-turn captured transient post-curator engine state and were wrongly marked clean; restoring one aborted the engine, and six crash cycles died invisibly because the engine's abort output was being discarded. Fixed by saving only at the clean unload seam and by capturing that output.
- **Manager hung on a dead engine.** When the idle model's engine died silently, the manager kept reusing the dead process for 10+ minutes instead of re-spawning it. It now detects the dead engine and recovers.
- **Restoring from SSD clobbered the fresh cache.** A size-mismatch "repair" step overwrote the current turn's cache with a stale pre-compression copy on every scan, pinning reuse at the stale version. The repair now only fills in a missing copy and never overwrites a live one.
- **Reasoning blocks vanished on tool-call turns.** A position-based strip was deleting reasoning from every saved cache on tool-call turns, so a later resend diverged from what was originally saved. Reasoning is now preserved.
- **Stall alarm fired on healthy work, and real stalls read as normal.** The token counter used to classify progress kept climbing through normal decoding, so a genuine stall was labelled "prefill" forever while healthy decoding painted a stuck "prefill 1%". The classifier now reads the counter that actually freezes on a stall.
- **The SSD cache-cap setting was a silent no-op.** The ceiling was read from the environment but never wired to the running config, so raising it did nothing. It's now applied at runtime and editable from the Settings tab.
- **Unknown model tags now return 404** on the chat endpoints instead of failing obscurely.
- **Oversized request bodies are now rejected with 413** at the middleware layer.
- **Fixed a delete-then-use race in manifest handling.**
- **Blocking file I/O moved off the event loop** so a slow disk no longer stalls request handling.
- **VRAM admission now counts both halves of the KV cache** (value cache as well as key cache), so the box no longer over-commits memory.
- **Guarded an empty-choices response** that could crash the caller.
- **Tool-call messages with `null` content no longer crash the chat endpoints.** Standard OpenAI-format clients that send a tool-call turn with `content: null` used to hit a server error; the content is now handled cleanly on both the streaming and non-streaming paths.

### Changed

- **The main agent's cache is now written only when the model unloads, not on every turn.** Per-turn work just keeps the live cache warm in VRAM; the single disk write happens once, cleanly, at unload. This is what ended the mid-turn poisoned-save crashes.
- **Warm restores now force a clean reload by default**, eliminating the residual "no cache data" errors that used to surface on reuse.
- **A tail-integrity check refuses to save** a cache whose reasoning primer wasn't properly stripped, preventing a known corruption from reaching disk.
- **KV save timeout raised from 10s to 120s.** A chronic timeout was freezing the cache at ~25 turns; save failures are now logged with enough detail to diagnose.
- **The curator can now ride the main agent's context read-only** when that route is enabled (off by default), guaranteed never to overwrite the main agent's saved cache.
- **Caches are only saved for single-series, large-enough contexts** — a multi-series slot won't save, and a context has to clear a minimum size before it's worth caching.
- **Garbage collection now piggybacks on the existing idle sweep**, throttled to run at most every 5 minutes.
- **The mismatched-engine purge is opt-in and off by default**; when enabled it drops caches whose build/model/context fingerprint doesn't match the running engine, and never touches the live idle model's cache.
- **Model scheduling no longer thrashes between same-model turns.** When several sub-agents on the same model were queued behind one request for a different model, the scheduler used to swap the model out and back in unnecessarily. It now drains queued same-model work first (still swapping promptly if a waiting request would otherwise starve), and briefly holds a model warm for a same-model request already in the queue.

### Known issues / limitations

- **The cache-restore verifier is observability-only for now.** The cold-spawn path doesn't yet carry expected token counts, so the per-model indicator confirms a restore was attempted and the slot is live but can't check restore depth against an expectation — it may report a restore as unverified even when reuse is actually working. The engine log and wave-return counters remain the authoritative reuse signals; real expected-versus-actual verification is a planned follow-up.
- **Some engine-side crash classes are mitigated in the manager, not root-fixed.** The strict-extension abort on hybrid-recurrent models and rare cache-save truncation are handled by rebuilding the cache and quarantining bad bins, but the underlying fixes belong in the vendored llama.cpp fork and remain open upstream.
- **The `monitor` config setting requires a restart** — it's part of runtime config but can't be changed live and isn't returned by the config read endpoint.
- **Most cache-tuning knobs are environment-only.** Only the SSD cache ceiling is adjustable from the Settings tab; the RAM ceiling, cleanup age and count thresholds, sweep interval, and purge settings all require environment variables.
- **Auxiliary background tasks are unlabeled for now.** Some calls an agent's harness makes on its own behind the scenes — title generation, session search, skill lookups, vision, web extraction, and approval checks — currently arrive without a role tag or session identity. They work correctly, but because they're unlabeled they can't ride the main agent's warm cache and instead take the slower full-prefill path. Threading the agent's session context through these auxiliary calls so they can reuse cache is a planned follow-up.
- **Not in this release:** further KV-cache work is still in review — request classification, a fuller completion cache, fingerprint-keyed saves, byte-match self-check, identity shadowing, stale-anchor age reclaim, and shadow re-prefill saves — targeted for a future release.

---

## [v0.2.3] - 2026-05-19

### Highlights

- **Tool-call recovery for jinja-templated GGUFs** — transparent post-processor restores structured `tool_calls` when `llama-server` jinja templates (notably some reasoning-model chat templates) emit calls as JSON text inside `message.content` instead of populating the structured field. See [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).
- **`/v1/chat/completions` tools-field forwarding fixed** — the OpenAI endpoint previously dropped `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` from `client_meta`, making the recovery layer unreachable on the OpenAI surface. Now mirrors the `/api/chat` Ollama pattern.
- **Multi-agent GPU sharing proven in production** (2026-05-19) — three agents serialized cleanly through one Blackwell card across a 27B -> 35B -> 27B model-swap smoke. See [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md).
- **Persistence migration** to a bind-mount at `/var/lib/turbohaul` on the host data directory. State + manifests + blobs now survive `docker rm` and container-layer corruption. New image tag `v0.2.3-cuda-bindmount` bakes the accumulated patches into the layer. See [docs/PERSISTENCE_CHECKLIST.md](docs/PERSISTENCE_CHECKLIST.md).
- **`response_format` validator** — `json_object` pass-through plus `json_schema` FULL validate + retry + thinking-strip + security hardening. See [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).
- **`/v1/embeddings`** — llama-server embeddings passthrough.
- **`/v1/logging`** — paginated audit-event endpoint, 20K-token envelope budget, recursive REDACTED scrub.
- **Logs tab + Schema editor in the frontend.**
- **Client-disconnect queue eviction** + background terminal-park sweeper.

### Added

- `src/turbohaul/api/tool_call_recovery.py` — `maybe_recover_tool_calls` post-processor (287 LoC) handling OpenAI canonical `{"name":..,"arguments":..}` shape + a reasoning-model XML wrapper. Reasoning-guard (only scans content after ``), parallel-call support (finditer + brace-balancer for nested args), idempotent skip when upstream populates `tool_calls`, name-allowlist gate against hallucinated tool names.
- `tests/test_tool_call_recovery.py` — 12 test functions / 18 sub-cases / 18-18 GREEN on host pytest.
- `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` keys added to the `/v1/chat/completions` `client_meta` dict at `chat_completion.py:468`.
- `docs/TOOL_CALL_HANDLING.md` — user-facing doc covering the two wire paths, the recovery post-processor mechanism, the closure-fix history, and testing. (this release)
- `docs/MULTI_AGENT_SHARING.md` — multi-agent serialization architecture + 2026-05-19 production proof.
- `docs/TURBOQUANT_FLAGS.md` — 5-flag doctrine for production manifests, spawn-vs-request distinction, patching + verification recipes.
- `docs/PERSISTENCE_CHECKLIST.md` — persistence audit, full migration log, image-vs-patches debt finding.
- `CHANGELOG.md` — this file. (this release)
- `response_format` validator — `json_object` MVP + `json_schema` FULL with validate + retry + thinking-strip.
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
- Doc errors caught by a pre-release audit:
  - `/api/admin/unload` claim replaced with the three real cold-spawn paths (Option A `keep_alive: 0` per-request body, Option B natural IDLE_HOT teardown, Option C `docker restart`). The `/api/admin/unload` endpoint does not exist.
  - Multi-agent claim sharpened to "multiplexed serialization" rather than "concurrent execution" — Turbohaul time-slices on a single GPU slot, not parallel tensor execution.

### Changed

- Image tag bumps: `turbohaul-manager:v0.2.2` -> `v0.2.3` references in README + AI_AGENT_SETUP recipes.
- Bind-mount migration baked into the persistent image: `turbohaul-manager:v0.2.3-cuda-bindmount` (a 2.59 GB image tarball).

### Known issues / limitations

- `jinja: true` in the model's manifest is still required for any tool-call work. Tool-call recovery (above) catches the case where a jinja template emits calls as text-JSON; it does not synthesize calls when the model never emits anything tool-call-shaped.
- Multi-residency (two models in VRAM simultaneously) is not supported in v0.2.x. Single-slot serialization is the v0.2 invariant; multi-residency is a v0.3 roadmap item.
- `--reload` uvicorn mode is banned in production. Production uses `docker restart <container>` for code changes.
- `image-vs-patches` debt: prior v0.2.x runtime updates were applied as `docker cp` overlays on the running container rather than baked into a new image. v0.2.3 closes this via the `v0.2.3-cuda-bindmount` commit. Going forward, any non-trivial production deploy MUST `docker commit` + re-save tarball + update auto-recovery references, OR rebuild from `Dockerfile.cuda-multi` against your current source tree.

### Upgrade path

```bash
# Pull the new image (tag may differ depending on registry mirror)
docker pull ghcr.io/mrtrenchtrucker/turbohaul-manager:v0.2.3

# Stop + remove the old container (state survives because of the bind-mount)
docker stop turbohaul-manager
docker rm turbohaul-manager

# Run the new container with the canonical bind-mount layout
docker run -d --name turbohaul-manager     --restart unless-stopped     --runtime nvidia --gpus all     -p 11401:11401     -p 11434:11434     -v /var/lib/turbohaul:/var/lib/turbohaul     -e TURBOHAUL_IDLE_HOT_SECONDS=600     -e TURBOHAUL_GRACE_SECONDS=30     ghcr.io/mrtrenchtrucker/turbohaul-manager:v0.2.3
```

Existing state (`state.sqlite`, `manifests/*.yaml`, `blobs/sha256/*`) is preserved through the bind-mount. First request to a new model may cold-load 30 to 60 seconds; subsequent same-thread follow-ups within the grace + IDLE_HOT windows reuse the warm slot.

---

## [v0.2.2] - earlier in May 2026

Initial public ship at `https://github.com/MrTrenchTrucker/turbohaul-manager`. See the git history at the `v0.2.2` tag for the v0.2.2 commit set. v0.2.2 included the Phase 0-6 management plane + CUDA Dockerfile + v0.2.1 bug-sweep waves.

---

## Contributors to this release

See [CONTRIBUTORS.md](CONTRIBUTORS.md). The v0.2.3 release added the tool-call recovery post-processor and the `/v1/chat/completions` tools-field forwarding fix, together with the response_format validator, embeddings and logging endpoints, and the persistence migration.

**v0.6.0 (2026-07-10) delta**: KV cache persistence and engine-hardening work (PRs #124-128), covering per-agent single-copy caching, model-swap-safe two-tier storage, dead-engine recovery, and the crash-safe KV-save path.