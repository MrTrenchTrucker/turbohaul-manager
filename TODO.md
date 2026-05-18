# Turbohaul-Manager — TODO / Phase tracker

**Project:** Turbohaul-Manager — Ollama-shape inference server using TurboQuant llama.cpp.
**Authority:** Cmdr MrTrench (2026-05-16 directive).
**Cmdr's Logbook RC:** RC-TURBOHAUL-V1 (to be filed alongside this doc).
**Architecture:** see [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Phase status

- [x] **Phase 0 — Forgejo prep + license audit** (DONE 2026-05-16)
  - [x] Pull Tom's Fork → matthew/llamacpp-turboquant-mirror (342 MB)
  - [x] Pull Ollama → matthew/ollama-mirror (356 MB; clone done, push complete)
  - [x] Tarball sidecar-manager → matthew/sidecar-manager-archive (commit 9eda7fa, 5 files)
  - [x] Create empty matthew/turbohaul-manager
  - [x] License audit via RBSRS Documentation Reader (#4) — GO-WITH-MOD 8.5/10. All MIT.
  - [x] Tom's Fork LICENSE on consumed branch `feature/turboquant-kv-cache` = MIT (blob e7dca554)
  - [ ] Trigger GitNexus index on all 4 new repos — **TODO**

- [ ] **Phase 1 — Architecture critique** (IN_PROGRESS — this doc just shipped)
  - [x] ARCHITECTURE.md written + committed
  - [x] TODO.md written + committed
  - [ ] RBSRS Architecture Brainstormer (#12) critique on the queue/state-machine design
  - [ ] RBSRS Devil's Advocate (#13) critique on the API surface + subprocess model
  - [ ] RBSRS Failure Predictor (#16) — what fails in production?
  - [ ] RBSRS Security Reviewer — Trusted-network assumption ok? Yaml-write attack surface?
  - [ ] Apply revisions → v0.2 doc
  - [ ] Cmdr final ✓

- [ ] **Phase 2 — Core queue + slot manager** (~3-5 days)
  - [ ] FastAPI app skeleton + project structure
  - [ ] State machine implementation (per-state class + transitions)
  - [ ] Subprocess management (Popen, health-poll, SIGTERM/SIGKILL)
  - [ ] Acceptance buffer + staging queue (asyncio.Queue + asyncio.Lock)
  - [ ] Grace period timer + thread_id matching
  - [ ] Idle hot-load timer + same-model-tag matching
  - [ ] Per-model manifest reader + flag-to-CLI translator
  - [ ] Pytest unit tests — state machine + queue ordering + grace + idle
  - [ ] Pytest integration tests — real subprocess spawn (use a tiny GGUF for fast tests)

- [ ] **Phase 3 — Ollama-compat API + OpenAI-compat surface** (~2-3 days)
  - [ ] POST /api/generate (Ollama single-turn)
  - [ ] POST /api/chat (Ollama multi-turn w/ thread_id)
  - [ ] POST /v1/chat/completions (OpenAI)
  - [ ] POST /v1/completions
  - [ ] GET /api/tags, GET /api/show, GET /api/version
  - [ ] GET /status (queue + active + grace + idle state)
  - [ ] WebSocket /ws/state with event types: queue_change, slot_transition, llama_stderr_line, model_pull_progress
  - [ ] GET/PUT /api/manifests/{tag}, GET/PUT /api/config (yaml edit endpoints)
  - [ ] Schema validation on yaml writes (pydantic models)
  - [ ] Pytest API endpoint tests

- [ ] **Phase 4 — Blob store + pull endpoints** (~3-5 days)
  - [ ] Content-addressed blob store (`/var/lib/turbohaul/blobs/sha256/AB/ABCD...`)
  - [ ] Manifest sidecar (json or yaml) per blob with metadata
  - [ ] POST /api/pull (Ollama registry — implement Ollama's pull protocol)
  - [ ] POST /api/pull-hf (HuggingFace API — uses HF_API_KEY env)
  - [ ] POST /api/pull-url (arbitrary URL — straight GET + stream)
  - [ ] POST /api/import (local file path — copy + hash)
  - [ ] DELETE /api/delete (remove from blob + manifest)
  - [ ] Pull progress events on /ws/state
  - [ ] VRAM-fit pre-check before stage (use manifest.expected_vram_bytes vs nvidia-smi)

- [ ] **Phase 5 — Frontend (React+Vite + WebSocket)** (~3-4 days)
  - [ ] Vite project scaffolded
  - [ ] WebSocket client (autoreconnect, event dispatcher)
  - [ ] Dashboard view (active sidecar + queue + throughput chart)
  - [ ] Queue view (FIFO list + ETAs)
  - [ ] Blob view (model list + Pull / Import / Delete forms)
  - [ ] Config view (main yaml editor + per-model yaml editor with model selector; monaco-editor or codemirror)
  - [ ] Logs view (tail llama-server stderr per slot)
  - [ ] Build output to /opt/turbohaul/ui_dist
  - [ ] FastAPI static mount /ui/\* + SPA route fallback
  - [ ] Restart-required field flagging on yaml save

- [ ] **Phase 6 — Dockerfile + smoke + ship** (~1-2 days)
  - [ ] Dockerfile (Python 3.12-slim + llama-server binary from Tom's Fork build stage)
  - [ ] docker-compose.yml (mounts /var/lib/turbohaul + GPU + 11401)
  - [ ] THIRD_PARTY_LICENSES file in image
  - [ ] README with attribution + quickstart
  - [ ] Smoke E2E: Secretary doc-to-ratecon via 11401 (parallel to 11400)
  - [ ] Tag v1.0.0, push to Forgejo
  - [ ] Image tarball → /mnt/AppData_SSD/Docker_Images/turbohaul-manager-v1.tar
  - [ ] Auto-recovery script entry (mirror sidecar-manager's pattern)

## Decision log

| Date       | Decision                                                                   | Rationale                                             |
| ---------- | -------------------------------------------------------------------------- | ----------------------------------------------------- |
| 2026-05-16 | Project name: Turbohaul-Manager                                            | Cmdr approved                                         |
| 2026-05-16 | Port 11401 (parallel to old 11400)                                         | Soak period before cutover                            |
| 2026-05-16 | Subprocess-per-slot (Option A)                                             | Matches current arch; simpler; debuggable             |
| 2026-05-16 | Unified queue (not per-model)                                              | 1-at-a-time on CONTAK-01; fairness moot               |
| 2026-05-16 | thread_id semantic for follow-ups                                          | Ollama already has the concept                        |
| 2026-05-16 | Two-tier: unbounded acceptance buffer + capped staging queue               | Cmdr directive — receive always, stage only when room |
| 2026-05-16 | Per-model yaml separate from main config                                   | Cmdr directive — flags are per-model                  |
| 2026-05-16 | FE + BE both edit yaml                                                     | Cmdr directive — match exactly                        |
| 2026-05-16 | React + Vite + FastAPI static mount /ui/\*                                 | Single-port deploy; matches Logbook fleet pattern     |
| 2026-05-16 | Subprocess `llama-server` from Tom's Fork (vendored binary in image)       | Build it ourselves, don't depend on external image    |
| 2026-05-16 | License: all upstream MIT → ship THIRD_PARTY_LICENSES + README attribution | RBSRS audit GO-WITH-MOD 8.5/10                        |

## Open questions for Cmdr

- v1 license for Turbohaul-Manager itself: proprietary (closed) or open-source (which license: MIT, Apache 2.0, ours)? Phase 6 closeout.
- Default `STAGING_QUEUE_DEPTH` = 100. Is that right for CONTAK-01? More? Less?
- BYOI fleet-wide audit (Phase 6) — also covers GEAR/advisor/Logbook hard-coded 11400 references. Need to confirm scope when we get there.
