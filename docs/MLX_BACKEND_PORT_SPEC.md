# MLX Backend Port Spec — v0.3.0-era `mlx_support_PR` → current `main` (v0.6.0)

Status: **IMPLEMENTED** (2026-07-12). Code ported to `main`, tests at
`tests/test_mlx_backend.py`, 17 MLX tests + 31 manifest tests passing. Was a
read-only spec; port is now done and this file is the record of what shipped.
Author: Hermes Agent (at Scott's request, 2026-07-12)
Branch constraint: `main` and `mlx_support_PR` share NO common ancestor
(`main` was remangled/force-rewritten; `mlx_support_PR` descends from v0.3.0,
`main` from v0.5.0→v0.6.0). A mechanical rebase/cherry-pick was NOT possible:
cherry-picking the MLX feature commit alone produced 9 content conflicts in
core files. A manual PORT was done instead.

## Goal
Make Turbohaul use MLX on Apple Silicon (no CUDA, no 8GB Docker image) and
verify, on a real macOS run, that it (a) prefers MLX when available, (b) does
NOT fall back to CPU/torch, and (c) performs comparably to the llama.cpp/Metal
path. This is the precondition for the "evaluate MLX vs CPU fallback" task.

## What the old PR actually delivered (and what it did NOT)

Present on `mlx_support_PR` @ `ff1c237` (now reverted to this point):
- `src/turbohaul/backends/__init__.py`, `base.py`, `llamacpp.py`, `mlx.py`
  - A `BackendInterface` ABC with `spawn(req)`, `wait_healthy`, `teardown`,
    `make_completion_fn`, `health_endpoint`.
  - `SidecarHandle` / `SpawnRequest` dataclasses (DIFFERENT shape from main's
    `subprocess_mgr.SidecarHandle`).
  - `MLXBackend.spawn` builds `python -m mlx_lm.server --model <repo|path>
    --host 127.0.0.1 --port <p>` + `req.mlx_flags`.
  - `_check_availability()` hard-gates to `Darwin` + `arm64`, refuses otherwise.
  - Module-level `SAFE_MLX_FLAGS` allowlist + `validate_mlx_flags()` +
    `mlx_flags_to_argv()` (snake_case → `--snake-case`, bool handling).
- `turbohaul-launcher.sh` (local-dev launcher).
- `tests/test_backends_mlx.py`.
- README/ARCHITECTURE/manifest/config/manager edits.

KNOWN GAPS in the old PR (must be fixed during the port):
1. `MLXBackend.spawn` passes `req.mlx_flags` RAW as a list — it never calls
   `mlx_flags_to_argv()` / `validate_mlx_flags()`. The injection-protected
   allowlist is dead code in the spawn path.
2. `MLXBackend.make_completion_fn` is a STUB that `await asyncio.sleep(0.001);
   return None`. The old PR NEVER wired MLX into the real completion path.
3. The old `backends/` abstraction differs in API from main's actual spawn
   contract (see Gap A). It cannot be dropped in.

## Current `main` (v0.6.0) spawn model — the seams to port against

Evidence (verified by reading the tree):
- `src/turbohaul/subprocess_mgr.py` (449 lines): exports `SidecarHandle`,
  `spawn_sidecar(binary, gguf_path, port, model_tag, argv_flags,
  binary_fd=None)`. llama-server-SPECIFIC: builds cmd with
  `--slot-save-path`, `--log-file`, execs via `/proc/self/fd/<fd>` (TOCTOU pin),
  polls `/health`, runs `nvidia-smi` VRAM verify. NO backend awareness.
- `src/turbohaul/manager.py:1300-1305` — the real integration surface is
  dependency-injected:
    self._spawn = spawn_fn or spawn_sidecar
    self._wait_healthy = health_fn or wait_until_healthy
    self._sigterm = sigterm_fn or drained_sigterm
    self._vram_verify = vram_fn or verify_vram_cleared
    self._complete_fn = complete_fn or self._default_complete
  Completion is an httpx proxy to `<host>:<port>/v1/chat/completions` (wired via
  DI, not via a backend `make_completion_fn`). mlx-lm serves the SAME OpenAI-
  compatible endpoint, so the existing `_complete_fn` proxy is REUSABLE as-is —
  the old stub `make_completion_fn` is obsolete.
- `src/turbohaul/manifest.py:583` — `class Manifest(BaseModel)` has
  `gguf_blob_sha256` (validator at line 605 REJECTS empty string: "must be 64
  hex chars"), `llama_server_flags: dict`, `expected_vram_bytes` (mandatory VRAM
  pre-check). NO `backend` / `model_repo` / `mlx_server_flags` / `mlx_*` fields.
- `src/turbohaul/config.py` — `RuntimePathsConfig` (line 53) has
  `llama_server_binary` + `llama_server_binary_sha256`. NO `mlx_python_binary` /
  `mlx_models_dir`.
- VRAM/safety model is NVIDIA-centric: `main_gpu`, `detect_foreign_gpu_apps()`,
  `_vram_admits_locked()`, `expected_vram_bytes`, nvidia-smi. On Apple Silicon
  there is no discrete VRAM — MLX uses unified memory. This logic must be
  bypassed/neutral for `backend: mlx`.

## Port plan (recommended shape)

### 1. Backend field on Manifest (manifest.py)
- Add optional `backend: str = "llama.cpp"` (or `Literal["llama.cpp","mlx"]`).
- Add optional `model_repo: str = ""` (HF repo id for MLX).
- Add optional `mlx_server_flags: dict[str, Any] = {}`.
- Add optional `mlx_model_path: str = ""` (local dir for MLX).
- Fix `gguf_blob_sha256` validator to allow empty string (MLX models have no
  GGUF blob). Reuse the v0.3.0 empty-string guard.

### 2. Config fields (config.py)
- `RuntimePathsConfig`: add `mlx_python_binary: Path | None = None` (null = use
  running interpreter) and `mlx_models_dir: Path | None = None`.
- Keep `llama_server_binary` (required by schema; ignored on macOS).

### 3. Spawn dispatch (manager.py)
- Replace the single `self._spawn = spawn_sidecar` with a backend-aware
  dispatcher: read `manifest.backend`; if `"mlx"` → call a new
  `mlx_spawn(...)`; else the existing `spawn_sidecar`.
- New `src/turbohaul/backends/mlx_spawn.py` (or fold into subprocess_mgr):
  build `python -m mlx_lm.server --model <repo|path> --host 127.0.0.1 --port <p>`
  + `mlx_flags_to_argv(manifest.mlx_server_flags)` run THROUGH `validate_mlx_flags`
  (close the old injection gap). Pop `PYTHONPATH` from the child env (the
  pythonpath-shadow-bug fix). Return a `subprocess_mgr.SidecarHandle` (SAME shape
  as the llama.cpp path) so the rest of the manager is backend-agnostic.

### 4. Health check
- mlx-lm health is `/v1/models`, not `/health`. Either:
  (a) give `SidecarHandle` an optional `health_endpoint` field, or
  (b) branch in `_wait_healthy` on `manifest.backend`.
  Recommend (a): add `health_endpoint: str = "/health"` to `SidecarHandle`, set
  it to `/v1/models` for MLX, and have `wait_until_healthy` probe that path.

### 5. Teardown
- Reuse `drained_sigterm` (backend-agnostic, operates on `SidecarHandle.pid` +
  pgid). No change needed if `SidecarHandle` is shared.

### 6. Completion
- REUSE the existing httpx `/v1/chat/completions` forwarder. Do NOT port the old
  stub `make_completion_fn`. This is the key simplification vs the old PR.

### 7. VRAM / safety for MLX
- For `backend: mlx`, skip `verify_vram_cleared` and the `expected_vram_bytes`
  pre-check (or remap to unified-memory RSS if available). `main_gpu` is N/A.
  Keep RAM safety (`safety_min_free_ram_mib`) since MLX shares system memory.

### 8. Platform gate
- Keep `_check_availability()` (Darwin + arm64). Surface a clear error if a
  manifest declares `backend: mlx` on non-Apple-Silicon. Log at boot which
  backends are available.

### 9. Launcher + tests + docs
- Port `turbohaul-launcher.sh`.
- Port/adapt `tests/test_backends_mlx.py` to the new `SidecarHandle` shape and
  the `validate_mlx_flags` wiring.
- Update README "macOS" section + ARCHITECTURE with the backend model.

## Things to verify on a real macOS run (the actual evaluation)
1. `backend: mlx` manifest spawns `python -m mlx_lm.server` (not llama.cpp, not
   CPU torch). Confirm via `ps` of the child cmdline.
2. No torch/CUDA import path is invoked. Confirm `mlx` is the loaded framework
   (e.g. Activity Monitor "Metal" / `sample` shows MLX, or `MLX` in the server
   startup log). 
3. Performance: tok/s on a small model (e.g. Qwen3-1.7B-4bit) vs the same model
   under llama.cpp/Metal. Flag any regression suggesting a CPU fallback (CPU
   would be 5-10x slower than MLX on Apple Silicon).
4. Health/teardown/idle-hot/KV reuse all work through the shared `SidecarHandle`.
5. `pythonpath-shadow` and `gguf validator` fixes are present and exercised by
   tests.

## Risks / decisions needed
- R: Whether to keep the `backends/` package abstraction (old PR style) or a
  thinner `manifest.backend`-switched dispatch inside `manager.py`. Recommend the
  thinner dispatch — main already centralizes spawn via DI; a whole `backends/`
  ABC layer is unused machinery (the old one even shipped a stub completion fn).
- R: Unified-memory accounting for MLX safety. Simplest: disable VRAM pre-check
  for MLX, keep RAM safety.
- R: The old PR's `SpawnRequest`/`BackendInterface` types are NOT reused; we map
  onto main's `spawn_sidecar` signature + `SidecarHandle`. Confirms this is a
  PORT, not a rebase.

## Verification that the port is "done"
- All new tests pass on macOS (Apple Silicon) with an MLX model.
- `curl .../v1/chat/completions` returns tokens; child process is `mlx_lm.server`.
- No `nvidia-smi` / CUDA calls on the macOS path.

## What actually shipped (update 2026-07-12, post real-run)
The port is implemented and verified end-to-end on Apple Silicon with a real
local MLX model (`mlx-community/Llama-3.2-1B-Instruct-4bit`, ~396 tok/s).

Two findings from the real run (not visible in mocked tests):
1. **Readiness gate is compatible as-is.** `mlx_lm server` serves `/health`
   returning `{"status":"ok"}`, which satisfies `subprocess_mgr.wait_until_healthy`
   (required field `status`, ok-status set). No `health_endpoint` field was
   needed — the `SidecarHandle` carries only `model_id`, not a health path.
2. **Completion `model` rewrite (critical).** `mlx_lm server` is single-model
   and re-resolves the request `model` field as a HuggingFace repo_id, so
   forwarding the Turbohaul tag 404s (it tries to download
   `huggingface.co/api/models/<tag>`). Fix: `SidecarHandle.model_id` advertises
   the `--model` arg (HF repo id or local path); `make_llama_server_complete_fn`
   sends `handle.model_id or slot.model_tag`. llama.cpp handles leave
   `model_id=None` → unchanged behavior. Covered by unit + e2e tests.

Files:
- `src/turbohaul/mlx_spawn.py` — MLX spawner (Darwin/arm64 gate, flag allowlist,
  PYTHONPATH pop, returns `SidecarHandle` with `model_id`).
- `src/turbohaul/manifest.py` — `backend`/`model_repo`/`model_path`/`mlx_server_flags`;
  relaxed `gguf_blob_sha256` for MLX.
- `src/turbohaul/config.py` — `mlx_python_binary`/`mlx_models_dir` (optional).
- `src/turbohaul/manager.py` — both spawn sites branch on `manifest.backend`;
  skip GGUF/KV-restore for MLX; VRAM gate is a no-op for MLX.
- `src/turbohaul/subprocess_mgr.py` — `SidecarHandle.model_id` (default None).
- `src/turbohaul/api/chat_completion.py` — `model` rewrite for MLX.
- `turbohaul-launcher.sh` — ported (macOS helper).
- `pyproject.toml` — `[mlx]` extra (`mlx-lm>=0.21.0`).
- `tests/test_mlx_backend.py` — 20 unit tests (allowlist, cmd build, PYTHONPATH
  pop, precondition gating, manifest validation, `model_id`, completion rewrite).
- `tests/e2e_mlx_smoke.py` — real supervised spawn + 200 OK `/v1/chat/completions`
  through `mlx_lm server` on Apple Silicon. Run manually (needs `mlx-lm` + a model).
- README + this spec — usage + record.

General-PR safety: MLX is hard-gated to Darwin/arm64 at spawn (`_check_mlx_preconditions`);
the module imports cleanly on any platform (no macOS-only import deps); the
`[mlx]` extra is optional so non-Apple-Silicon users are unaffected.

- The ~8GB CUDA Docker image is still excluded from macOS builds (already
  hardened in commit 88e9b97).
