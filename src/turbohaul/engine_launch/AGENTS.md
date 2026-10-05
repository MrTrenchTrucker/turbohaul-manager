# Module: engine_launch

## Purpose
Decide the environment each engine (llama-server) process starts with. Today that means putting a vision model's image
projector on the same card TurboHaul placed the model on. First module of TurboHaul's module system (decisions/ADR-001).

## Owns
- The full environment dict passed to one engine launch: the manager's own environment, plus the per-launch additions.
- The vision projector's device: for a model with a projector (`--mmproj`) placed on one card (`--split-mode none`,
  `--main-gpu N`), the engine gets `MTMD_BACKEND_DEVICE=CUDA<N>`.
- The rule for what happens when the operator already set `MTMD_BACKEND_DEVICE`, and the rule for split models.

## Does Not Own
- Choosing the card or the split for a model: that is the manager's placement (manager.py).
- Building the engine's command-line flags: manifest.py and the manager's argv assembly. This module reads the final flags; it never changes them.
- Starting, health-checking, reaping or stopping the process: subprocess_mgr.py (spawn_sidecar).
- The engine itself (engine/llama-cpp-turboquant).
- Queue behaviour after a failed load: manager.py.

## Public Interface
- `launch_env(argv, base_env) -> dict[str, str]`: the complete environment for one engine launch, built from its final
  argv and the manager's environment. For an engine without a projector it returns a copy of `base_env` with nothing added.
- `preset_device_mismatch(argv, base_env) -> str | None`: when an operator's preset `MTMD_BACKEND_DEVICE` names a different
  card than `--main-gpu` on a single-card or row placement of a model with a projector, the one-line warning text naming
  both; otherwise `None`. `spawn_sidecar` logs it; this module never logs.

## Depends On
- Nothing inside TurboHaul. It reads plain argv strings and an environment mapping.

## Invariants
- A text-only engine gets exactly the environment it would get without this module.
- The environment is never filtered: everything in `base_env` reaches the engine.
- The projector device names the same card as `--main-gpu` for a single-card placement, unless the operator set it first
  (see below). `--main-gpu` and the ggml
  CUDA device names (`CUDA<N>`) use the same CUDA device order, so the index carries over unchanged, as long as the engine is
  started with no `--rpc` servers and no `--device` list (true today: the manifest flag allowlist has no `device` entry and
  `rpc` is a denied flag). If either is ever added, this rule must be revisited:
  llama.cpp puts RPC devices first and `--device` reorders the list `--main-gpu` indexes.
- Split models: with `--split-mode row` the projector goes on the `--main-gpu` card (row mode keeps intermediate results and
  the KV cache there). With `--split-mode layer` (or any other split mode, or none given) the module sets nothing: the model
  spans every card, there is no home card, and the projector stays on the engine's default (the first GPU).
- An operator who already set `MTMD_BACKEND_DEVICE` in the manager's environment wins: the module never overrides it. When it
  names a different card than `--main-gpu` on a single-card or row placement, the launch logs one warning naming both.
- Pure functions: no I/O (logging included), no GPU access, no reading of live state.

## Test Locations
- Unit: tests/unit/engine_launch/
- Contract: tests/contract/test_engine_launch_contract.py

## Known Gotchas
- The engine's projector code (engine/llama-cpp-turboquant/tools/mtmd/clip.cpp:183-195) uses `MTMD_BACKEND_DEVICE` if it is
  set, and otherwise the FIRST GPU, whatever `--main-gpu` says. That is why the projector went to GPU 0, and why a
  vision model placed on an empty GPU 1 could fail to load while GPU 0 was full.
- A bad device name in `MTMD_BACKEND_DEVICE` does not fail: the engine logs a warning and falls back to the first GPU. Tests
  must check the exact value, not just that the variable is present.
- Tests must be hermetic: no nvidia-smi, no real engine, no live GPU state.
- A layer-split vision model still puts its projector on the first GPU. If that card is full, its load can still run out of
  memory; when the manager recognises the failure as out-of-memory it can put the request back in the queue instead of
  failing it (manager.py).
