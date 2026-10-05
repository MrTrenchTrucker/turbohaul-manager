# ADR-002: Retire the per-model max_instances setting; the box-wide sidecar budget decides (module engine_budget)

**Status:** accepted
**Date:** 2026-10-01

## Context
A model's manifest carried `max_instances`, a limit on how many engine processes (sidecars) that one model may run. The
box-wide `queue.max_parallel_sidecars` setting also limits engines, for all models together. The two disagreed: a manifest
with `max_instances: 1` got one engine even when `max_parallel_sidecars` was 2 or more and there was room for another
copy. Users and agents reading `max_parallel_sidecars = 2` reasonably expected two engines for a
busy model (for example several sub-agents on one model). A per-model engine limit also does not fit multi-GPU placement,
where the cards that can hold a copy are what really limit it.

## Decision
- `max_instances` is removed from the manifest. The box-wide `queue.max_parallel_sidecars` budget, the cards the model
  fits on, and a manifest placed one card per engine (`auto_place: true` with an explicit `split_mode: "none"`) are the
  only limits on how many engines one model runs.
- The per-model context settings stay exactly as they are: `llama_server_flags.parallel` (windows per engine) and
  `ctx_size`.
- The old key is not carried over. Saved manifests that still contain it are migrated (the key is removed), so they keep
  loading and saving, including through the manifest API.
- The engine-count decision becomes the module `engine_budget` (src/turbohaul/engine_budget): pure, with the measured
  inputs passed in by the manager.
- Every document that describes `max_instances` is updated in the same change.

## Reasons
- One setting for one question: how many engines the box runs is a box-wide decision. How many conversations one engine
  serves is a per-model decision, and it already has its own setting.
- It removes a silent override that made `max_parallel_sidecars` look broken.
- A pure module makes the rule testable without starting a manager.

## Consequences
- A manifest no longer limits its own engine count. To keep a model to one engine, an operator lowers
  `max_parallel_sidecars` or relies on what fits; a model whose manifest does not ask for that placement already keeps
  one engine.
- The manifest schema, validator, API and documentation stop mentioning `max_instances`, except to say it was removed.
- Changes to how many engines a model may run now go through engine_budget and its card.
