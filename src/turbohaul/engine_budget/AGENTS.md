# Module: engine_budget

## Purpose
Decide how many engine (llama-server sidecar) processes one model may run at the same time. The limits this module
applies are the box-wide `queue.max_parallel_sidecars` setting and the cards the model fits on (the manager separately
requires a one-card-per-engine placement before it asks). There is no per-model sidecar limit: the manifest setting
`max_instances` is retired (decisions/ADR-002).

## Owns
- The engine count for one model tag: the TOTAL number of engines it may run once every admitted one has loaded, from the
  box-wide sidecar budget, the engines already running on the box, the engines this tag already runs, and how many cards
  can host another copy of the model.
- The fail-safe for a manifest that cannot be read, when the caller reports it: keep the engines already running, never
  fewer. (The manager today handles an unreadable manifest before it calls this module: see Known Gotchas.)
- The reason attached to every decision, so a log can say which limit applied.
- The total number of context windows a tag can serve (engines times windows per engine), as a helper. The manager does
  not call it today.

## Does Not Own
- Which card an engine goes on, or how a model is split across cards: placement in manager.py (legacy).
- The context windows inside one engine (`llama_server_flags.parallel`) and the context size (`ctx_size`): the manifest
  (manifest.py). This module reads the windows-per-engine number only to report the context total; it never changes it.
- Measuring the inputs (engines live on the box and for the tag, the card-fit probe, the manifest read): the manager, which
  holds the registry lock and passes plain numbers in.
- Starting, health-checking or stopping engines: subprocess_mgr.py.
- Removing the retired `max_instances` key from saved manifests: manifest handling (manifest.py and the manifest store).

## Public Interface
- `EngineBudgetInputs`: the measured inputs: the box-wide sidecar budget, engines live on the box, engines live for this
  tag, cards that can host another engine, windows per engine, and whether the manifest could not be read.
- `EngineBudget`: the decision: total engines for the tag, engines already live, and the reason. `admits_additional_engine`
  and `additional_engines` are derived from those.
- `effective_engine_cap(inputs) -> EngineBudget`: the total number of engines this tag may run.
- `context_window_capacity(inputs, engines_total) -> int`: the total number of concurrent context windows for the tag.

## Depends On
- Nothing inside TurboHaul. Plain numbers in, a decision out.

## Invariants
- The answer never admits an engine beyond `queue.max_parallel_sidecars`: the engines it adds are limited to the free slots.
  A tag that already runs more engines than the budget (for example after the budget is lowered) keeps them, because of the
  next rule.
- The answer is never lower than the engines this tag already runs: a failed probe or an unreadable manifest must not shut
  down a running engine.
- No per-model sidecar limit: `max_instances` is not an input, and nothing in this module reads it.
- The number of windows per engine never changes the number of engines. They are two separate settings.
- A tag with nothing running and no card reported as fitting still gets one engine, provided the box budget has a free
  slot; the VRAM check at spawn time guards the actual load. With the budget spent, a tag with nothing running gets zero.
- Pure functions: no I/O, no logging, no GPU access, no reading of live state.

## Test Locations
- Unit: tests/unit/engine_budget/
- Contract: tests/contract/test_engine_budget_contract.py
- The manager's use of the number (legacy code): tests/test_multinstance_gate_vs_budget.py

## Known Gotchas
- `max_instances` sounded like "requests at once" but limited engine processes per model, and it silently overrode the
  box-wide budget: with `max_parallel_sidecars` at 2 or more, a model still got one engine. That is why it was retired.
  The windows-per-engine setting is `llama_server_flags.parallel`, and it is unchanged.
- The engine count and the placement path answer different questions; keep them apart. Whether a tag can be placed as more
  than one engine comes from its manifest (`auto_place: true` with an explicit `split_mode: "none"`, checked in
  manager.py), not from this count.
- The manager reads the manifest before it calls this module. If that read raises, the manager answers 1 engine and does not
  call the module, so the module's `manifest_unreadable` fail-safe is not exercised on that path today.
- `effective_engine_cap` returns the TOTAL for the tag, not the number still to start. Callers compare it with the engines
  already live; `additional_engines` gives the remainder.
