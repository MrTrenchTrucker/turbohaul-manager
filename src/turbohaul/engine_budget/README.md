# engine_budget

How many copies of one model's engine TurboHaul may run at the same time.

## What it decides
When requests for the same model pile up, TurboHaul can start a second engine (a "sidecar") for that model instead of
making everyone wait for the first one. This module answers one question: how many engines may this model have right now?

The answer comes from two limits:
- **The box-wide budget**, `queue.max_parallel_sidecars`: the most engines the whole box runs at once, summed across ALL
  models. Its value comes from three layers, lowest to highest: the config file, then the `TURBOHAUL_MAX_PARALLEL`
  environment variable, then a value saved in the Settings tab (which survives a restart).
- **What fits**: how many cards can hold another copy of the model right now.

A second engine of the same model also needs a one-card-per-engine placement, because only that placement can host a
second copy. The manager therefore offers a second engine only to a manifest with `auto_place: true` and an explicit
`llama_server_flags.split_mode: "none"` (an absent `split_mode` does not count). A pinned model, or a model that spans
several cards, keeps one engine. This module never looks at placement: it only reports how many engines the budget and
the cards allow, and the manager applies the placement condition.

This module never answers with fewer engines than a model already runs. If the card probe fails, the answer is "keep
what is running". The module has the same fail-safe for an unreadable manifest, but the manager reads the manifest first:
if that read fails, the manager treats the model as a single-engine model and does not call this module.

## What it does not decide
- **Context windows.** Each engine can serve several conversations at once, one per context window. That is the manifest's
  `llama_server_flags.parallel` setting (with `ctx_size` for the window size), and it is unchanged. Engines times windows
  is how many conversations a model can serve at once.
- **Placement.** Which card an engine goes on, and whether a model is eligible for a second engine at all (see above),
  is the manager's placement code.

## The retired setting
Manifests used to carry `max_instances`, a per-model limit on engines. It quietly overrode the box-wide budget, so a user
who set `max_parallel_sidecars` to 2 still got one engine per model. It is gone (decisions/ADR-002). Saved manifests that
still carry the key keep loading: the key is dropped, with a warning, and its value is not carried over to any other
setting.

## How it is used
The manager measures the inputs while it holds its registry lock, builds an `EngineBudgetInputs`, and calls
`effective_engine_cap`, but only for a tag whose manifest asks for one card per engine, and only while the box budget
has a slot this tag can use. When the budget is spent, the answer is the engines the tag already runs (none for a cold
tag), with no card probe. Any other tag, while a slot is free, keeps the engines it already runs (at least one, so a
cold tag can start its engine), also with no card probe. The result says how many engines the model may run in total and which limit applied.
The module does no I/O, so it is tested without a running manager.
