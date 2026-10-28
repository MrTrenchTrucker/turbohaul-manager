# co_residence

Starts a further engine for a model that already runs one, when the second engine fits.

## The problem it solves
When requests for one model pile up behind a busy engine, TurboHaul can start a second engine (a "sidecar") for that
model on another card. The arrival path asks that question once, when a request arrives. If the answer is no at that
moment, the request is parked in the busy engine's inbox and nothing asks again, so a card that frees up a minute later
stays empty while the request waits.

This module asks again. The dispatch loop calls `run_recheck` once per pass; the module decides whether this pass may do
anything, finds a waiting request that has waited long enough, checks whether another engine fits, and if so starts it.

## What decides "fits"
Two checks, both owned by the manager and called through the host:
- the card picker finds a card that can hold the model on its own;
- the cross-resident VRAM gate admits the model on that card next to the engines already loaded.

A decline carries one of two reasons in the re-check. `vram_refused`: cards are visible but none has room for one more
engine (the picker found no card, or the gate refused the one it chose). `no_card`: the model cannot be placed as a second
engine at all, or no card is visible. The picker keeps a safety floor of VRAM free that the gate does not, so a plain
shortage of VRAM makes the picker find no card; the re-check then reads the free VRAM per visible card to tell a shortage
(`vram_refused`) from no card at all (`no_card`).

The re-check uses nothing else. It does not read the box-wide budget `queue.max_parallel_sidecars` and does not compare the
number of engines with a per-model ceiling, so on a box whose budget is one engine, the re-check can still take the box from
one engine to two, three, and so on, for as long as each next engine fits. It adds no memory gate of its own. The arrival
path is unchanged: it still stops at the budget and the per-model ceiling, and says so with the reasons `cap` and `box_full`.

## How a pass works
1. **Gate.** At most one attempt per 10 seconds for the whole box. An attempt that finds nothing counts too.
2. **Scan.** Waiting requests come from the staging queue (for a model with a live engine) and from the inbox of an ACTIVE
   engine. A request counts as demand after 5 seconds, or 120 seconds when an engine's grace loop is about to claim it.
   The oldest request of the first model that may run is chosen; a model that found no room in the last 60 seconds is
   skipped.
3. **Phase A, probe.** Under the registry lock, ask the two fit questions. No request is taken. A no-fit answer is
   remembered for the model for 60 seconds and logged as one `CO_RESIDENCE_DECLINED` line.
4. **Phase B, take.** With no registry lock held, take the request out of the staging queue or the engine's inbox. If the
   dispatcher or the engine got it first, stop.
5. **Phase C, reserve.** Marked as the in-flight reservation for the status display (set before the lock, cleared on every
   exit), and under the registry lock, reserve the engine directly on the card the fit check chose. The request is
   never sent back through the arrival path, because that path enforces the budget and may evict an idle engine to make
   room, which would swap models instead of running them side by side. If another reservation took the card in the
   meantime, the reserve's own VRAM gate requeues the request and the model is remembered as having no room.

`run_recheck` never raises (a cancellation excepted): a failure fails that one request and the dispatch loop goes on.

## The two verdicts
- `fit_verdict` is the re-check's answer: card and VRAM only; reasons `no_card`, `vram_refused` (meanings above).
- `arrival_verdict` is the arrival path's answer: the model's own ceiling and the box budget first (`cap`, `box_full`), then
  the same two questions, plus the Fast Lane relocation override for a cold start. It writes the decline line once per
  60 seconds per model and reason, and nothing for a model with no live engine.

## The log lines
- `CO_RESIDENCE_DECLINED via=<recheck|arrival> tag=... reason=... eff_cap=... instances=... residents=... max_sidecars=...
  need_mib=... card=... split=...`: a further engine was declined, with the reason. A value that does not apply shows `-`.
- `CO_RESIDENCE_RECHECK tag=... source=<staged|inbox> slot=... waited_s=... instances=... residents=...`: the re-check is
  about to take a waiting request.

## The host
The module imports only `queue_liveness` (for the liveness predicate) from the rest of TurboHaul. It is handed the manager and may use only the members listed in
`host.py` (and in the card's Depends On). Requests and engines are described by small private views of the host
contract that name only the attributes that are read. A test can therefore drive the whole module with a small fake host that records every call.

## Not to be confused with
`coresident_admission` is a different module: it owns the probe-retry admission rule. This module owns the periodic
re-check and its log lines.

## Files
`constants.py` tuning numbers; `window.py` the debounce and the demand age; `decision.py` the per-tag decision; `memo.py`
the expiring no-room memo; `state.py` what the re-check remembers; `lines.py` the log text; `host.py` the host contract;
`scan.py` finding and taking waiting requests; `verdict.py` the two verdicts; `recheck.py` the three phases.
