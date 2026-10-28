# Module: co_residence

## Purpose
Start a further engine (llama-server sidecar) for a model tag that already runs one, when a second engine fits. A request
that arrives while the tag's engine is busy is declined once by the arrival path and parked behind that engine; without
this module nothing asks again, even when a card is free. The module re-asks on a bounded schedule (one attempt per 10 s
for the whole box, one probe per tag per 60 s no-room window) and starts the engine when the card picker finds a card and
the cross-resident VRAM gate admits it. The box-wide engine budget (`queue.max_parallel_sidecars`) does not limit that
re-check: the card and VRAM fit is the only ceiling. It also writes the `CO_RESIDENCE_*` log lines.
Not coresident_admission: that module owns the probe-retry admission rule; co_residence owns the fit-based periodic re-check and its CO_RESIDENCE_* lines.

## Owns
- The periodic re-check, end to end (`run_recheck`): the box-wide debounce, the scan for waiting demand (staging queue and
  the inbox of an ACTIVE engine), the per-tag decision, the probe under the registry lock, taking the waiting request out,
  and reserving the engine directly on the card the fit check chose.
- The re-check verdict (`fit_verdict`): placement plus the VRAM gate, with no box budget and no per-tag engine count.
  Its only decline reasons are `no_card` (the tag cannot be placed as a second engine, or no card is visible) and
  `vram_refused` (cards are visible but none has room for one more engine, or the cross-resident gate refused the
  chosen card).
- The arrival verdict (`arrival_verdict`): the logic of the multi-engine spawn step of the arrival path (the tag's own
  engine ceiling, the box budget, then the same fit questions, with the Fast Lane relocation override). Its decline
  reasons are `cap`, `box_full`, `no_card` and `vram_refused`.
- The no-room memo, the decline-line throttle and the attempt time (`RecheckState`).
- The two log lines, `CO_RESIDENCE_DECLINED` and `CO_RESIDENCE_RECHECK`, and the logging of them.
- Taking one request out of an engine's inbox without disturbing the others (`take_from_inbox`).
- The tuning numbers: 10 s debounce, 60 s memo, 5 s demand age (120 s for a protected request), 64 memo entries.

## Does Not Own
- The probe-retry admission rule: coresident_admission (a separate module).
- The per-tag engine ceiling and the box budget as a number: engine_budget and the manager's `_effective_cap`. The arrival
  verdict receives the ceiling as an argument and reads the budget from the host; the re-check reads neither.
- Placement, the card picker, the VRAM math and the reserve itself (the registry entry, the port, the driver): the manager
  (legacy) and its placement code. This module calls them through the host.
- The registry lock and the clock: the host owns them; the module takes the lock the host hands it and asks the host for
  the time.
- The queue, the engines' drivers, eviction and teardown, the Fast Lane claim: the manager and the queue.
- Calling the re-check: the dispatch loop in the manager does, once per pass.

## Public Interface
- `RECHECK_MIN_INTERVAL_S` (10.0), `NO_ROOM_MEMO_S` (60.0), `DEMAND_MIN_AGE_S` (5.0), `PROTECTED_DEMAND_MIN_AGE_S` (120.0),
  `MEMO_MAX_ENTRIES` (64): the tuning numbers.
- `RecheckGate`: the box-wide answer: `open` and `reason`.
- `recheck_window(*, now, last_recheck_at, min_interval_s) -> RecheckGate`: debounce only. Reasons: `debounced`, `window_open`.
- `demand_is_eligible(*, age_s, protected) -> bool`: has a waiting request waited long enough to count?
- `RecheckInputs`: what was measured for one tag: live engines, eligible waiting requests, whether the memo blocks.
- `RecheckDecision`: the per-tag answer: `run` and `reason`.
- `decide_recheck(inputs) -> RecheckDecision`: probe this tag now? Reasons: `no_live_instance`, `no_eligible_demand`,
  `no_room_memo`, `recheck_due`.
- `NoRoomMemo(ttl_s, max_entries)`: `note(key, now)`, `blocked(key, now)`, `clear(key)`, `len()` and `keys()`.
- `DeclineReason`: why a further engine was declined: `cap`, `box_full`, `no_card`, `vram_refused`.
- `format_step2_decline(*, via, tag, reason, eff_cap, instances, residents, max_sidecars, need_mib, card, split) -> str`: the
  one-line decline text.
- `format_recheck_action(*, tag, source, slot_id, waited_s, instances, residents) -> str`: the one-line re-check text.
- `RecheckState()`: `last_recheck_at`, `norom` (no-room memo per tag) and `declog` (decline-line throttle).
- `Candidate`: one waiting request: `age`, `rank`, `source`, `slot`, `owner`.
- `inbox_slots(r) -> tuple` and `take_from_inbox(r, slot) -> slot | None`: read, or take by identity, an engine's inbox.
- `gather_candidates(host, now, grace_pairs) -> list[Candidate]`: waiting requests old enough to count, oldest first.
- `pick_candidate(host, state, cands, now) -> Candidate | None`: the oldest candidate of the first tag allowed to run.
- `Verdict`: `admit`, `card`, `split`, `reason`, `need`, `eff_cap`.
- `fit_verdict(host, state, tag, insts, now) -> Verdict` (async): the re-check verdict. Caller holds the registry lock.
- `arrival_verdict(host, tag, insts, eff_cap, slot) -> Verdict` (async): the arrival verdict. Caller holds the registry lock.
- `run_recheck(host) -> None` (async): one re-check pass. Never raises except `asyncio.CancelledError`.
- `CoResidenceHost`: the host contract (see Depends On).

## Depends On
- `queue_liveness` (its package entry point only): `client_gone(slot)`, the shared liveness predicate, used by the scan
  and by the take-time re-check so a request whose client has hung up is neither counted as demand nor given an engine.
- Otherwise nothing inside TurboHaul. The module is handed one `host` (the manager) and may touch
  only the members declared in `host.py` (`CoResidenceHost`); the contract test records every member used on a recording
  host and fails on an undeclared one, and on a declared one that no scenario uses.
- Attributes: `queue` (`staging_snapshot()`, async `remove(slot_id)`), `runtime` (`runtime.queue.max_parallel_sidecars`,
  read by the arrival verdict only), `_registry_lock` (an async context manager).
- Methods: `_co_residence_now()`, `_co_residence_state()`, `_model_residents()`, `_instances_for(tag)`, async
  `_grace_active_exclusions()`, `_route_cap_and_placement(tag)`, `_resolve_placement_locked(tag)`,
  `_card_avail_mib()` (free MiB per visible card, read only when the picker found no card),
  `_pick_card_for_new_instance(tag, need_mib)`, `_vram_admits_locked(need, parallel, main_gpu, split_mode)`, async
  `_reserve_and_start_locked(slot, main_gpu_override, split_mode_override)`, `_remember_affinity(tag, session_key,
  resident_key)`, `_session_key_for(slot)`, `_reserving(tag)` (a context manager that names the in-flight reservation
  for the status display; the re-check sets it around its reserve), `_unpark_fastlane_claim(slot)`,
  `_handle_unloaded_slot(slot)`,
  `_fail_completion_future(slot, exc)`.
- Resident and slot objects are typed by small private views of the host contract (private to `host.py`; only
  `CoResidenceHost` is public) that list only the attributes read: resident `state.name`,
  `model_tag`, `inbox`, `idle_thread_id`, `resident_key`; slot `slot_id`, `model_tag`, `thread_id`, `created_at`,
  `is_evicted`, `fastlane_relocated_main_gpu`, `fastlane_relocated_split_mode`.

## Invariants
- The re-check never reads the box budget and never compares the number of engines with a ceiling: `fit_verdict` drops the
  cap-derived count that `_route_cap_and_placement` returns, and `recheck_window` takes neither the budget nor the live
  engine count. The reasons `box_full` and `cap` come only from `arrival_verdict`.
- The gate is a debounce only: closed for `RECHECK_MIN_INTERVAL_S` after the last attempt (a negative elapsed time included),
  open otherwise. An attempt is counted when it starts, so a scan that finds nothing is debounced too.
- A further engine is admitted only when the card picker returns a card and `_vram_admits_locked` passes for that card. No
  other gate and no fixed engine count.
- The re-check names its decline by what was seen: a tag that cannot be a second engine is `no_card`; a picker that finds no
  card is `vram_refused` when `_card_avail_mib()` reports at least one visible card and `no_card` when it reports none;
  a gate that refuses the chosen card is `vram_refused`. The arrival verdict keeps `no_card` for any picker miss.
- The three phases keep the lock order queue lock, then registry lock, never the reverse. Phase A probes under the registry
  lock and takes no request. Phase B takes the request out with no registry lock held. Phase C reserves under the registry
  lock, inside the `_reserving(tag)` marker (set before the lock, cleared on every exit; never set for a decline or a
  request that was not taken).
- Phase C reserves directly with the card and split of the verdict (`_reserve_and_start_locked`). It never sends the request
  back through the arrival path: that path enforces the budget and may evict an idle engine to make room, which would turn
  co-residence into a swap. The host handed to the module needs no routing method, and the module calls none.
- A declined verdict takes no request and notes the tag in the no-room memo; a successful reserve clears it; a reserve that
  places nothing notes it.
- `run_recheck` never raises except `asyncio.CancelledError`. A failure in the reserve fails that one request
  (`_fail_completion_future`) and the dispatch loop goes on. A logging failure never changes a verdict.
- The decline line is written once per 60 s window per tag and reason on the arrival path, on every decline on the
  re-check path (the memo already bounds it), and never for a tag with no live engine.
- The no-room memo is bounded and expiring; an entry lasts until `now + ttl_s`, an expired entry is removed when looked up,
  and over the bound the earliest-expiring keys go first, never the key just noted.
- A protected request (its thread is held by an engine's grace loop, or it is the engine's own idle thread) is not demand
  before 120 s. A non-finite or negative age is never eligible.
- A decline line is one line with a fixed field order. A blank or absent value shows as `-` and a run of whitespace as `_`.
- Every file is under 300 lines.

## Test Locations
- Unit: tests/unit/co_residence/ (shared fake host: tests/_co_residence_fakes.py)
- Contract: tests/contract/test_co_residence_contract.py
- The manager's use of the module (legacy code): tests/test_co_residence_fit_admission.py and
  tests/test_co_residence_safety.py

## Known Gotchas
- The two verdicts differ on purpose. A cap-1 box: the arrival path still stops at one engine, the re-check may take it from
  one engine to several while each one fits. After that the box holds more engines than the budget, and the arrival path
  keeps treating the budget as spent.
- The arrival path must not be fed the re-check's request. Handing it back through the arrival path on a box at or over the
  budget enters its make-room branch and evicts an idle engine.
- The card picker never returns a card-spanning placement: `(None, 'none')` means "no card of its own". It also keeps a
  safety floor of VRAM free that the cross-resident gate does not, so a plain VRAM shortage makes the picker return no card
  while the gate would not have been asked. That is why the re-check reads the visible cards on a picker miss: cards
  visible means `vram_refused`, none visible means `no_card`. A tag whose manifest does not ask for one card per engine is
  `no_card` without reading the cards.
- A card the fit check chose may be taken by another reservation before Phase C. The reserve's own VRAM gate then
  requeues the request and the module treats that as "placed nothing".
- A protected warm follow-up is never taken before 120 s: the engine's grace loop claims it within seconds.
- The debounce counts attempts, not successes.
- Only ACTIVE engines contribute inbox demand: an idle engine serves its inbox at once.
- The registry-lock-held verdicts run their card and VRAM probes in worker threads (`asyncio.to_thread`) and keep the lock
  across the await, as the arrival path does.
