"""Fast Lane victim-candidacy is inert at max_parallel_sidecars=1 (the shipped
default, config.py FastLane... max_parallel_sidecars=1). KILLING TEST: parameterised over cap in
{1, 2}, asserts (not merely prints) that a resident genuinely serving a Fast-Lane-ranked client's
turn is a member of the victim-candidate set used by _is_worst_ranked_loaded_locked.

RED at cap=1 on the current code (no fix ships with this test; the
sidecar-count unification is deferred to follow-on work).
GREEN at cap=2 as the positive control -- same assertions, the OTHER production code path
(worker_loop branches to _dispatch_loop at cap>=2 vs _process_slot at cap<=1 in
manager.py), proving the checker is not vacuously true or false regardless of what the code does.

Built on a standalone probe of the same defect and
the shared Fast Lane test fixture's validated wiring -- reused, not reinvented.

drive_to_active (the fixture's usual admission helper) CANNOT be used for the cap=1 arm: it polls
_model_residents() internally and TIMES OUT at cap=1 for the exact same reason
this test exists -- no model-keyed resident is ever created at cap=1, so the predicate it waits on
never becomes true. Both cap arms here instead use the "submit, wait wall-clock, then inspect real
state" pattern that probe established, per the fixture's own documented trap warning that such a
timeout can mean either a real bug OR "asked for a state the runtime cannot hold" -- the tolerant
pattern is what tells the two apart without depending on the very machinery under test.
"""

import asyncio
import pytest

from tests._fastlane_fixture import (
    boot_ranked_runtime, seed_manifest, make_fakes, high_vram,
)
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import TurbohaulManager, ResidentState


def _is_genuinely_serving(mgr) -> bool:
    """True iff SOME representation (singleton active_slot at cap 1, or a
    model-keyed resident's ACTIVE state at cap>=2) shows real work in flight
    right now. Deliberately checks BOTH so this non-vacuity guard does not
    itself depend on the cap-dependent code path under test."""
    singleton = mgr._residents.get("__phase0_singleton__")
    if singleton is not None and getattr(singleton, "active_slot", None):
        return True
    return any(r.state is ResidentState.ACTIVE for r in mgr._model_residents())


_ONE_RESIDENT_CAP_XFAIL_REASON = (
    "Confirmed by an independent code trace: "
    "_model_residents() (manager.py ~4928) excludes the Phase-0 singleton by "
    "key, so at max_parallel_sidecars=1 (the shipped default) the sole loaded resident is "
    "invisible to _is_worst_ranked_loaded_locked's candidate list -- Fast Lane's entire designated-victim "
    "preemption mechanism is structurally inert at this cap. The fix (sidecar-count "
    "unification) is deferred to follow-on work; the scope here is diagnosis + this test + "
    "the unification-shape proposal, not the refactor itself. strict=True is deliberate: the "
    "moment the unification fix lands this arm becomes an unexpected PASS, which strict turns "
    "into a failure -- forcing removal of this marker rather than letting it go stale."
)


@pytest.mark.parametrize("cap", [1, 2])
@pytest.mark.asyncio
async def test_DISCRIMINATOR_loaded_fastlane_resident_is_a_victim_candidate(tmp_path, cap):
    """The claim under test: a resident that is DEMONSTRABLY serving a
    Fast-Lane-ranked client's turn right now must be discoverable as a victim
    candidate -- _is_worst_ranked_loaded_locked must be able to see it at
    all, regardless of max_parallel_sidecars. At cap=1 it structurally
    cannot: the work runs on the Phase-0 singleton (manager.py), which
    _model_residents() (manager.py) excludes by key
    (_SINGLETON_RESIDENT_KEY), and _is_worst_ranked_loaded_locked
    (manager.py) builds its candidate list ONLY from
    _model_residents()."""
    rules = [
        FastLaneRule(container_name="open-webui", tag_ranks=FastLaneTagRanks(main=1)),
        FastLaneRule(container_name="agent-b", tag_ranks=FastLaneTagRanks(main=2)),
    ]
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=cap, grace_seconds=5)
    seed_manifest(boot, "m1", main_gpu=0)
    gate = asyncio.Event()  # unset: m1's turn stays mid-turn until we release it below
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task = asyncio.create_task(mgr.submit_and_wait(
            "m1", "p", thread_id="t1", client_meta={"container_name": "agent-b"}))
        try:
            await asyncio.sleep(3.0)  # wall-clock, not a predicate -- fixture's own documented trap

            # NON-VACUITY GUARD 1: prove the request was really admitted before
            # trusting an empty candidate set as meaningful, rather than as a
            # fixture/boot failure that would make it vacuous.
            assert not task.done(), (
                f"cap={cap}: the request never even got admitted in 3s -- this would make "
                f"an empty candidate set MEANINGLESS (nothing to be a candidate for), not "
                f"evidence of the defect. Check fixture wiring, not the hypothesis."
            )
            # NON-VACUITY GUARD 2: prove SOMETHING is really ACTIVE right now,
            # under either representation, before trusting the candidate-set
            # reading at all.
            assert _is_genuinely_serving(mgr), (
                f"cap={cap}: task is not done but neither the singleton's active_slot nor "
                f"any _model_residents() entry shows ACTIVE -- can't tell what's serving "
                f"it. Fixture problem, not evidence about the candidate set."
            )

            candidates = mgr._model_residents()
            worst = [r.model_tag for r in candidates if mgr._is_worst_ranked_loaded_locked(r)]

            # THE ASSERTION: work is genuinely being served for a Fast-Lane-ranked
            # client (the rank-2 client) right now. The victim-candidate
            # machinery must be able to see SOMETHING to designate. An empty
            # candidate set here means Fast Lane's victim preemption
            # mechanism is structurally inert at this cap, independent of
            # anything an arriving higher-priority claimant would otherwise do.
            singleton = mgr._residents.get("__phase0_singleton__")
            assert candidates, (
                f"cap={cap}: _model_residents() is EMPTY while work is demonstrably in "
                f"flight (task.done()={task.done()}, singleton active_slot="
                f"{getattr(singleton, 'active_slot', None)}). "
                f"_is_worst_ranked_loaded_locked can only select from an empty list, so no "
                f"resident can EVER be the designated victim at max_parallel_sidecars={cap}. "
                f"Fast Lane's victim preemption mechanism is structurally inert at this cap."
            )
            assert worst == ["m1"], (
                f"cap={cap}: expected the sole loaded resident 'm1' to be identified as "
                f"worst-ranked-loaded (it is the only candidate present), got {worst!r}."
            )
        finally:
            gate.set()
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except BaseException:
                pass
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except BaseException:
                pass
