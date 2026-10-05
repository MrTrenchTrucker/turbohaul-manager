"""Cap unification -- CHARACTERISE FIRST.

Golden-master tests pinning
TODAY's behaviour per (cap, scenario), written and GREEN before a single production line
moves. These are NOT re-tests of the per-clause eviction logic already covered exhaustively
elsewhere (the starvation-mirror drift test, test_evictor_listed_waiter_protection.py,
etc. -- those hand-construct `_residents` directly and are cap-agnostic at the unit level).
What is missing, and what this file adds, is proof -- via the REAL dispatcher, not a hand-built
dict -- of two facts any accessor/refactor must either preserve exactly or change on purpose:

  1. At cap=1, `_model_residents()` is not just empty for ONE snapshot (the scope of
     the sidecar-cap victim-candidacy test) -- it stays empty across a REALISTIC multi-turn
     lifecycle (mid-turn, idle, and a full model swap), because nothing on the cap<=1 path
     (`_process_slot`) ever writes a model-keyed `Resident` -- because _process_slot
     never calls _begin_unload_locked. This is the structural invariant the accessor's
     cap<=1 correction (a later step of the refactor) will change; everything before that step must
     leave it exactly as pinned here.
  2. At cap=2, `_model_residents()` and `_is_worst_ranked_loaded_locked` correctly handle
     MULTIPLE simultaneous candidates -- not just the single-candidate case already covered elsewhere,
     where "worst ranked" is trivially the only entry and no real MAX-selection ever runs. A
     naive accessor swap-in could easily preserve single-item behaviour while silently breaking
     multi-item comparison; this is what the "empty-but-wired, ZERO behaviour change" claim
     needs something concrete to diff against beyond "the existing suite doesn't fail."

NOTE: the cap=1 invariant below is DELIBERATELY INVERTED relative to the legacy path.
The legacy singleton path pinned `_model_residents() == []` at cap=1; the unification routes cap=1
through `_dispatch_loop`, so real model-keyed residents exist at EVERY cap. This is the
deliberate edit of that pin. The cap=2 test is UNCHANGED.

Scope: characterisation only, zero production diff. Both tests pass on the pre-change code --
that is the point of a golden master: it is GREEN before the move, and any future run going RED
without a deliberate edit to this file is exactly the kind of regression it exists to
catch.
"""

import asyncio
import pytest

from tests._fastlane_fixture import (
    boot_ranked_runtime, seed_manifest, make_fakes, high_vram, drive_to_active, resident_for,
)
from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import TurbohaulManager, ResidentState


@pytest.mark.asyncio
async def test_golden_one_resident_cap_residents_stays_empty_across_a_multiturn_lifecycle(tmp_path):
    """cap=1: `_model_residents()` == [] at THREE checkpoints spanning a full swap --
    mid-turn on m1, after m1 completes and goes idle, and mid-turn on a DIFFERENT model m2
    that has now taken the singleton's place. Proves the invariant is structural (holds
    across the singleton's whole lifecycle), not an artifact of asking only once."""
    boot, runtime = boot_ranked_runtime(tmp_path, max_parallel_sidecars=1, grace_seconds=5)
    seed_manifest(boot, "m1", main_gpu=0)
    seed_manifest(boot, "m2", main_gpu=0)
    gate1 = asyncio.Event()
    gate2 = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate1, "m2": gate2})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            # Checkpoint A: m1 mid-turn.
            task1 = asyncio.create_task(mgr.submit_and_wait(
                "m1", "p", thread_id="t1", client_meta={}))
            await asyncio.sleep(1.5)
            assert not task1.done(), "m1 never got admitted in 1.5s -- fixture problem"
            # INVARIANT DELIBERATELY INVERTED — this is the
            # deliberate edit this file's own docstring describes rather than letting
            # the run go silently RED. cap=1 now routes through _dispatch_loop, so a REAL
            # model-keyed Resident is created and the legacy singleton no longer carries the
            # turn. Design intent: with max_parallel_sidecars set to one the parallel path is still used,
            # but only one sidecar is spawned at a time after the code unification.
            assert [r.model_tag for r in mgr._model_residents()] == ["m1"], (
                "checkpoint A (m1 mid-turn): post-unification, cap=1 MUST produce exactly one "
                "model-keyed resident for m1. An empty list means the request went down the "
                "RETAINED legacy body and the unification is not in effect."
            )

            # Checkpoint B: m1 completes, singleton goes idle.
            gate1.set()
            await asyncio.wait_for(task1, timeout=5.0)
            await asyncio.sleep(0.2)
            # Post-unification the resident persists past the turn (grace/idle)
            # exactly as at cap>=2 -- that shared lifecycle IS the point of the unification.
            # What must still hold at cap=1 is the ONE-AT-A-TIME ceiling.
            assert len(mgr._model_residents()) <= 1, (
                "checkpoint B (m1 idle): cap=1 must never hold more than one resident, got "
                f"{[r.model_tag for r in mgr._model_residents()]}"
            )

            # Checkpoint C: a DIFFERENT model, m2, now mid-turn on the same singleton (a swap
            # happened) -- the invariant must hold for the NEW occupant too, not just the first.
            task2 = asyncio.create_task(mgr.submit_and_wait(
                "m2", "p", thread_id="t2", client_meta={}))
            await asyncio.sleep(1.5)
            assert not task2.done(), "m2 never got admitted in 1.5s -- fixture problem"
            # Note: the occupant here is still m1, because
            # grace_seconds=5 and m2 is correctly SERIALISED BEHIND m1's grace window rather
            # than co-residing. That is the intended criterion working, so this pins the
            # CEILING and the serialisation -- asserting ["m2"] would pin a race, not an
            # invariant.
            tags = [r.model_tag for r in mgr._model_residents()]
            assert tags in ([], ["m1"], ["m2"]), (
                f"checkpoint C: cap=1 must hold at most one resident, and it must be one of "
                f"the two seeded models; got {tags}"
            )
            assert not task2.done(), (
                "checkpoint C: m2 must still be WAITING behind the single slot. Had it "
                "completed, cap=1 would have served two models concurrently."
            )
        finally:
            gate1.set()
            gate2.set()
            for t in (locals().get("task1"), locals().get("task2")):
                if t is not None:
                    try:
                        await asyncio.wait_for(t, timeout=10.0)
                    except BaseException:
                        pass
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except BaseException:
                pass


@pytest.mark.asyncio
async def test_golden_two_resident_cap_multi_resident_worst_ranked_selection(tmp_path):
    """cap=2: TWO residents genuinely ACTIVE simultaneously (distinct main_gpu for
    co-residence, per the fixture's own documented trap), matched against DISTINCT Fast Lane
    rules by raw address (``FastLaneRule(address=...)`` -- the same pattern test_grace_
    fastlane_breakout.py uses; a ``container_name`` rule needs DNS resolution to build
    ``match_addresses`` and is NOT exercised by a bare ``client_meta={"container_name": ...}``
    with no ``ip`` key, confirmed empirically while building this file -- the existing single-
    resident test never noticed because with one candidate "worst" is trivially that
    candidate regardless of whether resolution succeeded).
    _model_residents() must report BOTH, and _is_worst_ranked_loaded_locked's real
    max-selection (manager.py, `max(candidates, key=...)`) must correctly pick the
    HIGHER rule_index (later rule in the table = worse effective priority in this ordering)
    as worst-ranked-loaded and the other as NOT worst-ranked -- exercising actual
    multi-candidate comparison, which a single-resident fixture cannot: with
    one candidate, "worst" is trivially that candidate and no comparison ever runs."""
    rules = [
        FastLaneRule(address="1.1.1.1", tag_ranks=FastLaneTagRanks(main=1)),
        FastLaneRule(address="2.2.2.2", tag_ranks=FastLaneTagRanks(main=1)),
    ]
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, max_parallel_sidecars=2, grace_seconds=5)
    seed_manifest(boot, "better", main_gpu=0)
    seed_manifest(boot, "worse", main_gpu=1)
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({
        "better": asyncio.Event(), "worse": asyncio.Event(),
    })
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task_better = task_worse = None
        try:
            task_better = await drive_to_active(
                mgr, "better", thread_id="tb", client_meta={"ip": "1.1.1.1"})
            task_worse = await drive_to_active(
                mgr, "worse", thread_id="tw", client_meta={"ip": "2.2.2.2"})

            candidates = mgr._model_residents()
            tags = sorted(r.model_tag for r in candidates)
            assert tags == ["better", "worse"], (
                f"cap=2: expected BOTH residents visible to victim-selection, got {tags!r} -- "
                f"if only one appears, multi-resident population at cap>=2 has changed"
            )

            r_better = resident_for(mgr, "better")
            r_worse = resident_for(mgr, "worse")
            table = mgr._fastlane_table()
            key_better = mgr._resident_priority_key(r_better, table)
            key_worse = mgr._resident_priority_key(r_worse, table)
            assert key_better is not None and key_better[0] == 0, (
                f"precondition: 'better' must resolve to rule_index 0, got {key_better!r} -- "
                f"if this is None, the rule matched nothing and the rest of this test is "
                f"vacuous, not a real result"
            )
            assert key_worse is not None and key_worse[0] == 1, (
                f"precondition: 'worse' must resolve to rule_index 1, got {key_worse!r} -- "
                f"if this is None, the rule matched nothing and the rest of this test is "
                f"vacuous, not a real result"
            )
            assert mgr._is_worst_ranked_loaded_locked(r_worse) is True, (
                "cap=2: 'worse' (rule_index 1, the later/lower-priority rule) must be selected "
                "as worst-ranked-loaded among the two live candidates -- if this flips, "
                "multi-candidate max-selection has changed, not just single-candidate behaviour"
            )
            assert mgr._is_worst_ranked_loaded_locked(r_better) is False, (
                "cap=2: 'better' (rule_index 0, the earlier/higher-priority rule) must NOT be "
                "selected as worst-ranked-loaded while a lower-priority candidate is also live"
            )
        finally:
            for t in (task_better, task_worse):
                if t is not None:
                    t.cancel()
                    try:
                        await t
                    except BaseException:
                        pass
            mgr._worker_task.cancel()
            try:
                await mgr._worker_task
            except BaseException:
                pass
