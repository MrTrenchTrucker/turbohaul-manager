"""Grace-hold half of the shared paired test for grace exclusion.

Built against the shared fixture module imported below (shared byte-for-byte with
the other half of this pair, which covers the manager side)
rather than this file's own ad hoc monkeypatched objects
(``test_queue_grace_exclusion.py`` / ``test_manager.py::TestGraceActiveExclusions``),
so that both halves test one construction of the world, not two that resemble
each other.

THE PAIR (neither half alone proves the seam):
  This file: a NON-VICTIM's grace-held follow-up IS protected from the pick.
  Other half: a DESIGNATED VICTIM's follow-up is NOT served -- the victim rule's
    no-grace-timer guarantee, requeued to the tail.

CONTRACT NAME: ``_is_designated_unload_target_locked(r)``, manager-side, under
``_registry_lock``. This file uses the fixture's ``install_victim_predicate``
stand-in, which installs under that exact name -- if either half ever renames
it, the exclusion silently protects everyone and this file's assertions would
start passing for the wrong reason (protecting-because-absent, not
protecting-because-correctly-not-the-victim). The absent-predicate case is
tested explicitly below specifically so that failure mode has a name.

FIXTURE REQUIREMENT: the fixture's ``assert_resolves`` calls
``mgr._resident_priority_key(r, mgr._fastlane_table())``, so the
tree under test must provide ``_resident_priority_key``. Against a
tree that lacks it, a direct call raises ``AttributeError``. The
fixture was built and self-tested against a revision that has the
method, so the method is a precondition of the fixture rather than
something this file adds. The fixture documents the requirement
and gives ``assert_resolves`` a named tree-mismatch error instead
of a bare ``AttributeError`` for any tree that still lacks the
method, which makes a mismatched tree easy to recognise from the
error alone.

This file does not call ``assert_resolves``: a direct call
against this file's own ``_real_grace_held_resident`` construction
returns ``None``, correctly, since ``boot_ranked_runtime`` here is
never given ``rules=`` and ``drive_to_active``'s ``client_meta={}``
matches nothing, so there is no Fast Lane match to resolve a priority
key from. ``_grace_active_exclusions`` -- the only manager method this
file's assertions exercise -- never calls ``_resident_priority_key``
or ``_fastlane_table`` either; it reads only ``r.grace``/``r.model_tag``
and the victim predicate. So there is no real priority claim in this
file for ``assert_resolves`` to guard, unlike the victim-designation
test in the other half,
which does need it.

FIXTURE PRECONDITION: ``drive_to_active``'s own docstring does not
mention that ``TurbohaulManager``'s dispatcher must already be running
as a background task for ``submit_and_wait`` to make any progress at
all; a bare call hangs until ``wait_until``'s 5s timeout. Starting
``mgr.worker_loop()`` as ``mgr._worker_task`` first avoids the hang,
matching the exact pattern ``tests/test_auto_placer.py``'s
``TestAutoPlaceIntegration`` already uses for the same reason
(``worker_loop`` routes to the cap>=2 dispatcher internally when
``max_parallel_sidecars >= 2``, which ``boot_ranked_runtime``
defaults to). The mechanism is the same in every tree that uses the
fixture.
"""
import asyncio

import pytest

from tests._fastlane_fixture import (
    boot_ranked_runtime,
    drive_to_active,
    high_vram,
    install_victim_predicate,
    make_fakes,
    resident_for,
    seed_manifest,
    victim_predicate_absent,
)
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot


async def _real_grace_held_resident(tmp_path, *, grace_seconds=5.0):
    """Drive ONE resident through a real admission -> ACTIVE -> turn-complete
    lifecycle via the shared fixture, so its GraceTimer is populated by
    production code (the manager's ``r.grace.start(...)`` call), never
    hand-constructed. Returns (mgr, resident) with the manager's dispatcher
    already running in the background and the request's own task already
    awaited to completion.
    """
    boot, runtime = boot_ranked_runtime(tmp_path, grace_seconds=grace_seconds)
    seed_manifest(boot, "m1", main_gpu=0)
    gate = asyncio.Event()
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes({"m1": gate})
    with high_vram():
        mgr = TurbohaulManager(
            boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
            sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete_fn,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        task = await drive_to_active(mgr, "m1", thread_id="t1", client_meta={})
        gate.set()
        await asyncio.wait_for(task, timeout=5.0)
    return mgr, resident_for(mgr, "m1")


@pytest.mark.asyncio
class TestGraceExclusionAgainstSharedFixture:
    """Set-membership only -- no staging, no pop_next, no live-poller race.
    ``_grace_active_exclusions`` never touches the queue; asserting on its
    returned set directly is the deterministic half of this pair. The
    end-to-end "and pop_next therefore leaves it alone" claim is the
    dedicated integration test below, which explains why it stays
    deterministic despite the live dispatcher task still running.
    """

    async def test_real_nonvictim_grace_pair_is_excluded_predicate_present(self, tmp_path):
        mgr, r = await _real_grace_held_resident(tmp_path)
        try:
            assert r.grace is not None and not r.grace.expired(), (
                "sanity: a real turn completion must leave a live, unexpired "
                "GraceTimer -- otherwise this test measures nothing"
            )
            assert (r.grace.thread_id, r.grace.model_tag) == ("t1", "m1")
            install_victim_predicate(mgr, ["some-other-tag"])  # m1 is NOT listed
            excluded = await mgr._grace_active_exclusions()
            assert ("t1", "m1") in excluded, (
                "a resident the contract-named predicate does NOT flag as the "
                "designated victim must have its grace-held pair protected -- "
                "the grace-hold protection rule"
            )
        finally:
            await mgr.shutdown()

    async def test_real_resident_flagged_as_the_victim_is_not_excluded(self, tmp_path):
        """The paired control: same real construction, but this time the
        contract-named predicate DOES flag m1. Proves ``_grace_active_exclusions``
        genuinely reads the predicate's answer for THIS resident rather than
        protecting unconditionally -- exercised here against a real resident,
        whereas other tests only exercise this branch
        against a hand-rolled stand-in object."""
        mgr, r = await _real_grace_held_resident(tmp_path)
        try:
            install_victim_predicate(mgr, ["m1"])  # m1 IS the designated victim
            excluded = await mgr._grace_active_exclusions()
            assert ("t1", "m1") not in excluded, (
                "Victim rule: the designated victim gets NO grace timer protection "
                "-- no timer, no warm-hold. Protecting it "
                "here would be the exact violation this seam exists to prevent"
            )
        finally:
            await mgr.shutdown()

    async def test_real_resident_protected_when_predicate_absent(self, tmp_path):
        """The tree's actual state today: no companion predicate exists yet.
        Must degrade toward protecting everyone, never toward failing to
        protect -- this is the path most likely to ship untested per the
        fixture's own docstring."""
        mgr, r = await _real_grace_held_resident(tmp_path)
        try:
            victim_predicate_absent(mgr)  # explicit -- the fixture's own note:
            # "the state the tree is actually in today"
            excluded = await mgr._grace_active_exclusions()
            assert ("t1", "m1") in excluded
        finally:
            await mgr.shutdown()


@pytest.mark.asyncio
class TestGraceExclusionEndToEndAgainstSharedFixture:
    async def test_nonvictim_followup_survives_a_real_pop_next_call(self, tmp_path):
        """The literal claim: 'protected from the pick', not just 'present in
        a set'. Stages a competing follow-up for the SAME (thread_id,
        model_tag) directly in the real manager's live queue, computes the
        exclusion set via the real ``_grace_active_exclusions()``, and calls
        ``pop_next`` with the EXACT keyword shape ``_dispatch_loop`` itself
        uses -- not a re-implementation of the call.

        DETERMINISM NOTE, disclosed rather than assumed: the manager's own
        background ``worker_loop`` task and the resident's per-resident
        grace-loop poller are both still alive during this test (killing
        them was considered and rejected -- it would stop this from being
        the shared, real construction this pair is meant to use). Staging the
        competing slot and taking this file's own ``pop_next`` measurement
        happens back-to-back with a single Python statement in between and
        no ``asyncio.sleep`` -- the only suspension points are this test's
        own two lock acquisitions (``_registry_lock`` then ``queue._lock``),
        which is several orders of magnitude tighter than the ~50ms poll
        granularity the codebase's OWN existing grace-loop tests already
        tolerate as adequately deterministic (see
        ``test_grace_fastlane_breakout.py``'s
        ``test_invariant_no_higher_priority_still_grants_match``). If the
        resident's own poller wins this race instead of the dispatcher, that
        is not a failure of what this test checks: either way, the
        dispatcher must not steal the entry, and this assertion is about the
        dispatcher's own ``pop_next`` call specifically, not about which
        consumer eventually claims the entry.
        """
        mgr, r = await _real_grace_held_resident(tmp_path)
        try:
            install_victim_predicate(mgr, ["some-other-tag"])
            followup = Slot.new("m1", prompt="again", thread_id="t1")
            await mgr.queue.enqueue(followup)

            grace_active = await mgr._grace_active_exclusions()
            assert (r.grace.thread_id, r.grace.model_tag) in grace_active
            popped = await mgr.queue.pop_next(
                warm_model_tag=None,
                fastlane_policy=mgr._fastlane_pop_kwarg(),
                grace_active=grace_active,
            )
            assert popped is None or popped.slot_id != followup.slot_id, (
                "the dispatcher's own pop_next call must not steal a "
                "grace-held non-victim's staged follow-up -- the grace-hold protection rule"
            )
        finally:
            await mgr.shutdown()

    async def test_control_without_the_exclusion_the_same_call_would_steal_it(self, tmp_path):
        """Non-vacuity control for the test above: same construction, same
        staged follow-up, but ``grace_active=None`` (the pre-fix
        call shape). Must show the OPPOSITE outcome, proving the previous
        test's pass is because of the exclusion, not because this staged
        entry was unreachable for some unrelated reason."""
        mgr, r = await _real_grace_held_resident(tmp_path)
        try:
            followup = Slot.new("m1", prompt="again", thread_id="t1")
            await mgr.queue.enqueue(followup)
            popped = await mgr.queue.pop_next(
                warm_model_tag=None,
                fastlane_policy=mgr._fastlane_pop_kwarg(),
                grace_active=None,
            )
            assert popped is not None and popped.slot_id == followup.slot_id, (
                "control: without grace_active this staged entry must still "
                "be poppable exactly as before the exclusion was added, or this file's "
                "protected-case assertions above would be vacuous"
            )
        finally:
            await mgr.shutdown()
