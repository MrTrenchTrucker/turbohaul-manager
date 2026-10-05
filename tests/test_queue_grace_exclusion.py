"""Grace-window exclusion — pop_next's ``grace_active``
parameter makes a resident's own grace-held (thread_id, model_tag) pair
INVISIBLE to the whole pick ladder for one call, so no rung of it (Fast Lane,
main-lane, compression, FIFO, model-affinity/starvation) can be the one that
steals a warm same-thread follow-up away from the resident's own grace-loop
poll. ``grace_active=None``/empty must be byte-for-byte the behavior without it.

Also carries a fan-out probe: a reproduction of
the exact call shape manager.py's two same-turn rider fan-out sites use
when they pass no grace_active, proving the starvation fallback can steal a
grace-held pair regardless of which pop_next caller triggers it.
"""
import asyncio
import time

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


@pytest.mark.asyncio
class TestGraceActiveDegradationControl:
    """grace_active=None/empty must reproduce the ungated behavior exactly --
    and this control must be able to actually FAIL against a plausible bug
    (e.g. treating None/empty as "exclude everything")."""

    async def test_none_does_not_exclude_anything(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m1", thread_id="t1", prompt="only")
        await q.enqueue(s)

        got = await q.pop_next(grace_active=None)

        assert got is not None and got.slot_id == s.slot_id, (
            "a mutant that treats grace_active=None as 'protect everyone' "
            "would return None here instead of the only staged slot"
        )

    async def test_empty_frozenset_does_not_exclude_anything(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m1", thread_id="t1", prompt="only")
        await q.enqueue(s)

        got = await q.pop_next(grace_active=frozenset())

        assert got is not None and got.slot_id == s.slot_id

    async def test_none_leaves_staging_max_untouched(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m1", thread_id="t1", prompt="only")
        await q.enqueue(s)

        await q.pop_next(grace_active=None)

        assert q.staging_max == 10


@pytest.mark.asyncio
class TestGraceActiveProtectsAcrossEveryLadderRung:
    """The exclusion must hold regardless of WHICH rung of pop_next's ladder
    would otherwise have popped the protected entry."""

    async def test_protects_the_fifo_head(self):
        q = TurbohaulQueue(staging_max=10)
        protected = Slot.new(model_tag="m1", thread_id="p", prompt="protected")
        other = Slot.new(model_tag="m1", thread_id="q", prompt="other")
        await q.enqueue(protected)
        await q.enqueue(other)

        got = await q.pop_next(grace_active=frozenset({("p", "m1")}))

        assert got is not None and got.slot_id == other.slot_id, (
            "FIFO head is protected -- must fall through to the next entry"
        )

    async def test_protects_the_model_affinity_preferred_entry(self):
        # warm_model_tag=m1 with staging non-empty engages the affinity
        # branch (not FIFO) -- the protected entry is the one affinity would
        # otherwise prefer (same model as warm_model_tag, not the FIFO head).
        # max_other_model_wait_s must be large: the default (0.0) makes the
        # other-model head starved almost immediately, which forces the
        # SEPARATE starvation branch instead of the affinity-preference one
        # this test means to exercise.
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=9999.0)
        head_other_model = Slot.new(model_tag="m2", thread_id="head", prompt="head")
        protected = Slot.new(model_tag="m1", thread_id="p", prompt="protected")
        fallback = Slot.new(model_tag="m1", thread_id="q", prompt="fallback")
        for s in (head_other_model, protected, fallback):
            await q.enqueue(s)

        got = await q.pop_next(
            warm_model_tag="m1", grace_active=frozenset({("p", "m1")}),
        )

        assert got is not None and got.slot_id == fallback.slot_id, (
            "affinity must skip the protected same-model entry and prefer "
            "the next same-model candidate, not fall back to the other-model head"
        )

    async def test_protects_the_fastlane_pick(self):
        q = TurbohaulQueue(staging_max=10)
        protected = Slot.new(model_tag="m1", thread_id="p", prompt="protected")
        protected.fastlane = _match(rule_index=0, rank=1)  # top priority
        fallback = Slot.new(model_tag="m1", thread_id="q", prompt="fallback")
        fallback.fastlane = _match(rule_index=1, rank=1)
        await q.enqueue(protected)
        await q.enqueue(fallback)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=0)

        got = await q.pop_next(
            fastlane_policy=policy, grace_active=frozenset({("p", "m1")}),
        )

        assert got is not None and got.slot_id == fallback.slot_id, (
            "Fast Lane's own pick must not be exempt from the grace exclusion"
        )

    async def test_control_without_grace_active_the_fastlane_pick_wins(self):
        # Green control for the test above: without grace_active, the
        # top-priority fastlane candidate DOES win -- proving the previous
        # test's outcome is caused by the exclusion, not some unrelated
        # reason "protected" never wins fastlane picks anyway.
        q = TurbohaulQueue(staging_max=10)
        top = Slot.new(model_tag="m1", thread_id="p", prompt="top")
        top.fastlane = _match(rule_index=0, rank=1)
        other = Slot.new(model_tag="m1", thread_id="q", prompt="other")
        other.fastlane = _match(rule_index=1, rank=1)
        await q.enqueue(top)
        await q.enqueue(other)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=0)

        got = await q.pop_next(fastlane_policy=policy)

        assert got is not None and got.slot_id == top.slot_id


@pytest.mark.asyncio
class TestGraceActiveNoStarvation:
    """A held-aside entry must survive the call (never dropped) and become
    ordinarily poppable again the instant it is no longer protected."""

    async def test_protected_entry_still_staged_afterward(self):
        q = TurbohaulQueue(staging_max=10)
        protected = Slot.new(model_tag="m1", thread_id="p", prompt="protected")
        other = Slot.new(model_tag="m1", thread_id="q", prompt="other")
        await q.enqueue(protected)
        await q.enqueue(other)

        await q.pop_next(grace_active=frozenset({("p", "m1")}))

        staged = await q.peek_staging()
        assert protected.slot_id in {s.slot_id for s in staged}, (
            "held-aside entries must be spliced back, never dropped"
        )

    async def test_once_unprotected_it_is_poppable_next_call(self):
        q = TurbohaulQueue(staging_max=10)
        protected = Slot.new(model_tag="m1", thread_id="p", prompt="protected")
        other = Slot.new(model_tag="m1", thread_id="q", prompt="other")
        await q.enqueue(protected)
        await q.enqueue(other)
        await q.pop_next(grace_active=frozenset({("p", "m1")}))  # holds `protected` aside once

        got = await q.pop_next(grace_active=None)  # grace lapsed -- no exclusion this call

        assert got is not None and got.slot_id == protected.slot_id, (
            "no starvation: the instant protection lifts, the entry is an "
            "ordinary poppable slot again"
        )


@pytest.mark.asyncio
class TestGraceActiveStagingMaxInvariant:
    """The one real correctness risk of the hold-aside design: replenishing
    from accept_buf while an entry is held aside must never let staging grow
    past its real cap once the held-aside entries return."""

    async def test_staging_never_exceeds_its_real_cap_across_the_call(self):
        q = TurbohaulQueue(staging_max=3)
        protected = Slot.new(model_tag="m1", thread_id="p", prompt="protected")
        filler1 = Slot.new(model_tag="m1", thread_id="f1", prompt="filler1")
        filler2 = Slot.new(model_tag="m1", thread_id="f2", prompt="filler2")
        overflow = Slot.new(model_tag="m1", thread_id="of", prompt="overflow")
        for s in (protected, filler1, filler2):
            await q.enqueue(s)  # fills staging_max=3 exactly
        await q.enqueue(overflow)  # staging full -> lands in accept_buf

        got = await q.pop_next(grace_active=frozenset({("p", "m1")}))

        assert got is not None and got.slot_id == filler1.slot_id  # FIFO head, protected skipped
        staged = await q.peek_staging()
        assert len(staged) <= 3, (
            f"staging grew to {len(staged)} > staging_max=3 -- the held-aside "
            "entry plus an over-eager accept_buf replenish overshot the cap"
        )

    async def test_staging_max_is_restored_to_its_original_value(self):
        q = TurbohaulQueue(staging_max=3)
        protected = Slot.new(model_tag="m1", thread_id="p", prompt="protected")
        other = Slot.new(model_tag="m1", thread_id="q", prompt="other")
        await q.enqueue(protected)
        await q.enqueue(other)

        await q.pop_next(grace_active=frozenset({("p", "m1")}))

        assert q.staging_max == 3, "staging_max must never leak its temporary shrink"


@pytest.mark.asyncio
class TestFanoutStyleCallProbe:
    """Probe whether a fan-out-shaped pop_next
    call (warm_model_tag=anchor's own tag, no grace_active -- the exact
    shape manager.py's two same-turn rider fan-out sites use when they omit
    it) can steal a grace-held pair via the starvation fallback."""

    async def test_unwired_fanout_shaped_call_steals_a_starved_grace_pair(self):
        """This is the RED arm for the fan-out wiring: run against the
        pre-wiring shape (no grace_active passed at all, matching what the
        two manager.py call sites did without the wiring) and confirm the theft.
        """
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=0.0)
        grace_pair_slot = Slot.new(model_tag="model-b", thread_id="victim-thread", prompt="follow-up")
        await q.enqueue(grace_pair_slot)
        await asyncio.sleep(0.01)  # elapsed > 0.0 -> starved by construction

        got = await q.pop_next(warm_model_tag="model-a")  # the fan-out sites' call shape, unwired

        assert got is not None and got.slot_id == grace_pair_slot.slot_id, (
            "REPRODUCED: an unwired fan-out-shaped call steals a starved "
            "other-model entry with no way to know it was someone's grace pair"
        )

    async def test_wiring_grace_active_into_the_same_call_shape_fixes_it(self):
        """The fix: the SAME call shape, with grace_active supplied (exactly
        what manager.py's two fan-out sites now pass) -- the pair survives."""
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=0.0)
        grace_pair_slot = Slot.new(model_tag="model-b", thread_id="victim-thread", prompt="follow-up")
        rider = Slot.new(model_tag="model-a", thread_id="rider", prompt="rider")
        await q.enqueue(grace_pair_slot)
        await q.enqueue(rider)
        await asyncio.sleep(0.01)

        got = await q.pop_next(
            warm_model_tag="model-a",
            grace_active=frozenset({("victim-thread", "model-b")}),
        )

        assert got is not None and got.slot_id == rider.slot_id
        staged = await q.peek_staging()
        assert grace_pair_slot.slot_id in {s.slot_id for s in staged}
