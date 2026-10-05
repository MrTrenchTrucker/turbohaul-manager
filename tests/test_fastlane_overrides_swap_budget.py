"""Fast Lane overrides the cross-model swap budget entirely.

Design rule: Fast Lane must be able to
override that setting entirely -- a Fast Lane-matched claimant must never be
refused a cross-model swap on `cross_model_switches_per_min` grounds, at any
value including 0. This is a standing requirement of the Fast Lane.

Design (four mechanisms, not three): the budget is
RELOCATED, not deleted -- it now gates ordinary/unregistered cross-model
swaps instead of lane ones, EXCEPT the one-turn fairness floor's designated
client (always protected, even over a higher-priority resident). That
exemption is free by construction -- see the code comments at the gate site
(`pop_next`'s affinity-path `else` arm) -- not an additional condition.

⚠ NOTE: the design does NOT name a
second exemption for `max_other_model_wait_s` starvation breakout: that would be
a code detail, not a design clause, so none is tested.
Rationale: the design
already calls for ~1hr, and the CODE sets it accordingly
(`max_other_model_wait_s` is set to 3600.0), which removes the
collision such a clause would address. The starvation-breakout
branch (`starved_slot`) still never reaches the new gate, but that is a
STRUCTURAL fact of the ladder's if/elif/else shape (see the code comment
there), not a spec-backed exemption -- do not test it as one. The control
that would prove it (control "(d)") is therefore omitted;
only THREE controls exist.

⚠ TEST-CONSTRUCTION TRAP:
`TurbohaulQueue`'s `max_other_model_wait_s` defaults to 0.0 -- ANY
different-model staged entry is trivially "starved" and drained via the
EXEMPT `starved_slot` branch before ever reaching the gated `else` branch
below it. Every scenario below that means to exercise the new gate
explicitly passes `max_other_model_wait_s=9999.0` to rule this out; verified
with a direct (non-pytest) trace, not assumed; without it the
"ordinary traffic unaffected" control can pass for the wrong reason (never
reaching the gate at all).
"""
import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


@pytest.mark.asyncio
class TestFastLaneOverridesSwapBudget:
    async def test_discriminator_non_exempt_lane_claimant_granted_at_budget_zero(self):
        """The core claim. budget=0 is the trap value
        (`_fastlane_swap_allowed_locked` always refuses at <=0) and `cand` is
        deliberately NOT exempt (its (rule_index, rank) does not strictly
        outrank the loaded resident) -- the exact combination the
        NON-VACUITY check needs. Pre-fix this is refused (see
        `test_non_exempt_candidate_still_obeys_the_budget`,
        `TestSwapBudget`); post-fix it must be granted."""
        q = TurbohaulQueue(staging_max=10)
        cand = Slot.new(model_tag="m2", prompt="cand")
        cand.fastlane = _match(rule_index=2, rank=1)
        assert cand.fastlane is not None  # non-vacuity: really Fast-Lane-matched
        await q.enqueue(cand)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=0,
            loaded_priority_keys=((1, 1),),  # outranks cand -> genuinely NOT exempt
        )
        # non-vacuity: candidate key really does not strictly outrank the loaded key
        assert not q._fastlane_strictly_higher((2, 1), (1, 1))

        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got is not None
        assert got.slot_id == cand.slot_id  # granted, not refused, not falling through
        assert q._fastlane_refusal_counts.get("budget", 0) == 0  # never refused
        # Sharper than the slot_id check alone (which a FIFO-fallback would
        # also satisfy with only one slot staged): confirm it was actually
        # won via the Fast Lane grant path, with the new telemetry reason.
        assert q.fastlane_swap_counts() == {"budget_exempt": 1}

    async def test_still_refused_and_falls_through_when_fastlane_disabled(self):
        """CONTROL, other half: this override is scoped to Fast Lane's own
        pick -- a plain call with no fastlane_policy at all (the feature-off
        shape) must be completely untouched by this fix, proven by an
        UNREGISTERED slot's FIFO order being unaffected regardless of
        `cross_model_switches_per_min`."""
        q = TurbohaulQueue(staging_max=10)
        normal_first = Slot.new(model_tag="m2", prompt="first")
        normal_second = Slot.new(model_tag="m1", prompt="second")
        assert normal_first.fastlane is None and normal_second.fastlane is None
        await q.enqueue(normal_first)
        await q.enqueue(normal_second)
        got = await q.pop_next(warm_model_tag="m1")  # no fastlane_policy at all
        assert got.slot_id == normal_first.slot_id  # plain FIFO head, untouched

    async def test_GREEN_CONTROL_same_model_ordinary_traffic_outcome_identical_across_budget(self):
        """Narrowed from an over-broad first claim (see module
        docstring's TEST-CONSTRUCTION TRAP note): SAME-model ordinary traffic
        never reaches the gate at all (`_pop_first_matching_non_unloaded_from`
        succeeds, `returned is not None`, the gated `else` never runs) -- so
        THIS scenario's outcome really is budget-independent, verified by
        running it both ways rather than assumed. Cross-model ordinary
        traffic is NOT byte-for-byte unchanged any more -- that's the new
        gate below, proven separately, not this control's claim."""
        outcomes = []
        for budget in (0, 999):
            q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=9999.0)
            a = Slot.new(model_tag="m2", prompt="a")
            b = Slot.new(model_tag="m1", prompt="b")  # matches warm -- same-model path
            assert a.fastlane is None and b.fastlane is None  # non-vacuity
            await q.enqueue(a)
            await q.enqueue(b)
            policy = FastLanePopPolicy(
                max_normal_wait_s=9999.0, cross_model_switches_per_min=budget,
            )
            got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
            outcomes.append(got.slot_id == b.slot_id if got else None)
        assert outcomes == [True, True]  # same-model match won both times, budget irrelevant
        # non-vacuity: confirm this scenario really never reached the new gate
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 0

    async def test_STRONG_CONTROL_ordinary_cross_model_swap_now_refused_at_exhausted_budget(self):
        """The strong control (b): a routine (not starved, not one-turn-
        floor-eligible) ordinary cross-model swap is REFUSED once the budget
        is exhausted -- pre-fix (verified by reverting just this hunk) this
        always swapped regardless of budget. `max_normal_wait_s`
        and `max_other_model_wait_s` both huge so NEITHER exemption fires --
        isolates the new gate itself, not an exemption path."""
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=9999.0)
        a = Slot.new(model_tag="m2", prompt="a")  # only staged entry, cross-model from warm
        assert a.fastlane is None  # non-vacuity: genuinely non-lane
        await q.enqueue(a)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,       # one-turn floor never fires
            cross_model_switches_per_min=0,  # exhausted
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got is None  # refused, not served this cycle
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 1
        assert [s.slot_id for s in q._staging] == [a.slot_id]  # left staged, not dropped

    async def test_GREEN_one_turn_floor_client_still_served_at_exhausted_budget(self):
        """Control (c): the one-turn fairness floor's carve-out, proven not asserted. An unregistered
        slot already aged past `max_normal_wait_s` is ALSO a cross-model
        candidate (different model than warm) -- must still be served via
        the floor, untouched by the new gate, even at budget=0."""
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=9999.0)
        old = Slot.new(model_tag="m2", prompt="old")
        assert old.fastlane is None  # non-vacuity
        await q.enqueue(old)
        policy = FastLanePopPolicy(
            max_normal_wait_s=0.0,           # already "aged past" instantly
            cross_model_switches_per_min=0,  # exhausted -- would refuse if this reached the gate
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got is not None
        assert got.slot_id == old.slot_id
        assert got.floor_promoted is True  # served via the floor, not the gated fallback
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 0  # gate never reached

    # Control "(d)" (starvation-breakout exemption) is omitted along
    # with the spec clause it would prove -- see the module docstring note.
    # Not replaced with a weaker "structural fact" test: the original
    # intent is that this control is omitted, not downgraded.


@pytest.mark.asyncio
class TestOrdinaryGateRespectsFreeRoom:
    """REAL REGRESSION + FIX: the room-first rule ("room
    check comes first ... if there is a free sidecar slot ... nothing is
    evicted") means the ordinary swap-budget gate (queue.py's `else` arm)
    must never refuse a pick that lands in a genuinely FREE registry slot --
    the spec defines the budget's subject as a swap costing an
    unload+load+re-prefill, and a free-slot placement costs none of that, so
    it is not a swap at all. The manager computes the fact under
    `_registry_lock` (queue.py must not do it itself) and hands it down
    as `room_available`, same HINT contract as `warm_model_tag`
    (`_dispatch_room_hint`, manager.py). End-to-end proof of the real bug
    this closes:
    the turn-boundary handoff test, class TestStalenessProtectedNotEvictedViaNotify,
    test_end_to_end_real_park_wakes_waiter_who_still_skips_a_protected_sibling
    -- goes RED when just the `room_available` gate condition is reverted and
    GREEN when it is restored. The tests below isolate the SAME mechanism
    at the unit level, cheaply and mutation-testably."""

    async def test_room_available_grants_even_at_exhausted_budget(self):
        """The core fix. Identical setup to
        test_STRONG_CONTROL_ordinary_cross_model_swap_now_refused_at_exhausted_budget
        (budget=0, neither the one-turn floor nor the starvation breakout
        fires) EXCEPT `room_available=True` -- the ONLY variable that flips
        refused -> granted, isolating this exact mechanism and nothing else."""
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=9999.0)
        a = Slot.new(model_tag="m2", prompt="a")
        assert a.fastlane is None  # non-vacuity: genuinely non-lane
        await q.enqueue(a)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,       # one-turn floor never fires
            cross_model_switches_per_min=0,  # exhausted -- would refuse without room_available
        )
        got = await q.pop_next(
            warm_model_tag="m1", fastlane_policy=policy, room_available=True,
        )
        assert got is not None
        assert got.slot_id == a.slot_id
        # never refused: room, not budget, decided this pick
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 0

    async def test_room_available_omitted_still_refuses_at_exhausted_budget(self):
        """CONTROL, other half: `room_available` defaults to False (every
        caller except the cap>=2 dispatch loop) -- byte-identical to
        test_STRONG_CONTROL_ordinary_cross_model_swap_now_refused_at_exhausted_budget,
        restated here explicitly alongside its own mirror image so the
        contrast is visible in one place, not to duplicate that test's
        authority."""
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=9999.0)
        a = Slot.new(model_tag="m2", prompt="a")
        assert a.fastlane is None
        await q.enqueue(a)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0, cross_model_switches_per_min=0,
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got is None
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 1
