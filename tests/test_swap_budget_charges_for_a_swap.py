"""The swap budget must charge for a SWAP,
not for a GRANT.

DEFECT: `queue.py`'s `_fastlane_swap_allowed_locked` appends its rolling-window
token INSIDE the permission check, so the token is spent the instant the gate
says yes. When the grant does not become a swap, the budget is burned for an
unload+load+re-prefill that never happened, and a LATER, REAL swap is refused.

DESIGN, verbatim, twice over:
  * the cost it names is the swap itself — "a swap is expensive — an unload, a
    load and a re-prefill — and an unbounded swap rate spends the server's time
    moving models instead of generating";
  * and it ALREADY declines to charge a non-swap — "A pick that lands in a
    genuinely free sidecar slot is not a swap at all — nothing is unloaded — so
    the cap does not apply to it either".
The precedent for refunding a not-a-swap is inside the clause, so the rule
is: REFUND, not attempts.

WHO IS HARMED, an unregistered client rather than a lane claimant: the
gate's ONLY live call site is `pop_next`'s ORDINARY ladder, and
it counts `ordinary_budget`. A Fast-Lane claimant can never reach it — the design says
"Fast Lane is exempt from this cap entirely", and the lane side makes no
call to it. **The party refused a real swap by a phantom token is an
UNREGISTERED client**, which is exactly who the design says the cap is FOR ("The cap
governs unregistered traffic instead"). The tests below assert that harm.

⚠ THE REFUND CONDITION HAS TWO DISJUNCTS AND ONLY ONE IS REACHABLE HERE.
`returned is None or returned.is_evicted`, as specified. Measured on the
code before the change:
  * `returned.is_evicted` (a2) — REACHABLE, and the whole of the live defect.
    `_pop_first_non_unloaded_from` hands back a slot it has ITSELF flagged
    evicted; `manager.py` then does `continue`, skipping
    `_route_or_reserve` entirely. Nothing unloaded, nothing loaded, nothing
    re-prefilled — and the token is gone. Tested below.
  * `returned is None` (a1) — **UNREACHABLE at this call site, therefore
    UNFALSIFIABLE, and deliberately NOT claimed as tested.** The helper's loop
    returns on its first iteration on BOTH branches, so `return None` needs an
    EMPTY deque; and staging is non-empty here (`head = self._staging[0]` runs
    first, and `_pop_first_matching_non_unloaded_from` removes nothing when it
    returns None). Retained as specified, because it costs one token and guards a
    future caller. `TestWhyTheIsNoneDisjunctIsUnfalsifiableHere` pins the two
    structural facts that make it unreachable, which is the honest thing a test
    CAN do here — it does not pretend to exercise the branch.
"""
import asyncio
from collections import deque

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


async def _ordinary_cross_model_queue(budget, evicted_head=False, staging_max=10):
    """The fixture shape PROVEN to reach the ordinary gate by
    the fast-lane swap-budget override test (TestOrdinaryGateRespectsFreeRoom):
    a non-lane slot for a model that is NOT warm, no room hint, and no
    same-model entry for the affinity scan to prefer. Reusing a proven
    gate-reaching fixture rather than inventing one is deliberate — a fixture
    that quietly stopped reaching the gate would make every assertion vacuous.
    """
    q = TurbohaulQueue(staging_max=staging_max, max_other_model_wait_s=9999.0)
    slot = Slot.new(model_tag="m2", prompt="ordinary")
    assert slot.fastlane is None, "non-vacuity: this slot must be genuinely NON-lane"
    await q.enqueue(slot)
    if evicted_head:
        slot.disconnect_event = asyncio.Event()
        slot.disconnect_event.set()
    policy = FastLanePopPolicy(
        max_normal_wait_s=9999.0,  # the one-turn floor never fires
        cross_model_switches_per_min=budget,
    )
    return q, slot, policy


# ---------------------------------------------------------------------------
# The defect: a grant that produced no swap must not keep the token.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAGrantThatBecameNoSwapIsRefunded:
    async def test_an_evicted_slot_handed_back_does_not_keep_the_token(self):
        """RED pre-fix: 1 token burned for a slot the manager throws away."""
        q, slot, policy = await _ordinary_cross_model_queue(budget=5, evicted_head=True)

        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

        # The gate was genuinely reached and genuinely granted -- assert the
        # preconditions by hand so a fixture that stopped reaching the gate
        # fails HERE, loudly, instead of making the real assertion vacuous.
        assert got is not None, "fixture no longer reaches the ordinary gate"
        assert got.slot_id == slot.slot_id
        assert got.is_evicted is True, (
            "fixture precondition: the head must come back already flagged "
            "evicted -- that is the case whose token must be refunded"
        )
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 0, (
            "fixture precondition: this pick must have been GRANTED, not refused"
        )

        assert len(q._fastlane_swap_times) == 0, (
            "The design charges for a swap -- 'an unload, a load and a re-prefill'. "
            "This pick produced an already-evicted slot that manager.py "
            "discards without ever calling _route_or_reserve, so NOTHING was "
            "unloaded, loaded or re-prefilled. The token must have been given "
            f"back; the rolling window still holds {len(q._fastlane_swap_times)}."
        )

    async def test_the_phantom_token_refuses_a_later_real_swap_by_an_unregistered_client(self):
        """The HARM, end to end, and the assertion it makes:
        the client refused is UNREGISTERED, not a lane claimant.

        budget=1. The first pick grants, produces an evicted slot, no swap
        happens. A second, healthy, genuinely-cross-model pick then arrives and
        is entitled to the budget. Pre-fix it is REFUSED by a token that paid
        for nothing.
        """
        q, dead, policy = await _ordinary_cross_model_queue(budget=1, evicted_head=True)

        first = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert first is not None and first.is_evicted is True, (
            "fixture precondition: first pick must return the evicted head"
        )

        live = Slot.new(model_tag="m2", prompt="a real swap, entitled to run")
        assert live.fastlane is None, (
            "the harmed party must be UNREGISTERED -- the design exempts the lane "
            "entirely, so a claimant could never be refused here"
        )
        await q.enqueue(live)

        second = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 0, (
            "an UNREGISTERED client's real cross-model swap was refused on "
            "budget grounds by a token spent on a pick that never became a "
            "swap. The design puts the cap on unregistered traffic to throttle real "
            "churn, not to invent refusals out of grants that did nothing."
        )
        assert second is not None and second.slot_id == live.slot_id, (
            "the later, real swap must be served once the phantom token is "
            "refunded"
        )


# ---------------------------------------------------------------------------
# Controls. Each must pass on BOTH arms; their teeth are proven by mutants
# recorded separately, not by their own green.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestARealSwapStillPaysForItself:
    async def test_a_healthy_cross_model_pop_keeps_its_token(self):
        """The over-refund guard. If the refund fired unconditionally the cap
        would stop capping anything, which is a worse bug than the one fixed."""
        q, slot, policy = await _ordinary_cross_model_queue(budget=5, evicted_head=False)

        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

        assert got is not None and got.is_evicted is False, (
            "fixture precondition: a genuinely healthy cross-model pop"
        )
        assert len(q._fastlane_swap_times) == 1, (
            "a real swap DID happen -- this pick is routed and costs an "
            "unload+load+re-prefill -- so its token must stay spent. A refund "
            "here would defeat the cap entirely."
        )

    async def test_a_refused_pick_refunds_nothing_because_it_took_nothing(self):
        q, slot, policy = await _ordinary_cross_model_queue(budget=0, evicted_head=False)

        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

        assert got is None
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 1, (
            "fixture precondition: budget=0 must REFUSE, exercising the arm "
            "where no token was ever taken"
        )
        assert len(q._fastlane_swap_times) == 0, (
            "a refused pick never took a token, so there is nothing to give "
            "back -- the refund must not invent a credit"
        )

    async def test_room_available_neither_takes_nor_refunds(self):
        """Design: a pick into genuinely free room is not a swap, so the gate
        is skipped and no token is involved in either direction."""
        q, slot, policy = await _ordinary_cross_model_queue(budget=1, evicted_head=True)

        got = await q.pop_next(
            warm_model_tag="m1", fastlane_policy=policy, room_available=True,
        )

        assert got is not None
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 0
        assert len(q._fastlane_swap_times) == 0, (
            "room_available skips the gate entirely: no token taken, so the "
            "refund must not run and must not underflow"
        )

    async def test_feature_off_neither_takes_nor_refunds(self):
        """queue.py's own convention: feature-off is zero behaviour change."""
        q, slot, policy = await _ordinary_cross_model_queue(budget=1, evicted_head=True)

        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=None)

        assert got is not None
        assert len(q._fastlane_swap_times) == 0
        assert q._fastlane_refusal_counts.get("ordinary_budget", 0) == 0


# ---------------------------------------------------------------------------
# The refund helper itself.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTheRefundHelper:
    async def test_refund_returns_the_most_recent_token_not_the_oldest(self):
        """Which end matters. The oldest token belongs to a DIFFERENT, real
        swap whose 60s window must keep expiring on its own schedule; taking
        it back would extend that swap's charge by the age gap."""
        q = TurbohaulQueue(staging_max=10)
        q._fastlane_swap_times.extend([100.0, 200.0, 300.0])

        q._fastlane_swap_refund_locked()

        assert list(q._fastlane_swap_times) == [100.0, 200.0], (
            "the refund must pop the token this caller just took (the most "
            "recent), never the oldest one, which belongs to an earlier real "
            f"swap; window is now {list(q._fastlane_swap_times)}"
        )

    async def test_refund_on_an_empty_window_cannot_go_negative(self):
        q = TurbohaulQueue(staging_max=10)
        assert len(q._fastlane_swap_times) == 0

        q._fastlane_swap_refund_locked()

        assert len(q._fastlane_swap_times) == 0, (
            "a refund with nothing to give back must be a no-op, so a double "
            "refund can never credit the budget below zero"
        )


# ---------------------------------------------------------------------------
# Why the `returns None` disjunct is UNFALSIFIABLE at this call site.
# These do NOT test that branch. They pin the two structural facts that make
# it unreachable, so the UNFALSIFIABLE verdict stays true — and so a
# future change that makes it reachable trips something.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWhyTheIsNoneDisjunctIsUnfalsifiableHere:
    async def test_the_helper_returns_none_only_for_an_empty_deque(self):
        """FACT 1. Both branches of the loop's `if` return, so the loop never
        reaches a second iteration: `return None` needs an empty deque.
        (`max_drain` is consequently inert. NOT this change's to fix — it is
        a separate unit, and widening here is the silent-rider shape.)"""
        q = TurbohaulQueue(staging_max=10)

        assert q._pop_first_non_unloaded_from(deque()) is None, (
            "an empty deque is the ONLY input that yields None"
        )

        healthy = Slot.new(model_tag="m1", prompt="healthy")
        assert q._pop_first_non_unloaded_from(deque([healthy])) is healthy

        dead = Slot.new(model_tag="m1", prompt="dead")
        dead.disconnect_event = asyncio.Event()
        dead.disconnect_event.set()
        got = q._pop_first_non_unloaded_from(deque([dead]))
        assert got is dead and got.is_evicted is True, (
            "an all-evicted deque still returns the SLOT (flagged), not None -- "
            "this is exactly why the is-None disjunct cannot fire at the gate"
        )

    async def test_the_gate_site_always_sees_a_non_empty_staging(self):
        """FACT 2, measured rather than derived: at the moment the ordinary
        gate grants, staging still holds the cross-model head."""
        q, slot, policy = await _ordinary_cross_model_queue(budget=5, evicted_head=True)

        seen = []
        real = q._pop_first_non_unloaded_from

        def spy(buf, max_drain=10):
            seen.append(len(buf))
            return real(buf, max_drain)

        q._pop_first_non_unloaded_from = spy
        await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

        assert seen, "fixture no longer reaches the gate's pop at all"
        assert all(n > 0 for n in seen), (
            "staging was empty at the gate's pop, which would make the is-None "
            f"disjunct REACHABLE and the UNFALSIFIABLE verdict wrong; "
            f"observed deque lengths {seen}"
        )
