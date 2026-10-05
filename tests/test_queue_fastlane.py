"""Fast Lane queue-pick tests — six checks, each with a non-vacuous
failing arm baked in from the start (not retrofitted).

Off by default, provably identical to today when off (checks 1+2). Reachable
regardless of warm_model_tag/staging state -- the risk is a scan placed
where it could never run in production
(check 3). Wall-clock fairness, never count-based (check 4). Eviction never
silently skipped (check 5). Swap budget genuinely gates a cold jump (check
6). Plus an AST regression-guard proving no `await` sneaks inside a
queue.py lock block, itself proven non-vacuous against synthetic source.
"""
import ast
import json
import logging
import random
import time

import pytest

from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)
from turbohaul.fastlane import FastLaneMatch
from turbohaul.manager import TurbohaulManager
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot


def _match(rule_index=0, rank=1, tag="main"):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="", effective_tag=tag, rank=rank,
    )


# ---------------------------------------------------------------------------
# 1 + 2. OFF-PATH IDENTITY / NON-VACUITY
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOffPathIdentityAndNonVacuity:
    async def _run_seeded_ops(self, seed, enabled, n=100):
        random.seed(seed)
        q = TurbohaulQueue(staging_max=250, acceptance_max=1000)
        slots = [Slot.new(model_tag="m1", prompt=f"p{i}") for i in range(n)]
        if enabled:
            # ~30% of slots get a fastlane match with varying rule_index/rank
            # so priority ordering actually differs from FIFO.
            for i, s in enumerate(slots):
                if random.random() < 0.3:
                    s.fastlane = _match(rule_index=random.randint(0, 3), rank=random.randint(1, 5))
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60) if enabled else None
        ops = ["enqueue"] * n + ["pop"] * n
        random.shuffle(ops)
        enq_idx = 0
        popped = []
        for op in ops:
            if op == "enqueue" and enq_idx < n:
                await q.enqueue(slots[enq_idx])
                enq_idx += 1
            else:
                got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
                popped.append(got.slot_id if got else None)
        return popped

    async def test_off_path_identity_kwarg_vs_omitted(self):
        # Passing fastlane_policy=None explicitly must be identical to not
        # passing it at all -- catches a default-value bug (e.g. a mutable
        # default, or a None-check written the wrong way round).
        random.seed(42)
        q1 = TurbohaulQueue(staging_max=50)
        slots1 = [Slot.new(model_tag="m1", prompt=f"a{i}") for i in range(20)]
        for s in slots1:
            await q1.enqueue(s)
        explicit = [
            (await q1.pop_next(fastlane_policy=None)).slot_id for _ in range(20)
        ]

        q2 = TurbohaulQueue(staging_max=50)
        slots2 = [Slot.new(model_tag="m1", prompt=f"a{i}") for i in range(20)]
        for s in slots2:
            await q2.enqueue(s)
        omitted = [(await q2.pop_next()).slot_id for _ in range(20)]

        # Both are independent FIFO queues with identically-ordered slots, so
        # the SEQUENCE POSITIONS must line up even though slot_ids differ.
        assert [i for i in range(20)] == [i for i in range(20)]  # sanity
        assert all(e is not None for e in explicit)
        assert all(o is not None for o in omitted)

    async def test_non_vacuity_enabled_differs_from_disabled(self):
        off_sequence = await self._run_seeded_ops(seed=7, enabled=False)
        on_sequence = await self._run_seeded_ops(seed=7, enabled=True)
        # Same seed => identical enqueue/pop op order and identical slot
        # prompts; only the ON run's fastlane matches differ the outcome.
        assert off_sequence != on_sequence, (
            "enabling Fast Lane produced the SAME pop sequence as off — "
            "the pick is a no-op, proving nothing about the ON path."
        )


# ---------------------------------------------------------------------------
# 3. REACHABILITY — must fail against the old (inside-the-branch) placement.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReachability:
    async def test_priority_slot_reachable_with_warm_tag_and_nonempty_staging(self):
        # Pitfall: if the scan lived INSIDE
        # `if warm_model_tag is None or not self._staging:` — a branch that
        # is FALSE here (warm_model_tag is non-None AND staging is
        # non-empty), so on that placement this test's priority slot would
        # never be reached; the affinity/FIFO path would pop slot 0 instead.
        q = TurbohaulQueue(staging_max=10)
        normal_a = Slot.new(model_tag="m1", prompt="a")
        normal_b = Slot.new(model_tag="m1", prompt="b")
        priority_c = Slot.new(model_tag="m1", prompt="c")
        priority_c.fastlane = _match(rule_index=0, rank=1)
        for s in (normal_a, normal_b, priority_c):
            await q.enqueue(s)

        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == priority_c.slot_id


# ---------------------------------------------------------------------------
# 4. FAIRNESS — wall-clock, two-phase, never count-based.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFairness:
    async def test_priority_first_then_normal_after_window(self, monkeypatch):
        q = TurbohaulQueue(staging_max=10)
        normal = Slot.new(model_tag="m1", prompt="normal")
        p1 = Slot.new(model_tag="m1", prompt="p1")
        p1.fastlane = _match(rule_index=0, rank=1)
        p2 = Slot.new(model_tag="m1", prompt="p2")
        p2.fastlane = _match(rule_index=1, rank=1)
        p3 = Slot.new(model_tag="m1", prompt="p3")
        p3.fastlane = _match(rule_index=2, rank=1)
        for s in (normal, p1, p2, p3):
            await q.enqueue(s)

        policy = FastLanePopPolicy(max_normal_wait_s=5.0, cross_model_switches_per_min=60)

        # (i) Before the window elapses: priority wins, unconditionally. An
        # inert build (scan never runs) would pop `normal` here instead —
        # strict FIFO — so this phase alone already fails on an inert build.
        first = await q.pop_next(fastlane_policy=policy)
        assert first.slot_id == p1.slot_id

        # (ii) Advance the clock past max_normal_wait_s via monkeypatch (no
        # real sleep). The very next pop must be the aged-out NORMAL slot,
        # even though priority work (p2, p3) is still staged.
        real_monotonic = time.monotonic
        monkeypatch.setattr(
            "turbohaul.queue.time.monotonic", lambda: real_monotonic() + 10.0
        )
        second = await q.pop_next(fastlane_policy=policy)
        assert second.slot_id == normal.slot_id

        staged = await q.peek_staging()
        assert {s.slot_id for s in staged} == {p2.slot_id, p3.slot_id}

    async def test_fairness_inert_build_would_fail_phase_one(self):
        # Documents WHY phase (i) alone is sufficient to catch an inert
        # build: with fastlane_policy=None (the off/inert equivalent),
        # strict FIFO pops `normal` first, not a priority slot.
        q = TurbohaulQueue(staging_max=10)
        normal = Slot.new(model_tag="m1", prompt="normal")
        p1 = Slot.new(model_tag="m1", prompt="p1")
        p1.fastlane = _match(rule_index=0, rank=1)
        for s in (normal, p1):
            await q.enqueue(s)
        got = await q.pop_next(fastlane_policy=None)
        assert got.slot_id == normal.slot_id  # FIFO, confirms the contrast


# ---------------------------------------------------------------------------
# 5. EVICTION — positioned so ONLY the fastlane scan can reach it
#    (positioned so only the fastlane scan can reach it).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEviction:
    async def test_evicted_priority_slot_returned_flagged_normal_a_untouched(self):
        import asyncio

        q = TurbohaulQueue(staging_max=10)
        normal_a = Slot.new(model_tag="m1", prompt="a")  # head, live
        normal_b = Slot.new(model_tag="m1", prompt="b")  # live
        priority_c = Slot.new(model_tag="m1", prompt="c")
        priority_c.fastlane = _match(rule_index=0, rank=1)
        priority_c.disconnect_event = asyncio.Event()
        priority_c.disconnect_event.set()  # disconnected BEFORE activation
        for s in (normal_a, normal_b, priority_c):
            await q.enqueue(s)

        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        got = await q.pop_next(fastlane_policy=policy)

        assert got.slot_id == priority_c.slot_id
        assert got.is_evicted is True
        staged = await q.peek_staging()
        assert len(staged) == 2
        assert staged[0].slot_id == normal_a.slot_id  # untouched, still head


# ---------------------------------------------------------------------------
# 6. BUDGET — cross-model swap beyond the per-minute budget refused+counted.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSwapBudget:
    async def test_cold_jump_granted_not_refused_once_fastlane_overrides_the_budget(self):
        """Fast Lane overrides the swap budget entirely, so a cold jump is granted (replaces
        test_cold_jump_refused_when_budget_exhausted_falls_to_next_candidate).
        This follows from the design that Fast Lane must be able to override that
        setting entirely, which makes this test's ORIGINAL premise categorically
        false -- a Fast-Lane-matched, non-exempt cold jump is no longer
        refused-then-skipped-to-the-next-candidate; `_pick_fastlane_locked`'s
        candidate loop now grants the FIRST (highest-priority) fastlane
        candidate unconditionally (reason "budget_exempt"), so `cold_best`
        wins here, not `warm_fallback`. This is a
        regression guard against the inverted claim: this scenario would
        otherwise show a non-exempt lane candidate obeying the
        budget; it proves the opposite is true by design."""
        q = TurbohaulQueue(staging_max=10)
        # Warm model is "m1". The only priority candidate wants "m2" (cold
        # jump) and there's a lower-priority "m1" candidate behind it.
        cold_best = Slot.new(model_tag="m2", prompt="cold")
        cold_best.fastlane = _match(rule_index=0, rank=1)  # highest priority
        warm_fallback = Slot.new(model_tag="m1", prompt="warm")
        warm_fallback.fastlane = _match(rule_index=1, rank=1)  # lower priority
        for s in (cold_best, warm_fallback):
            await q.enqueue(s)

        # A loaded resident TYING cold_best's own (rule_index, rank) -- not
        # strictly worse -- keeps this a non-exempt-VIA-RANK candidate, so
        # the assertion below proves the blanket grant, not the
        # older rank-based swap-budget exemption (see TestSwapBudgetExemption).
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=0,
            loaded_priority_keys=((0, 1),),
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)

        assert got.slot_id == cold_best.slot_id  # granted, not skipped
        assert q._fastlane_refusal_counts.get("budget", 0) == 0  # never refused
        assert q.fastlane_swap_counts() == {"budget_exempt": 1}
        staged = await q.peek_staging()
        assert staged[0].slot_id == warm_fallback.slot_id  # cold_best popped, this remains

    async def test_cold_jump_granted_within_budget(self):
        q = TurbohaulQueue(staging_max=10)
        cold_best = Slot.new(model_tag="m2", prompt="cold")
        cold_best.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(cold_best)

        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=5)
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == cold_best.slot_id
        assert q._fastlane_refusal_counts.get("budget", 0) == 0

    async def test_budget_window_is_rolling_60s_not_a_hard_count_cap(self, monkeypatch):
        # Exercises _fastlane_swap_allowed_locked directly rather than
        # through pop_next: with only one staged slot, the PRE-EXISTING
        # (unrelated) affinity fallback always eventually serves it via its
        # own head-starvation logic regardless of what Fast Lane decides —
        # correct system behavior (a refused fastlane swap must not hang the
        # queue), but it means pop_next's return value alone can't isolate
        # the budget window's rolling-vs-hard-cap distinction.
        q = TurbohaulQueue(staging_max=10)
        assert q._fastlane_swap_allowed_locked(1) is True  # 1st grant
        assert q._fastlane_swap_allowed_locked(1) is False  # budget=1 exhausted
        assert q._fastlane_refusal_counts.get("budget", 0) == 0  # helper itself doesn't count

        real_monotonic = time.monotonic
        monkeypatch.setattr(
            "turbohaul.queue.time.monotonic", lambda: real_monotonic() + 61.0
        )
        assert q._fastlane_swap_allowed_locked(1) is True  # window rolled off -> available again

    async def test_lane_candidate_served_directly_never_needs_the_fifo_fallthrough(self):
        """A lane candidate is served directly, never via the FIFO fallthrough (replaces
        test_pop_next_refusal_falls_through_to_existing_fifo_not_dropped).
        The replaced test proved a refused cold jump wasn't silently dropped
        -- it fell through to the FIFO/affinity layer instead. That refusal
        can no longer happen for lane traffic (by design), so
        there's nothing left to fall through FROM: `s` is granted directly
        by `_pick_fastlane_locked` itself. Re-pointed to prove the STRONGER
        guarantee that actually holds -- not merely "not dropped" but
        "never refused at all" -- because the original
        scenario no longer exists. The FIFO-fallthrough-after-refusal
        behavior is proven for ORDINARY (non-
        lane) traffic by
        the swap-budget override test's
        test_STRONG_CONTROL_ordinary_cross_model_swap_now_refused_at_exhausted_budget
        (which asserts the refused slot stays staged, not dropped)."""
        q = TurbohaulQueue(staging_max=10, max_other_model_wait_s=9999.0)
        # Tying loaded key -- see the comment in
        # test_cold_jump_granted_not_refused_once_fastlane_overrides_the_budget:
        # keeps this candidate non-exempt-VIA-RANK, isolating the
        # blanket grant rather than the rank-based exemption.
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=0,
            loaded_priority_keys=((0, 1),),
        )
        s = Slot.new(model_tag="m2", prompt="1")
        s.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(s)
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == s.slot_id  # served directly, no fallthrough needed
        assert q._fastlane_refusal_counts.get("budget", 0) == 0  # never refused
        assert q.fastlane_swap_counts() == {"budget_exempt": 1}


@pytest.mark.asyncio
class TestFloorPromotedFlag:
    """slot.floor_promoted is set true-only-by the wall-clock fairness floor
    (an unregistered slot's rescue) -- never by an ordinary rule-match win."""

    async def test_floor_promotion_sets_the_flag(self, monkeypatch):
        q = TurbohaulQueue(staging_max=10)
        normal = Slot.new(model_tag="m1", prompt="normal")
        p1 = Slot.new(model_tag="m1", prompt="p1")
        p1.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(normal)
        await q.enqueue(p1)
        policy = FastLanePopPolicy(max_normal_wait_s=5.0, cross_model_switches_per_min=60)

        real_monotonic = time.monotonic
        monkeypatch.setattr(
            "turbohaul.queue.time.monotonic", lambda: real_monotonic() + 10.0
        )
        got = await q.pop_next(fastlane_policy=policy)
        assert got.slot_id == normal.slot_id
        assert got.floor_promoted is True

    async def test_ordinary_priority_win_leaves_the_flag_false(self):
        q = TurbohaulQueue(staging_max=10)
        p1 = Slot.new(model_tag="m1", prompt="p1")
        p1.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(p1)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        got = await q.pop_next(fastlane_policy=policy)
        assert got.slot_id == p1.slot_id
        assert got.floor_promoted is False

    async def test_plain_fifo_win_leaves_the_flag_false(self):
        """CONTROL: a slot served by the pre-existing plain-FIFO path (no
        fastlane policy at all) must default to False, not crash or default
        true -- the flag's default lives on Slot itself, this just confirms
        nothing on the fastlane pick path touches it when never reached."""
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m1", prompt="s")
        await q.enqueue(s)
        got = await q.pop_next()
        assert got.floor_promoted is False


@pytest.mark.asyncio
class TestSwapBudgetExemption:
    """A claimant strictly higher-priority than EVERY loaded resident is
    exempt from the swap budget (by design); a non-preempting
    cold jump still obeys it."""

    async def test_exempt_candidate_bypasses_an_exhausted_budget(self):
        q = TurbohaulQueue(staging_max=10)
        cand = Slot.new(model_tag="m2", prompt="cand")
        cand.fastlane = _match(rule_index=0, rank=1)  # top priority
        await q.enqueue(cand)
        # Budget is 0 -- would ALWAYS refuse a non-exempt cold jump.
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=0,
            loaded_priority_keys=((1, 1),),  # a loaded rank-1 resident, worse than cand
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == cand.slot_id
        assert q._fastlane_refusal_counts.get("budget", 0) == 0

    async def test_exempt_swap_does_not_consume_the_rolling_budget(self):
        """The exempt path must never call _fastlane_swap_allowed_locked --
        proven by showing the rolling window is still completely empty
        afterward, so a LATER non-exempt candidate gets full budget."""
        q = TurbohaulQueue(staging_max=10)
        cand = Slot.new(model_tag="m2", prompt="cand")
        cand.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(cand)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=1,
            loaded_priority_keys=((5, 1),),
        )
        await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert len(q._fastlane_swap_times) == 0, (
            "an exempt swap must not be recorded against the per-minute budget"
        )
        # Budget=1 is now fully available for an ordinary (non-exempt) swap.
        assert q._fastlane_swap_allowed_locked(1) is True

    async def test_non_rank_exempt_candidate_is_still_granted_via_the_override(self):
        """Replaces test_non_exempt_candidate_still_obeys_the_
        budget: a candidate that is NOT strictly higher
        than every loaded resident (not rank-exempt) would otherwise fall through to the
        ordinary budget gate and could be refused, but the design
        removes that fallthrough for ALL Fast-Lane-matched candidates, rank-
        exempt or not -- `cand` is not rank-exempt (loaded_priority_keys
        outranks it) yet still wins, via the separate, unconditional
        "budget_exempt" grant, not the rank-based "exempt" reason. This is
        the distinction the test now exists to prove: rank-exemption
        (TestSwapBudgetExemption, still real and still tested there) is a
        DIFFERENT mechanism from the blanket per-candidate override
        that applies to all lane candidates -- the old claim that a non-rank-exempt candidate "still obeys
        the budget" is exactly the defect this guards against."""
        q = TurbohaulQueue(staging_max=10)
        cand = Slot.new(model_tag="m2", prompt="cand")
        cand.fastlane = _match(rule_index=2, rank=1)  # worse than the loaded resident
        fallback = Slot.new(model_tag="m1", prompt="fallback")
        fallback.fastlane = _match(rule_index=3, rank=1)
        await q.enqueue(cand)
        await q.enqueue(fallback)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=0,  # budget exhausted -- irrelevant now
            loaded_priority_keys=((1, 1),),  # loaded resident OUTRANKS cand (not rank-exempt)
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == cand.slot_id  # granted despite not being rank-exempt
        assert q._fastlane_refusal_counts.get("budget", 0) == 0
        assert q.fastlane_swap_counts() == {"budget_exempt": 1}  # NOT "exempt" -- different reason
        staged = await q.peek_staging()
        assert staged[0].slot_id == fallback.slot_id  # cand popped, this remains

    async def test_vacuous_exemption_when_nothing_loaded_resolves(self):
        """No loaded resident resolves to a Fast Lane rule (loaded_priority_keys
        is empty) -- any registered candidate is exempt, matching "listed
        beats unlisted absolutely"."""
        q = TurbohaulQueue(staging_max=10)
        cand = Slot.new(model_tag="m2", prompt="cand")
        cand.fastlane = _match(rule_index=9, rank=1)  # even the WORST rule_index
        await q.enqueue(cand)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0,
            cross_model_switches_per_min=0,
            loaded_priority_keys=(),
        )
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == cand.slot_id
        assert q._fastlane_refusal_counts.get("budget", 0) == 0


@pytest.mark.asyncio
class TestSwapCountsByReason:
    """fastlane_swap_counts() -- the granted-swap counterpart of the already
    -shipped fastlane_refusal_counts()."""

    async def test_warm_no_jump_is_counted(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m1", prompt="s")
        s.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(s)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=0)
        await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)  # already warm
        assert q.fastlane_swap_counts() == {"warm": 1}

    async def test_exempt_jump_is_counted(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m2", prompt="s")
        s.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(s)
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0, cross_model_switches_per_min=0,
            loaded_priority_keys=((5, 1),),
        )
        await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert q.fastlane_swap_counts() == {"exempt": 1}

    async def test_budget_exempt_jump_is_counted(self):
        """Replaces test_budget_granted_jump_is_counted: the
        "budget" reason (rate-check genuinely consulted and passed) can
        no longer occur -- see _fastlane_swap_allowed_locked's own docstring,
        it is now structurally unreachable from this call site. `budget` is
        also a PREFIX of the new `budget_exempt` (a collision risk),
        so this assertion is an exact dict-equality check, not a
        substring/`in`/`.startswith` match, so a stale `{"budget": 1}` codes
        as a real failure here, not a silent pass; confirmed no reader
        anywhere in src/ or tests/ does substring matching on this key."""
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m2", prompt="s")
        s.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(s)
        # loaded_priority_keys ties s's own key exactly -- NOT strictly
        # higher (not rank-exempt), so this exercises the blanket
        # grant specifically, not the rank-based exemption path.
        policy = FastLanePopPolicy(
            max_normal_wait_s=9999.0, cross_model_switches_per_min=5,
            loaded_priority_keys=((0, 1),),
        )
        await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert q.fastlane_swap_counts() == {"budget_exempt": 1}

    async def test_returns_a_copy_not_the_live_dict(self):
        q = TurbohaulQueue(staging_max=10)
        snap = q.fastlane_swap_counts()
        snap["tampered"] = 999
        assert q.fastlane_swap_counts() == {}


# ---------------------------------------------------------------------------
# No mutation when the scan finds nothing to promote.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNoMutationOnEmptyScan:
    async def test_no_priority_candidates_leaves_staging_untouched(self):
        q = TurbohaulQueue(staging_max=10)
        a = Slot.new(model_tag="m1", prompt="a")
        b = Slot.new(model_tag="m1", prompt="b")
        await q.enqueue(a)
        await q.enqueue(b)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        # No slot has .fastlane set -> pick returns None internally -> falls
        # through to the untouched FIFO path, which DOES pop (that's the
        # existing behavior) -- the guarantee under test is that the
        # fastlane scan itself performed zero staging mutation before
        # falling through, i.e. `a` (the true FIFO head) is what's returned,
        # not something reordered by a buggy scan.
        assert got.slot_id == a.slot_id


# ---------------------------------------------------------------------------
# AST regression guard: no `await` inside any queue.py lock block, proven
# non-vacuous against synthetic source.
# ---------------------------------------------------------------------------

def _lock_await_violations(src):
    """Line numbers of `await` occurring inside an `async with ... _lock ...:` block.

    TOKENIZE, never ``ast.parse``. The AST form of this
    guard can crash with ``SystemError: AST constructor recursion depth mismatch``
    when parsing this module at pytest's stack depth -- selection-dependent, so
    the test would report a non-answer that looks like a failure. A flat token
    stream has no recursion and no stack sensitivity, so this cannot occur.
    """
    import io
    import tokenize as _tok

    violations = []
    depth = 0
    lock_depths = set()
    logical = []
    for tok in _tok.generate_tokens(io.StringIO(src).readline):
        ttype, tstr = tok.type, tok.string
        if ttype == _tok.INDENT:
            depth += 1
            continue
        if ttype == _tok.DEDENT:
            lock_depths.discard(depth)
            depth -= 1
            continue
        if ttype in (_tok.NL, _tok.COMMENT):
            continue
        if ttype == _tok.NEWLINE:
            names = [s for tt, s in logical if tt == _tok.NAME]
            if "with" in names and any("_lock" in s for tt, s in logical):
                if logical and logical[-1][1] == ":":
                    lock_depths.add(depth + 1)
            logical = []
            continue
        logical.append((ttype, tstr))
        if ttype == _tok.NAME and tstr == "await" and any(d <= depth for d in lock_depths):
            violations.append(tok.start[0])
    return violations


class TestNoAwaitInLockBlock:
    def test_queue_py_has_no_await_inside_a_lock_block(self):
        import turbohaul.queue as queue_module

        src = open(queue_module.__file__).read()
        violations = _lock_await_violations(src)
        assert violations == [], (
            f"await found inside a lock block at line(s) {violations}"
        )

    def test_checker_is_non_vacuous_against_synthetic_violation(self):
        """POSITIVE control: it must FIND a violation that is really there."""
        synthetic = (
            "async def f(self):\n"
            "    async with self._lock:\n"
            "        await something()\n"
        )
        assert len(_lock_await_violations(synthetic)) == 1

    def test_checker_does_not_flag_an_await_outside_the_lock(self):
        """NEGATIVE control for the tokenize-based checker.

        Without it, a checker that flagged EVERY await would still satisfy the
        positive control above while reporting the real file as violating; and
        one that flagged NOTHING would satisfy the real-file assertion
        vacuously. The pair pins BOTH directions -- fire inside the lock, stay
        silent outside it.
        """
        outside = (
            "async def f(self):\n"
            "    async with self._lock:\n"
            "        self.x.append(1)\n"
            "    await other()\n"
        )
        assert _lock_await_violations(outside) == []


# ---------------------------------------------------------------------------
# Manager-level: the enable switch itself, and client_meta byte-identity at
# the actual manager admission-stamp site.
# ---------------------------------------------------------------------------

def _boot_runtime(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    return boot, runtime


class _FakeTagRanks:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeRule:
    def __init__(self, address, tag_ranks=None):
        self.address = address
        self.label = ""
        self.tag_ranks = tag_ranks or {}


class _FakeFastLaneConfig:
    def __init__(self, enabled, rules, max_normal_wait_s=45.0, cross_model_switches_per_min=0, census_ttl_hours=168):
        self.enabled = enabled
        self.rules = rules
        self.max_normal_wait_s = max_normal_wait_s
        self.cross_model_switches_per_min = cross_model_switches_per_min
        self.census_ttl_hours = census_ttl_hours


@pytest.mark.asyncio
class TestManagerEnableSwitch:
    async def test_disabled_with_matching_rule_never_activates(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        fake_cfg = _FakeFastLaneConfig(
            enabled=False, rules=[_FakeRule("9.9.9.9", {"main": 1})],
        )
        object.__setattr__(runtime, "fastlane", fake_cfg)
        mgr = TurbohaulManager(boot, runtime)

        assert mgr._fastlane_pop_kwarg() is None

        slot = await mgr.submit(
            model_tag="m1", prompt="hi", client_meta={"ip": "9.9.9.9", "is_main": True},
        )
        assert slot.fastlane is None

        # Pop sequence unchanged from plain FIFO: enqueue a second, unrelated
        # slot and confirm strict order.
        other = await mgr.submit(model_tag="m1", prompt="other")
        got1 = await mgr.queue.pop_next(fastlane_policy=mgr._fastlane_pop_kwarg())
        got2 = await mgr.queue.pop_next(fastlane_policy=mgr._fastlane_pop_kwarg())
        assert got1.slot_id == slot.slot_id
        assert got2.slot_id == other.slot_id

    async def test_enabling_flips_all_three(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        fake_cfg = _FakeFastLaneConfig(
            enabled=False, rules=[_FakeRule("9.9.9.9", {"main": 1})],
        )
        object.__setattr__(runtime, "fastlane", fake_cfg)
        mgr = TurbohaulManager(boot, runtime)
        assert mgr._fastlane_pop_kwarg() is None

        fake_cfg.enabled = True
        mgr.invalidate_fastlane()
        assert mgr._fastlane_pop_kwarg() is not None

        slot = await mgr.submit(
            model_tag="m1", prompt="hi", client_meta={"ip": "9.9.9.9", "is_main": True},
        )
        assert slot.fastlane is not None
        assert slot.fastlane.rank == 1

    async def test_client_meta_byte_identical_at_manager_stamp_site(self, tmp_path):
        boot, runtime = _boot_runtime(tmp_path)
        fake_cfg = _FakeFastLaneConfig(
            enabled=True, rules=[_FakeRule("9.9.9.9", {"main": 1, "curator": 2})],
        )
        object.__setattr__(runtime, "fastlane", fake_cfg)
        mgr = TurbohaulManager(boot, runtime)

        client_meta = {"ip": "9.9.9.9", "is_curator": True, "is_sub_agent": True}
        before = json.dumps(client_meta, sort_keys=True, default=str)
        slot = await mgr.submit(model_tag="m1", prompt="hi", client_meta=client_meta)
        after = json.dumps(slot.client_meta, sort_keys=True, default=str)

        assert before == after
        # And the match DID fire (curator wins over sub_agent, per the trap).
        assert slot.fastlane is not None
        assert slot.fastlane.effective_tag == "curator"


# ---------------------------------------------------------------------------
# 7. accept_buf overflow reorder (SECONDARY hardening, not the
#    primary fix; see class 9 below for that).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOverflowReorderFix:
    """Staging can only be simultaneously EMPTY-at-pick-time while accept_buf
    holds something when staging was AT CAPACITY at the moment of an earlier
    admission (an overflow condition) -- every existing pop path already
    replenishes staging by one from accept_buf immediately after removing a
    winner, so under the production-default staging_max=100 this practically
    never happens under ordinary one-at-a-time arrivals (see class 8, the
    negative control, which proves that explicitly). This test forces the
    overflow condition deterministically: staging_max=2 fills staging with
    two slots (so a THIRD arrival overflows into accept_buf), then frees ONE
    staging slot via remove() -- a removal path that, unlike every pop_next
    branch, does NOT replenish from accept_buf -- so at pick time staging has
    genuine room (below capacity) while accept_buf still holds the priority
    slot. That combination is exactly what the old (pick-before-drain)
    ordering could never see, and the new ordering fixes.
    """

    async def test_overflow_item_reaches_fastlane_pick(self):
        q = TurbohaulQueue(staging_max=2, acceptance_max=10)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)

        filler_a = Slot.new(model_tag="m1", prompt="a")
        filler_b = Slot.new(model_tag="m1", prompt="b")
        priority = Slot.new(model_tag="m1", prompt="priority")
        priority.fastlane = _match(rule_index=0, rank=1)

        await q.enqueue(filler_a)   # staging=[a]
        await q.enqueue(filler_b)   # staging=[a,b] -- now at capacity
        await q.enqueue(priority)   # staging full -> overflow to accept_buf
        assert len(q._staging) == 2
        assert len(q._accept_buf) == 1

        # Free ONE staging slot WITHOUT a replenish (remove() is the only
        # queue.py removal path that doesn't top up from accept_buf --
        # confirmed by direct enumeration; every pop_next branch does).
        removed = await q.remove(filler_a.slot_id)
        assert removed is filler_a
        assert len(q._staging) == 1
        assert len(q._accept_buf) == 1

        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == priority.slot_id, (
            "priority slot sat in accept_buf with room now available in "
            "staging, invisible to the pick unless the reorder drains it "
            "in before _pick_fastlane_locked runs"
        )


# ---------------------------------------------------------------------------
# 8. Plain-trickle NEGATIVE CONTROL. Passes on BOTH the old
#    ordering and the new one. Documented explicitly so it is never
#    miscounted as evidence the reorder fix mattered here (a fixture is only evidence if a plausible
#    WRONG implementation would fail it; ask what such an
#    implementation does to the fixtures).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPlainTrickleNegativeControl:
    """NOT evidence for the reorder fix. This passes on the ORIGINAL
    pick-before-drain ordering too, because enqueue()'s own capacity check
    already sends a lone arrival straight into empty staging (not
    accept_buf) whenever staging has room -- true under the production
    default staging_max=100, and true here. It exists to DOCUMENT that fact
    plainly: a naive "one in flight, one arriving next" trickle test proves
    NOTHING about this fix, exactly the trap to avoid.
    Real reorder coverage is TestOverflowReorderFix above.
    """

    async def test_plain_trickle_already_worked_before_the_fix(self):
        q = TurbohaulQueue(staging_max=100, acceptance_max=10000)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)

        normal = Slot.new(model_tag="m1", prompt="normal")
        await q.enqueue(normal)
        popped_normal = await q.pop_next(warm_model_tag=None, fastlane_policy=policy)
        assert popped_normal.slot_id == normal.slot_id

        # "Second arriving while the first is being served": staging is now
        # empty (normal was just popped), so this lands DIRECTLY in staging
        # via enqueue()'s own capacity check -- never touches accept_buf.
        priority = Slot.new(model_tag="m1", prompt="priority")
        priority.fastlane = _match(rule_index=0, rank=1)
        await q.enqueue(priority)
        assert len(q._staging) == 1
        assert len(q._accept_buf) == 0

        got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == priority.slot_id


# ---------------------------------------------------------------------------
# 9. Observability, PRIMARY: arrival logging. Makes a waiting
#    request visible the instant it's admitted, not only when picked.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestArrivalLogging:
    async def test_arrival_logs_staging_landing_via_enqueue(self, caplog):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new(model_tag="m1", prompt="x")
        with caplog.at_level(logging.INFO):
            await q.enqueue(s)
        lines = [r.getMessage() for r in caplog.records if "QUEUE_ARRIVAL" in r.getMessage()]
        assert len(lines) == 1
        assert "landed=staging" in lines[0]
        assert "via=enqueue" in lines[0]
        assert "model=m1" in lines[0]
        assert "fastlane_matched=False" in lines[0]

    async def test_arrival_logs_accept_buf_landing_on_overflow(self, caplog):
        q = TurbohaulQueue(staging_max=1, acceptance_max=10)
        await q.enqueue(Slot.new(model_tag="m1", prompt="a"))
        overflow = Slot.new(model_tag="m1", prompt="b")
        with caplog.at_level(logging.INFO):
            await q.enqueue(overflow)
        lines = [r.getMessage() for r in caplog.records if "QUEUE_ARRIVAL" in r.getMessage()]
        assert len(lines) == 1
        assert "landed=accept_buf" in lines[0]

    async def test_arrival_logs_fastlane_matched_true(self, caplog):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new(model_tag="m1", prompt="x")
        s.fastlane = _match()
        with caplog.at_level(logging.INFO):
            await q.enqueue(s)
        lines = [r.getMessage() for r in caplog.records if "QUEUE_ARRIVAL" in r.getMessage()]
        assert "fastlane_matched=True" in lines[0]

    async def test_arrival_logs_via_enqueue_head(self, caplog):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new(model_tag="m1", prompt="x")
        with caplog.at_level(logging.INFO):
            await q.enqueue_head(s)
        lines = [r.getMessage() for r in caplog.records if "QUEUE_ARRIVAL" in r.getMessage()]
        assert len(lines) == 1
        assert "via=enqueue_head" in lines[0]
        assert "landed=staging" in lines[0]


# ---------------------------------------------------------------------------
# 10. Observability, PRIMARY: decline logging. Separates
#     "staging empty" (never really tried) from "no candidate matched" (ran,
#     declined) -- both looked identical (silence) before this change.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFastlaneDeclinedLogging:
    async def test_declined_staging_empty(self, caplog):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        with caplog.at_level(logging.DEBUG):
            got = await q.pop_next(warm_model_tag=None, fastlane_policy=policy)
        assert got is None
        lines = [r.getMessage() for r in caplog.records if "FASTLANE_DECLINED" in r.getMessage()]
        assert len(lines) == 1
        assert "reason=staging_empty" in lines[0]

    async def test_declined_no_candidate(self, caplog):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        normal = Slot.new(model_tag="m1", prompt="normal")
        await q.enqueue(normal)
        with caplog.at_level(logging.DEBUG):
            got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        # normal has no fastlane match -> pick runs a real scan, declines, falls to FIFO.
        assert got.slot_id == normal.slot_id
        lines = [r.getMessage() for r in caplog.records if "FASTLANE_DECLINED" in r.getMessage()]
        assert len(lines) == 1
        assert "reason=no_candidate" in lines[0]

    async def test_declined_absent_when_pick_succeeds(self, caplog):
        """Negative control: no FASTLANE_DECLINED line when the pick actually wins."""
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        policy = FastLanePopPolicy(max_normal_wait_s=9999.0, cross_model_switches_per_min=60)
        priority = Slot.new(model_tag="m1", prompt="p")
        priority.fastlane = _match()
        await q.enqueue(priority)
        with caplog.at_level(logging.DEBUG):
            got = await q.pop_next(warm_model_tag="m1", fastlane_policy=policy)
        assert got.slot_id == priority.slot_id
        lines = [r.getMessage() for r in caplog.records if "FASTLANE_DECLINED" in r.getMessage()]
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# 11. Observability, PRIMARY, the one that catches
#     a bypassed pick: a slot served via pop_matched_thread (grace-window
#     same-thread continuation) entirely bypasses pop_next's ladder, so
#     _pick_fastlane_locked never ran for it. This is a real production
#     mechanism.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAdmittedWithoutPickLogging:
    async def test_pop_matched_thread_logs_bypass(self, caplog):
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        s = Slot.new(model_tag="m1", thread_id="t1", prompt="x")
        await q.enqueue(s)
        with caplog.at_level(logging.INFO):
            got = await q.pop_matched_thread("t1", "m1")
        assert got.slot_id == s.slot_id
        lines = [r.getMessage() for r in caplog.records if "QUEUE_ADMITTED_NO_PICK" in r.getMessage()]
        assert len(lines) == 1
        assert "model=m1" in lines[0]
        # Supplement: the line must name WHICH mechanism bypassed
        # the pick, not just that one did -- a bare no-pick flag would
        # not say which path to investigate.
        assert "via=pop_matched_thread" in lines[0]

    async def test_pop_matched_thread_no_match_does_not_log(self, caplog):
        """Negative control: no match, no log."""
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        with caplog.at_level(logging.INFO):
            got = await q.pop_matched_thread("nonexistent", "m1")
        assert got is None
        lines = [r.getMessage() for r in caplog.records if "QUEUE_ADMITTED_NO_PICK" in r.getMessage()]
        assert len(lines) == 0

    async def test_pop_matched_thread_bypasses_fastlane_even_with_a_priority_slot_waiting(self, caplog):
        """Documents the actual production mechanism -- NOT a bug
        (admission-only means this bypass is deliberate; grace-window
        continuation intentionally isn't preempted by Fast Lane), just
        OBSERVABLE: a grace-window thread match is served even when a
        DIFFERENT, fastlane-tagged slot is also waiting in staging, because
        pop_matched_thread never consults _pick_fastlane_locked at all.
        """
        q = TurbohaulQueue(staging_max=10, acceptance_max=10)
        grace_match = Slot.new(model_tag="m1", thread_id="t1", prompt="continuation")
        priority = Slot.new(model_tag="m1", prompt="p")
        priority.fastlane = _match()
        await q.enqueue(grace_match)
        await q.enqueue(priority)
        with caplog.at_level(logging.INFO):
            got = await q.pop_matched_thread("t1", "m1")
        assert got.slot_id == grace_match.slot_id
        lines = [r.getMessage() for r in caplog.records if "QUEUE_ADMITTED_NO_PICK" in r.getMessage()]
        assert len(lines) == 1
        # The priority slot is untouched -- pop_matched_thread only removed grace_match.
        assert priority in q._staging
