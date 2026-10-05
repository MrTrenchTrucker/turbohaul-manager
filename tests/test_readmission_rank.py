"""Priority governs RE-admission.

⛔ THIS BUILDS ON AN EARLIER FIX. The earlier change (a
rank-admission change) enforces that PRIORITY MUST
GOVERN RE-ADMISSION, NOT ONLY FIRST ADMISSION, and its fix is
correct and running. It does not cover this: both of its halves are wired to
other questions.
  * manager side (``_rank_admit_woken``) ranks the slot handed to a PARKED
    ``r.inbox.get()``. ``_route_or_reserve`` does ``r.inbox.put_nowait(slot)``
    and on the NEXT line ``_release_fastlane_claim_locked(slot, "admitted")``,
    which is what emits the FASTLANE_CLAIM ADM line -- so that helper runs
    strictly DOWNSTREAM of the admission this file is about.
  * queue side (``_has_strictly_higher_priority_waiting_locked`` widened to
    scan ``_accept_buf``) has exactly ONE live caller,
    ``_same_model_residency_floor``; its other two callers
    (``_legacy_fastlane_grace_breakout``, ``_process_slot``) are both marked
    UNCALLED and retained but unused. Nothing live
    asks it to order an admission.

THE DEFECT
(visible in the log of a real run):
``_defer_unroutable`` registers a claim and then hands the slot to
``_requeue_after_backoff``, which AWAITS ``_wait_for_park_or_timeout(backoff)``
BEFORE it calls ``enqueue_head``. For that whole backoff the slot is in NO
buffer -- not ``_staging``, not ``_accept_buf``. ``_pick_fastlane_locked``
sorts correctly by ``(rule_index, rank, created_at)`` over a set that does not
contain the waiter. Duty cycle: two clients of different rank can each be
absent for most of their wait, and the contested admission is then split by a few ms of
backoff phase -- the lower-ranked client can be admitted soon after its own defer while the higher-ranked one
sits in an identical backoff. Rank never enters the decision.

⭐ WHY THE ASSERTION IS ON THE SYMPTOM, NOT THE MECHANISM. This file asserts
SERVICE ORDER -- "a lower-ranked client is not served while a higher-ranked one
is already waiting" -- rather than any particular mechanism,
so it stays valid under any fix shape.

★ SHAPE (B), the chosen design. The fix WITHHOLDS service from an
outranked slot (it is held aside for one call, exactly as a grace-protected
slot is) rather than PROMOTING the claimant. Shape (C) -- pull the best live
claim when capacity frees -- was DEFERRED, not rejected on merit. That
distinction is load-bearing here and is recorded executably in
``TestShapeCIsDeferredNotAbandoned`` below.

⛔ (B) CAN FAIL IN TWO WAYS THAT EVERY ORDERING TEST ABOVE WOULD MISS, because
a hold that never lifts serves nobody and therefore never inverts anybody:
  1. a hold that is PERMANENT rather than bounded -> ``TestTheHoldIsBounded``
  2. a DEAD claim that blocks forever -> ``TestADeadClaimDoesNotBlockService``
Each converts an ordering defect into a strictly worse LIVENESS defect, and
each needs its own test. Neither is redundant with the other.

⛔ FIXTURE TRAP, inherited deliberately from
the reserve-VRAM-gate requeue tests: a bare ``Slot.new(...)``
has ``fastlane=None``, so ``_fastlane_claim_key`` returns None and claim
registration SILENTLY NO-OPS. Every listed slot here is constructed directly
with a real ``FastLaneMatch``, and
``test_the_fixture_actually_registers_claims`` below fails loudly if that ever
stops being true.

Behaviour-changing fix: the tests below pin the new behaviour.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import pathlib
import time

import pytest

import turbohaul.manager as manager_mod
import turbohaul.queue as queue_mod
from turbohaul.config import (
    BootConfig,
    FastLaneConfig,
    FastLaneRule,
    FastLaneTagRanks,
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
from turbohaul.queue import FastLanePopPolicy
from turbohaul.slot import Slot, SlotState

# rules list index IS the priority: index 0 outranks index 1 (config.py,
# "rules: list[FastLaneRule] -- LIST INDEX IS THE PRIORITY").
HIGH_ADDR = "10.0.0.1"     # rule_index 0 -- stands in for a high-priority client
LOW_ADDR = "10.0.0.2"      # rule_index 1 -- stands in for a lower-priority client
UNLISTED_ADDR = "10.9.9.9"  # matches no rule -- ordinary traffic

_RULES = [
    FastLaneRule(address=HIGH_ADDR, tag_ranks=FastLaneTagRanks(main=1)),
    FastLaneRule(address=LOW_ADDR, tag_ranks=FastLaneTagRanks(main=1)),
]


@pytest.fixture
def mgr(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    for sub in ("blobs", "manifests", "import-staging"):
        (storage_root / sub).mkdir()
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
            default_port_base=59900,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),
    )
    runtime = RuntimeConfig(
        queue=QueueConfig(safety_enabled=False),
        pull=PullConfig(),
        fastlane=FastLaneConfig(enabled=True, rules=_RULES),
    )
    return TurbohaulManager(boot, runtime)


def _slot(slot_id: str, addr: str, rule_index: int, model_tag: str) -> Slot:
    """A REAL fastlane-matched slot. See the FIXTURE TRAP note in the module
    docstring: fastlane=None would make claim registration a silent no-op."""
    return Slot(
        slot_id=slot_id,
        model_tag=model_tag,
        state=SlotState.RECEIVED,
        thread_id=f"t-{slot_id}",
        client_meta={"ip": addr},
        fastlane=FastLaneMatch(
            # rank is the WITHIN-IP sub-rank and is equal for both clients on
            # purpose: this file is about rule_index (which client outranks
            # which), not about tag sub-ranks. Config validation requires >= 1.
            rule_index=rule_index, raw_address=addr, label="",
            effective_tag="main", rank=1,
        ),
    )


def _unlisted_slot(slot_id: str, model_tag: str) -> Slot:
    """Ordinary traffic: matches no Fast Lane rule, so ``fastlane`` is None and
    ``_fastlane_priority_key`` returns None for it. Deliberately NOT built by
    ``_slot`` -- the whole point is the absent match."""
    return Slot(
        slot_id=slot_id,
        model_tag=model_tag,
        state=SlotState.RECEIVED,
        thread_id=f"t-{slot_id}",
        client_meta={"ip": UNLISTED_ADDR},
        fastlane=None,
    )


def _policy(mgr: TurbohaulManager) -> FastLanePopPolicy:
    """⛔ BUILT BY THE MANAGER, NOT BY HAND, AND THAT IS LOAD-BEARING.

    Constructing ``FastLanePopPolicy(...)``
    directly would make the fix UNREACHABLE BY CONSTRUCTION: the policy is the
    only channel by which the pick learns about live claims, so a hand-built
    one is permanently claim-blind and the test could not tell a working fix
    from a missing one (it would go red identically on both trees). Driving
    ``_fastlane_pop_kwarg`` is what puts the production wiring under test.
    ``test_the_policy_carries_the_claim_channel`` below fails loudly if that
    channel ever disappears again.

    ``max_normal_wait_s`` defaults to 3600.0, so the wall-clock fairness
    floor cannot fire inside a sub-second test and rescue anyone -- this file
    is about priority order and a floor promotion answers a different
    question.
    """
    policy = mgr._fastlane_pop_kwarg()
    assert policy is not None, "fastlane must be enabled for this fixture"
    return policy


async def _defer(mgr: TurbohaulManager, slot: Slot, *, evict_pending: bool = True) -> None:
    """Put ``slot`` into the exact production state a real run reaches: a live
    fastlane claim, and the slot itself handed to a backoff requeue task that
    has NOT yet re-enqueued it. That is `live-but-unstaged`.

    ``evict_pending`` picks the production regime: True is the VRAM path
    (``_VRAM_DEFER_BACKOFF_S``, 1.0s -- what a real run used) and
    False is the busy/count-cap path (``_DISPATCH_DEFER_BACKOFF_S``, 0.05s).
    Tests that must WAIT OUT a real backoff use False so they cost 50ms
    instead of a second; the ordering property is identical in both regimes.
    """
    async with mgr._registry_lock:
        mgr._defer_unroutable(slot, evict_pending=evict_pending)


async def _cancel_bg(mgr: TurbohaulManager) -> None:
    for t in list(getattr(mgr, "_bg_tasks", ())):
        if not t.done():
            t.cancel()
    await asyncio.sleep(0)


async def _drain_bg(mgr: TurbohaulManager, timeout: float = 5.0) -> None:
    """Let every spawned background task RUN TO COMPLETION -- in particular
    ``_requeue_after_backoff``, which is what re-enqueues a deferred claimant.

    Deliberately awaits the real tasks instead of ``asyncio.sleep(backoff)``:
    sleeping a guessed interval is a race that passes on a fast box and flakes
    on a loaded one, and it would also pass if the requeue never happened at
    all. ``_bg_tasks`` self-removes on completion, so the list is snapshotted
    first.
    """
    tasks = [t for t in list(getattr(mgr, "_bg_tasks", ())) if not t.done()]
    if tasks:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
        )


def _kill_claim(mgr: TurbohaulManager, slot: Slot, how: str) -> None:
    """Make ``slot``'s claim DEAD by one of ``_claim_is_live``'s own release
    reasons, without pruning it from the registry -- the claim must remain
    PRESENT so the test distinguishes 'liveness is checked' from 'the entry
    happened to be gone'.

    ``queue_closed`` is deliberately absent: closing the queue disables
    ``pop_next`` itself, so that arm could not tell a liveness check from a
    closed queue and would pass for the wrong reason.
    """
    if how == "disconnected":
        slot.disconnect_event = asyncio.Event()
        slot.disconnect_event.set()
    elif how == "evicted":
        slot.is_evicted = True
    elif how == "failed_future":
        fut = asyncio.get_running_loop().create_future()
        fut.set_exception(RuntimeError("client went away"))
        slot.completion_future = fut
    elif how == "ttl_expired":
        key = mgr._fastlane_claim_key(slot)
        # the registry's own field, read by _claim_is_live; moving the deadline
        # into the past is the only way to reach that branch without faking the
        # clock for every other consumer of time.monotonic().
        mgr._fastlane_claims[key]["ttl_deadline_monotonic"] = time.monotonic() - 1.0
    else:  # pragma: no cover - guards a typo in a parametrize list
        raise AssertionError(f"unknown kill mode {how!r}")


# --------------------------------------------------------------------------
# Gates. These are not the main assertion -- they are what stops it being
# vacuous. Each one guards against a false reading seen before.
# --------------------------------------------------------------------------
class TestFixtureIsNotVacuous:
    async def test_the_fixture_actually_registers_claims(self, mgr):
        """If this fails, every assertion below is meaningless: no claim means
        nothing is 'already waiting' and the rule has nothing to govern."""
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        try:
            await _defer(mgr, high)
            assert mgr._fastlane_claims, (
                "no claim registered -- the fastlane=None trap fired, or "
                "_fastlane_claim_eligible rejected this slot"
            )
            key = mgr._fastlane_claim_key(high)
            assert key in mgr._fastlane_claims, f"claim keyed unexpectedly: {key!r}"
        finally:
            await _cancel_bg(mgr)

    async def test_the_policy_carries_the_claim_channel(self, mgr):
        """⛔ THE REACHABILITY GATE. The pick can only honour a live claim if
        the policy bundle has somewhere to put it. If this attribute vanishes,
        every ordering assertion below silently becomes untestable rather than
        failing -- which is how a fixture hides a missing fix."""
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        try:
            await _defer(mgr, high)
            policy = _policy(mgr)
            assert hasattr(policy, "governing_claim_key"), (
                "FastLanePopPolicy has no governing_claim_key -- the channel "
                "carrying live-claim priority into the pick is GONE, and the "
                "ordering tests below cannot detect anything"
            )
            assert policy.governing_claim_key is not None, (
                "a live claim exists but the manager published no governing "
                "key -- the pick will be claim-blind"
            )
        finally:
            await _cancel_bg(mgr)

    async def test_the_deferred_slot_is_genuinely_unstaged(self, mgr):
        """The premise of the whole fix: while the backoff task sleeps, the
        slot is in NEITHER queue buffer. If it were staged, the picker would
        see it and there would be no defect to fix."""
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        try:
            await _defer(mgr, high)
            staged = list(mgr.queue._staging) + list(mgr.queue._accept_buf)
            assert high not in staged, (
                "the deferred slot IS in a buffer -- the live-but-unstaged "
                "window this fix is about does not exist in this fixture"
            )
            assert mgr._fastlane_claims, "…but it must still hold a live claim"
        finally:
            await _cancel_bg(mgr)


# --------------------------------------------------------------------------
# The core assertion.
# --------------------------------------------------------------------------
class TestPriorityGovernsReAdmission:
    async def test_lower_rank_is_not_served_while_higher_rank_is_waiting(self, mgr):
        """⛔ THE CORE ASSERTION, as a rule: "A lower-ranked
        client is not admitted ahead of a higher-ranked client that is already
        waiting."

        HIGH (rule_index 0) is already waiting -- it holds a live claim, and is
        mid-backoff exactly as in a real run. LOW
        (rule_index 1) is staged. Serving LOW here is the production
        inversion.

        RED on the pre-fix code: pop_next returns LOW, because the pick's candidate set
        is the two buffers and HIGH is in neither.
        """
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        low = _slot("low", LOW_ADDR, 1, "model-b")
        try:
            await _defer(mgr, high)
            await mgr.queue.enqueue(low)

            assert mgr._fastlane_claims, "gate: HIGH must hold a live claim"
            served = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))

            assert served is not low, (
                "PRIORITY VIOLATION: rule_index 1 was served while rule_index 0 "
                "held a live claim and was merely mid-backoff. This is the "
                "production inversion: the pick ranked correctly over a "
                "candidate set that did not contain the higher-ranked waiter."
            )
        finally:
            await _cancel_bg(mgr)


# --------------------------------------------------------------------------
# ⛔ WHICH live claim governs. A selector that returns the worst claim
# (``key < best[0]`` -> ``key >`` in _governing_claim_locked) would go UNDETECTED by any
# file made only of single-claim tests.
# ⭐ THE REASON IT GOES UNDETECTED IS STRUCTURAL, NOT AN OVERSIGHT ABOUT ASSERTIONS:
# every other test here builds exactly ONE live claim, and over a set of one
# the best and the worst are the SAME OBJECT -- so the selection step is
# unobservable no matter how hard those tests assert. A comparator is only
# under test when the set it ranks has TWO members with different keys.
# --------------------------------------------------------------------------
class TestTheBestLiveClaimGoverns:
    """★ The rule does not say "a live claim governs", it says the HIGHEST-RANKED
    one does. One claim cannot tell those two readings apart."""

    async def test_the_governing_key_is_the_best_claim_not_the_worst(self, mgr):
        """Direct on the selector, with two live claims at rule_index 0 and 2.

        Catches a selector that returns the WORST claim's key. That bug is
        otherwise invisible: every single-claim test passes with it, and the
        production symptom (the wrong client is protected) looks identical to
        a correct hold from outside.
        """
        best = _slot("best", HIGH_ADDR, 0, "model-a")
        worst = _slot("worst", LOW_ADDR, 2, "model-c")
        try:
            await _defer(mgr, best)
            await _defer(mgr, worst)

            assert len(mgr._fastlane_claims) == 2, (
                "gate: this test is VACUOUS below two live claims -- selection "
                f"is only observable over a set. got {sorted(mgr._fastlane_claims)}"
            )
            best_key = mgr.queue._fastlane_priority_key(best)
            worst_key = mgr.queue._fastlane_priority_key(worst)
            assert best_key is not None and worst_key is not None
            assert best_key < worst_key, (
                "gate: the fixture must present a STRICT ordering, else 'best' "
                f"and 'worst' are interchangeable. got {best_key} vs {worst_key}"
            )

            governing = mgr._governing_claim_priority_key_locked()

            assert governing == best_key, (
                f"the WRONG live claim governs: got {governing}, expected the "
                f"best {best_key}, not the worst {worst_key}. A selector that "
                "returns the worst claim protects the client the rule exists to "
                "outrank, and starves the one it exists to protect."
            )
        finally:
            await _cancel_bg(mgr)

    async def test_a_mid_ranked_arrival_is_held_by_the_best_claim(self, mgr):
        """The same defect at the queue boundary, so this class does not rest
        on the selector's return value alone.

        MID (rule_index 1) sits BETWEEN the two live claims. Governed by the
        best claim (0) it is outranked and must be held; governed by the worst
        (2) it outranks the holder and is served. Serving MID is the bug, and
        MID's rank is chosen so the two readings disagree -- with MID outside
        the interval both would hold, and the test would pass on the mutant.
        """
        best = _slot("best", HIGH_ADDR, 0, "model-a")
        worst = _slot("worst", LOW_ADDR, 2, "model-c")
        mid = _slot("mid", LOW_ADDR, 1, "model-b")
        try:
            await _defer(mgr, best)
            await _defer(mgr, worst)
            await mgr.queue.enqueue(mid)

            assert len(mgr._fastlane_claims) == 2, "gate: two live claims required"

            served = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))

            assert served is not mid, (
                "rule_index 1 was served while rule_index 0 held a live claim. "
                "The hold was computed against the WORST live claim "
                "(rule_index 2), which MID outranks, instead of against the "
                "best -- the selection step, not the comparison step."
            )
        finally:
            await _cancel_bg(mgr)

# --------------------------------------------------------------------------
# ⛔ LIVENESS HALF 1 -- the hold must LIFT.
# --------------------------------------------------------------------------
class TestTheHoldIsBounded:
    """★ THE TEST THAT MAKES SHAPE (B) SAFE. (B) withholds service rather than
    promoting the claimant, so its catastrophic failure mode is a hold that is
    PERMANENT rather than bounded -- and a permanent hold PASSES every ordering
    assertion in this file, because nothing lower-ranked is ever served ahead
    of anything. The ordering tests are structurally blind to it. This is not
    a duplicate of the dead-claim class below: that one covers a DEAD claim
    blocking, this one covers a LIVE claim blocking FOREVER.
    """

    async def test_the_hold_lifts_once_the_claimant_re_enqueues(self, mgr):
        """The full production cycle, end to end: defer -> hold -> backoff
        expires -> ``enqueue_head`` -> the claimant is served. The busy regime
        (0.05s) is used so this waits out a REAL backoff cheaply.
        """
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        low = _slot("low", LOW_ADDR, 1, "model-b")
        try:
            await _defer(mgr, high, evict_pending=False)
            await mgr.queue.enqueue(low)

            first = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            assert first is not low, (
                "gate: the hold must be ON for this test to mean anything -- "
                "if LOW is served here the inversion is simply unfixed"
            )

            # Wait out the REAL backoff by awaiting the REAL requeue task.
            await _drain_bg(mgr)
            staged = list(mgr.queue._staging) + list(mgr.queue._accept_buf)
            assert high in staged, (
                "gate: _requeue_after_backoff never re-enqueued the claimant, "
                "so this test cannot observe whether the hold lifts"
            )

            second = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            assert second is high, (
                "THE HOLD DID NOT LIFT. The claimant is back in the queue and "
                "still was not served -- shape (B) has converted an ordering "
                f"defect into a LIVENESS defect. got "
                f"{getattr(second, 'slot_id', second)!r}"
            )
        finally:
            await _cancel_bg(mgr)

    async def test_the_held_slot_is_deferred_not_dropped(self, mgr):
        """The hold-aside splices back in the ``finally``, so LOW must still be
        there afterwards. If a held slot were dropped, every ordering test
        above would still pass while the queue silently ate requests."""
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        low = _slot("low", LOW_ADDR, 1, "model-b")
        try:
            await _defer(mgr, high, evict_pending=False)
            await mgr.queue.enqueue(low)

            await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            await _drain_bg(mgr)
            assert await mgr.queue.pop_next(fastlane_policy=_policy(mgr)) is high

            # HIGH is served and its claim released on admission in a real run;
            # here nothing is left claiming, so LOW is now the best waiter.
            remaining = list(mgr.queue._staging) + list(mgr.queue._accept_buf)
            assert low in remaining, (
                "LOW was DROPPED by the hold-aside rather than spliced back -- "
                "the queue is eating requests, which is worse than the "
                "inversion this hold exists to fix"
            )
        finally:
            await _cancel_bg(mgr)


# --------------------------------------------------------------------------
# ⛔ LIVENESS HALF 2 -- a DEAD claim must not block.
# --------------------------------------------------------------------------
class TestADeadClaimDoesNotBlockService:
    """⛔ THE REGISTRY HAS NO SWEEPER. ``_fastlane_claims``' only pruner is
    ``fastlane_claims_snapshot``, whose single production caller is
    ``status_snapshot`` -- i.e. an HTTP /status or /ws/state request. Nothing
    periodic prunes. So a claim keyed on PRESENCE rather than LIVENESS could
    block service indefinitely on a host nobody is polling, which is strictly
    worse than the ordering defect the hold fixes.

    The fix must therefore consult ``_claim_is_live`` -- and it does so by
    delegating to ``_governing_claim_priority_key_locked``, the SAME predicate
    the eviction path uses, rather than inventing a second liveness notion.
    """

    @pytest.mark.parametrize(
        "how", ["disconnected", "evicted", "failed_future", "ttl_expired"]
    )
    async def test_a_dead_claim_does_not_hold_the_queue(self, mgr, how):
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        low = _slot("low", LOW_ADDR, 1, "model-b")
        try:
            await _defer(mgr, high)
            assert mgr._fastlane_claims, "gate: the claim must start out LIVE"
            assert _policy(mgr).governing_claim_key is not None, (
                "gate: a LIVE claim must govern, or this test proves nothing "
                "about what happens when it dies"
            )

            _kill_claim(mgr, high, how)

            # ⭐ THE NON-VACUITY GATE. The claim must still be PRESENT in the
            # registry -- otherwise this test would pass because the entry
            # vanished, not because its liveness was checked, and it would go
            # on passing against a presence-keyed implementation.
            assert mgr._fastlane_claims, (
                "the claim was PRUNED, so this test can no longer distinguish "
                "'liveness is checked' from 'the entry is simply gone'"
            )
            assert _policy(mgr).governing_claim_key is None, (
                f"a {how} claim is still being published as the governing "
                "key -- the fix is keyed on PRESENCE, not LIVENESS"
            )

            await mgr.queue.enqueue(low)
            served = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            assert served is low, (
                f"a DEAD ({how}) claim blocked service. Nothing sweeps this "
                "registry outside a /status request, so this is an indefinite "
                f"stall, not a delay. got {getattr(served, 'slot_id', served)!r}"
            )
        finally:
            await _cancel_bg(mgr)


# --------------------------------------------------------------------------
# ⭐ NON-VACUITY CONTROLS. A test that flags every re-admission is not a fix,
# it is an alarm.
# --------------------------------------------------------------------------
class TestCorrectlyOrderedAdmissionStaysSilent:
    async def test_low_rank_IS_served_when_no_higher_claim_is_waiting(self, mgr):
        """No live HIGH claim => LOW is the best waiter and MUST be served.
        If this ever fails, the fix has become 'never admit anybody' and the
        test above is passing for the wrong reason."""
        low = _slot("low", LOW_ADDR, 1, "model-b")
        try:
            await mgr.queue.enqueue(low)
            assert not mgr._fastlane_claims, "control: no claim may be live here"
            served = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            assert served is low, (
                "a correctly-ordered admission was refused -- the guard is "
                "firing on every admission, not on inversions"
            )
        finally:
            await _cancel_bg(mgr)

    async def test_equal_rank_is_not_an_inversion(self, mgr):
        """Two claimants at the SAME rule_index: the rule governs priority order,
        and an equal rank is not 'lower'. Serving the staged one is correct
        and must stay silent -- this is the boundary the guard must not
        over-reach past."""
        peer_waiting = _slot("peer-waiting", LOW_ADDR, 1, "model-a")
        peer_staged = _slot("peer-staged", LOW_ADDR, 1, "model-b")
        try:
            await _defer(mgr, peer_waiting)
            await mgr.queue.enqueue(peer_staged)
            assert mgr._fastlane_claims, "gate: the peer must hold a live claim"
            served = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            assert served is peer_staged, (
                "an EQUAL-ranked waiter blocked the pick -- the guard is "
                "treating a tie as a violation, which the rule does not say"
            )
        finally:
            await _cancel_bg(mgr)

    async def test_unlisted_traffic_is_never_held(self, mgr):
        """⛔ THE STARVATION BOUNDARY, and it is one comparator argument wide.

        ``_fastlane_strictly_higher(candidate, holder)`` returns True whenever
        ``holder_key is None`` -- "listed beats unlisted absolutely". That rule
        is correct for PRIORITY, but applied to the hold-aside it would mean a
        single listed claimant makes EVERY ordinary request invisible for the
        whole backoff, and claims live up to ``_FASTLANE_CLAIM_TTL_S`` (1800s)
        with no periodic sweeper. The rule orders ranked clients against each
        other; an unlisted client is not "lower-ranked", it is unranked.

        So the hold is guarded by ``s.fastlane is not None``. Delete that one
        conjunct and this test is the only thing in the suite that fails.
        """
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        plain = _unlisted_slot("plain", "model-c")
        try:
            await _defer(mgr, high)
            await mgr.queue.enqueue(plain)
            assert mgr._fastlane_claims, "gate: a listed claimant must be waiting"
            served = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            assert served is plain, (
                "ordinary unlisted traffic was held aside by a Fast Lane "
                "claim. This converts an ordering fix into a starvation bug: "
                "one listed claimant would silence the whole box for the "
                f"life of its claim. got {getattr(served, 'slot_id', served)!r}"
            )
        finally:
            await _cancel_bg(mgr)


# --------------------------------------------------------------------------
# ⛔ Alternative (C): deferred, not abandoned.
# --------------------------------------------------------------------------
class TestShapeCIsDeferredNotAbandoned:
    """Shape (C) is deferred, explicitly NOT
    on merit: the smaller change goes first; (C) is not judged wrong.

    ⭐ WHY A TEST THAT IS EXPECTED TO FAIL IS WORTH SHIPPING: a deferral that
    lives only in a chat message evaporates. ``strict=True`` means the day (C)
    lands this test XPASSes, the suite goes RED, and somebody must remove this
    marker deliberately -- so the deferral cannot expire by inattention.

    ⛔ This is NOT a retained test for a dead subject. The property below is
    REAL and still wanted; only its delivery is postponed. Precedent: the
    cap-1 xfail(strict=True) in this repo whose removal IS the designed
    proof-of-fix.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="shape (C) (promote the claimant on free) deferred "
               "until later; shape (B) withholds service "
               "instead, so the claimant is served on the NEXT pick (see "
               "TestTheHoldIsBounded) rather than this one",
    )
    async def test_the_claimant_is_served_in_the_SAME_tick(self, mgr):
        """(C)'s property: the rightful claimant is served IMMEDIATELY, not
        merely protected from being overtaken. (B) cannot satisfy this -- the
        claimant is mid-backoff and in NEITHER buffer, so the queue has no
        slot object to return and correctly returns None."""
        high = _slot("high", HIGH_ADDR, 0, "model-a")
        low = _slot("low", LOW_ADDR, 1, "model-b")
        try:
            await _defer(mgr, high)
            await mgr.queue.enqueue(low)
            served = await mgr.queue.pop_next(fastlane_policy=_policy(mgr))
            assert served is high, (
                f"expected the rule_index 0 claimant, got "
                f"{getattr(served, 'slot_id', served)!r}"
            )
        finally:
            await _cancel_bg(mgr)


# --------------------------------------------------------------------------
# ⛔ the lock-order proof, made EXECUTABLE.
# --------------------------------------------------------------------------
def _class_methods(path: pathlib.Path, classname: str) -> dict:
    tree = ast.parse(path.read_text(), filename=str(path))
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == classname:
            return {
                m.name: m for m in n.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"class {classname} not found in {path}")


def _self_call_edges(node: ast.AST) -> list:
    """``self.X(...)`` -> ("mgr", X); ``self.queue.X(...)`` -> ("que", X).
    CALLS only -- a bare attribute reference is not an invocation, and
    counting one as an edge is what made an earlier version of this analysis
    report three false positives."""
    out = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        v = n.func.value
        if isinstance(v, ast.Name) and v.id == "self":
            out.append(("mgr", n.func.attr))
        elif (isinstance(v, ast.Attribute) and v.attr == "queue"
              and isinstance(v.value, ast.Name) and v.value.id == "self"):
            out.append(("que", n.func.attr))
    return out


def _closure_from_pop_kwarg() -> list:
    mgr_methods = _class_methods(pathlib.Path(manager_mod.__file__), "TurbohaulManager")
    que_methods = _class_methods(pathlib.Path(queue_mod.__file__), "TurbohaulQueue")
    tables = {"mgr": mgr_methods, "que": que_methods}
    start = ("mgr", "_fastlane_pop_kwarg")
    seen, stack, resolved = {start}, [start], []
    while stack:
        owner, name = stack.pop()
        node = tables[owner].get(name)
        if node is None:
            continue  # not one of ours (stdlib, dataclass, local) -- not followed
        resolved.append((owner, name, node))
        for edge in _self_call_edges(node):
            if edge not in seen:
                seen.add(edge)
                stack.append(edge)
    return resolved


class TestTheLockFreeReadIsStructurallyPinned:
    """⛔ WHY THERE IS NO LOCK HERE, AND WHY THAT IS THE SAFE CHOICE.

    ``_governing_claim_priority_key_locked`` documents "CALLER HOLDS
    ``_registry_lock``". ``_fastlane_pop_kwarg`` calls it WITHOUT that lock,
    deliberately, because the precondition cannot be honoured at any
    placement that is also deadlock-safe. Both alternatives are closed:

      * inside ``_pick_fastlane_locked`` (which holds the correct OUTER lock,
        ``queue._lock``) -- impossible: that method is SYNC and
        ``_registry_lock`` is an ``asyncio.Lock``;
      * here, wrapped in ``async with self._registry_lock`` -- that takes
        ``_registry_lock`` and then ``queue._lock`` inside ``pop_next``,
        INVERTING the established ``queue._lock -> _registry_lock`` order.

    Taking NO lock is therefore not the weakest option, it is the only safe
    one: a path that acquires nothing cannot invert an order. What makes the
    lock-free read correct is that the whole path is SYNC and AWAIT-FREE, so
    no other coroutine can interleave DURING it -- the same justification
    ``_model_residents`` states for its own ``list()`` snapshot ("Await-free
    ``list()`` snapshot so a concurrent mutation can't error").

    ⛔ AND THAT ARGUMENT HOLDS ONLY AGAINST COROUTINES. It buys nothing
    against a THREAD, so ``test_no_thread_mutates_the_claim_registry`` below
    carries the other half. Both must hold; neither alone is sufficient.
    """

    def test_the_path_is_sync_await_free_and_lock_free(self):
        """If anyone adds an ``await`` or a lock acquisition anywhere on this
        path, this goes RED here instead of deadlocking."""
        resolved = _closure_from_pop_kwarg()
        names = {f"{o}.{n}" for o, n, _ in resolved}
        assert "mgr._governing_claim_priority_key_locked" in names, (
            "the claim read is NOT on _fastlane_pop_kwarg's call path -- the "
            f"fix is not wired in. reached: {sorted(names)}"
        )
        offenders = []
        for owner, name, node in resolved:
            if isinstance(node, ast.AsyncFunctionDef):
                offenders.append(f"{owner}.{name}: is `async def`")
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Await, ast.AsyncWith, ast.AsyncFor)):
                    offenders.append(f"{owner}.{name}: contains {type(sub).__name__}")
                if isinstance(sub, (ast.With, ast.AsyncWith)):
                    for item in sub.items:
                        expr = ast.unparse(item.context_expr)
                        if "lock" in expr.lower():
                            offenders.append(f"{owner}.{name}: acquires {expr}")
        assert not offenders, (
            "the lock-free claim read is no longer safe -- its synchrony "
            "argument requires the WHOLE path to be sync, await-free and "
            "lock-free:\n  " + "\n  ".join(sorted(set(offenders)))
        )

    def test_fastlane_pop_kwarg_is_not_a_coroutine(self):
        """A redundant, cheap, DIFFERENT instrument for the same property --
        runtime introspection rather than AST. Two patterns that can
        disagree are worth more than one that cannot."""
        assert not inspect.iscoroutinefunction(
            TurbohaulManager._fastlane_pop_kwarg
        ), "_fastlane_pop_kwarg became a coroutine; the synchrony argument is void"

    def test_no_thread_mutates_the_claim_registry(self):
        """⛔ THE OTHER HALF OF THE SYNCHRONY ARGUMENT.
        Await-freeness buys atomicity against COROUTINES only. If any code
        mutated ``_fastlane_claims`` from a thread, this lock-free read could
        observe a torn dict or raise "dictionary changed size during
        iteration" -- rarely, and only under load.

        Audited at the time of writing: the registry is mutated in exactly three methods
        (``_register_fastlane_claim_locked``, ``_release_fastlane_claim_locked``,
        ``fastlane_claims_snapshot``), every one reached only from event-loop
        code. ``asyncio.to_thread`` is not the tree's only thread user. The
        audit found two more, and both call only socket lookups: the Fast Lane
        client namer's own two-thread executor (its two callables,
        ``socket.getnameinfo`` and ``socket.getaddrinfo``, are pinned below;
        its name callback runs on the event loop and touches only the
        census), and the forward resolver's ``loop.getaddrinfo`` on the loop's
        default executor. None of these reaches a mutator.

        ⭐ The subtle one this asserts, which a to_thread grep does NOT catch:
        FastAPI runs a route handler declared ``def`` in a worker THREAD. A
        single sync ``def`` route calling ``status_snapshot()`` would put the
        registry's own pruner on a thread. Every route is ``async def`` today;
        this fails the day one is not.
        """
        src_root = pathlib.Path(manager_mod.__file__).parent
        MUTATORS = {
            "_register_fastlane_claim_locked",
            "_release_fastlane_claim_locked",
            "fastlane_claims_snapshot",
        }

        # (a) the mutator set itself has not grown unnoticed.
        found = set()
        for path in sorted(src_root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for n in ast.walk(fn):
                    hit = (
                        (isinstance(n, ast.Delete) and any(
                            isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Attribute)
                            and t.value.attr == "_fastlane_claims"
                            for t in n.targets))
                        or (isinstance(n, ast.Assign) and any(
                            (isinstance(t, ast.Subscript)
                             and isinstance(t.value, ast.Attribute)
                             and t.value.attr == "_fastlane_claims")
                            or (isinstance(t, ast.Attribute)
                                and t.attr == "_fastlane_claims")
                            for t in n.targets))
                        or (isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and n.func.attr in ("pop", "clear", "setdefault", "update")
                            and isinstance(n.func.value, ast.Attribute)
                            and n.func.value.attr == "_fastlane_claims")
                    )
                    if hit:
                        found.add(fn.name)
        assert found <= MUTATORS | {"__init__"}, (
            "a NEW mutator of _fastlane_claims appeared and has not been "
            f"audited for thread-safety: {sorted(found - MUTATORS - {'__init__'})}"
        )

        # (b) no thread-offload target reaches a mutator. Outside the one
        #     file below, the text ban on every thread word stands.
        #     The Fast Lane client namer is the one place allowed an executor,
        #     because it submits only two C-level socket lookups that cannot
        #     reach the manager or the registry. The exception is kept narrow
        #     on purpose: only that file, only those callables, only that
        #     executor, built in only that constructor. Anything else in
        #     the file is a new thread user and needs a fresh audit.
        _NAMER_FILE = "fastlane_resolve.py"
        _NAMER_EXECUTOR_CALLABLES = ("socket.getnameinfo", "socket.getaddrinfo")
        for path in sorted(src_root.rglob("*.py")):
            text = path.read_text()
            rel = path.relative_to(src_root).as_posix()
            if rel == _NAMER_FILE:
                banned_words = ("threading.Thread", "run_in_threadpool")
            else:
                banned_words = ("run_in_executor", "ThreadPoolExecutor",
                                "threading.Thread", "run_in_threadpool")
            for banned in banned_words:
                assert banned not in text, (
                    f"{path.name} introduces {banned}; the claim registry's "
                    "lock-free read was audited against asyncio.to_thread "
                    "only -- re-run the thread audit"
                )
            if rel != _NAMER_FILE:
                continue

            tree = ast.parse(text, filename=str(path))
            # Calls inside FastLaneClientNamer.__init__ are the only place an
            # executor may be built; every other Call is judged against that.
            in_namer_init = set()
            enclosing = {}

            def _walk(node, scope, in_init):
                for child in ast.iter_child_nodes(node):
                    sub_scope, sub_init = scope, in_init
                    if isinstance(child, ast.ClassDef):
                        sub_scope = scope + [f"class {child.name}"]
                        sub_init = False
                    elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_scope = scope + [f"def {child.name}"]
                        sub_init = (
                            isinstance(node, ast.ClassDef)
                            and node.name == "FastLaneClientNamer"
                            and child.name == "__init__"
                        )
                    if isinstance(child, ast.Call):
                        enclosing[id(child)] = " / ".join(scope) or "module level"
                        if in_init:
                            in_namer_init.add(id(child))
                    _walk(child, sub_scope, sub_init)

            _walk(tree, [], False)

            thread_findings = []
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                fn = n.func
                fn_name = (fn.attr if isinstance(fn, ast.Attribute)
                           else fn.id if isinstance(fn, ast.Name) else None)
                if fn_name == "run_in_executor":
                    args = [ast.unparse(a) for a in n.args]
                    executor = args[0] if len(args) >= 1 else "<missing>"
                    target = args[1] if len(args) >= 2 else "<missing>"
                    if (len(args) < 2 or executor != "self._executor"
                            or target not in _NAMER_EXECUTOR_CALLABLES):
                        thread_findings.append(
                            f"{path.name}:{n.lineno} run_in_executor with "
                            f"callable {target} on executor {executor} (in "
                            f"{enclosing.get(id(n))}); only "
                            f"{list(_NAMER_EXECUTOR_CALLABLES)} on "
                            "self._executor are audited"
                        )
                elif fn_name == "ThreadPoolExecutor":
                    if id(n) not in in_namer_init:
                        thread_findings.append(
                            f"{path.name}:{n.lineno} ThreadPoolExecutor built "
                            f"in {enclosing.get(id(n))}; only "
                            "FastLaneClientNamer.__init__ is audited"
                        )
            assert not thread_findings, (
                "the Fast Lane client namer's executor is no longer the "
                "audited one; the claim registry's lock-free read was audited "
                "against it only -- re-run the thread audit:\n  "
                + "\n  ".join(thread_findings)
            )

        # (c) ⭐ every route handler must stay `async def`.
        sync_routes = []
        for path in sorted(src_root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for n in ast.walk(tree):
                if not isinstance(n, ast.FunctionDef):   # sync defs only
                    continue
                for d in n.decorator_list:
                    dump = ast.dump(d)
                    if any(f"'{verb}'" in dump for verb in (
                            "get", "post", "put", "delete", "patch",
                            "websocket", "api_route", "head", "options")):
                        sync_routes.append(f"{path.name}:{n.lineno} {n.name}")
        assert not sync_routes, (
            "a SYNC `def` route handler appeared -- FastAPI runs these in a "
            "worker THREAD, so any of them reaching status_snapshot() would "
            "mutate _fastlane_claims off the event loop and break the "
            "lock-free read's synchrony argument:\n  " + "\n  ".join(sync_routes)
        )
