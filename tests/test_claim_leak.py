"""A Fast Lane claim
registered at staging arrival is never released when its request is later
admitted via the grace-window "matched thread" shortcut.

This test reproduces the leak; it does not fix it.

THE LEAK, BY SYMBOL: `manager.py`'s three "admitted" release call sites --
`_route_to`'s closure, the legacy single-instance HIT block,
and `_reserve_and_start_locked`'s fresh-spawn success return -- are
the ONLY places `_release_fastlane_claim_locked(slot, "admitted")` is ever
called. `queue.pop_matched_thread` (and its hash-chain-fallback twin,
`drain_inbox_and_staging_match_hash_chain`) is, by its own docstring, "a
SECOND, SEPARATE removal path from pop_next -- a match here is served
WITHOUT pop_next, _pick_fastlane_locked, or any of the main-lane/
compression/FIFO ladder ever running" (queue.py). The ONLY LIVE
grace loop that consumes its result is `_serve_on_resident` (in manager.py)
-- `worker_loop` (manager.py) unconditionally delegates to
`_dispatch_loop`, since `max_parallel_sidecars` has `ge=1` (config.py),
so `_process_slot` / "Loop B" and its own copy of this same shape are
PROVABLY DEAD CODE (retained in the tree)
-- not a second live leak site. `_serve_on_resident` promotes the matched
slot straight through ACTIVE_MATCH -> ACTIVE -> completion -> GRACE ->
POPPED without ever routing through `_route_or_reserve`, so none of the
three release call sites is ever reached for it. A claim registered for
that slot at staging arrival (`_register_staged_claim`, manager.py)
is left live in `_fastlane_claims` until the 1800s TTL sweep -- exactly
the shape a TTL-only release produces (a served slot whose claim was
released only by ttl_expired long after the slot was served).

The comment in `_register_staged_claim` is STALE and hid this leak:
"the existing 'admitted' release at every HIT path ... cleans it
up when the slot is served" is false for this path -- the grace-window
match is not a HIT path in that sense at all.

Assertions are on the claim REGISTRY
(`mgr._fastlane_claims` / `mgr._fastlane_claim_key`), never on the
ttl_expired log line (it lags the true leak by up to the full TTL).
"""
import asyncio

import pytest

from turbohaul.fastlane import FastLaneMatch, match_fastlane
from turbohaul.manager import TurbohaulManager
from turbohaul.queue import FastLanePopPolicy
from turbohaul.slot import Slot, SlotState

from _fastlane_fixture import (
    boot_ranked_runtime,
    ranked_rules,
    seed_manifest,
    make_fakes,
    high_vram,
    wait_until,
    resident_for,
)


def _boot_and_manager(tmp_path, *, rules, grace_seconds=5, max_parallel_sidecars=1,
                       gates=None, start_worker=True):
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=rules, grace_seconds=grace_seconds,
        max_parallel_sidecars=max_parallel_sidecars,
    )
    seed_manifest(boot, "m1", main_gpu=0)
    spawn_fn, health_fn, sigterm_fn, vram_fn, complete_fn = make_fakes(gates or {})
    mgr = TurbohaulManager(
        boot, runtime,
        spawn_fn=spawn_fn, health_fn=health_fn, sigterm_fn=sigterm_fn,
        vram_fn=vram_fn, complete_fn=complete_fn,
    )
    if start_worker:
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    return boot, mgr


async def _noop_pop_next(*a, **k):
    return None


@pytest.mark.asyncio
class TestClaimLeakOnMatchedThreadAdmission:
    async def test_matched_thread_admission_leaks_the_staged_claim(self, tmp_path):
        """THE bug.

        VERIFIED FIRST, not assumed: with the top-level dispatcher
        (`_dispatch_loop`, reached from every `worker_loop` call) left
        running and free to poll, its own `pop_next` -> `_route_or_reserve`
        legacy single-instance HIT branch (manager.py) reliably wins
        the race against the resident's own grace loop for a freshly
        enqueued same-thread follow-up and releases the claim correctly
        (see the CONTROL below, `test_same_thread_followup_with_dispatcher_
        alive_releases_the_claim`, same construction, dispatcher left
        alive). That HIT branch is not the bug. The bug is in the OTHER
        consumer of the same staged claim -- `pop_matched_thread`, served
        directly by the resident's own grace loop -- which can win
        this race in practice and leak a large share of
        claims. To exercise that specific consumer
        deterministically rather than depend on an unreliable race, this
        test neutralizes ONLY `mgr.queue.pop_next` (the one entry point
        `_dispatch_loop` can ever admit anything through) with a no-op
        stub -- the resident's driver task, `_serve_on_resident`'s grace
        loop, and `pop_matched_thread` itself are all left completely real
        and untouched. The follow-up slot is placed directly in
        `queue._staging` (bypassing only the `_accept_buf` queueing detail
        `submit()` would otherwise use, which nothing would ever drain once
        `pop_next` is neutralized) and its claim registered via the REAL
        `_register_fastlane_claim_locked` call with the REAL
        "staging_arrival" reason string, matching `_register_staged_claim`'s
        own call exactly. Every step downstream of that point -- the grace
        loop noticing it, `pop_matched_thread` popping it, ACTIVE_MATCH
        promotion, completion, GRACE, POPPED -- is the real, unmodified
        manager code under test.

        The anchor shares the SAME fastlane identity as the follow-up
        (both `10.0.0.1`/`is_main`) rather than being unlisted: an
        unlisted resident is "unresolvable", which the designation rule's own victim
        selection sorts WORST -- the instant the follow-up's claim
        registers, an unlisted anchor would be trivially designated the
        eviction victim and its grace loop would `break` immediately
        (correct behaviour, but a different code path than the one
        under test here, and it would exit before ever calling
        `pop_matched_thread`). Tying the anchor's own identity to the
        follow-up's rule keeps the anchor un-designated (a tied key is not
        "strictly higher", so it is never outranked) and lets the grace
        loop reach `pop_matched_thread` as intended.

        MUST FAIL ON UNMODIFIED CODE: no release call in the only live
        grace loop's matched branch means the claim key is never removed
        from `mgr._fastlane_claims` for this admission."""
        gate = asyncio.Event()
        rules = ranked_rules(("10.0.0.1", 1))
        boot, mgr = _boot_and_manager(tmp_path, rules=rules, gates={"m1": gate})
        try:
            with high_vram():
                anchor = await mgr.submit(
                    model_tag="m1", prompt="hi", thread_id="tA",
                    client_meta={"ip": "10.0.0.1", "is_main": True},
                )
                await wait_until(
                    lambda: any(rr.model_tag == "m1" for rr in mgr._model_residents()),
                    timeout=5.0,
                )
                r = resident_for(mgr, "m1")
                gate.set()  # anchor completes -> GRACE
                await wait_until(
                    lambda: getattr(r, "in_grace_loop", False), timeout=5.0,
                )

                # Neutralize ONLY the top-level dispatcher's single admission
                # entry point -- driver tasks and pop_matched_thread are real.
                mgr.queue.pop_next = _noop_pop_next

                match = match_fastlane(
                    mgr._fastlane_table(), "10.0.0.1",
                    {"ip": "10.0.0.1", "is_main": True},
                )
                assert match is not None and match.rank != 6, (
                    "fixture precondition: 10.0.0.1/is_main must resolve to a "
                    "RANKED fastlane match or this proves nothing about claims"
                )
                followup = Slot(
                    slot_id="followup-tA-2", model_tag="m1", state=SlotState.STAGED,
                    thread_id="tA", client_meta={"ip": "10.0.0.1", "is_main": True},
                    fastlane=match,
                )
                mgr.queue._staging.append(followup)
                async with mgr._registry_lock:
                    mgr._register_fastlane_claim_locked(followup, "staging_arrival")
                key = mgr._fastlane_claim_key(followup)
                assert key is not None and key in mgr._fastlane_claims, (
                    "fixture precondition: follow-up claim must be registered "
                    "before the grace loop can possibly leak it"
                )

                await wait_until(
                    lambda: followup.state is SlotState.POPPED, timeout=6.0,
                )

                assert key not in mgr._fastlane_claims, (
                    f"CLAIM LEAK: slot {followup.slot_id}'s Fast Lane claim "
                    f"(key={key}) is STILL registered in mgr._fastlane_claims "
                    f"after being fully served via the pop_matched_thread grace "
                    f"shortcut (state={followup.state}). No admitted-release "
                    f"call site is reached on this path -- the claim will now "
                    f"sit live for up to the full 1800s TTL, governing eviction "
                    f"decisions and pop_next hold-asides for a request that has "
                    f"already completed."
                )
        finally:
            # ALWAYS shut down, even on the RED assertion above -- an
            # un-shut-down manager leaves its driver tasks/background tasks
            # running and can pollute the NEXT test's fixture.
            await mgr.shutdown()

    async def test_queue_priority_admission_releases_the_claim(self, tmp_path):
        """CONTROL, must PASS both before and after any fix. A fastlane
        request admitted the ORDINARY way -- cold-spawn onto a fresh
        resident, never matched via pop_matched_thread -- has its claim
        released the instant it is admitted, via `_reserve_and_start_
        locked`'s "admitted" release call site. Proves the registry
        assertion technique is sound (it CAN observe a release) and that
        the leak above is specific to the matched-thread path, not a
        property of the registry helper or the fixture.

        Registration is observed via a TRACE on `_register_fastlane_claim_
        locked` rather than by reading `mgr._fastlane_claims` mid-flight --
        `_dispatch_loop` can register-then-release a cold-spawn's claim
        (reservation-time release, in manager.py) between two lines of
        this test's OWN coroutine (an asyncio scheduling race, not a bug),
        which made a mid-flight dict read of "was it ever registered"
        itself flaky. The trace makes that fact non-racy: it always sees
        the call, however fast the release that follows it is."""
        rules = ranked_rules(("10.0.0.1", 1))
        boot, mgr = _boot_and_manager(tmp_path, rules=rules, max_parallel_sidecars=1)
        registered_slot_ids = []
        orig_register = mgr._register_fastlane_claim_locked

        def _traced_register(slot, reason):
            registered_slot_ids.append(slot.slot_id)
            return orig_register(slot, reason)

        mgr._register_fastlane_claim_locked = _traced_register
        try:
            with high_vram():
                slot = await mgr.submit(
                    model_tag="m1", prompt="cold-start", thread_id="tB",
                    client_meta={"ip": "10.0.0.1", "is_main": True},
                )
                assert slot.fastlane is not None, (
                    "fixture precondition: slot must resolve to the fastlane rule"
                )
                key = mgr._fastlane_claim_key(slot)

                await wait_until(lambda: key not in mgr._fastlane_claims, timeout=5.0)
                assert slot.slot_id in registered_slot_ids, (
                    "fixture precondition: a fresh cold-start submission must "
                    "register a staging_arrival claim at some point -- if this "
                    "fails, the control never exercised claim registration at "
                    "all and proves nothing"
                )
                assert key not in mgr._fastlane_claims, (
                    "QUEUE_PRIORITY control failed to release its own claim -- "
                    "if THIS fails, something is wrong with the fixture/registry "
                    "helper itself, not with the matched-thread leak above"
                )
        finally:
            await mgr.shutdown()

    async def test_same_thread_followup_with_dispatcher_alive_releases_the_claim(
        self, tmp_path,
    ):
        """SINGLE-VARIABLE CONTROL. Identical construction to the leak test
        above -- anchor and follow-up share thread tA and the same fastlane
        identity, anchor driven to GRACE -- with exactly one thing
        different: `mgr.queue.pop_next` is left REAL instead of
        neutralized, and the follow-up is submitted the ordinary way (via
        `mgr.submit`, landing in `_accept_buf` like any real request) rather
        than placed directly in `_staging`. With the dispatcher able to
        admit it, its ordinary HIT path wins the race and releases the
        claim correctly. This isolates the one variable that matters --
        whether `pop_matched_thread`'s grace-loop consumer is the SOLE
        consumer or merely a racing one -- and proves the leak is a
        property of that specific consumer, not of same-thread admission
        in general."""
        gate = asyncio.Event()
        rules = ranked_rules(("10.0.0.1", 1))
        boot, mgr = _boot_and_manager(tmp_path, rules=rules, gates={"m1": gate})
        try:
            with high_vram():
                await mgr.submit(
                    model_tag="m1", prompt="hi", thread_id="tA",
                    client_meta={"ip": "10.0.0.1", "is_main": True},
                )
                await wait_until(
                    lambda: any(rr.model_tag == "m1" for rr in mgr._model_residents()),
                    timeout=5.0,
                )
                gate.set()

                followup = await mgr.submit(
                    model_tag="m1", prompt="fastlane-followup", thread_id="tA",
                    client_meta={"ip": "10.0.0.1", "is_main": True},
                )
                assert followup.fastlane is not None
                key = mgr._fastlane_claim_key(followup)

                await wait_until(
                    lambda: followup.state is SlotState.POPPED, timeout=15.0,
                )
                assert key not in mgr._fastlane_claims, (
                    "control failed: with the dispatcher alive, a same-thread "
                    "follow-up's claim should be released by the ordinary HIT "
                    "admission path that wins the race against pop_matched_"
                    "thread -- if this fails, the leak test's isolation premise "
                    "(neutralizing pop_next changes which consumer serves the "
                    "slot) is wrong, not merely unlucky timing"
                )
        finally:
            await mgr.shutdown()


@pytest.mark.asyncio
class TestLeakedClaimHoldsAsideAnotherThread:
    """The leaked claim does not sit inert --
    while live, it is the GOVERNING claim (`_governing_claim_locked` has no
    special-case for "already served"; `_claim_is_live` does not treat a
    successfully-completed slot as dead -- that is explicitly release-on-
    admission's job, per its own docstring, and release-on-admission is
    exactly what never runs on this path). `queue.pop_next`'s hold-aside
    (queue.py) makes ANY staged slot whose Fast Lane key is
    STRICTLY worse than the governing key invisible to the entire pick
    ladder for that call -- keyed purely on `(rule_index, rank)`, not on
    thread identity.

    "own-thread claims do not block" is explained by
    `_fastlane_strictly_higher`'s own documented tie rule: an EQUAL key is
    deliberately NOT "strictly higher" ("an agent's own follow-up turns
    keep the warm slot; two equal-ranked agents don't bump each other",
    queue.py) -- so a second request from the SAME fastlane
    identity as the leaked/governing claim is never held aside by it. A
    request from a DIFFERENT, lower-ranked identity is held aside exactly
    as if the governing claimant were still genuinely waiting, even though
    its request already completed.

    This is pure queue.py state -- no manager admission, no dispatcher, no
    async grace loop needed to exercise `pop_next`'s hold-aside directly.
    Slots are built directly (matching the Fast Lane claim registry test's
    own `_slot()`/`_match()` convention) and placed straight
    in `_staging`, bypassing `submit()`/`_accept_buf` entirely so the
    hold-aside is observed against a known, controlled snapshot rather
    than whatever `pop_next`'s own accept-buffer drain-and-retry happens
    to do."""

    async def test_leaked_claim_holds_aside_a_lower_ranked_other_thread(self, tmp_path):
        """A leaked claim from thread A (rule_index=0, the best rule) is
        still the governing key. A DIFFERENT thread B's staged request, on
        a WORSE rule (rule_index=1), is held aside by pop_next -- invisible
        to the pick ladder -- purely because of the ghost claim, even
        though thread A's real request already finished."""
        rules = ranked_rules(("10.0.0.1", 1), ("10.0.0.2", 2))
        boot, mgr = _boot_and_manager(tmp_path, rules=rules, start_worker=False)

        governing_key = (0, 1)  # best rule, matches the leaked claim
        other_key = (1, 2)      # worse rule
        assert mgr.queue._fastlane_strictly_higher(governing_key, other_key), (
            "fixture precondition: governing key must outrank the other "
            "thread's key or the hold-aside has nothing to prove"
        )

        other_thread_slot = Slot(
            slot_id="other-thread-slot", model_tag="m1", state=SlotState.STAGED,
            thread_id="tB-other", client_meta={"ip": "10.0.0.2", "is_main": True},
            fastlane=FastLaneMatch(
                rule_index=1, raw_address="10.0.0.2", label="",
                effective_tag="main", rank=2,
            ),
        )
        mgr.queue._staging.append(other_thread_slot)

        picked = await mgr.queue.pop_next(
            fastlane_policy=FastLanePopPolicy(
                max_normal_wait_s=3600.0, cross_model_switches_per_min=999,
                governing_claim_key=governing_key,
            ),
        )
        assert picked is None or picked.slot_id != other_thread_slot.slot_id, (
            "the other thread's lower-ranked staged request was popped by "
            "pop_next despite a strictly-higher governing claim being live "
            "-- the hold-aside did not fire; either the claim genuinely "
            "isn't governing any more (leak already fixed) or this test's "
            "construction is wrong"
        )
        # non-vacuity: the entry must still be sitting in staging, held
        # aside, not simply lost
        assert any(
            s.slot_id == other_thread_slot.slot_id for s in mgr.queue._staging
        ), "other thread's slot vanished from staging entirely -- fixture bug"

        await mgr.shutdown()

    async def test_same_rank_thread_is_not_held_aside_by_its_own_leaked_claim(
        self, tmp_path,
    ):
        """CONTROL: a second request under the SAME fastlane rule as the
        (leaked) governing claim is a tie, not "strictly higher" on either
        side -- `_fastlane_strictly_higher` requires a strict `<`. It must
        NOT be held aside. This is the "own-thread claims do not block"
        case."""
        rules = ranked_rules(("10.0.0.1", 1))
        boot, mgr = _boot_and_manager(tmp_path, rules=rules, start_worker=False)

        governing_key = (0, 1)
        same_rule_slot = Slot(
            slot_id="same-rule-slot", model_tag="m1", state=SlotState.STAGED,
            thread_id="tA-2", client_meta={"ip": "10.0.0.1", "is_main": True},
            fastlane=FastLaneMatch(
                rule_index=0, raw_address="10.0.0.1", label="",
                effective_tag="main", rank=1,
            ),
        )
        same_key = mgr.queue._fastlane_priority_key(same_rule_slot)
        assert same_key == governing_key, (
            "fixture precondition: second request must tie the governing "
            "key exactly (same rule) for this control to mean anything"
        )
        assert not mgr.queue._fastlane_strictly_higher(governing_key, same_key), (
            "fixture precondition: a tie must not be 'strictly higher' -- "
            "if this fails, _fastlane_strictly_higher's own contract "
            "changed and this control no longer tests what it claims to"
        )
        mgr.queue._staging.append(same_rule_slot)

        picked = await mgr.queue.pop_next(
            fastlane_policy=FastLanePopPolicy(
                max_normal_wait_s=3600.0, cross_model_switches_per_min=999,
                governing_claim_key=governing_key,
            ),
        )
        assert picked is not None and picked.slot_id == same_rule_slot.slot_id, (
            "a same-rank (tied) request was wrongly held aside by its own "
            "rule's governing claim -- 'own-thread claims do not block' "
            "does not hold on this tree"
        )

        await mgr.shutdown()


@pytest.mark.asyncio
class TestClaimReleasedWhileFollowupStillServing:
    """KILLER: the claim must be released WHILE the matched follow-up is
    still being served (mid-complete), not only after it finishes -- and
    not just before the end-of-branch POPPED transition either.

    A fix that releases the claim only at completion -- after the follow-up
    is already done -- still leaves the claim live during the entire serving
    window, where it can still govern eviction decisions and pop_next
    hold-asides for a request that is actively being served. The release
    must fire at the moment the match is admitted, before completion.

    Kills TWO mutants:
    - (a): no release at the match site (claim leaks entirely)
    - (b): release moved to just before the end-of-branch POPPED transition
      (claim lives for the whole serving window, dies only at the very end)

    The assertion fires at ACTIVE_MATCH -- the moment the follow-up is
    promoted into service -- NOT at POPPED. Mutant (b) survives a test that waits
    for POPPED; it cannot survive one that asserts at ACTIVE_MATCH.
    """

    async def test_claim_released_while_followup_still_serving(self, tmp_path):
        """Hold the follow-up mid-serving (via the fixture's gate mechanism)
        and assert the claim is already gone at ACTIVE_MATCH, before
        completion fires."""
        rules = ranked_rules(("10.0.0.1", 1))
        followup_gate = asyncio.Event()

        # Custom complete: anchor (slot_a) completes instantly; followup
        # blocks on followup_gate until we release it. The fixture's
        # make_fakes gates by model_tag, but both slots are m1 -- so a
        # custom complete is needed to hold only the followup.
        def fake_spawn(binary, gguf, port, model_tag, argv, **_kw):
            from unittest.mock import MagicMock
            from turbohaul.subprocess_mgr import SidecarHandle
            proc = MagicMock()
            proc.pid = 90000
            proc.poll.return_value = None
            return SidecarHandle(proc=proc, port=port, model_tag=model_tag)

        async def fake_health(*a, **k):
            return True

        async def fake_sigterm(*a, **k):
            return True, "sigterm-clean"

        async def fake_vram(**k):
            return True, 100

        async def fake_complete(slot, handle):
            if slot.slot_id == "followup-tA-5":
                await followup_gate.wait()
            return {"ok": True, "model": handle.model_tag}

        boot, runtime = boot_ranked_runtime(
            tmp_path, rules=rules, grace_seconds=5, max_parallel_sidecars=1,
        )
        seed_manifest(boot, "m1", main_gpu=0)
        mgr = TurbohaulManager(
            boot, runtime,
            spawn_fn=fake_spawn, health_fn=fake_health,
            sigterm_fn=fake_sigterm, vram_fn=fake_vram,
            complete_fn=fake_complete,
        )
        mgr._worker_task = asyncio.create_task(mgr.worker_loop())
        try:
            with high_vram():
                anchor = await mgr.submit(
                    model_tag="m1", prompt="hi", thread_id="tA",
                    client_meta={"ip": "10.0.0.1", "is_main": True},
                )
                await wait_until(
                    lambda: any(rr.model_tag == "m1" for rr in mgr._model_residents()),
                    timeout=5.0,
                )
                r = resident_for(mgr, "m1")
                # Anchor completes -> GRACE. Follow-up will block on
                # followup_gate, holding it mid-serving at ACTIVE_MATCH.
                await wait_until(
                    lambda: getattr(r, "in_grace_loop", False), timeout=5.0,
                )

                # Neutralize ONLY the top-level dispatcher
                mgr.queue.pop_next = _noop_pop_next

                match = match_fastlane(
                    mgr._fastlane_table(), "10.0.0.1",
                    {"ip": "10.0.0.1", "is_main": True},
                )
                assert match is not None and match.rank != 6
                followup = Slot(
                    slot_id="followup-tA-5", model_tag="m1",
                    state=SlotState.STAGED, thread_id="tA",
                    client_meta={"ip": "10.0.0.1", "is_main": True},
                    fastlane=match,
                )
                mgr.queue._staging.append(followup)
                async with mgr._registry_lock:
                    mgr._register_fastlane_claim_locked(followup, "staging_arrival")
                key = mgr._fastlane_claim_key(followup)
                assert key is not None and key in mgr._fastlane_claims

                # Wait for the follow-up to reach ACTIVE -- the state when
                # fake_complete is called and blocks on followup_gate.
                # At this point the followup is held mid-serving.
                await wait_until(
                    lambda: followup.state is SlotState.ACTIVE,
                    timeout=6.0,
                )

                # KILLER: while the follow-up is held at ACTIVE (mid-serving,
                # gate NOT set, completion blocked), the claim must already be
                # gone. A release at completion or at the end-of-branch POPPED
                # transition is TOO LATE -- the claim governed eviction and
                # hold-asides for the entire serving window.
                assert key not in mgr._fastlane_claims, (
                    f"EARLY RELEASE: claim (key={key}) for follow-up "
                    f"{followup.slot_id} is STILL registered while the "
                    f"follow-up is held at ACTIVE (state={followup.state}, "
                    f"completion blocked on followup_gate). The release must "
                    f"fire at admission time, not at completion or at the "
                    f"end-of-branch POPPED transition."
                )

                # Release the gate to let the follow-up finish cleanly
                followup_gate.set()
                await wait_until(
                    lambda: followup.state is SlotState.POPPED, timeout=6.0,
                )
        finally:
            await mgr.shutdown()
