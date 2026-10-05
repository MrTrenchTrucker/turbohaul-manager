"""The one-turn wall-clock fairness floor grants ONE turn per
CLIENT and then re-arms that client's timer.

THE DEFECT. queue.py's floor was a pure age test on the winning REQUEST,
recomputed from scratch on every pop_next and holding no state at all. Nothing
recorded that a promotion had happened, so nothing could re-arm. The fairness
rule's subject is a CLIENT (the longest-waiting queued client is admitted for
exactly one full turn, then its timer re-arms); the code's subject was a
request. One client with N aged unregistered requests therefore took N
consecutive turns ahead of a rank-0 registered claimant.

THE TWO DESIGN DECISIONS THESE TESTS ENCODE:
  * PER CLIENT, not global. The rule bounds the
    worst-case time a queued CLIENT goes without work; a global
    one-promotion-at-a-time timer makes that bound k x max_normal_wait_s for k
    aged clients, so it does not implement the rule, it contradicts it. Global also
    over-corrects into starving a second legitimately-aged client behind the
    first -- the opposite of the original defect.
    ``TestADifferentClientIsNotCollateral`` is the control for that, and
    ``test_control_has_teeth_a_global_rearm_fails_it`` proves the control can
    actually fail rather than merely passing.
  * WIDE, not narrow. The rule's condition is that a client goes
    "without being SERVED a turn",
    by ANY path, so every serve resets the clock, not only a floor promotion.
    ``TestWideRearmEveryServeResetsTheClock`` is that half.

WHAT IS *NOT* CHANGED, and is asserted here so a later reader cannot mistake
the scope: WHO is eligible. The floor still promotes unregistered
(``slot.fastlane is None``) traffic only, as before.

NON-VACUITY. Every RED test in this file fails against the pre-change
queue.py, and for the intended reason rather than an incidental one,
so a pass against the changed queue.py is a real signal.
(RED = fails without the change.)
"""
import time

import pytest

from turbohaul.fastlane import FastLaneMatch
from turbohaul.queue import FastLanePopPolicy, TurbohaulQueue
from turbohaul.slot import Slot

WINDOW_S = 5.0
CLIENT_A = "10.77.0.7"
CLIENT_B = "10.77.0.8"


def _policy(max_normal_wait_s=WINDOW_S):
    """A live policy. cross_model_switches_per_min is high so the ordinary
    swap-budget gate can never be what decides any pick below."""
    return FastLanePopPolicy(
        max_normal_wait_s=max_normal_wait_s, cross_model_switches_per_min=9999,
    )


def _match(rule_index=0, rank=0):
    return FastLaneMatch(
        rule_index=rule_index, raw_address="1.2.3.4", label="",
        effective_tag="main", rank=rank,
    )


def _normal(ip, *, aged_by=0.0, model_tag="m1", prompt="n"):
    """An UNREGISTERED slot (fastlane is None -- the only population the floor
    may promote) belonging to client ``ip``.

    ``aged_by`` backdates ``created_at`` directly instead of monkeypatching the
    clock. That matters: these tests need a request that is ALREADY old while
    the wall clock has barely moved, which is exactly the shape a clock jump
    cannot produce -- a jump ages the request and the re-arm window together.
    """
    s = Slot.new(model_tag=model_tag, prompt=prompt, client_meta={"ip": ip})
    assert s.fastlane is None, "non-vacuity: the floor only ever promotes these"
    if aged_by:
        s.created_at = time.monotonic() - aged_by
    return s


def _registered(ip, *, rule_index=0, rank=0, model_tag="m1", prompt="r"):
    """A rank-0 Fast Lane claimant -- the party the defect let an aged client
    displace over and over."""
    s = Slot.new(model_tag=model_tag, prompt=prompt, client_meta={"ip": ip})
    s.fastlane = _match(rule_index=rule_index, rank=rank)
    return s


@pytest.mark.asyncio
class TestOneClientGetsOneTurnThenRearms:
    """The defect itself."""

    async def test_second_aged_request_from_the_same_client_does_not_take_a_second_turn(self):
        q = TurbohaulQueue(staging_max=10)
        aged_1 = _normal(CLIENT_A, aged_by=600.0, prompt="a1")
        aged_2 = _normal(CLIENT_A, aged_by=590.0, prompt="a2")
        claimant = _registered(CLIENT_B, prompt="rank0")
        for s in (aged_1, aged_2, claimant):
            await q.enqueue(s)

        first = await q.pop_next(fastlane_policy=_policy())
        assert first.slot_id == aged_1.slot_id, "the floor's one turn"
        assert first.floor_promoted is True, (
            "non-vacuity: this must be the FLOOR serving it, not the priority "
            "rung -- an unregistered slot reaching the pick any other way "
            "would make the rest of this test meaningless"
        )

        second = await q.pop_next(fastlane_policy=_policy())

        assert second is not None
        assert second.slot_id == claimant.slot_id, (
            "the floor granted client A a SECOND consecutive turn: the rule "
            "grants 'exactly one full turn' per client and then the timer "
            "re-arms, so the rank-0 claimant is next -- not client A's other "
            f"aged request (got {second.prompt!r})"
        )
        assert second.floor_promoted is False

    async def test_the_client_gets_another_turn_once_its_window_has_actually_elapsed(self):
        """The re-arm is a DELAY, never a permanent lockout -- the other way
        this could be got wrong, and the reason the test above is not enough on
        its own."""
        q = TurbohaulQueue(staging_max=10)
        aged_1 = _normal(CLIENT_A, aged_by=600.0, prompt="a1")
        aged_2 = _normal(CLIENT_A, aged_by=590.0, prompt="a2")
        for s in (aged_1, aged_2):
            await q.enqueue(s)

        first = await q.pop_next(fastlane_policy=_policy())
        assert first.slot_id == aged_1.slot_id

        # Rewind the credit by more than the window: the same thing the passage
        # of WINDOW_S real seconds would do, without sleeping for it.
        for key in list(q._floor_last_served_at):
            q._floor_last_served_at[key] -= (WINDOW_S + 1.0)

        second = await q.pop_next(fastlane_policy=_policy())
        assert second is not None and second.slot_id == aged_2.slot_id
        assert second.floor_promoted is True, (
            "once its own window has elapsed the client is floor-eligible "
            "again -- the re-arm delays a turn, it does not cancel one"
        )


@pytest.mark.asyncio
class TestADifferentClientIsNotCollateral:
    """The control for the PER CLIENT vs GLOBAL decision. A second, genuinely
    aged client must still get its own floor turn immediately -- a global timer
    would make it wait behind the first client's window."""

    async def test_a_second_aged_client_still_gets_its_floor_turn_immediately(self):
        q = TurbohaulQueue(staging_max=10)
        aged_a = _normal(CLIENT_A, aged_by=600.0, prompt="a")
        aged_b = _normal(CLIENT_B, aged_by=590.0, prompt="b")
        claimant = _registered("10.77.0.9", prompt="rank0")
        for s in (aged_a, aged_b, claimant):
            await q.enqueue(s)

        first = await q.pop_next(fastlane_policy=_policy())
        assert first.slot_id == aged_a.slot_id

        second = await q.pop_next(fastlane_policy=_policy())

        assert second is not None
        assert second.slot_id == aged_b.slot_id, (
            "client B has waited past the window on its own account and must "
            "get its own floor turn now; making it wait out client A's window "
            "is the starvation this fix removes, pointed the other way "
            f"(got {second.prompt!r})"
        )
        assert second.floor_promoted is True

    async def test_control_has_teeth_a_global_rearm_fails_it(self, monkeypatch):
        """Proof the control above is not vacuous, by the only means that can
        prove it: collapse every client onto ONE identity -- which is exactly
        what a global 'one floor promotion at a time' re-arm is -- and watch
        the same scenario stop serving client B.

        This monkeypatches a helper inside the test; it does not mutate
        shipped code. The file's own
        ``test_fairness_inert_build_would_fail_phase_one`` sets the precedent
        for stating a contrast this way.
        """
        monkeypatch.setattr(
            TurbohaulQueue, "_floor_client_key", lambda self, slot: "GLOBAL",
        )
        q = TurbohaulQueue(staging_max=10)
        aged_a = _normal(CLIENT_A, aged_by=600.0, prompt="a")
        aged_b = _normal(CLIENT_B, aged_by=590.0, prompt="b")
        claimant = _registered("10.77.0.9", prompt="rank0")
        for s in (aged_a, aged_b, claimant):
            await q.enqueue(s)

        first = await q.pop_next(fastlane_policy=_policy())
        assert first.slot_id == aged_a.slot_id

        second = await q.pop_next(fastlane_policy=_policy())

        assert second is not None and second.slot_id == claimant.slot_id, (
            "under a GLOBAL re-arm client B is locked out behind client A's "
            "window -- if this ever returns client B, the per-client control "
            "above has stopped discriminating and is proving nothing"
        )


@pytest.mark.asyncio
class TestWideRearmEveryServeResetsTheClock:
    """The WIDE rule: the fairness clock is 'without being SERVED a turn', by
    ANY path -- not only 'without being floor-promoted'."""

    async def test_an_ordinary_priority_serve_rearms_the_floor_for_that_client(self):
        q = TurbohaulQueue(staging_max=10)
        # Client A is served first by the ORDINARY priority rung, with nothing
        # aged in staging, so the floor plays no part in that pop.
        a_registered = _registered(CLIENT_A, prompt="a-rank0")
        await q.enqueue(a_registered)
        served = await q.pop_next(fastlane_policy=_policy())
        assert served.slot_id == a_registered.slot_id
        assert served.floor_promoted is False, (
            "non-vacuity: this pop must NOT be a floor promotion, or the test "
            "proves the narrow reading rather than the wide one"
        )

        # Only now does client A's long-queued backlog request appear.
        a_backlog = _normal(CLIENT_A, aged_by=600.0, prompt="a-backlog")
        claimant = _registered(CLIENT_B, prompt="b-rank0")
        for s in (a_backlog, claimant):
            await q.enqueue(s)

        got = await q.pop_next(fastlane_policy=_policy())

        assert got is not None
        assert got.slot_id == claimant.slot_id, (
            "client A was SERVED a turn moments ago, so its floor clock has "
            "been reset and its aged backlog request must not displace a "
            f"rank-0 claimant (got {got.prompt!r})"
        )

    async def test_a_grace_rematch_serve_also_rearms_the_floor(self):
        """pop_matched_thread is a SECOND, SEPARATE removal path that never
        reaches pop_next -- and a same-thread continuation is the ORDINARY
        shape of one agent's sequential turns. Uncredited, a client whose whole
        workload is continuations would look permanently unserved."""
        q = TurbohaulQueue(staging_max=10)
        # Arm the ledger the way production does: one pop_next carrying a live
        # policy. (Feature-off writes nothing -- see TestFeatureOffIsInert.)
        warmup = _registered("10.77.0.99", prompt="warmup")
        await q.enqueue(warmup)
        assert (await q.pop_next(fastlane_policy=_policy())).slot_id == warmup.slot_id

        a_continuation = Slot.new(
            model_tag="m1", prompt="a-cont", thread_id="thread-a",
            client_meta={"ip": CLIENT_A},
        )
        await q.enqueue(a_continuation)
        matched = await q.pop_matched_thread("thread-a", "m1")
        assert matched is not None and matched.slot_id == a_continuation.slot_id

        a_backlog = _normal(CLIENT_A, aged_by=600.0, prompt="a-backlog")
        claimant = _registered(CLIENT_B, prompt="b-rank0")
        for s in (a_backlog, claimant):
            await q.enqueue(s)

        got = await q.pop_next(fastlane_policy=_policy())

        assert got is not None
        assert got.slot_id == claimant.slot_id, (
            "a grace-window rematch IS being served a turn; leaving it "
            "uncredited lets a client draw floor turns on top of work it is "
            f"already getting (got {got.prompt!r})"
        )


@pytest.mark.asyncio
class TestEveryPopNextRungCredits:
    """The credit lives at ONE choke point (_update_run_length_locked) rather
    than at six call sites. These pin each rung individually so a rung added
    later without its credit fails loudly instead of silently under-counting.
    """

    async def _served_keys(self, q):
        return set(q._floor_last_served_at)

    async def test_fastlane_priority_rung_credits(self):
        q = TurbohaulQueue(staging_max=10)
        await q.enqueue(_registered(CLIENT_A))
        got = await q.pop_next(fastlane_policy=_policy(max_normal_wait_s=9999.0))
        assert got is not None
        assert await self._served_keys(q) == {f"ip:{CLIENT_A}"}

    async def test_main_lane_rung_credits(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=True)
        s = Slot.new(
            model_tag="m1", prompt="main",
            client_meta={"ip": CLIENT_A, "is_main": True},
        )
        await q.enqueue(s)
        got = await q.pop_next(fastlane_policy=_policy(max_normal_wait_s=9999.0))
        assert got is not None and got.slot_id == s.slot_id
        assert await self._served_keys(q) == {f"ip:{CLIENT_A}"}

    async def test_compression_rung_credits(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
        s = Slot.new(
            model_tag="m1", prompt="comp",
            client_meta={"ip": CLIENT_A, "is_compression": True},
        )
        await q.enqueue(s)
        got = await q.pop_next(fastlane_policy=_policy(max_normal_wait_s=9999.0))
        assert got is not None and got.slot_id == s.slot_id
        assert await self._served_keys(q) == {f"ip:{CLIENT_A}"}

    async def test_fifo_rung_credits(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
        await q.enqueue(_normal(CLIENT_A))
        got = await q.pop_next(fastlane_policy=_policy(max_normal_wait_s=9999.0))
        assert got is not None
        assert await self._served_keys(q) == {f"ip:{CLIENT_A}"}

    async def test_affinity_rung_credits(self):
        q = TurbohaulQueue(staging_max=10, main_lane_reserved=False)
        await q.enqueue(_normal(CLIENT_A, model_tag="m1"))
        got = await q.pop_next(
            warm_model_tag="m1", fastlane_policy=_policy(max_normal_wait_s=9999.0),
        )
        assert got is not None
        assert await self._served_keys(q) == {f"ip:{CLIENT_A}"}


@pytest.mark.asyncio
class TestClientKeyResolution:
    """The identity key:
    client_meta['ip'] -> thread_id -> one shared 'unresolved' bucket."""

    async def test_ip_wins_over_thread_id(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(
            model_tag="m1", prompt="s", thread_id="t1",
            client_meta={"ip": CLIENT_A},
        )
        assert q._floor_client_key(s) == f"ip:{CLIENT_A}"

    async def test_thread_id_is_the_fallback(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(model_tag="m1", prompt="s", thread_id="t1")
        assert q._floor_client_key(s) == "thread:t1"

    async def test_neither_lands_in_the_shared_unresolved_bucket(self):
        """The consequence, asserted rather than left to be found:
        two wholly-unidentifiable clients SHARE one floor timer. More
        restrictive than per-client, never less."""
        q = TurbohaulQueue(staging_max=10)
        one = Slot.new(model_tag="m1", prompt="one")
        two = Slot.new(model_tag="m1", prompt="two")
        assert q._floor_client_key(one) == "unresolved"
        assert q._floor_client_key(two) == q._floor_client_key(one)

    async def test_an_ip_can_never_collide_with_an_equal_thread_id(self):
        """kv_classify notes a thread_id can itself BE an IP -- the per-source
        prefix is why that cannot merge two different clients."""
        q = TurbohaulQueue(staging_max=10)
        by_ip = Slot.new(model_tag="m1", prompt="a", client_meta={"ip": CLIENT_A})
        by_thread = Slot.new(model_tag="m1", prompt="b", thread_id=CLIENT_A)
        assert q._floor_client_key(by_ip) != q._floor_client_key(by_thread)

    async def test_the_logged_key_never_carries_the_raw_ip(self):
        q = TurbohaulQueue(staging_max=10)
        hashed = q._floor_key_hash(f"ip:{CLIENT_A}")
        assert CLIENT_A not in hashed
        assert len(hashed) == 12
        # The bucket that identifies nobody stays readable.
        assert q._floor_key_hash("unresolved") == "unresolved"


@pytest.mark.asyncio
class TestEligibilityIsUnchanged:
    """Eligibility scope, asserted so a later reader can see the scope line
    was held: the floor promotes UNREGISTERED traffic only. The re-arm logic
    only changes when a client's timer resets, never who is eligible."""

    async def test_an_aged_registered_client_is_still_never_floor_promoted(self):
        q = TurbohaulQueue(staging_max=10)
        aged_registered = _registered(CLIENT_A, rule_index=5, rank=9, prompt="aged-reg")
        aged_registered.created_at = time.monotonic() - 600.0
        top = _registered(CLIENT_B, rule_index=0, rank=0, prompt="rank0")
        for s in (aged_registered, top):
            await q.enqueue(s)

        got = await q.pop_next(fastlane_policy=_policy())

        assert got is not None and got.slot_id == top.slot_id, (
            "a low-ranked REGISTERED client must not displace a high-ranked "
            "one purely by waiting -- the fairness rule's own rationale, and the "
            "half the eligibility scope forbids touching"
        )
        assert got.floor_promoted is False


@pytest.mark.asyncio
class TestFeatureOffIsInert:
    """Feature-off = zero behaviour change AND zero memory cost, this file's
    own convention."""

    async def test_no_policy_writes_nothing_to_the_ledger(self):
        q = TurbohaulQueue(staging_max=10)
        await q.enqueue(_normal(CLIENT_A, aged_by=600.0))
        got = await q.pop_next(fastlane_policy=None)
        assert got is not None
        assert q._floor_last_served_at == {}
        assert q._floor_ledger_active is False

    async def test_pop_matched_thread_writes_nothing_while_the_feature_is_off(self):
        q = TurbohaulQueue(staging_max=10)
        s = Slot.new(
            model_tag="m1", prompt="s", thread_id="t1",
            client_meta={"ip": CLIENT_A},
        )
        await q.enqueue(s)
        assert (await q.pop_matched_thread("t1", "m1")) is not None
        assert q._floor_last_served_at == {}


@pytest.mark.asyncio
class TestLedgerPruneIsBehaviourNeutral:
    """The prune only ever removes entries that already read as ARMED, so no
    reader can tell a pruned entry from an absent one."""

    async def test_prune_only_drops_entries_that_are_already_rearmed(self):
        q = TurbohaulQueue(staging_max=10)
        q._floor_ledger_active = True
        now = time.monotonic()
        fresh = "ip:10.77.1.1"
        stale = "ip:10.77.1.2"
        q._floor_last_served_at = {
            f"ip:10.77.9.{i}": now for i in range(200)  # push past the threshold
        }
        q._floor_last_served_at[fresh] = now
        q._floor_last_served_at[stale] = now - (WINDOW_S + 10.0)

        q._prune_floor_ledger_locked(now, WINDOW_S)

        assert fresh in q._floor_last_served_at, "a live credit must survive"
        assert stale not in q._floor_last_served_at
        # ...and dropping it changed nothing a reader can observe:
        stale_slot = Slot.new(model_tag="m1", prompt="s", client_meta={"ip": "10.77.1.2"})
        assert q._floor_timer_armed_locked(stale_slot, now, WINDOW_S) is True

    async def test_a_small_ledger_is_left_alone_entirely(self):
        q = TurbohaulQueue(staging_max=10)
        now = time.monotonic()
        q._floor_last_served_at = {"ip:10.77.1.2": now - (WINDOW_S + 10.0)}

        q._prune_floor_ledger_locked(now, WINDOW_S)

        assert q._floor_last_served_at, (
            "below the threshold the walk is skipped -- the threshold decides "
            "WHEN the prune runs, never what it would leave behind"
        )
