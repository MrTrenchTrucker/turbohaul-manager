"""End to end: the grace window's follow-up match must be handed the resident's
inbox, so that a better-tag request of the SAME client waiting there holds back
a worse-tag same-thread follow-up even when that request has no live claim.

Why the claim is removed: a better-tag request parked in the inbox WITH a live
claim already ends the holder's grace at the turn boundary, so the follow-up
match is never reached and the inbox argument would not matter. Here the claim
registry is emptied after the request parked, so the request is still in the
inbox but nothing outranks the holder through a claim. The only thing left that
can hold the follow-up back is the inbox contents handed to the follow-up match
in the grace loop. The real way to reach this state is a claim eviction at
capacity; this test empties the registry by a direct edit and does not drive
that eviction.

Real manager, dispatcher and grace loop; only the engine process, the health
probe, the teardown call and the completion call are faked. One client (one
rule: main = 1, sub_agent = 3). The resident serves a sub_agent turn (W1) that
is held open. While it is busy, a request M of the same client is submitted and
the real dispatcher parks it in the resident's inbox (asserted, not assumed).
W1 is released; the resident enters its grace window, and a same-thread
sub_agent follow-up F arrives. The assertion is the ORDER in which W1, M and F
start, taken from the completion fake at the moment the real code reached it.

Waits are polls of an observable condition with a timeout; no sleep decides a
result. Free VRAM: this file never reads it; the autouse fixture below makes
any read of it, through either binding, fail the test.
"""
import asyncio
import contextlib

import pytest

from _fastlane_fixture import (
    boot_ranked_runtime, make_fakes, resident_for, seed_manifest, wait_until,
)

from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState


# Nothing in this file may read the machine's real free VRAM, through either the
# manager's own binding or the safety module's: both are replaced by a function
# that fails the test if it is ever called.
@pytest.fixture(autouse=True)
def _pin_free_vram_autouse(monkeypatch):
    import turbohaul.manager as manager_module
    import turbohaul.safety as safety_module

    def _must_not_read(*args, **kwargs):
        raise AssertionError("a test of this file read free VRAM")

    monkeypatch.setattr(manager_module, "_read_free_vram_all_mib", _must_not_read)
    monkeypatch.setattr(safety_module, "_read_free_vram_all_mib", _must_not_read)


MODEL = "m1"
IP = "192.0.2.10"
RULES = [FastLaneRule(address=IP, tag_ranks=FastLaneTagRanks(main=1, sub_agent=3))]
MAIN = {"ip": IP, "is_main": True}
SUB = {"ip": IP, "is_sub_agent": True}
THREAD_SUB = "thread-sub"
THREAD_M = "thread-m"
SETUP_WAIT_S = 3.0
GRACE_S = 2


class World:
    def __init__(self, mgr):
        self.mgr = mgr
        self.labels = {}      # slot_id -> label
        self.starts = []      # slot ids in the order the completion fake was reached
        self.gates = {}

    def gate(self, label):
        return self.gates.setdefault(label, asyncio.Event())

    async def complete(self, slot, handle):
        self.starts.append(slot.slot_id)
        if self.labels.get(slot.slot_id) == "W1":
            await self.gate("W1").wait()
        return {"ok": True}

    async def send(self, label, meta, thread):
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread, client_meta=dict(meta),
            wait_for_completion=True,
        )
        self.labels[slot.slot_id] = label
        return slot

    def order(self):
        return [self.labels.get(s, s) for s in self.starts]


@contextlib.asynccontextmanager
async def build_world(tmp_path):
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=RULES, max_parallel_sidecars=1, grace_seconds=GRACE_S,
        max_grace_extensions=50, idle_hot_load_seconds=600,
    )
    # A layer-split manifest is not auto-placed, and the memory-fit check made
    # before a load is replaced below by one that always admits, so loading the
    # model never asks for the cards' free memory.
    seed_manifest(boot, MODEL, split_mode="layer", main_gpu=0)
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})
    holder = {}

    async def complete(slot, handle):
        return await holder["w"].complete(slot, handle)

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    mgr._vram_admits_locked = lambda *args, **kwargs: True
    w = World(mgr)
    holder["w"] = w
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        yield w
    finally:
        for ev in w.gates.values():
            ev.set()
        await mgr.shutdown()


async def _run(tmp_path, parked_meta):
    """W1 held, M parked in the inbox, M's claim removed, W1 released, F sent once
    the resident is in its grace window. Returns the order of starts once M and F
    have both started."""
    async with build_world(tmp_path) as w:
        w1 = await w.send("W1", SUB, THREAD_SUB)
        await wait_until(lambda: "W1" in w.order(), timeout=5.0)
        m = await w.send("M", parked_meta, THREAD_M)
        r = resident_for(w.mgr, MODEL)
        await wait_until(
            lambda: r.inbox.qsize() >= 1 and m not in w.mgr.queue._staging,
            timeout=SETUP_WAIT_S)
        assert r.state is ResidentState.ACTIVE and r.active_slot is w1, (
            "setup: the holder must be mid-turn")
        assert m.state is SlotState.STAGED, "setup: the parked request must still be queued"
        assert w.order() == ["W1"], f"setup: only W1 may have started, got {w.order()}"
        assert w.mgr._fastlane_claims, "setup: the parked request must have registered a claim"

        # The claim registry is emptied while M stays in the inbox: no live claim
        # outranks the holder, so the turn boundary will not admit M early.
        w.mgr._fastlane_claims.clear()
        assert r.inbox.qsize() == 1, "setup: M must still be in the inbox"
        assert not w.mgr._parked_claim_outranks_holder_locked(r), (
            "setup: with no claim nothing may end the holder's grace through a claim")

        w.gate("W1").set()
        await wait_until(lambda: r.in_grace_loop, timeout=SETUP_WAIT_S)
        assert w.order() == ["W1"], (
            f"setup: M must not have been admitted before the follow-up is sent: {w.order()}")
        assert r.inbox.qsize() == 1, "setup: M must still be waiting in the inbox in the grace window"
        await w.send("F", SUB, THREAD_SUB)
        try:
            await wait_until(lambda: {"M", "F"} <= set(w.order()), timeout=GRACE_S + 8.0)
        except AssertionError:
            raise AssertionError(f"not every request started: {w.order()}") from None
        return w.order()


async def test_better_tag_in_inbox_without_a_claim_still_holds_back_the_follow_up(tmp_path):
    """M is tagged main (rank 1), waits in the inbox and has no live claim; F is
    a sub_agent (rank 3) same-thread follow-up in the grace window. Expected: the
    follow-up is not served warm ahead of M, so M starts before F (order W1, M,
    F) and F is not lost."""
    order = await _run(tmp_path, MAIN)
    assert order == ["W1", "M", "F"], (
        f"start order {order}: the worse-tag follow-up was served ahead of the better-tag "
        "request of the same client waiting in the resident's inbox")


async def test_control_equal_tag_in_inbox_without_a_claim_follow_up_still_rides_warm(tmp_path):
    """Control: M is tagged sub_agent like F (equal rank) and has no live claim.
    Expected: an equal rank does not hold the follow-up back, so it keeps the
    warm slot and starts first (order W1, F, M); both requests start."""
    order = await _run(tmp_path, SUB)
    assert order == ["W1", "F", "M"], f"start order {order}"
