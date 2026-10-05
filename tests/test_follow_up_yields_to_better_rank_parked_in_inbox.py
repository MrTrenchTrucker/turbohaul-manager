"""End to end: a worse-tag same-thread follow-up in the grace window must not
jump a better-tag request of the SAME client that is parked in the busy
resident's inbox.

Real manager, dispatcher and grace loop; only the engine process, the health
probe, the teardown call and the completion call are faked. One client (one
rule: main = 1, sub_agent = 3). The resident serves a sub_agent turn (W1) that
is held open. While it is busy, a request M of the same client is submitted and
the real dispatcher parks it in the resident's inbox (asserted, not assumed).
W1 is released; once the resident is either in its grace window or has already
admitted M at its turn boundary, a same-thread sub_agent follow-up F arrives.
The assertion is the ORDER in which W1, M and F start, taken from the
completion fake at the moment the real code reached it.

Waits are polls of an observable condition with a timeout; no sleep decides a
result. Free VRAM: both `turbohaul.safety._read_free_vram_all_mib` and
`turbohaul.manager._read_free_vram_all_mib` are pinned to the same single-card
model, so the decisions never read the machine's real GPUs.
"""
import asyncio
import contextlib
from unittest.mock import patch

from _fastlane_fixture import (
    boot_ranked_runtime, make_fakes, resident_for, seed_manifest, wait_until,
)

from turbohaul.config import FastLaneRule, FastLaneTagRanks
from turbohaul.manager import ResidentState, TurbohaulManager
from turbohaul.slot import SlotState

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
    seed_manifest(boot, MODEL, main_gpu=0)
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})
    holder = {}

    async def complete(slot, handle):
        return await holder["w"].complete(slot, handle)

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    w = World(mgr)
    holder["w"] = w
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with patch("turbohaul.safety._read_free_vram_all_mib", return_value=[80000]), \
             patch("turbohaul.manager._read_free_vram_all_mib", return_value=[80000]):
            yield w
    finally:
        for ev in w.gates.values():
            ev.set()
        await mgr.shutdown()


async def _run(tmp_path, parked_meta):
    """W1 held, M parked in the inbox, W1 released, F sent once the resident is
    in its grace window or has already admitted M. Returns the order of starts
    once M and F have both started."""
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
        w.gate("W1").set()
        # The follow-up is sent once the resident is either in its grace window or
        # has already admitted M at its turn boundary; the test does not depend on
        # the grace window surviving.
        await wait_until(lambda: r.in_grace_loop or "M" in w.order(), timeout=SETUP_WAIT_S)
        await w.send("F", SUB, THREAD_SUB)
        try:
            await wait_until(lambda: {"M", "F"} <= set(w.order()), timeout=GRACE_S + 8.0)
        except AssertionError:
            raise AssertionError(f"not every request started: {w.order()}") from None
        return w.order()


async def test_better_tag_parked_in_inbox_starts_before_worse_tag_follow_up(tmp_path):
    """M is tagged main (rank 1) and parked in the inbox; F is a sub_agent
    (rank 3) same-thread follow-up in the grace window. Expected: M starts
    before F (order W1, M, F); F is not lost."""
    order = await _run(tmp_path, MAIN)
    assert order == ["W1", "M", "F"], (
        f"start order {order}: the worse-tag follow-up ran before the better-tag request "
        "of the same client parked in the resident's inbox")


async def test_control_equal_tag_parked_in_inbox_follow_up_still_starts_first(tmp_path):
    """Control: M is tagged sub_agent like F (equal rank). Expected: an equal
    rank does not outrank, so the follow-up keeps the warm slot and starts
    first (order W1, F, M); both requests start."""
    order = await _run(tmp_path, SUB)
    assert order == ["W1", "F", "M"], f"start order {order}"
