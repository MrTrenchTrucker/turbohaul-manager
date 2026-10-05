"""Accounting and edge cases of a Fast Lane claim kept while its request waits in a resident's inbox.

Three behaviours are pinned here, each by outcome:

1. A request parked in a resident's inbox with its claim kept is ONE waiting request. The
   headline queue depth must count it once, not once as an inbox entry and again as a claim.
2. A parked request whose client has disconnected is dropped when the resident picks its next
   request, and the resident never goes idle-wedged because of it: if the disconnected request
   is the only candidate it is still handed back (and left alone), if there is a live
   neighbour the live one is returned and the disconnected one is failed with the evicted
   error. The same drop applies when the inbox is drained directly.
3. When two requests of one client share a dedup key and the one that owned the claim has
   already started its turn, a sibling that is routed and parked afterwards still gets a claim
   of its own (the dedup key is free again), and releases it when its own turn starts.

The first and third tests run real manager code (real dispatcher, resident driver, claim
registry) with a fake engine process and a fake completion call that blocks per request, so a
test decides exactly when each turn ends. The second group calls the two admit helpers
directly on a hand-built resident. One process, asyncio only, at most three requests per
test, bounded waits, no sleep above one second. The engine's slot save is replaced by a
recorder so no request ever leaves the process.
"""

import asyncio
import contextlib

from _fastlane_fixture import (
    boot_ranked_runtime,
    high_vram,
    make_fakes,
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)

from turbohaul.manager import Resident, ResidentState, TurbohaulManager
from turbohaul.slot import Slot, SlotEvictedError, SlotState

MODEL = "m1"

# Priority order: index 0 outranks index 1, which outranks index 2.
CLIENT_A_IP = "10.0.0.4"
CLIENT_B_IP = "10.0.0.3"
HOLDER_IP = "10.0.0.5"
RULES = ranked_rules((CLIENT_A_IP, 1), (CLIENT_B_IP, 1), (HOLDER_IP, 1))

HOLDER_THREAD = "thread-holder"

# Bounds for things that must happen promptly or after a short grace window; not delays.
PROMPT_S = 1.5
AFTER_GRACE_S = 8.0


def meta(ip):
    return {"ip": ip, "is_main": True}


class World:
    """One manager, one box slot, a fake engine whose turns end only when told to.

    ``send`` submits a request through the real ``submit`` and labels it. The fake completion
    call records the order in which turns start (``starts``) and, at the moment each turn
    starts, the slot ids that hold a claim in the registry (``registry_at_start``). It then
    blocks until the test releases that request.
    """

    def __init__(self, mgr):
        self.mgr = mgr
        self.slots = {}               # label -> Slot
        self.labels = {}              # slot_id -> label
        self.starts = []              # slot_ids in the order their turns started
        self.registry_at_start = {}   # slot_id -> slot ids holding a claim when the turn began
        self._gates = {}
        self._open = False

    # -- fake completion -------------------------------------------------
    def gate(self, slot_id):
        return self._gates.setdefault(slot_id, asyncio.Event())

    async def complete(self, slot, handle):
        self.starts.append(slot.slot_id)
        self.registry_at_start[slot.slot_id] = self.registry_slot_ids()
        if not self._open:
            await self.gate(slot.slot_id).wait()
        return {"ok": True, "model": handle.model_tag}

    def release(self, label):
        self.gate(self.slots[label].slot_id).set()

    def open_everything(self):
        self._open = True
        for ev in self._gates.values():
            ev.set()

    # -- requests --------------------------------------------------------
    async def send(self, label, ip, thread):
        slot = await self.mgr.submit(
            MODEL, prompt=label, thread_id=thread, client_meta=meta(ip),
            wait_for_completion=True,
        )
        self.slots[label] = slot
        self.labels[slot.slot_id] = label
        return slot

    def order(self):
        """Labels of the turns that have started, in start order."""
        return [self.labels.get(s, s) for s in self.starts]

    async def started(self, label, timeout=PROMPT_S):
        """True when ``label``'s turn starts within ``timeout`` seconds."""
        try:
            await wait_until(lambda: label in self.order(), timeout=timeout)
        except AssertionError:
            return False
        return True

    async def parked(self, labels, timeout=PROMPT_S):
        """Wait until every request in ``labels`` sits in the resident's inbox.

        The real HIT path of the dispatcher put them there: each one has left the staging
        queue and the inbox holds as many requests as were asked for.
        """
        r = resident_for(self.mgr, MODEL)
        slots = [self.slots[lbl] for lbl in labels]
        await wait_until(
            lambda: r.inbox.qsize() >= len(slots)
            and all(s not in self.mgr.queue._staging for s in slots),
            timeout=timeout,
        )
        return r

    async def turn_done(self, label, timeout=PROMPT_S):
        """Wait until ``label``'s turn has completed (it left ACTIVE)."""
        slot = self.slots[label]
        await wait_until(
            lambda: slot.state in (SlotState.GRACE, SlotState.POPPED), timeout=timeout,
        )

    # -- single-stepping the dispatcher ----------------------------------
    async def pause_dispatcher(self):
        """Stop the dispatcher task (the resident's driver keeps running).

        Afterwards a request submitted with ``send`` stays in staging, so a test can look at
        requests that are queued but not routed.
        """
        task = self.mgr._worker_task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def resume_dispatcher(self):
        """Start the dispatcher again; whatever waits in staging is routed from now on."""
        self.mgr._worker_task = asyncio.create_task(self.mgr.worker_loop())

    # -- claims ----------------------------------------------------------
    def registry_slot_ids(self):
        """Slot ids holding a registered claim right now (a plain read, never prunes)."""
        return [c["slot"].slot_id for c in self.mgr._fastlane_claims.values()]

    def snapshot_ids(self):
        """Slot ids the claims snapshot lists (the surface the queue tab reads)."""
        return [row["slot_id"] for row in self.mgr.fastlane_claims_snapshot()]

    def names(self, slot_ids):
        return [self.labels.get(s, s) for s in slot_ids]

    def queue_status(self):
        """The ``queue`` block of the status payload, as the operator surface reads it."""
        return self.mgr.status_snapshot()["queue"]


@contextlib.asynccontextmanager
async def build_world(tmp_path, *, grace_seconds=1, max_grace_extensions=50,
                      idle_hot_load_seconds=600):
    """A manager with ONE box slot and the dispatcher running, over three ranked clients."""
    boot, runtime = boot_ranked_runtime(
        tmp_path, rules=RULES, max_parallel_sidecars=1, grace_seconds=grace_seconds,
        max_grace_extensions=max_grace_extensions, idle_hot_load_seconds=idle_hot_load_seconds,
    )
    seed_manifest(boot, MODEL, main_gpu=0)
    spawn_fn, health_fn, sigterm_fn, vram_fn, _unused = make_fakes({})
    holder = {}

    async def complete(slot, handle):
        return await holder["world"].complete(slot, handle)

    mgr = TurbohaulManager(
        boot, runtime, spawn_fn=spawn_fn, health_fn=health_fn,
        sigterm_fn=sigterm_fn, vram_fn=vram_fn, complete_fn=complete,
    )
    world = World(mgr)
    holder["world"] = world

    async def fake_save(port, model_tag, slot=None, **kwargs):
        return True

    mgr._save_slot_kv = fake_save
    mgr._worker_task = asyncio.create_task(mgr.worker_loop())
    try:
        with high_vram():
            yield world
    finally:
        world.open_everything()
        await mgr.shutdown()


async def start_holder(world):
    """The holder's first turn is running (its fake completion is blocked)."""
    h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
    assert await world.started("H1", timeout=5.0), "setup: the holder's turn never started"
    return h1


# --------------------------------------------------------------------------------------
# 1. a parked request is one waiting request in the headline queue depth
# --------------------------------------------------------------------------------------

async def test_a_parked_request_is_counted_once_in_the_queue_depth(tmp_path):
    """A higher-ranked request is parked in the holder's inbox through the real HIT path and
    keeps its claim. It is ONE waiting request, so the headline queue depth of the status
    payload must read 1: nothing in staging or the accept buffer, the claims snapshot lists it
    once, and the inbox part of the count leaves it out (otherwise it would be counted as an
    inbox entry and again as a claim, a depth of 2). A second parked request of another
    client makes the total 2, not 4. Once the parked requests have started their turns, the
    depth reads 0 again.

    Fails when a request that is both in an inbox and in the claim registry is counted by both
    terms of the total."""
    async with build_world(tmp_path) as world:
        await start_holder(world)
        mgr = world.mgr

        await world.send("CA", CLIENT_A_IP, "thread-CA")
        r = await world.parked(["CA"])
        ca = world.slots["CA"]
        assert r.state is ResidentState.ACTIVE and r.inbox.qsize() == 1, (
            "setup: the holder must still be mid-turn with exactly one request parked"
        )
        assert ca.slot_id in world.registry_slot_ids(), (
            "setup: the parked request must have kept its claim"
        )

        queue = world.queue_status()
        assert queue["staging_queue_depth"] == 0 and queue["acceptance_buffer_depth"] == 0
        assert world.snapshot_ids() == [ca.slot_id], (
            f"the claims snapshot should list the parked request once, listed {world.names(world.snapshot_ids())}"
        )
        assert queue["queue_depth_total"] == 1, (
            f"one parked request must read as a queue depth of 1, got {queue['queue_depth_total']}"
        )
        assert mgr._inbox_waiting_count() == 0, (
            "the inbox part of the depth counted a request that is already counted as a claim: "
            f"{mgr._inbox_waiting_count()}"
        )

        await world.send("CB", CLIENT_B_IP, "thread-CB")
        await world.parked(["CA", "CB"])
        assert r.inbox.qsize() == 2
        assert world.queue_status()["queue_depth_total"] == 2, (
            "two parked requests of two clients must read as a queue depth of 2, got "
            f"{world.queue_status()['queue_depth_total']}"
        )

        world.release("H1")
        assert await world.started("CA", timeout=PROMPT_S), f"order {world.order()}"
        world.release("CA")
        assert await world.started("CB", timeout=AFTER_GRACE_S), f"order {world.order()}"
        assert world.queue_status()["queue_depth_total"] == 0, (
            "the queue depth is not 0 while the last parked request's own turn is running: "
            f"{world.queue_status()['queue_depth_total']}"
        )
        world.release("CB")


# --------------------------------------------------------------------------------------
# 2. a woken (or drained) parked request whose client has gone
# --------------------------------------------------------------------------------------

def make_manager(tmp_path):
    boot, runtime = boot_ranked_runtime(tmp_path)
    return TurbohaulManager(boot, runtime)


def make_slot(slot_id, created_at, *, disconnected=False):
    """A queued request with a completion future; ``disconnected`` sets its disconnect event."""
    slot = Slot(
        slot_id=slot_id,
        model_tag=MODEL,
        state=SlotState.ACTIVE,
        created_at=created_at,
        fastlane=None,
        is_evicted=False,
        client_meta={},
        completion_future=asyncio.Future(),
    )
    slot.disconnect_event = asyncio.Event()
    if disconnected:
        slot.disconnect_event.set()
    return slot


def make_resident():
    return Resident(model_tag=MODEL, inbox=asyncio.Queue())


def evicted_error(slot):
    """The exception its completion future was failed with (fails the test if there is none)."""
    assert slot.completion_future.done(), f"{slot.slot_id}: the completion future is still pending"
    return slot.completion_future.exception()


async def test_a_disconnected_woken_slot_is_dropped_when_another_slot_is_admittable(tmp_path):
    """The resident's parked ``inbox.get()`` hands it a slot whose client has already
    disconnected, and a live request is waiting in the inbox behind it. The live request is
    the one admitted. The disconnected slot is marked evicted and its completion future is
    failed with the evicted error, so its caller learns it was dropped; the inbox is left
    empty (the live request was taken, nothing was put back).

    Fails when the disconnected slot is returned for service while a live one waits, or when
    its future is left pending or failed with another error."""
    mgr = make_manager(tmp_path)
    r = make_resident()
    woken = make_slot("woken-gone", 1.0, disconnected=True)
    live = make_slot("neighbour-live", 2.0)
    r.inbox.put_nowait(live)

    admitted = mgr._rank_admit_woken(r, woken)

    assert admitted is live, (
        f"the admitted slot was {getattr(admitted, 'slot_id', admitted)!r}, expected the live neighbour"
    )
    assert woken.is_evicted, "the disconnected woken slot was not marked evicted"
    assert isinstance(evicted_error(woken), SlotEvictedError), (
        f"the disconnected slot's future was failed with {evicted_error(woken)!r}"
    )
    assert not live.completion_future.done(), "the live neighbour's future must stay pending"
    assert r.inbox.empty(), "nothing may be left in the inbox"


async def test_a_sole_disconnected_woken_slot_is_returned_and_its_future_left_alone(tmp_path):
    """The only candidate is a woken slot whose client has disconnected. The admit helper must
    still hand a slot back (returning nothing would leave the resident marked busy with an
    empty inbox, never idle-unloading), and it hands back that same slot: it is served as an
    evicted woken slot is, and its completion future is untouched here (still pending, not
    failed), because the turn that serves it owns that future.

    Fails when the sole slot is dropped, or its future is failed, before it is returned."""
    mgr = make_manager(tmp_path)
    r = make_resident()
    woken = make_slot("woken-gone", 1.0, disconnected=True)

    admitted = mgr._rank_admit_woken(r, woken)

    assert admitted is woken, (
        f"the helper must hand back the sole woken slot, got {getattr(admitted, 'slot_id', admitted)!r}"
    )
    assert not woken.completion_future.done(), (
        "the sole woken slot's future was settled before it was returned: "
        f"{woken.completion_future!r}"
    )
    assert r.inbox.empty()


async def test_a_direct_drain_returns_the_live_slot_and_fails_the_disconnected_one(tmp_path):
    """The inbox holds a disconnected request and a live one and is drained directly (the
    admit used at a turn boundary). The live one is returned; the disconnected one is marked
    evicted and its future is failed with the evicted error; the inbox ends empty. The order
    in the inbox does not matter, so the disconnected one is put first: it is not skipped
    merely for being first, nor admitted for being the earliest arrival."""
    mgr = make_manager(tmp_path)
    r = make_resident()
    dead = make_slot("parked-gone", 1.0, disconnected=True)
    live = make_slot("parked-live", 2.0)
    r.inbox.put_nowait(dead)
    r.inbox.put_nowait(live)

    admitted = mgr._priority_admit_from_inbox(r)

    assert admitted is live, (
        f"the admitted slot was {getattr(admitted, 'slot_id', admitted)!r}, expected the live one"
    )
    assert dead.is_evicted, "the disconnected parked slot was not marked evicted"
    assert isinstance(evicted_error(dead), SlotEvictedError), (
        f"the disconnected slot's future was failed with {evicted_error(dead)!r}"
    )
    assert not live.completion_future.done()
    assert r.inbox.empty()


# --------------------------------------------------------------------------------------
# 3. a sibling that parks after its representative has already started
# --------------------------------------------------------------------------------------

async def test_a_sibling_that_parks_after_its_representative_started_still_holds_a_claim(tmp_path):
    """Two requests of ONE client for one model. The first is parked in the holder's inbox
    and owns the client's claim; the second arrives while the dispatcher is stopped, so it
    stays in staging and collapses onto the first one's claim (it has none of its own). The
    holder's turn ends and the first request's turn starts: its claim is released and passes
    to the second request, which still waits; that claim then runs out its time limit, so the
    dedup key is free again. Only now is the dispatcher started again and routes the second
    request into the inbox of the resident that is busy with the first one. At that moment
    the claims snapshot lists a claim for the SECOND slot (by slot identity), so the Fast Lane
    machinery still sees it. When the second request's own turn starts that claim is gone,
    and none is left at the end.

    Fails when a request parked while the dedup key is free gets no claim."""
    async with build_world(tmp_path) as world:
        await start_holder(world)

        # The first request parks behind the holder and owns the claim of its client.
        await world.send("C1", CLIENT_A_IP, "thread-C1")
        await world.parked(["C1"])
        c1 = world.slots["C1"]
        assert world.registry_slot_ids() == [c1.slot_id], (
            f"setup: the first request must own the client's claim, registry {world.names(world.registry_slot_ids())}"
        )

        # The second request stays in staging and collapses onto that claim.
        await world.pause_dispatcher()
        c2 = await world.send("C2", CLIENT_A_IP, "thread-C2")
        assert c2 in world.mgr.queue._staging, "setup: the second request must still be queued"
        assert c2.slot_id not in world.snapshot_ids(), (
            "setup: the second request collapses onto the first one's claim, it has none of its own"
        )

        # The first request's turn starts and releases its claim; the claim passes to the
        # second request, which still waits. That claim then runs out its time limit while the
        # second request waits, so the key is free again, as the rest of this test needs.
        world.release("H1")
        assert await world.started("C1", timeout=PROMPT_S), f"order {world.order()}"
        assert world.registry_slot_ids() == [c2.slot_id], (
            "setup: the first request's claim must pass to the waiting second request once its "
            f"own turn runs, registry {world.names(world.registry_slot_ids())}"
        )
        for claim in world.mgr._fastlane_claims.values():
            claim["ttl_deadline_monotonic"] = 0.0
        assert world.snapshot_ids() == [], "setup: the expired claim must be pruned, the key free"
        assert c2 in world.mgr.queue._staging, "setup: the second request must still be in staging"

        # Now the dispatcher routes the second request into the busy resident's inbox.
        world.resume_dispatcher()
        r = await world.parked(["C2"])
        assert r.active_slot is c1, "setup: the first request's turn must still be running"
        listed = world.snapshot_ids()
        assert c2.slot_id in listed, (
            "the request parked after its representative started holds no claim; the claims "
            f"snapshot listed {world.names(listed)}"
        )
        assert c1.slot_id not in listed, "the first request's claim was listed during its own turn"

        # Its own turn starts: the claim is released, and nothing leaks.
        world.release("C1")
        assert await world.started("C2", timeout=AFTER_GRACE_S), f"order {world.order()}"
        assert c2.slot_id not in world.registry_at_start[c2.slot_id], (
            "the second request's claim was still registered after its own turn started"
        )
        world.release("C2")
        await world.turn_done("C2")
        assert world.registry_slot_ids() == [], (
            f"a claim leaked: {world.names(world.registry_slot_ids())}"
        )
        assert world.snapshot_ids() == []
