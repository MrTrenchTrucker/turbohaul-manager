"""Two windows in the life of a parked claim, and the promise that a parked claim holds nothing aside.

A request for an already-loaded model tag is handed to the resident's inbox and keeps its Fast
Lane claim, marked as parked on that resident, until its own turn starts. Three things around
that are pinned here by outcome:

1. The swap window. When the holder's turn ends and a higher-ranked parked request takes the
   engine next, the holder's KV is saved first, and that save can take a long time. The
   claimant has already left the inbox by then and its turn has not started. In that window the
   resident must not look free: a make-room pass must not choose it as the victim to unload,
   and a mid-ranked claim waiting for room must not see it as the designated victim either.
   A resident that is unloaded in that window leaves the highest-ranked request neither served,
   nor failed, nor queued.
2. The out-of-memory re-queue. A request whose engine failed to load for lack of memory goes
   back to waiting for room. It is not in any inbox then, so its claim has to be an ordinary
   waiting claim again (not marked as parked, with the claim time-to-live running). And the
   claim's mark is only a resident key, which a later resident can reuse: a claim marked as
   parked on a resident that is gone must not make a new resident that has the same key look
   outranked, nor be subtracted from its inbox count.
3. A claim parked in a resident's own inbox needs no room, so the queue must not hold lower
   ranked, unrelated requests aside because of it. A waiting claim of the same rank, not
   parked, does hold them aside (the control).

Everything runs real manager code (real dispatcher, drivers, grace loop, claim registry, queue
pick) with a fake engine process and a fake completion call that blocks per request. One
process, asyncio only, at most four requests per test, bounded waits, no sleep above one
second. The completion call, the slot save and (for the re-queue) the queue's re-enqueue are the
only seams replaced.
"""

import asyncio
import contextlib
import time

from _fastlane_fixture import (
    ranked_rules,
    resident_for,
    seed_manifest,
    wait_until,
)
from test_parked_claim_lifetime_and_kv_save import (
    AFTER_GRACE_S,
    CLAIMANT_IP,
    CLAIMANT_THREAD,
    HOLDER_IP,
    HOLDER_THREAD,
    MODEL,
    PAST_TTL_S,
    PROMPT_S,
    SHORT_TTL_S,
    build_world,
    hold_and_park,
    meta,
)
from test_oom_requeue_resident_driver import (
    _OOM_SCAN,
    _make_manager,
    _run_worker,
    _wait_until_parked,
)

from turbohaul import manager as manager_module
from turbohaul.manager import Resident, ResidentState
from turbohaul.slot import SlotState

# A third client that ranks between the claimant (index 0) and the holder (index 2), and a
# second model a request of that client can ask for.
MID_IP = "10.0.0.6"
OTHER_MODEL = "m2"
THREE_RULES = ranked_rules((CLAIMANT_IP, 1), (MID_IP, 1), (HOLDER_IP, 1))

# How long the replaced slot save may block before it gives up by itself, and how long a test
# waits for something that must have happened by then. Bounds, not delays.
SAVE_BOUND_S = 10.0
ENTER_S = 5.0
OOM_POLL_S = 0.01


class BlockedSave:
    """The manager's slot save, replaced by one that blocks until the test lets it go.

    ``entered`` is set when a save has begun, ``go`` lets it finish. The save is recorded in
    the world's event list like the world's own recorder does. ``go`` is also set when the
    world is torn down, so a failing test can never leave a save blocked.
    """

    def __init__(self, world):
        self.world = world
        self.entered = asyncio.Event()
        self.go = asyncio.Event()

        async def blocking_save(port, model_tag, slot=None, **kwargs):
            world.events.append(("save", kwargs.get("thread_id_override")))
            self.entered.set()
            # Not ``wait_for``: it can swallow a cancellation that arrives in the same loop
            # turn as the event, and a driver that is cancelled at teardown must stay cancelled.
            waiter = asyncio.ensure_future(self.go.wait())
            try:
                await asyncio.wait({waiter}, timeout=SAVE_BOUND_S)
            finally:
                waiter.cancel()
            return True

        world.mgr._save_slot_kv = blocking_save
        world.cleanups.append(self.go.set)

    async def wait_entered(self):
        await asyncio.wait_for(self.entered.wait(), ENTER_S)


async def blocked_swap(world):
    """Reach the swap window: the holder's turn has ended, the higher-ranked parked request has
    been taken out of the inbox, and the holder's KV save is blocked.

    Returns (holder slot, claimant slot, resident, the blocked save).
    """
    blocked = BlockedSave(world)
    h1, claimant, r = await hold_and_park(world)
    world.release("H1")
    await blocked.wait_entered()
    assert "C" not in world.order(), "setup: the claimant's turn must not have started yet"
    assert world.has_claim("C"), "setup: the claimant keeps its claim until its own turn starts"
    return h1, claimant, r, blocked


def in_some_inbox(mgr, slot):
    return any(slot in list(rr.inbox._queue) for rr in mgr._residents.values()
               if getattr(rr, "inbox", None) is not None)


# --------------------------------------------------------------------------------------
# the swap window: the holder's KV save is running
# --------------------------------------------------------------------------------------

async def test_a_make_room_pass_cannot_choose_the_resident_while_the_holders_kv_is_saved(tmp_path):
    """The holder's turn has ended and a higher-ranked request that was parked in its inbox is
    about to take the engine, but the holder's KV save is still running. The claimant has left
    the inbox and its turn has not started, so nothing in the resident looks busy. A make-room
    pass (the victim picker a request for another model uses to free a slot) must NOT be able
    to choose this resident in that window: unloading it would throw away the engine that the
    highest-ranked request is about to use. After the save is let go the claimant's turn starts
    on the same engine, with no respawn (control half, green with or without the fix)."""
    async with build_world(tmp_path) as world:
        _h1, _claimant, r, blocked = await blocked_swap(world)

        victim = world.mgr._lru_idle_unloadable()
        assert victim is not r, (
            "the make-room victim picker offers the resident during the swap save: state "
            f"{r.state.name}, active slot {r.active_slot}, in-flight {r.inflight}"
        )

        blocked.go.set()
        assert await world.started("C", timeout=AFTER_GRACE_S), (
            f"the claimant's turn never started after the save ended; events {world.named_events()}"
        )
        assert world.spawns == [MODEL] and world.unloads == [], (
            f"the resident was respawned or unloaded: spawns={world.spawns} unloads={world.unloads}"
        )


async def test_the_resident_is_marked_as_taken_by_the_claimant_while_the_holders_kv_is_saved(tmp_path):
    """Same window as above, looked at from the resident itself. While the holder's KV save
    runs, the claimant that is about to start is already the resident's active request, and the
    resident is not parked as an idle, evictable engine. An idle-evictable resident with no
    active request and no riders is exactly what a make-room pass takes."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r, blocked = await blocked_swap(world)

        assert r.state is not ResidentState.IDLE_EVICTABLE, (
            "the resident is still parked as idle and evictable while its engine is being "
            "handed to the claimant"
        )
        assert r.active_slot is claimant, (
            f"the resident's active request is {r.active_slot}, not the claimant that is "
            "about to start on it"
        )
        blocked.go.set()
        assert await world.started("C", timeout=AFTER_GRACE_S)


async def test_consequence_an_unload_during_the_swap_save_does_not_strand_the_claimant(tmp_path):
    """The consequence of the two tests above, driven on purpose. While the holder's KV save
    runs, an unload of the resident is requested (what a make-room pass does to the engine it
    picks). Whatever happens to the engine, the highest-ranked request must not be lost: once
    the save ends and every turn is allowed to run, the claimant's request is answered (served,
    or failed with an error), or at least still waiting in a queue or an inbox. Its future
    must not be left pending with no queue membership at all, because nothing would ever wake
    it and only the client giving up would end it. The outcome is asserted, not how the
    resident got there."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r, blocked = await blocked_swap(world)

        async with world.mgr._registry_lock:
            world.mgr._begin_unload_locked(r)

        blocked.go.set()
        world.open_everything()
        fut = claimant.completion_future
        done, _ = await asyncio.wait({fut}, timeout=AFTER_GRACE_S)
        await world.drain_background()
        queued = claimant in world.mgr.queue._staging or in_some_inbox(world.mgr, claimant)
        assert done or queued, (
            "the claimant is stranded: its future is pending, it is in no queue and in no "
            f"inbox (state {claimant.state.name}, residents {list(world.mgr._residents)}, "
            f"claims {[c['slot'] is claimant for c in world.mgr._fastlane_claims.values()]})"
        )


async def idle_holder_then_hit_swap(world):
    """The idle-holder-then-hit shape, up to the blocked holder save.

    Three clients: the claimant ranks highest, a middle client second, the holder lowest. The
    holder's turn has ended and its grace window has run out, so the resident sits idle. A
    request of the highest-ranked client hits it: the hit flips the resident to active and parks
    the request in its inbox, the driver wakes, takes the request, and blocks in the holder's KV
    save. Returns (claimant slot, resident, the blocked save). The world must have been built
    with the three-client rules.
    """
    seed_manifest(world.mgr.boot, OTHER_MODEL, main_gpu=1)
    blocked = BlockedSave(world)
    h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
    assert await world.started("H1", timeout=ENTER_S), "setup: the holder's turn never started"
    r = resident_for(world.mgr, MODEL)
    world.release("H1")
    await world.turn_done("H1")
    await wait_until(lambda: r.state is ResidentState.IDLE_EVICTABLE, timeout=AFTER_GRACE_S)
    assert r.active_slot is None and not r.inflight, "setup: the resident must be idle"

    claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
    await blocked.wait_entered()
    assert "C" not in world.order(), "setup: the claimant's turn must not have started yet"
    assert world.has_claim("C"), "setup: the claimant keeps its claim until its turn starts"
    assert claimant.state is SlotState.STAGED and h1.state is not SlotState.ACTIVE
    return claimant, r, blocked


async def ask_for_another_model(world):
    """The middle client asks for a model that is not loaded and has to wait for room."""
    return await world.mgr.submit(
        OTHER_MODEL, prompt="M", thread_id="thread-mid", client_meta=meta(MID_IP),
        wait_for_completion=True,
    )


async def test_a_mid_ranked_claim_does_not_make_the_resident_its_victim_during_the_swap_save(
    tmp_path, monkeypatch,
):
    """The idle-holder-then-hit shape (see ``idle_holder_then_hit_swap``). While the driver is
    blocked in the holder's KV save, the middle client asks for ANOTHER model and has to wait
    for room, and the real dispatcher acts on its claim. The resident's identity still names
    the holder (the lowest rank), but the engine is about to go to the highest-ranked client,
    who outranks the middle one. The resident must therefore not be unloaded for the middle
    client's claim. After the save is let go the highest-ranked request starts on the same
    engine, with no respawn (control half)."""
    monkeypatch.setattr("test_parked_claim_lifetime_and_kv_save.RULES", THREE_RULES)
    async with build_world(tmp_path, grace_seconds=1) as world:
        _claimant, r, blocked = await idle_holder_then_hit_swap(world)
        mid = await ask_for_another_model(world)
        await wait_until(
            lambda: any(c["slot"] is mid for c in world.mgr._fastlane_claims.values()),
            timeout=PROMPT_S,
        )
        # Give the dispatcher time to act on the middle client's claim (a bounded sleep).
        await asyncio.sleep(0.3)

        assert world.unloads == [] and r.state is not ResidentState.DEAD, (
            "the resident was unloaded for a middle-ranked claim while a higher-ranked "
            f"request was about to start on it: unloads {world.unloads}, state {r.state.name}"
        )
        blocked.go.set()
        assert await world.started("C", timeout=AFTER_GRACE_S), (
            f"the claimant's turn never started after the save ended; events {world.named_events()}"
        )
        assert world.spawns == [MODEL], f"the resident was respawned: {world.spawns}"


async def test_a_mid_ranked_claim_does_not_designate_the_resident_during_the_swap_save(
    tmp_path, monkeypatch,
):
    """Same shape as above with the dispatcher stopped after the middle client's request is
    queued, so that nothing can act on its claim and the two questions a make-room pass asks
    can be read directly. While the driver is blocked in the holder's KV save, the middle
    client's waiting claim must NOT make the resident its designated victim (the resident
    that loses its grace and may be unloaded), and the victim picker must not offer it."""
    monkeypatch.setattr("test_parked_claim_lifetime_and_kv_save.RULES", THREE_RULES)
    async with build_world(tmp_path, grace_seconds=1) as world:
        _claimant, r, blocked = await idle_holder_then_hit_swap(world)
        await world.pause_dispatcher()
        mid = await ask_for_another_model(world)
        assert mid in world.mgr.queue._staging and any(c['slot'] is mid for c in world.mgr._fastlane_claims.values()), (
            "setup: the middle client's request must be queued with a live claim"
        )
        assert world.unloads == [] and r.state is not ResidentState.DEAD

        async with world.mgr._registry_lock:
            designated = world.mgr._is_designated_unload_target_locked(r)
        assert not designated, (
            "the middle-ranked claim sees the resident as its designated victim although "
            "the highest-ranked request is about to start on it"
        )
        assert world.mgr._lru_idle_unloadable() is not r, (
            "the make-room victim picker offers the resident to the middle-ranked claim"
        )
        blocked.go.set()
        assert await world.started("C", timeout=AFTER_GRACE_S)


# --------------------------------------------------------------------------------------
# the out-of-memory re-queue
# --------------------------------------------------------------------------------------

async def test_an_oom_requeued_request_holds_an_ordinary_waiting_claim_again(tmp_path, monkeypatch):
    """A request for a cold model is handed to a new resident's inbox while the engine starts
    (its claim is kept, parked on that resident). The engine fails to load for lack of memory,
    the resident is torn down, and the request goes back to waiting for room. The re-queue
    runs on a real make-room wake. It is not in any inbox then, so its claim must be an
    ordinary waiting claim: no longer marked as parked on the dead resident, and subject to
    the claim time-to-live again. A claim that kept the mark would never expire, and a resident
    that later reuses the key would be treated as having a parked request. The queue's
    re-enqueue is replaced by a recorder, so nothing re-admits the request and the claim is
    read exactly as the re-queue left it; the failure scan, the wake and the companion that
    waits for it are the real ones."""
    monkeypatch.setattr(manager_module, "_FASTLANE_CLAIM_TTL_S", SHORT_TTL_S)
    rules = ranked_rules(("10.0.0.5", 1))
    mgr, _calls, _boot = _make_manager(
        tmp_path, monkeypatch, scan_result=_OOM_SCAN, fastlane_rules=rules, poll_s=OOM_POLL_S,
    )
    enqueued = []
    parks = []

    async def record_enqueue_head(slot):
        enqueued.append(slot)

    real_park = mgr._park_fastlane_claim_locked

    def spy_park(slot, r):
        parks.append((slot, r.resident_key))
        return real_park(slot, r)

    mgr.queue.enqueue_head = record_enqueue_head
    mgr._park_fastlane_claim_locked = spy_park
    disconnect_event = asyncio.Event()
    try:
        slot = await mgr.submit(
            model_tag="m1", prompt="x", wait_for_completion=True,
            disconnect_event=disconnect_event, client_meta={"ip": "10.0.0.5"},
        )
        assert slot.fastlane is not None, "setup: an unlisted request holds no claim at all"
        await _run_worker(mgr)
        assert await _wait_until_parked(slot), "setup: the load failure did not re-queue the request"
        assert [s for s, _key in parks] == [slot], f"setup: the request was not parked on the new resident: {parks}"

        # A real make-room wake releases the companion that waits for it, which re-enqueues.
        deadline = time.monotonic() + 5.0
        while not enqueued and time.monotonic() < deadline:
            async with mgr._make_room_signal:
                mgr._make_room_signal.notify_all()
            await asyncio.sleep(0.01)
        assert enqueued == [slot], f"setup: the re-queue never ran: {enqueued}"
        for _ in range(20):
            pending = [t for t in list(mgr._bg_tasks) if not t.done()]
            if not pending:
                break
            await asyncio.wait(pending, timeout=0.25)
        await wait_until(lambda: "m1" not in mgr._residents, timeout=PROMPT_S)

        key = mgr._fastlane_claim_key(slot)
        claim = mgr._fastlane_claims.get(key)
        assert claim is not None and claim["slot"] is slot, "setup: the re-queued request holds no claim"
        assert not claim.get("parked_on"), (
            "the re-queued request's claim is still marked as parked on a resident that is "
            f"gone: parked_on={claim.get('parked_on')!r}, residents {list(mgr._residents)}"
        )

        await asyncio.sleep(PAST_TTL_S)
        assert mgr.fastlane_claims_snapshot() is not None
        assert key not in mgr._fastlane_claims, (
            "the claim of a request waiting for room again did not expire after the claim "
            "time-to-live"
        )
    finally:
        disconnect_event.set()
        await asyncio.wait_for(mgr.shutdown(), timeout=1.5)


@contextlib.asynccontextmanager
async def stale_parked_claim(tmp_path, *, with_inbox_waiter=False, grace_seconds=2):
    """A live claim marked as parked on a resident key that is gone, then a NEW resident that
    reuses the key (the lowest free index is reused) with a lower-ranked holder mid-turn.

    The claim is the highest-ranked client's; it is registered through the real submit (the
    dispatcher is stopped first, so the request stays in staging and nothing routes it), then
    parked with the manager's own park helper on a dead resident object that carries the key.
    ``with_inbox_waiter`` first puts one unlisted request in the live resident's inbox through
    the real hit path, so the inbox count is not trivially zero. Yields
    (world, live resident, claimant slot, dead resident).
    """
    async with build_world(tmp_path, grace_seconds=grace_seconds) as world:
        await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=ENTER_S), "setup: the holder's turn never started"
        r = resident_for(world.mgr, MODEL)
        if with_inbox_waiter:
            await world.send("D", "10.0.0.9", "thread-other")
            await world.parked("D")
            assert r.inbox.qsize() == 1 and not world.has_claim("D"), "setup: one unlisted waiter"
        await world.pause_dispatcher()
        claimant = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD)
        assert claimant in world.mgr.queue._staging and world.has_claim("C"), (
            "setup: the claimant must be queued with its own claim"
        )
        dead = Resident(model_tag=MODEL, resident_key=MODEL, state=ResidentState.DEAD)
        assert dead is not r and world.mgr._residents[MODEL] is r
        async with world.mgr._registry_lock:
            world.mgr._park_fastlane_claim_locked(claimant, dead)
        claim = next(c for c in world.mgr._fastlane_claims.values() if c["slot"] is claimant)
        assert claim.get("parked_on") == MODEL, "setup: the claim must carry the old resident's key"
        assert r.inbox.qsize() == (1 if with_inbox_waiter else 0)
        yield world, r, claimant, dead


async def test_a_claim_parked_on_a_gone_resident_does_not_outrank_a_new_resident_with_the_same_key(tmp_path):
    """A live, highest-ranked claim is marked as parked on a resident key, but the resident it
    was parked on is gone, and a new resident now has that key. The new resident's inbox is
    empty and its holder ranks lower than the claim. Nothing is waiting for the new resident,
    so it is NOT outranked by a parked request: the question the holder's grace and the
    immediate-evict check both ask must say no."""
    async with stale_parked_claim(tmp_path) as (world, r, _claimant, _dead):
        assert r.inbox.qsize() == 0
        assert not world.mgr._parked_claim_outranks_holder_locked(r), (
            "a claim marked as parked on a resident that is gone makes the new resident with "
            "the same key look outranked, although its inbox is empty"
        )


async def test_a_claim_parked_on_a_gone_resident_is_not_subtracted_from_a_new_residents_inbox_count(tmp_path):
    """Same stale claim, seen by the queue-depth display. One unlisted request really waits in
    the new resident's inbox. The claim is not in that inbox, so the count of requests waiting
    in inboxes stays at one. Subtracting the claim would hide a request that really waits."""
    async with stale_parked_claim(tmp_path, with_inbox_waiter=True) as (world, r, _claimant, _dead):
        assert r.inbox.qsize() == 1
        assert world.mgr._inbox_waiting_count() == 1, (
            "the inbox count subtracted a claim that is not parked in any live inbox: "
            f"{world.mgr._inbox_waiting_count()} for one real waiter"
        )


async def test_the_holder_keeps_its_grace_window_when_the_only_parked_claim_belongs_to_a_gone_resident(tmp_path):
    """Same stale claim, seen by the holder's own turn boundary. The holder's turn ends with its
    inbox empty and a two-second grace window configured: it must enter its grace window
    (its slot goes to the grace state) as it does when nothing is claiming. A holder that
    skips the window because of a claim that is not parked in its inbox loses the warm
    follow-ups the window exists for. The slot state is read right after the turn ends, well
    inside the window."""
    async with stale_parked_claim(tmp_path, grace_seconds=2) as (world, r, _claimant, _dead):
        h1 = world.slots["H1"]
        world.release("H1")
        await world.turn_done("H1")
        assert h1.state is SlotState.GRACE, (
            f"the holder did not enter its grace window: slot state {h1.state.name}"
        )
        assert r.state is not ResidentState.DEAD and world.unloads == []


async def test_control_with_no_claim_at_all_the_holder_enters_its_grace_window(tmp_path):
    """Control for the test above, green with or without the fix. With nothing claiming, the
    holder's slot goes to the grace state when its turn ends, so the state read in the test
    above can tell a skipped window from a taken one."""
    async with build_world(tmp_path, grace_seconds=2) as world:
        h1 = await world.send("H1", HOLDER_IP, HOLDER_THREAD)
        assert await world.started("H1", timeout=ENTER_S)
        world.release("H1")
        await world.turn_done("H1")
        assert h1.state is SlotState.GRACE, f"slot state {h1.state.name}"


async def test_control_a_claim_really_parked_in_the_inbox_does_outrank_the_holder_and_counts_once(tmp_path):
    """Control for the stale-claim tests, green with or without the fix. A higher-ranked request
    really parked in the live resident's inbox by the real hit path DOES outrank the lower
    ranked holder, and is counted once in the inbox count (the request sits in the inbox and its
    claim is subtracted: the claims are counted with the claims)."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r = await hold_and_park(world)
        assert r.inbox.qsize() == 1 and world.has_claim("C")
        assert world.mgr._parked_claim_outranks_holder_locked(r), (
            "a request really parked in the inbox must outrank the lower-ranked holder"
        )
        assert world.mgr._inbox_waiting_count() == 0
        assert claimant.state is SlotState.STAGED


# --------------------------------------------------------------------------------------
# the queue's hold-aside does not see parked claims
# --------------------------------------------------------------------------------------

async def test_a_parked_claim_holds_no_lower_ranked_request_aside_but_a_waiting_claim_does(tmp_path):
    """The queue holds a request aside for one pick when a strictly higher-ranked live claim
    is governing, because that claimant is waiting for room. A claim parked in a live
    resident's inbox needs no room, so it must hold nothing aside.

    Part one: the highest-ranked client's request is parked in the live resident's inbox; the
    holder's client (lower rank) asks for another model and waits in staging. The governing
    claim the queue is handed is not the parked one (it is the lower-ranked client's own
    claim), and the real queue pick returns the lower-ranked request.
    Part two (control): the SAME claim is made an ordinary waiting claim (un-parked, as if it
    had been re-queued). It now governs: it is the governing claim and the queue holds the next
    lower-ranked request aside, so the pick returns nothing and the request is still queued.

    The reader that decides this is the hold-aside's governing-claim read, which asks for
    claims that are not parked. If that read were changed to include parked claims, part one
    would fail: the parked claim would govern and the lower-ranked request would be held aside
    for a claimant that needs no room. Green on the current code."""
    async with build_world(tmp_path) as world:
        seed_manifest(world.mgr.boot, OTHER_MODEL, main_gpu=1)
        mgr = world.mgr
        _h1, claimant, _r = await hold_and_park(world)
        await world.pause_dispatcher()
        claim = next(c for c in mgr._fastlane_claims.values() if c["slot"] is claimant)
        assert claim.get("parked_on"), "setup: the claimant's claim must be parked"
        claimant_key = mgr.queue._fastlane_priority_key(claimant)

        lower = await mgr.submit(
            OTHER_MODEL, prompt="L1", thread_id="thread-lower", client_meta=meta(HOLDER_IP),
            wait_for_completion=True,
        )
        lower_key = mgr.queue._fastlane_priority_key(lower)
        assert lower in mgr.queue._staging
        assert mgr.queue._fastlane_strictly_higher(claimant_key, lower_key), (
            "setup: the parked claimant must outrank the staged request, or the test is vacuous"
        )

        assert mgr._governing_claim_priority_key_locked() != claimant_key, (
            "the parked claim governs the queue's hold-aside"
        )
        picked = await mgr.queue.pop_next(fastlane_policy=mgr._fastlane_pop_kwarg())
        assert picked is lower, (
            "the lower-ranked request was held aside for a claim that is parked in a live "
            f"inbox and needs no room (picked {picked})"
        )

        # Control: the same claim, now an ordinary waiting claim, governs and holds aside.
        mgr._unpark_fastlane_claim(claimant)
        assert not claim.get("parked_on"), "control setup: the claim must no longer be parked"
        lower2 = await mgr.submit(
            OTHER_MODEL, prompt="L2", thread_id="thread-lower-2", client_meta=meta(HOLDER_IP),
            wait_for_completion=True,
        )
        assert lower2 in mgr.queue._staging
        assert mgr._governing_claim_priority_key_locked() == claimant_key, (
            "control: a waiting claim of the highest rank must govern"
        )
        held = await mgr.queue.pop_next(fastlane_policy=mgr._fastlane_pop_kwarg())
        assert held is None and lower2 in mgr.queue._staging, (
            "control: a waiting higher-ranked claim must hold the lower-ranked request aside "
            f"(picked {held})"
        )


# --------------------------------------------------------------------------------------
# the engine dies while the holder's KV is being saved
# --------------------------------------------------------------------------------------

ENGINE_GONE = "engine process exited"
DONE_S = 8.0


def engine_dies_in_the_save(world, blocked, r):
    """Make the engine die while the blocked save runs.

    The process reports itself gone from now on, and the blocked save, once let go, fails the
    way a save against a dead engine does (it raises a connection error). A completion call that
    is made on the dead engine fails the same way a real one does. Returns the dead handle.
    """
    handle = r.handle
    assert handle is not None and handle.is_alive(), "setup: the engine must be up before it dies"
    handle.proc.poll.return_value = 1
    inner_save = world.mgr._save_slot_kv
    inner_complete = world.mgr._complete_fn

    async def save_on_dead_engine(*args, **kwargs):
        await inner_save(*args, **kwargs)
        raise ConnectionError(ENGINE_GONE)

    async def complete_on_dead_engine(slot, h):
        if not h.is_alive():
            raise ConnectionError(ENGINE_GONE)
        return await inner_complete(slot, h)

    world.mgr._save_slot_kv = save_on_dead_engine
    world.mgr._complete_fn = complete_on_dead_engine
    return handle


async def assert_claimant_not_stranded(world, claimant):
    """The claimant's request has reached an end, or at the very least is not lost.

    How it ended is not pinned: it may have been served again on a later engine or failed with
    an error. What is pinned is that the caller's future is done within a bound, that a failure
    is a visible error (an exception that says something, not a cancellation or a silent
    drop), that no queue or inbox still holds it, that its claim is gone from the registry and
    from the claims snapshot, and that nothing for another model was started or unloaded.
    """
    world.open_everything()
    fut = claimant.completion_future
    done, _ = await asyncio.wait({fut}, timeout=DONE_S)
    await world.drain_background()
    assert done, (
        "the claimant is stranded: its future is still pending after the engine died during "
        f"the swap save (slot state {claimant.state.name}, in staging "
        f"{claimant in world.mgr.queue._staging}, in an inbox "
        f"{in_some_inbox(world.mgr, claimant)}, residents {list(world.mgr._residents)})"
    )
    assert not fut.cancelled(), "the claimant's request was cancelled instead of answered or failed"
    exc = fut.exception()
    if exc is not None:
        assert isinstance(exc, Exception) and str(exc), (
            f"the claimant failed without a visible error: {exc!r}"
        )
    assert claimant not in world.mgr.queue._staging and not in_some_inbox(world.mgr, claimant), (
        "a finished claimant is still sitting in a queue or an inbox"
    )
    assert not world.has_claim("C"), "the claimant's claim is still registered"
    assert claimant.slot_id not in world.snapshot_ids(), (
        "the claimant's claim is still listed in the claims snapshot"
    )
    assert set(world.spawns) <= {MODEL} and set(world.unloads) <= {MODEL}, (
        f"an engine for another model was started or unloaded: spawns={world.spawns} "
        f"unloads={world.unloads}"
    )


async def test_an_engine_that_dies_during_the_swap_save_does_not_strand_the_claimant(tmp_path):
    """The holder's turn has ended, the higher-ranked request has been taken out of the inbox
    and the holder's KV save is blocked. The engine dies while the save runs: the process is
    gone and the save fails with a connection error. The save is best-effort, so the swap goes
    on, and the claimant's turn runs (or would run) on a dead engine. The claimant is the
    highest-ranked request and has not been answered, so however the death is handled its
    future must end within a bound: served again later, or failed with a visible error. It must
    not stay pending with nothing left to wake it, must not be dropped silently, must not keep
    a claim, and must not cost an unrelated engine."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r, blocked = await blocked_swap(world)
        engine_dies_in_the_save(world, blocked, r)
        blocked.go.set()
        await assert_claimant_not_stranded(world, claimant)


async def test_an_engine_that_dies_during_the_swap_save_and_is_reaped_does_not_strand_the_claimant(tmp_path):
    """Same death during the swap save, but the resident is also torn down while the save
    runs, which is how a make-room pass or the reaper removes a resident whose engine is gone:
    the resident is begun-unloaded (marked dead, removed from the registry, its detached
    teardown started) before the save is let go. The claimant is out of the inbox and has not
    been answered, so the unload cannot hand it back through the inbox. After the save is let
    go it must still end within a bound, served again later or failed with a visible error,
    with its claim released and no unrelated engine started or unloaded."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r, blocked = await blocked_swap(world)
        engine_dies_in_the_save(world, blocked, r)
        async with world.mgr._registry_lock:
            world.mgr._begin_unload_locked(r)
        blocked.go.set()
        await assert_claimant_not_stranded(world, claimant)


async def test_a_driver_that_is_torn_down_during_the_swap_save_does_not_strand_the_claimant(tmp_path):
    """Same window, and the resident's driver task itself is cancelled while the save runs (what
    a shutdown or a supervisor teardown does to a driver). The save is blocked on its event, so
    the cancellation lands inside the save. The claimant has been taken out of the inbox and its
    turn has not begun, so nothing else owns it: its future must still end within a bound, served
    again later or failed with a visible error, with its claim released."""
    async with build_world(tmp_path) as world:
        _h1, claimant, r, blocked = await blocked_swap(world)
        engine_dies_in_the_save(world, blocked, r)
        task = r.driver_task
        assert task is not None and not task.done(), "setup: the driver must be running"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait({task}, timeout=DONE_S)
        assert task.done(), "setup: the cancelled driver did not exit"
        blocked.go.set()
        await assert_claimant_not_stranded(world, claimant)
