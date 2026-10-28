"""A waiting request whose client has gone away must not cost the box anything.

Scope: requests that have not started (they wait in the staging queue or in the inbox of a
busy engine). The client's departure is modelled the way the route layer leaves it: the
request's ``disconnect_event`` is set and nothing else is touched, so the evicted flag is
still False. Every behaviour has a control with the same shape and a client that stays.

Four groups, all against the real manager:

* the periodic co-residence re-check must not start a further engine for such a request,
  must not count it as demand, and must not let it take the turn of a live request;
* the Fast Lane rules about claims: a claim of a request whose client is gone does not end
  the holder's grace ("a better claim waits in its own inbox"), does not designate an engine
  to unload, and moves to the next live request of the same client and model;
* the admission instant: a request that was live when the dispatcher picked it, whose client
  goes away before it is admitted, must not get an engine turn and must not leave a claim.
"""
from __future__ import annotations

import asyncio

import pytest
from _dead_client_recheck_and_claims_support import (
    CLAIMANT_IP,
    CLAIMANT_THREAD,
    HOLDER_MODEL,
    OTHER_MODEL,
    PROMPT_S,
    TAG,
    build_recheck_box,
    build_world,
    finish_box,
    hold_engine,
    parked_slot,
    pause_dispatcher,
    recheck_pass,
    staged_slot,
    start_busy_engine,
)
from _fastlane_fixture import wait_until

from turbohaul.slot import SlotState

SHAPES = [
    pytest.param("staged", id="staged_in_the_queue"),
    pytest.param("inbox", id="parked_in_a_busy_engines_inbox"),
]


async def _wait_in_shape(box, r1, shape, thread, *, event, age):
    """A request that waits behind the busy engine, in the queue or in the engine's inbox."""
    if shape == "staged":
        return await staged_slot(box, thread, event=event)
    slot = parked_slot(box, thread, event=event, age=age)
    r1.inbox.put_nowait(slot)
    return slot


# --------------------------------------------------------------------------------------
# the periodic co-residence re-check
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
async def test_recheck_starts_no_further_engine_for_a_request_whose_client_is_gone(
    tmp_path, monkeypatch, caplog, shape,
):
    """One engine is busy and a second one would fit. The only waiting request belongs to a client
    that has gone: no engine is reserved or started for it."""
    mgr, box = await build_recheck_box(tmp_path, monkeypatch, caplog)
    try:
        r1 = await start_busy_engine(box)
        gone = asyncio.Event()
        dead = await _wait_in_shape(box, r1, shape, "thread-gone", event=gone, age=30.0)
        gone.set()
        assert dead.is_evicted is False, "the departure leaves the evicted flag alone"
        spawned_before = len(box.spawned)

        await recheck_pass(box)

        reserved, dead_id = box.reserved_ids(), dead.slot_id
        assert dead_id not in reserved, "an engine was reserved for a request whose client is gone"
        assert len(box.spawned) == spawned_before, "an engine process was started for a request whose client is gone"
        assert len(mgr._instances_for(TAG)) == 1
    finally:
        await finish_box(box)


@pytest.mark.parametrize("shape", SHAPES)
async def test_recheck_does_not_count_a_request_whose_client_is_gone_as_demand(
    tmp_path, monkeypatch, caplog, shape,
):
    """With only that request waiting there is no demand: the re-check does not even ask the card
    picker, and writes no re-check line."""
    mgr, box = await build_recheck_box(tmp_path, monkeypatch, caplog)
    try:
        r1 = await start_busy_engine(box)
        gone = asyncio.Event()
        await _wait_in_shape(box, r1, shape, "thread-gone", event=gone, age=30.0)
        gone.set()
        box.picks.clear()

        await recheck_pass(box)

        assert box.picks == [], "the re-check probed for room because of a request whose client is gone"
        assert box.recheck_lines(caplog) == [], "the re-check acted on a request whose client is gone"
    finally:
        await finish_box(box)


@pytest.mark.parametrize("shape", SHAPES)
async def test_recheck_gives_the_further_engine_to_the_live_request_not_to_the_older_gone_one(
    tmp_path, monkeypatch, caplog, shape,
):
    """A request whose client is gone has waited longer than a live one. The further engine goes
    to the live request."""
    mgr, box = await build_recheck_box(tmp_path, monkeypatch, caplog)
    try:
        r1 = await start_busy_engine(box)
        gone = asyncio.Event()
        dead = await _wait_in_shape(box, r1, shape, "thread-gone", event=gone, age=40.0)
        live = await _wait_in_shape(box, r1, shape, "thread-live", event=asyncio.Event(), age=30.0)
        gone.set()

        await recheck_pass(box)

        reserved, dead_id, live_id = box.reserved_ids(), dead.slot_id, live.slot_id
        assert dead_id not in reserved, "the engine was reserved for the older gone request"
        assert reserved == [live_id], "the live request did not get the further engine"
        assert len(mgr._instances_for(TAG)) == 2
        assert live.is_evicted is False
    finally:
        await finish_box(box)


@pytest.mark.parametrize("shape", SHAPES)
async def test_recheck_control_starts_a_further_engine_for_a_request_whose_client_stays(
    tmp_path, monkeypatch, caplog, shape,
):
    """The control of the three tests above: the same request, the client stays. It is picked, an
    engine is reserved and started for it, and exactly one re-check line names its source."""
    mgr, box = await build_recheck_box(tmp_path, monkeypatch, caplog)
    try:
        r1 = await start_busy_engine(box)
        live = await _wait_in_shape(box, r1, shape, "thread-live", event=asyncio.Event(), age=30.0)
        box.picks.clear()

        await recheck_pass(box)

        reserved, live_id = box.reserved_ids(), live.slot_id
        assert reserved == [live_id], "the fitting engine was not reserved for the waiting request"
        assert len(mgr._instances_for(TAG)) == 2 and len(box.spawned) == 2
        assert box.picks, "the re-check never probed for room"
        (line,) = box.recheck_lines(caplog)
        assert f"source={'staged' if shape == 'staged' else 'inbox'}" in line, line
        assert live.is_evicted is False
    finally:
        await finish_box(box)


# --------------------------------------------------------------------------------------
# claims: a request whose client is gone is not a better claim, not a reason to unload,
# and hands its claim to the next live request
# --------------------------------------------------------------------------------------

async def _park_a_better_request(world, event):
    """The holder is mid-turn; a request of a better-ranked client waits in the holder's inbox with its claim."""
    h1, r = await hold_engine(world)
    c = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD, disconnect_event=event)
    await world.parked("C")
    assert world.claim_points_at(c), "setup: the parked request must hold a claim"
    assert r.active_slot is h1
    return h1, c, r


async def test_a_parked_request_whose_client_is_gone_does_not_end_the_holders_grace(tmp_path):
    """Grace ends early for a holder whose own inbox holds a LIVE claim that outranks it. When the
    client of that parked request is gone, the holder keeps its grace window and the parked request
    is not served."""
    async with build_world(tmp_path, grace_seconds=30) as world:
        gone = asyncio.Event()
        h1, c, _r = await _park_a_better_request(world, gone)
        gone.set()
        assert c.is_evicted is False, "the departure leaves the evicted flag alone"

        world.release("H1")
        await world.turn_done("H1")

        for _ in range(10):
            assert h1.state is SlotState.GRACE, f"the holder's grace ended early: {h1.state.name}"
            assert "C" not in world.order(), "the request whose client is gone was served"
            await asyncio.sleep(0.1)


async def test_a_parked_request_with_a_live_better_client_ends_the_holders_grace(tmp_path):
    """The control: the same parked request with its client still there ends the holder's grace
    at the turn boundary and is served next."""
    async with build_world(tmp_path, grace_seconds=30) as world:
        h1, c, _r = await _park_a_better_request(world, asyncio.Event())

        world.release("H1")
        await world.turn_done("H1")

        assert h1.state is SlotState.POPPED, f"the holder waited for follow-ups: {h1.state.name}"
        assert await world.started("C", timeout=3.0), "the live better request was not served"


async def _wait_for_room(world, event, label="C", thread=CLAIMANT_THREAD):
    """The holder is mid-turn on the only engine and the dispatcher is busy elsewhere. A request of a
    better-ranked client for another model waits in the queue with its claim."""
    h1, r = await hold_engine(world)
    await pause_dispatcher(world.mgr)
    c = await world.send(label, CLAIMANT_IP, thread, model=OTHER_MODEL, disconnect_event=event)
    assert c in world.mgr.queue.staging_snapshot(), "setup: the request must wait in the queue"
    assert world.claim_points_at(c), "setup: the waiting request must hold a claim"
    return h1, c, r


async def test_no_engine_is_designated_to_unload_for_a_request_whose_client_is_gone(tmp_path):
    """Rule: the engine to unload is chosen fresh, and only a live claim can designate one. With the
    only waiting claim belonging to a gone client, the engine is not designated and its turn end
    gives it the normal grace window."""
    async with build_world(tmp_path, grace_seconds=30) as world:
        gone = asyncio.Event()
        h1, c, r = await _wait_for_room(world, gone)
        gone.set()
        assert c.is_evicted is False, "the departure leaves the evicted flag alone"
        async with world.mgr._registry_lock:
            assert world.mgr._is_designated_unload_target_locked(r) is False

        world.release("H1")
        await world.turn_done("H1")

        assert h1.state is SlotState.GRACE, f"the holder lost its grace for a gone client: {h1.state.name}"


async def test_an_engine_is_designated_to_unload_for_a_live_waiting_request(tmp_path):
    """The control: the same waiting claim with its client present designates the engine, which then
    gives up its grace at its turn end."""
    async with build_world(tmp_path, grace_seconds=30) as world:
        h1, c, r = await _wait_for_room(world, asyncio.Event())
        async with world.mgr._registry_lock:
            assert world.mgr._is_designated_unload_target_locked(r) is True

        world.release("H1")
        await world.turn_done("H1")

        assert h1.state is SlotState.POPPED, f"the designated engine entered grace: {h1.state.name}"


async def _two_waiting_requests(world, first_event, second_event):
    """Two waiting requests of one client for one model; the claim points at the first."""
    h1, r = await hold_engine(world)
    await pause_dispatcher(world.mgr)
    c1 = await world.send("C1", CLAIMANT_IP, "thread-first", model=OTHER_MODEL, disconnect_event=first_event)
    c2 = await world.send("C2", CLAIMANT_IP, "thread-second", model=OTHER_MODEL, disconnect_event=second_event)
    assert world.claim_points_at(c1) and not world.claim_points_at(c2), "setup: the claim points at the first"
    assert len(world.mgr._fastlane_claims) == 1, "setup: one claim for one client and model"
    return h1, c1, c2, r


async def test_the_claim_moves_to_the_next_live_request_when_the_one_it_points_at_goes_away(tmp_path):
    """The first request's client goes away while another request of the same client and model still
    waits. The claim that the grace and unload decisions read points at the live one, the claims
    listing shows the live one, and the claim still designates the engine to unload."""
    async with build_world(tmp_path, grace_seconds=30) as world:
        first_gone = asyncio.Event()
        h1, c1, c2, _r = await _two_waiting_requests(world, first_gone, asyncio.Event())
        first_gone.set()

        world.release("H1")
        await world.turn_done("H1")

        assert world.claim_points_at(c2), "the claim does not point at the live request"
        assert not world.claim_points_at(c1), "the claim still points at the request whose client is gone"
        rows = world.mgr.fastlane_claims_snapshot()
        assert [row["slot_id"] for row in rows] == [c2.slot_id]
        assert h1.state is SlotState.POPPED, f"the live claim did not designate the engine: {h1.state.name}"


async def test_the_claim_stays_on_the_first_request_while_its_client_is_there(tmp_path):
    """The control: with both clients present the claim stays on the first (equally ranked) request."""
    async with build_world(tmp_path, grace_seconds=30) as world:
        h1, c1, c2, _r = await _two_waiting_requests(world, asyncio.Event(), asyncio.Event())

        world.release("H1")
        await world.turn_done("H1")

        assert world.claim_points_at(c1) and not world.claim_points_at(c2)
        assert [row["slot_id"] for row in world.mgr.fastlane_claims_snapshot()] == [c1.slot_id]
        assert h1.state is SlotState.POPPED


async def test_the_claim_is_released_when_no_live_request_is_left(tmp_path):
    """Both requests' clients go away: no claim remains, and the engine keeps its grace."""
    async with build_world(tmp_path, grace_seconds=30) as world:
        first_gone, second_gone = asyncio.Event(), asyncio.Event()
        h1, c1, c2, _r = await _two_waiting_requests(world, first_gone, second_gone)
        first_gone.set()
        second_gone.set()

        world.release("H1")
        await world.turn_done("H1")

        assert h1.state is SlotState.GRACE, f"the holder lost its grace for gone clients: {h1.state.name}"
        assert world.mgr.fastlane_claims_snapshot() == []
        assert not world.claim_points_at(c1) and not world.claim_points_at(c2)
        assert not world.mgr._fastlane_claims


# --------------------------------------------------------------------------------------
# the admission instant
# --------------------------------------------------------------------------------------

def _hang_up_after_the_pick(world, event):
    """Wrap the queue's pick so that the client goes away right after the dispatcher has been handed
    its request and before it routes it. Records nothing, calls through, returns the pick unchanged."""
    queue = world.mgr.queue
    real_pop = queue.pop_next

    async def pop_next(*a, **kw):
        slot = await real_pop(*a, **kw)
        if slot is not None and slot.disconnect_event is event:
            event.set()
        return slot

    queue.pop_next = pop_next


async def test_a_request_whose_client_leaves_at_the_admission_instant_gets_no_engine_turn(tmp_path):
    """The dispatcher picks a live request for an empty box; the client goes away before the engine
    is reserved. No turn runs for it and no claim stays behind."""
    async with build_world(tmp_path, grace_seconds=1) as world:
        gone = asyncio.Event()
        _hang_up_after_the_pick(world, gone)
        c = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD, disconnect_event=gone)

        outcome = await world.settled("C", timeout=1.5)

        assert gone.is_set(), "setup: the hang-up at the pick never happened"
        assert outcome != "started", "an engine turn ran for a request whose client had gone"
        assert "C" not in world.order()
        assert world.claims_of(c) == [], "a claim stayed behind for a request whose client had gone"


async def test_a_request_whose_client_leaves_at_the_admission_instant_gets_no_engine(tmp_path):
    """Same instant: no engine process is started and the engine reserve is not called for it."""
    async with build_world(tmp_path, grace_seconds=1) as world:
        gone = asyncio.Event()
        _hang_up_after_the_pick(world, gone)
        await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD, disconnect_event=gone)

        await world.settled("C", timeout=1.5)
        await asyncio.sleep(0.3)

        assert gone.is_set(), "setup: the hang-up at the pick never happened"
        assert not world.reserved("C"), "an engine was reserved for a request whose client had gone"
        assert world.spawns == [], "an engine process was started for a request whose client had gone"


async def test_a_request_whose_client_stays_is_admitted_at_the_admission_instant(tmp_path):
    """The control: the same pick, the client stays. An engine is reserved, started, runs the turn,
    and the claim is released as admitted."""
    async with build_world(tmp_path, grace_seconds=1) as world:
        c = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD, disconnect_event=asyncio.Event())

        outcome = await world.settled("C", timeout=3.0)

        assert outcome == "started"
        assert world.reserved("C") and world.spawns == [HOLDER_MODEL]
        assert world.release_reasons("C") == ["admitted"]
        assert world.claims_of(c) == []


async def _routed_to_a_busy_engine(world, event, *, hang_up):
    """The holder is mid-turn; the dispatcher picks a request for the same model and routes it into the
    holder's inbox. With ``hang_up`` the client goes away at the pick."""
    h1, _r = await hold_engine(world)
    if hang_up:
        _hang_up_after_the_pick(world, event)
    c = await world.send("C", CLAIMANT_IP, CLAIMANT_THREAD, disconnect_event=event)
    if hang_up:
        # Where the request ends up after the hang-up is not this test's business (it may be dropped
        # at once); the setup only needs the hang-up at the pick to have happened.
        await wait_until(event.is_set, timeout=PROMPT_S)
    else:
        await world.parked("C")
    return h1, c


async def test_a_request_whose_client_leaves_at_the_pick_is_not_served_at_the_next_turn_boundary(tmp_path):
    """The request is routed behind a busy engine and its client goes away at the pick. When the
    engine's turn ends, no turn runs for it, and its claim is released."""
    async with build_world(tmp_path, grace_seconds=1) as world:
        gone = asyncio.Event()
        _h1, c = await _routed_to_a_busy_engine(world, gone, hang_up=True)
        assert gone.is_set(), "setup: the hang-up at the pick never happened"

        world.release("H1")
        outcome = await world.settled("C", timeout=5.0)

        assert outcome != "started", "an engine turn ran for a request whose client had gone"
        assert "C" not in world.order()
        assert world.claims_of(c) == [], "a claim stayed behind for a request whose client had gone"


async def test_a_request_routed_behind_a_busy_engine_is_served_when_its_client_stays(tmp_path):
    """The control: the same routing, the client stays: served at the turn boundary."""
    async with build_world(tmp_path, grace_seconds=1) as world:
        h1, c = await _routed_to_a_busy_engine(world, asyncio.Event(), hang_up=False)

        world.release("H1")

        assert await world.started("C", timeout=5.0), "the live request was not served"
        assert world.release_reasons("C") == ["admitted"]
        assert h1.state in (SlotState.GRACE, SlotState.POPPED)
