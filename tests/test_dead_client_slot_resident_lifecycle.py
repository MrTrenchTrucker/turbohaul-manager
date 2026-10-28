"""What happens to a waiting request whose client hangs up, across the life of an engine.

The take-path tests pin that a request whose client is gone is never taken for service. These
tests pin the other half: the request is removed where it waits as soon as the hang-up is seen,
and removing it never harms the engine or the box.

* an engine whose driver is parked on its inbox must not lose its last queued request: a producer
  flips an idle engine to ACTIVE before it puts, and only the driver's own loop flips it back, so
  an emptied inbox would leave the engine ACTIVE and unable to idle-unload or be chosen to make room;
* an engine whose driver is busy with a turn does lose the request from its inbox, and nothing runs
  for it;
* an engine still loading, whose first request's client goes before the driver takes it, is never
  started and its reservation is freed;
* a request whose client leaves while the dispatcher waits for the registry lock gets no engine;
* the hang-up signal reaches a request that waits in the acceptance buffer, once, and never touches
  a request that has already started;
* the disconnect watcher the routes share passes the hang-up on to the manager.

Every behaviour has a control with the same shape and a client that stays (or a hang-up of some
other client), so a path that simply refuses or evicts everything cannot pass. One process, asyncio
only, bounded waits, no network.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from _dead_client_resident_lifecycle_support import (
    IP_A,
    IP_B,
    MODEL,
    bring_up_driver,
    build_box,
    claims_of,
    settle_background,
    wait_until,
    waiting_slot,
)

from turbohaul import queue_liveness
from turbohaul.api import chat_completion
from turbohaul.manager import ResidentState
from turbohaul.slot import SlotEvictedError, SlotState

LONG_KEEP_ALIVE_S = 30
SHORT_KEEP_ALIVE_S = 1
PROMPT_S = 3.0


def _failed_with_eviction(slot):
    fut = slot.completion_future
    return fut.done() and not fut.cancelled() and isinstance(fut.exception(), SlotEvictedError)


# --------------------------------------------------------------------------------------
# L1. An engine whose driver is parked on its inbox
# --------------------------------------------------------------------------------------

async def _put_into_a_parked_inbox(d, event, *, name="rider"):
    """The engine is idle with its driver parked. A producer wakes it and puts a request whose
    client may leave. Returns before the driver has had a chance to run."""
    rider = waiting_slot(name, 1.0, ip=IP_A, event=event)
    await d.produce(rider)
    d.park_claim(rider)
    assert d.r.state is ResidentState.ACTIVE and d.r.active_slot is None, "setup: a parked, woken engine"
    assert d.in_inbox(rider), "setup: the request must wait in the inbox"
    return rider


async def test_a_gone_client_in_a_parked_engines_inbox_is_failed_and_released_but_left_in_the_inbox(
    tmp_path, monkeypatch,
):
    """The request is flagged, its claim released and its future failed with the eviction error, but
    it stays in the inbox of the parked driver, so the inbox is not emptied under the driver.

    Fails when the request is taken out of a parked driver's inbox, or is not flagged, released or failed."""
    d = await bring_up_driver(tmp_path, monkeypatch, keep_alive_s=LONG_KEEP_ALIVE_S)
    try:
        event = asyncio.Event()
        rider = await _put_into_a_parked_inbox(d, event)

        evicted = await queue_liveness.evict_unstarted(d.mgr, rider)

        assert evicted is True
        assert event.is_set() and rider.is_evicted, "the request must be flagged before anything else"
        assert _failed_with_eviction(rider), "the request's future was not failed with the eviction error"
        assert [reason for s, reason in d.releases if s is rider] == ["disconnected"]
        assert claims_of(d.mgr, rider) == [], "the claim of a request whose client is gone was kept"
        assert d.in_inbox(rider), "the last queued request was taken away from a parked driver"
    finally:
        await d.stop()


async def test_a_parked_engine_whose_rider_hangs_up_does_not_wedge(tmp_path, monkeypatch):
    """With a long idle window, the engine must leave ACTIVE promptly after the hang-up: back to
    IDLE_EVICTABLE with an empty inbox, where it can be chosen to make room.

    Fails when the engine stays ACTIVE with nothing to serve (it would hold its memory until the
    idle window ends, here thirty seconds)."""
    d = await bring_up_driver(tmp_path, monkeypatch, keep_alive_s=LONG_KEEP_ALIVE_S)
    try:
        rider = await _put_into_a_parked_inbox(d, asyncio.Event())

        await queue_liveness.evict_unstarted(d.mgr, rider)

        await wait_until(
            lambda: d.r.state is ResidentState.IDLE_EVICTABLE and d.r.inbox.empty(), timeout=PROMPT_S,
        )
        assert d.task.done() is False, "the driver must still be alive and parked"
    finally:
        await d.stop()


async def test_a_parked_engine_whose_rider_hangs_up_still_idle_unloads(tmp_path, monkeypatch):
    """With a short idle window the engine idles out: its driver finishes and the engine is gone from
    the registry.

    Fails when the engine can never idle-unload after the hang-up."""
    d = await bring_up_driver(tmp_path, monkeypatch, keep_alive_s=SHORT_KEEP_ALIVE_S)
    try:
        rider = await _put_into_a_parked_inbox(d, asyncio.Event())

        await queue_liveness.evict_unstarted(d.mgr, rider)

        await wait_until(lambda: d.task.done(), timeout=8.0)
        assert d.r.state is ResidentState.DEAD
        assert d.r.resident_key not in d.mgr._residents
    finally:
        await d.stop()


async def test_a_parked_engine_serves_a_rider_when_only_another_client_hangs_up(tmp_path, monkeypatch):
    """Control of the three tests above: the same rider, the same inbox. A different client's hang-up
    changes nothing for it: it is not flagged or failed, its claim is released as admitted, and it is served."""
    d = await bring_up_driver(tmp_path, monkeypatch, keep_alive_s=LONG_KEEP_ALIVE_S)
    try:
        rider = await _put_into_a_parked_inbox(d, asyncio.Event())

        queue_liveness.notify_client_gone(d.mgr, asyncio.Event())
        await settle_background(d.mgr)

        await wait_until(lambda: "rider" in d.served_ids(), timeout=PROMPT_S)
        assert rider.is_evicted is False and not rider.completion_future.done()
        assert [reason for s, reason in d.releases if s is rider] == ["admitted"]
        assert d.audited == []
    finally:
        await d.stop()


# --------------------------------------------------------------------------------------
# L2. An engine whose driver is busy with a turn
# --------------------------------------------------------------------------------------

async def _busy_engine_with_two_riders(tmp_path, monkeypatch, gone_event):
    """The engine is mid-turn on the first request. A rider whose client may leave waits in its inbox,
    and a live rider of another client waits behind it."""
    d = await bring_up_driver(tmp_path, monkeypatch, keep_alive_s=LONG_KEEP_ALIVE_S, hold_anchor=True)
    dead = waiting_slot("dead", 1.0, ip=IP_A, event=gone_event)
    live = waiting_slot("live", 2.0, ip=IP_B, event=asyncio.Event())
    for s in (dead, live):
        await d.produce(s)
        d.park_claim(s)
    assert d.r.state is ResidentState.ACTIVE and d.r.active_slot is not None, "setup: the engine is mid-turn"
    assert d.in_inbox(dead) and d.in_inbox(live)
    return d, dead, live


async def test_a_gone_client_in_a_busy_engines_inbox_is_removed_and_nothing_runs_for_it(
    tmp_path, monkeypatch,
):
    """The hang-up removes the request from the busy engine's inbox at once, releases its claim and
    fails its future. The live rider behind it stays. When the turn ends, only the live rider runs.

    Fails when the request stays in the inbox, keeps its claim, or gets a turn."""
    gone = asyncio.Event()
    d, dead, live = await _busy_engine_with_two_riders(tmp_path, monkeypatch, gone)
    try:
        queue_liveness.notify_client_gone(d.mgr, gone)
        assert gone.is_set(), "the event is set synchronously, before any await"
        await wait_until(lambda: dead.completion_future.done(), timeout=PROMPT_S)

        assert _failed_with_eviction(dead)
        assert dead.is_evicted
        assert not d.in_inbox(dead), "the request of a gone client stayed in a busy engine's inbox"
        assert d.in_inbox(live), "the live rider was lost"
        assert claims_of(d.mgr, dead) == [] and claims_of(d.mgr, live), "claims after the hang-up are wrong"
        assert d.served_ids() == ["anchor"], "nothing may run before the held turn ends"

        d.hold("anchor").set()
        await wait_until(lambda: d.served_ids() == ["anchor", "live"], timeout=PROMPT_S)
        await wait_until(
            lambda: d.r.state is ResidentState.IDLE_EVICTABLE and d.r.inbox.empty(), timeout=PROMPT_S,
        )
        await settle_background(d.mgr)
        assert "dead" not in d.served_ids(), "a turn ran for a request whose client had gone"
        assert len(d.audited) == 1, "the failed request must be audited exactly once"
    finally:
        await d.stop()


async def test_a_busy_engine_keeps_both_riders_when_only_another_client_hangs_up(tmp_path, monkeypatch):
    """Control: the same inbox, a hang-up of some other client. Both riders stay, keep their claims,
    and run in order."""
    d, dead, live = await _busy_engine_with_two_riders(tmp_path, monkeypatch, asyncio.Event())
    try:
        queue_liveness.notify_client_gone(d.mgr, asyncio.Event())
        await settle_background(d.mgr)

        assert d.in_inbox(dead) and d.in_inbox(live)
        assert not dead.completion_future.done() and not live.completion_future.done()
        assert claims_of(d.mgr, dead) and claims_of(d.mgr, live)

        d.hold("anchor").set()
        await wait_until(lambda: d.served_ids() == ["anchor", "dead", "live"], timeout=PROMPT_S)
        assert d.audited == []
    finally:
        await d.stop()


# --------------------------------------------------------------------------------------
# L3. An engine still loading: its first request's client goes
# --------------------------------------------------------------------------------------

async def _reserved_engine_with_a_first_request(box, event):
    """The dispatcher has picked a request for an empty box and reserved an engine for it. The
    engine's driver has not yet taken the request."""
    first = await box.send("first", IP_A, event=event)
    assert (await box.pop()) is first, "setup: the request must be the one the dispatcher picks"
    async with box.mgr._registry_lock:
        r = await box.mgr._reserve_and_start_locked(first)
    assert r is not None and r.state is ResidentState.RESERVED_LOADING, "setup: an engine reserved"
    assert box.mgr._inbox_holds_slot(r, first) and box.spawns == [], "setup: nothing started yet"
    assert claims_of(box.mgr, first), "setup: the request must hold a claim"
    return first, r


@pytest.mark.parametrize("how", ["event_only", "evicted_at_source"])
async def test_no_engine_is_started_for_a_first_request_whose_client_goes_and_the_reservation_is_freed(
    tmp_path, how,
):
    """The client goes before the driver takes the first request, either only by its event or through
    the eviction at the source. The engine is never started, the reservation is freed, the claim is
    released, and a later request for the same model is reserved and served with no budget leaked.

    Fails when an engine process is started, or the reservation or the claim is left behind."""
    async with build_box(tmp_path, dispatcher=False) as box:
        mgr = box.mgr
        event = asyncio.Event()
        first, r = await _reserved_engine_with_a_first_request(box, event)

        if how == "event_only":
            event.set()
        else:
            assert await queue_liveness.evict_unstarted(mgr, first) is True
        done, _pending = await asyncio.wait({r.driver_task}, timeout=PROMPT_S)

        assert done, "the driver went on to start an engine for a request whose client had gone"
        await settle_background(mgr)
        assert box.spawns == [], "an engine process was started for a request whose client had gone"
        assert r.state is ResidentState.DEAD and r.resident_key not in mgr._residents
        assert mgr._model_residents() == [], "the reservation was not freed"
        assert _failed_with_eviction(first)
        assert claims_of(mgr, first) == [] and "admitted" not in box.release_reasons("first")
        assert len(box.audited) == 1 and mgr._unload_count == 1, "the request must be failed exactly once"

        box.start_dispatcher()
        live = await box.send("live", IP_A, event=asyncio.Event())
        assert await box.started("live"), "a later request was not served: the budget leaked"
        assert box.spawns == [MODEL]
        box.release("live")
        result = await asyncio.wait_for(live.completion_future, timeout=PROMPT_S)
        assert result["ok"] is True


async def test_a_first_request_whose_client_stays_starts_the_engine_and_is_served(tmp_path):
    """Control of the test above: the same reservation, the client stays. The engine is started, the
    request is served, and its claim is released as admitted."""
    async with build_box(tmp_path, dispatcher=False) as box:
        first, r = await _reserved_engine_with_a_first_request(box, asyncio.Event())

        await wait_until(lambda: "first" in box.order(), timeout=PROMPT_S)

        assert box.spawns == [MODEL]
        assert r.state is not ResidentState.DEAD
        assert box.release_reasons("first") == ["admitted"]
        assert claims_of(box.mgr, first) == []
        box.release("first")
        result = await asyncio.wait_for(first.completion_future, timeout=PROMPT_S)
        assert result["ok"] is True


# --------------------------------------------------------------------------------------
# L4. The dispatcher waits for the registry lock
# --------------------------------------------------------------------------------------

async def _dispatcher_waits_for_the_lock(box, event):
    """Send a request while another task holds the registry lock from the moment the dispatcher has
    picked it until the test lets go, so that the dispatcher reaches the lock and waits for it.

    Only a holder is added: the queue's own call just before the lock is wrapped to start the holder
    and wait until it has the lock, then returns its result unchanged. Returns (request, holder task,
    release event). The request has been picked (it is out of the queue) when this returns.
    """
    mgr = box.mgr
    held, release = asyncio.Event(), asyncio.Event()
    state = {}
    real = mgr.queue.listed_waiter_model_tags

    async def holder():
        async with mgr._registry_lock:
            held.set()
            await release.wait()

    async def listed(*a, **kw):
        result = await real(*a, **kw)
        if "task" not in state:
            state["task"] = asyncio.create_task(holder())
            await held.wait()
        return result

    mgr.queue.listed_waiter_model_tags = listed
    slot = await box.send("C", IP_A, event=event)
    await wait_until(lambda: held.is_set(), timeout=PROMPT_S)
    assert slot not in mgr.queue.waiting_snapshot(), "setup: the dispatcher must already hold the request"
    assert box.reserve_calls == [], "setup: the engine must not be reserved yet"
    return slot, state["task"], release


async def test_a_client_that_leaves_while_the_dispatcher_waits_for_the_lock_gets_no_engine(tmp_path):
    """The dispatcher has picked a live request for an empty box and waits for the registry lock. The
    client goes away meanwhile. When the lock is free: no reserve, no engine, no turn, claim released,
    request failed with the eviction error.

    Fails when the lock is taken and the request is reserved an engine, or its claim is left behind."""
    async with build_box(tmp_path) as box:
        event = asyncio.Event()
        slot, holder, release = await _dispatcher_waits_for_the_lock(box, event)

        event.set()
        release.set()
        await holder
        outcome = await box.settled("C", timeout=1.5)
        await asyncio.sleep(0.3)

        assert outcome != "started", "an engine turn ran for a request whose client had gone"
        assert not box.reserved("C"), "an engine was reserved for a request whose client had gone"
        assert box.spawns == [], "an engine process was started for a request whose client had gone"
        assert _failed_with_eviction(slot)
        assert box.release_reasons("C") == ["disconnected"]
        assert claims_of(box.mgr, slot) == []


async def test_a_client_that_stays_while_the_dispatcher_waits_for_the_lock_is_admitted(tmp_path):
    """Control: the same wait for the lock, the client stays. The request gets an engine and its turn,
    and its claim is released as admitted."""
    async with build_box(tmp_path) as box:
        slot, holder, release = await _dispatcher_waits_for_the_lock(box, asyncio.Event())

        release.set()
        await holder

        assert await box.settled("C", timeout=PROMPT_S) == "started"
        assert box.reserved("C") and box.spawns == [MODEL]
        assert box.release_reasons("C") == ["admitted"]
        assert claims_of(box.mgr, slot) == []


# --------------------------------------------------------------------------------------
# L5. The hang-up signal at manager level
# --------------------------------------------------------------------------------------

async def _full_staging_with_a_buffered_request(box, buffered_event):
    """No dispatcher; a staging queue of one. The first request fills it, the second waits in the
    acceptance buffer holding a claim."""
    first = await box.send("first", IP_A, event=asyncio.Event())
    buffered = await box.send("buffered", IP_B, event=buffered_event)
    assert first.state is SlotState.STAGED and buffered.state is SlotState.ACCEPT_BUFFER, "setup: states"
    assert buffered not in box.mgr.queue.staging_snapshot() and buffered in box.mgr.queue.waiting_snapshot()
    box.mgr._register_fastlane_claim_locked(buffered, "waiting_for_room")
    assert claims_of(box.mgr, buffered), "setup: the buffered request must hold a claim"
    return first, buffered


async def test_a_hang_up_removes_a_request_from_the_acceptance_buffer_and_fails_it_once(tmp_path):
    """The request waits in the acceptance buffer, not in staging. The hang-up removes it, releases
    its claim and fails its future with the eviction error; the live request in staging is untouched.
    Two notifications that race for the registry lock still fail it, audit it and count it once.

    Fails when the buffer is not searched, or the request is failed or counted twice."""
    async with build_box(tmp_path, dispatcher=False, staging_depth=1) as box:
        mgr = box.mgr
        gone = asyncio.Event()
        first, buffered = await _full_staging_with_a_buffered_request(box, gone)

        async with mgr._registry_lock:
            queue_liveness.notify_client_gone(mgr, gone)
            queue_liveness.notify_client_gone(mgr, gone)
            assert gone.is_set()
            await asyncio.sleep(0.05)
        await settle_background(mgr)

        assert buffered not in mgr.queue.waiting_snapshot(), "the request stayed in the acceptance buffer"
        assert _failed_with_eviction(buffered) and buffered.is_evicted
        assert claims_of(mgr, buffered) == []
        assert len(box.audited) == 1 and mgr._unload_count == 1, "the request was failed more than once"
        assert first in mgr.queue.staging_snapshot() and not first.completion_future.done()
        assert first.is_evicted is False

        queue_liveness.notify_client_gone(mgr, gone)
        await settle_background(mgr)
        assert len(box.audited) == 1 and mgr._unload_count == 1, "a later notification must do nothing"


async def test_a_hang_up_of_another_client_leaves_the_buffered_request_alone(tmp_path):
    """Control: the same buffer, a different client's hang-up. The buffered request stays, keeps its
    claim, and nothing is failed or audited."""
    async with build_box(tmp_path, dispatcher=False, staging_depth=1) as box:
        mgr = box.mgr
        first, buffered = await _full_staging_with_a_buffered_request(box, asyncio.Event())

        queue_liveness.notify_client_gone(mgr, asyncio.Event())
        await settle_background(mgr)

        assert buffered in mgr.queue.waiting_snapshot() and first in mgr.queue.waiting_snapshot()
        assert not buffered.completion_future.done() and buffered.is_evicted is False
        assert claims_of(mgr, buffered)
        assert box.audited == [] and mgr._unload_count == 0


async def test_a_hang_up_does_not_touch_a_request_that_has_already_started(tmp_path):
    """The request is mid-turn on the engine when its client's hang-up is reported, both by a direct
    eviction call and by the notification. It is not flagged, failed, audited or released again, and
    its turn ends with its answer. (The notification does not even find a started request, since
    none waits in a queue or an inbox; the direct call is what the state check guards.)

    Fails when a started request is evicted."""
    async with build_box(tmp_path) as box:
        event = asyncio.Event()
        slot = await box.send("H", IP_A, event=event)
        assert await box.started("H"), "setup: the turn never started"

        assert await queue_liveness.evict_unstarted(box.mgr, slot) is False
        assert not event.is_set() and slot.is_evicted is False, "the eviction touched a request mid-turn"
        assert box.release_reasons("H") == ["admitted"], "the claim of a request mid-turn was released again"
        queue_liveness.notify_client_gone(box.mgr, event)
        await settle_background(box.mgr)

        assert not slot.completion_future.done(), "a request mid-turn was failed"
        assert slot.is_evicted is False, "a request mid-turn was flagged evicted"
        assert slot.state in (SlotState.LOADING, SlotState.ACTIVE)
        assert box.audited == [] and box.mgr._unload_count == 0
        box.release("H")
        result = await asyncio.wait_for(slot.completion_future, timeout=PROMPT_S)
        assert result["ok"] is True


# --------------------------------------------------------------------------------------
# L6. The disconnect watcher the routes share
# --------------------------------------------------------------------------------------

class _Request:
    """A request whose client is connected for the first ``connected_polls`` polls, then gone."""

    def __init__(self, manager="absent", connected_polls=0):
        if manager != "absent":
            self.app = SimpleNamespace(state=SimpleNamespace(manager=manager))
        self.polls = 0
        self.connected_polls = connected_polls

    async def is_disconnected(self):
        self.polls += 1
        return self.polls > self.connected_polls


@pytest.fixture
def fast_polling(monkeypatch):
    monkeypatch.setattr(chat_completion, "_DISCONNECT_POLL_INTERVAL_S", 0.01)


async def _run_watcher(request, event):
    """Run the shared watcher to its end; it must end by itself once the client has gone."""
    watcher = asyncio.create_task(chat_completion.watch_disconnect(request, event))
    done, _pending = await asyncio.wait({watcher}, timeout=PROMPT_S)
    if not done:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
    assert done, "the watcher kept polling after the client had gone"
    watcher.result()


def _record_notifications(monkeypatch):
    calls = []
    real = queue_liveness.notify_client_gone

    def spy(host, event):
        calls.append((host, event))
        return real(host, event)

    monkeypatch.setattr(queue_liveness, "notify_client_gone", spy)
    return calls


async def test_the_watcher_tells_the_manager_once_and_the_waiting_request_is_removed(
    tmp_path, monkeypatch, fast_polling,
):
    """The route's watcher sees the client go: the event is set, the manager is told once with that
    event, and the request that carries it is taken out of the queue and failed.

    Fails when the watcher only sets the event."""
    async with build_box(tmp_path, dispatcher=False) as box:
        calls = _record_notifications(monkeypatch)
        event = asyncio.Event()
        slot = await box.send("W", IP_A, event=event)
        request = _Request(manager=box.mgr, connected_polls=2)

        await _run_watcher(request, event)
        await settle_background(box.mgr)

        assert event.is_set()
        assert calls == [(box.mgr, event)], "the manager must be told exactly once, with the request's event"
        assert slot not in box.mgr.queue.waiting_snapshot()
        assert _failed_with_eviction(slot)


async def test_the_watcher_tells_nobody_while_the_client_is_connected(tmp_path, monkeypatch, fast_polling):
    """Control: while the client stays, the watcher keeps polling and says nothing: no event, no
    notification, the request keeps waiting."""
    async with build_box(tmp_path, dispatcher=False) as box:
        calls = _record_notifications(monkeypatch)
        event = asyncio.Event()
        slot = await box.send("W", IP_A, event=event)
        request = _Request(manager=box.mgr, connected_polls=10_000)

        watcher = asyncio.create_task(chat_completion.watch_disconnect(request, event))
        try:
            await wait_until(lambda: request.polls >= 5, timeout=PROMPT_S)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        await settle_background(box.mgr)

        assert not event.is_set() and calls == []
        assert slot in box.mgr.queue.staging_snapshot() and not slot.completion_future.done()


@pytest.mark.parametrize("request_kind", ["no_app", "app_without_manager"])
async def test_the_watcher_without_a_manager_sets_the_event_and_ends(
    monkeypatch, fast_polling, request_kind,
):
    """A request that cannot reach a manager (the embeddings route and tests build such requests):
    the watcher sets the event, tells nobody, does not raise and does not keep polling.

    Fails when it raises, or spins after the client has gone."""
    calls = _record_notifications(monkeypatch)
    event = asyncio.Event()
    request = _Request()
    if request_kind == "app_without_manager":
        request.app = SimpleNamespace(state=SimpleNamespace())

    await _run_watcher(request, event)

    assert event.is_set() and calls == []
    assert request.polls == 1, "the watcher kept polling after it had seen the client go"
