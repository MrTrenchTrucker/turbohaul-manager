"""Places where a request whose client has gone must be dropped, and that the other dead-client tests do not reach.

Three behaviours, each with a control of the same shape in which the client stays:

* A non-streaming chat request whose handler task is cancelled while the request still waits
  behind a busy engine (the OpenAI route and the Ollama-style route). The disconnect watcher
  never fires here (it reports a connected client), so the only thing that can tell the manager
  is the handler's own cancellation path. The waiting request must be flagged, removed from where
  it waits, lose its claim, and get no engine turn.
* The co-residence re-check, when the client leaves after the request was gathered as demand and
  before it is taken out of its container (staged in the queue, or parked in a busy engine's
  inbox). The departure is signalled by the disconnect event alone; no engine may be reserved.
* The "still waiting" test the grace gate and the hand-off registry use: a handed-off request
  counts as waiting only while its client is there.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace

import pytest
from _co_residence_fakes import FakeHost, FakeResident, FakeSlot
from _dead_client_routes_support import (
    LOOPBACK_IP,
    MODEL,
    OBSERVE_WINDOW_S,
    SERVED_WINDOW_S,
    end_holder_turn_and_watch,
    route_world,
)
from _fastlane_fixture import wait_until

from turbohaul import co_residence
from turbohaul.api import chat_completion
from turbohaul.manager import TurbohaulManager
from turbohaul.slot import Slot, SlotState

ROUTES = [
    pytest.param(chat_completion.openai_chat_completions, id="openai_route"),
    pytest.param(chat_completion.ollama_chat, id="ollama_route"),
]
_STARTED = (SlotState.LOADING, SlotState.ACTIVE, SlotState.GRACE, SlotState.POPPED)
_SETTLE_S = 0.4


# --------------------------------------------------------------------------------------
# a cancelled non-streaming handler
# --------------------------------------------------------------------------------------

class _ConnectedRequest:
    """A request whose connection stays open: the disconnect watcher never has anything to report."""

    def __init__(self, manager) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(manager=manager))
        self.client = SimpleNamespace(host=LOOPBACK_IP)
        self.headers = {}

    async def is_disconnected(self) -> bool:
        return False


def _payload():
    return {"model": MODEL, "stream": False, "messages": [{"role": "user", "content": "hello"}]}


async def _park_behind_the_busy_engine(world, handler):
    """Run ``handler`` as a task; return (task, slot) once its request waits in the busy engine's inbox."""
    await world.start_holder()
    before = len(world.http_slots)
    task = asyncio.create_task(handler(_payload(), _ConnectedRequest(world.mgr)))
    try:
        await wait_until(lambda: len(world.http_slots) > before, timeout=5.0)
        slot = world.http_slots[before]
        await wait_until(lambda: bool(world.places(slot)), timeout=5.0)
    except BaseException:
        task.cancel()
        raise
    assert slot.state not in _STARTED, "setup: the request already started"
    assert world.claim_rows(slot), "setup: the waiting request holds no claim"
    assert not slot.disconnect_event.is_set(), "setup: the disconnect event is already set"
    assert slot.is_evicted is False, "setup: the request is already marked evicted"
    return task, slot


async def _first_seen(world, slot):
    """Sample every 50 ms for up to the observe window; return the seconds at which each property first held."""
    checks = {
        "flag_set": lambda: slot.disconnect_event.is_set(),
        "marked_evicted": lambda: slot.is_evicted is True,
        "left_queue": lambda: not world.places(slot),
        "claim_released": lambda: not world.claim_rows(slot),
    }
    t0 = time.monotonic()
    first = {}
    while len(first) < len(checks) and time.monotonic() - t0 < OBSERVE_WINDOW_S:
        for name, check in checks.items():
            if name not in first and check():
                first[name] = round(time.monotonic() - t0, 2)
        await asyncio.sleep(0.05)
    return first


@pytest.mark.parametrize("handler", ROUTES)
async def test_cancelling_a_non_streaming_request_that_still_waits_removes_it(tmp_path, handler):
    """The handler task is cancelled while its request waits behind a busy engine and the watcher
    reports a connected client. The request is flagged, leaves the engine's inbox, loses its claim,
    and when the busy turn ends no engine turn runs for it and no engine is started."""
    async with route_world(tmp_path) as world:
        task, slot = await _park_behind_the_busy_engine(world, handler)
        spawns_before = len(world.spawns)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        first = await _first_seen(world, slot)
        print(f"[{handler.__name__}] after the cancellation: {first}")
        turn_ran = await end_holder_turn_and_watch(world, slot)

        assert set(first) == {"flag_set", "marked_evicted", "left_queue", "claim_released"}, (
            f"after a cancellation of the waiting request, only {sorted(first)} held within "
            f"{OBSERVE_WINDOW_S}s; it still waits in {world.places(slot)}"
        )
        assert not turn_ran, "an engine turn ran for a request whose handler was cancelled"
        assert len(world.spawns) == spawns_before, "an engine was started for a cancelled request"


@pytest.mark.parametrize("handler", ROUTES)
async def test_a_non_streaming_request_that_completes_is_not_flagged_or_removed(tmp_path, handler):
    """The control: the same request, never cancelled, waits behind the busy engine, is served when
    the busy turn ends and answers. Its disconnect event stays unset and it is never marked evicted."""
    async with route_world(tmp_path) as world:
        task, slot = await _park_behind_the_busy_engine(world, handler)
        await asyncio.sleep(_SETTLE_S)
        still_waiting = world.places(slot)
        flag_while_waiting = slot.disconnect_event.is_set()

        served = await end_holder_turn_and_watch(world, slot, window_s=SERVED_WINDOW_S)
        result = await asyncio.wait_for(task, timeout=SERVED_WINDOW_S)

        assert still_waiting, "the request left the queue while its handler was running"
        assert flag_while_waiting is False, "the disconnect event of a request that stays was set"
        assert served, "the request got no engine turn after the busy turn ended"
        assert isinstance(result, dict) and "error" not in result, f"unexpected answer: {result!r}"
        assert slot.disconnect_event.is_set() is False and slot.is_evicted is False


# --------------------------------------------------------------------------------------
# the co-residence re-check: the client leaves between the gather and the take
# --------------------------------------------------------------------------------------

def _recheck_host(shape):
    """A host with one engine and one old waiting request, in the queue or in the engine's inbox."""
    event = asyncio.Event()
    waiting = FakeSlot(slot_id="w", created_at=970.0, disconnect_event=event)
    if shape == "staged":
        host = FakeHost(residents=[FakeResident()], staged=[waiting], now=1000.0)
    else:
        host = FakeHost(residents=[FakeResident().put(waiting)], now=1000.0)
    host.cap = 3
    return host, waiting, event


def _hang_up_when_taken(host, shape, event):
    """Set the disconnect event at the moment the request has been taken out of its container."""
    if shape == "staged":
        real_remove = host.queue.remove

        async def remove(slot_id):
            taken = await real_remove(slot_id)
            event.set()
            return taken

        host.queue.remove = remove
    else:
        real_unpark = host._unpark_fastlane_claim

        def unpark(slot):
            real_unpark(slot)
            event.set()

        host._unpark_fastlane_claim = unpark


@pytest.mark.parametrize("shape", ["staged", "inbox"])
async def test_recheck_reserves_no_engine_for_a_request_whose_client_leaves_as_it_is_taken(shape):
    """The request is live when it is gathered and its client is gone when the re-check holds it. It is
    handed to the host to be failed, and no engine is reserved for it."""
    host, waiting, event = _recheck_host(shape)
    _hang_up_when_taken(host, shape, event)

    await co_residence.run_recheck(host)

    assert event.is_set(), "setup: the hang-up never happened"
    assert waiting.is_evicted is False, "setup: the departure is signalled by the event alone"
    assert "_reserve_and_start_locked" not in host.calls, "an engine was reserved for a request whose client had gone"
    assert host.args.get("_handle_unloaded_slot") == [(waiting,)], "the gone request was not handed to the host to fail"


@pytest.mark.parametrize("shape", ["staged", "inbox"])
async def test_recheck_reserves_an_engine_for_a_request_whose_client_stays_as_it_is_taken(shape):
    """The control: the same flow with the client staying. The engine is reserved for the request and
    nothing is failed."""
    host, waiting, event = _recheck_host(shape)

    await co_residence.run_recheck(host)

    assert not event.is_set()
    reserved = host.args.get("_reserve_and_start_locked", [(None,)])[0][0]
    assert reserved is waiting, "no engine was reserved for the request"
    assert "_handle_unloaded_slot" not in host.calls


# --------------------------------------------------------------------------------------
# "still waiting" for a request handed to an engine's inbox
# --------------------------------------------------------------------------------------

def _handed_off_manager():
    """A manager object with exactly the registry state the hand-off bookkeeping reads."""
    mgr = TurbohaulManager.__new__(TurbohaulManager)
    mgr._handed_off_waiting = {}
    mgr._handed_off_prune_at = 64
    return mgr


def _handed_off_slot(slot_id, *, created_at=10.0, event=None, evicted=False):
    return Slot(
        slot_id=slot_id, model_tag="m", state=SlotState.STAGED, created_at=created_at,
        client_meta={"ip": "10.0.0.7"}, is_evicted=evicted, disconnect_event=event,
    )


def _gone_by_event():
    event = asyncio.Event()
    event.set()
    return _handed_off_slot("gone", event=event)


def _gone_by_flag():
    return _handed_off_slot("gone", event=asyncio.Event(), evicted=True)


GONE = [pytest.param(_gone_by_event, id="event_set"), pytest.param(_gone_by_flag, id="marked_evicted")]


@pytest.mark.parametrize("make", GONE)
def test_a_handed_off_request_whose_client_is_gone_is_not_counted_as_waiting(make):
    mgr = _handed_off_manager()
    assert mgr._handed_off_still_waiting(make()) is False


def test_a_handed_off_request_whose_client_is_connected_is_counted_as_waiting():
    """The control of the test above."""
    mgr = _handed_off_manager()
    assert mgr._handed_off_still_waiting(_handed_off_slot("live", event=asyncio.Event())) is True
    assert mgr._handed_off_still_waiting(_handed_off_slot("live-no-event")) is True


@pytest.mark.parametrize("make", GONE)
def test_a_returning_turn_count_leaves_out_a_waiting_request_whose_client_is_gone(make):
    """Two requests of one address wait in an engine's inbox, both newer than the grace start. Only the
    one whose client is there is a returning turn."""
    mgr = _handed_off_manager()
    gone, live = make(), _handed_off_slot("live", event=asyncio.Event())
    for slot in (gone, live):
        mgr._note_handed_off_waiting(slot)

    assert mgr._inbox_returning_turns("10.0.0.7", 1.0) == 1


def test_a_returning_turn_count_includes_every_waiting_request_whose_client_is_connected():
    """The control of the test above: with both clients there, both count."""
    mgr = _handed_off_manager()
    for name in ("a", "b"):
        mgr._note_handed_off_waiting(_handed_off_slot(name, event=asyncio.Event()))

    assert mgr._inbox_returning_turns("10.0.0.7", 1.0) == 2


@pytest.mark.parametrize("make", GONE)
def test_the_hand_off_sweep_drops_a_request_whose_client_is_gone_and_keeps_a_live_one(make):
    mgr = _handed_off_manager()
    gone, live = make(), _handed_off_slot("live", event=asyncio.Event())
    mgr._handed_off_waiting = {gone.slot_id: gone, live.slot_id: live}

    mgr._prune_handed_off_waiting()

    assert list(mgr._handed_off_waiting) == ["live"]
